#!/usr/bin/env python3
"""Collect independent ROS interface evidence for a running fusion pipeline."""
import argparse,json,time
from pathlib import Path
from collections import Counter,deque
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile,DurabilityPolicy,ReliabilityPolicy
from nav_msgs.msg import Odometry,Path as RosPath
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import String
from tf2_msgs.msg import TFMessage
from grid_map_msgs.msg import GridMap
from t3_interfaces.msg import IncrementalSemanticMap
from visual_evidence import summarize_visual_evidence


def main():
    a=argparse.ArgumentParser();a.add_argument("--seconds",type=float,default=90)
    a.add_argument("--report",required=True)
    a.add_argument("--visual-source",choices=["vins","learned","none","roma_external"],default="vins")
    a.add_argument("--ready-file",default="");a.add_argument("--stop-file",default="")
    x=a.parse_args()
    rclpy.init();n=Node("fusion_interface_verifier")
    counts=Counter();details={};states=deque(maxlen=6000);pose=deque(maxlen=12000);edges=set();static_edges=set();frames=set()
    measurements=deque(maxlen=20000);raw_poses=deque(maxlen=6000);started=time.monotonic()
    tf_at={};fused_at={}
    def retain(table,key,value):
        table[key]=value
        while len(table)>3000:table.pop(next(iter(table)))

    qos=QoSProfile(depth=1,durability=DurabilityPolicy.TRANSIENT_LOCAL,reliability=ReliabilityPolicy.RELIABLE)
    def odometry(key,m):
        counts[key]+=1
        if key in ("lio_raw","learned_raw","vins_raw","roma_raw"):
            p=m.pose.pose.position;q=m.pose.pose.orientation
            raw_poses.append(dict(source=key,stamp_sec=m.header.stamp.sec+m.header.stamp.nanosec*1e-9,
                frame=m.header.frame_id,child=m.child_frame_id,pose=[p.x,p.y,p.z,q.x,q.y,q.z,q.w],pose_covariance=list(m.pose.covariance)))
        if key in ("lio_guarded","vision_accepted","fused","ekf_output"):
            evidence=dict(source=key,stamp_sec=m.header.stamp.sec+m.header.stamp.nanosec*1e-9,
                          received_monotonic_sec=time.monotonic())
            if key in ("vision_accepted", "lio_guarded", "ekf_output"):
                p=m.pose.pose.position;q=m.pose.pose.orientation
                evidence["pose"]=[p.x,p.y,p.z,q.x,q.y,q.z,q.w]
                evidence["pose_covariance"]=list(m.pose.covariance)
                evidence["pose_covariance_diagonal"]=[float(m.pose.covariance[i*7]) for i in range(6)]
                v=m.twist.twist.linear;w=m.twist.twist.angular
                evidence["twist"]=[v.x,v.y,v.z,w.x,w.y,w.z]
                evidence["twist_covariance"]=list(m.twist.covariance)
                evidence["twist_covariance_diagonal"]=[float(m.twist.covariance[i*7]) for i in range(6)]
            measurements.append(evidence)
        if key=="fused":
            p=m.pose.pose.position;q=m.pose.pose.orientation
            pose.append([m.header.stamp.sec+m.header.stamp.nanosec*1e-9,p.x,p.y,p.z,q.x,q.y,q.z,q.w])
            frames.add(m.header.frame_id)
            retain(fused_at,m.header.stamp.sec*1000000000+m.header.stamp.nanosec,[p.x,p.y,p.z,q.x,q.y,q.z,q.w])
    for key,topic in [("lio_raw","/fusion/lio_raw"),("vins_raw","/fusion/vins_raw"),
                      ("learned_raw","/fusion/learned_raw"),("roma_raw","/fusion/roma_raw"),
                      ("lio_guarded","/lio/odom"),("vision_accepted","/fusion/vision_odom_guarded"),
                      ("ekf_output","/fusion/ekf"),
                      ("fused","/T3/semantic/current_pose")]:
        n.create_subscription(Odometry,topic,lambda m,k=key:odometry(k,m),100)
    def path(m):counts["path"]+=1;details["path_poses"]=len(m.poses)
    n.create_subscription(RosPath,"/T3/semantic/trajectory",path,qos)
    def cloud(m):
        counts["cloud"]+=1;details["cloud_points"]=m.width*m.height;frames.add(m.header.frame_id)
    n.create_subscription(PointCloud2,"/T3/mapping/lidar_map",cloud,qos)
    def grid(m):
        counts["grid"]+=1;details["layers"]=list(m.layers);frames.add(m.header.frame_id)
        i=m.layers.index("elevation")
        a=np.asarray(m.data[i].data)
        details["finite_elevation_cells"]=int(np.isfinite(a).sum())
        if np.isfinite(a).any():details["height_range"]=[float(np.nanmin(a)),float(np.nanmax(a))]
    n.create_subscription(GridMap,"/T3/mapping/elevation_map",grid,qos)
    def incremental(m):
        counts["incremental"]+=1
        details["incremental_type"]="t3_interfaces/msg/IncrementalSemanticMap"
        details["incremental_shape"]=[m.width,m.height]
        details["incremental_arrays_valid"]=all(len(v)==m.width*m.height for v in [
            m.elevation,m.elevation_variance,m.occupancy,m.semantic,m.observation_count])
    n.create_subscription(IncrementalSemanticMap,"/T3/semantic/incremental_map",incremental,qos)
    def status(m):
        d=json.loads(m.data);d["wall_time"]=time.time();states.append(d)
    n.create_subscription(String,"/fusion/status",status,100)
    n.create_subscription(String,"/fusion/map_status",lambda m:details.update(map_status=json.loads(m.data)),30)
    n.create_subscription(String,"/fusion/sensor_status",lambda m:details.update(sensor_status=json.loads(m.data)),10)
    def dynamic_tf(m):
        for t in m.transforms:
            edges.add((t.header.frame_id,t.child_frame_id))
            if (t.header.frame_id,t.child_frame_id)==("odom","base_link"):
                p=t.transform.translation;q=t.transform.rotation
                retain(tf_at,t.header.stamp.sec*1000000000+t.header.stamp.nanosec,[p.x,p.y,p.z,q.x,q.y,q.z,q.w])
    n.create_subscription(TFMessage,"/tf",dynamic_tf,100)
    n.create_subscription(TFMessage,"/tf_static",lambda m:static_edges.update(
        (t.header.frame_id,t.child_frame_id) for t in m.transforms),\
        QoSProfile(depth=100,durability=DurabilityPolicy.TRANSIENT_LOCAL,reliability=ReliabilityPolicy.RELIABLE))
    if x.ready_file:Path(x.ready_file).write_text("ready\n")
    deadline=time.monotonic()+x.seconds
    while time.monotonic()<deadline:
        if x.stop_file and Path(x.stop_file).exists():break
        rclpy.spin_once(n,timeout_sec=.1)
    publishers=n.get_publishers_info_by_topic("/tf")
    checks={"lio_ran":counts["lio_raw"]>5,"fused_pose":counts["fused"]>5,
            "trajectory":details.get("path_poses",0)>5,"elevation":details.get("finite_elevation_cells",0)>100,
            "lidar_map":details.get("cloud_points",0)>1000,"incremental_interface":details.get("incremental_arrays_valid",False),
            "finite_poses":bool(pose) and bool(np.isfinite(np.asarray(pose)).all()),
            "consistent_frames":frames=={"odom","map"} and ("map","odom") in static_edges,
            "only_ekf_main_tf":edges=={("odom","base_link")} and len(publishers)==1}
    shared=sorted(set(tf_at)&set(fused_at))
    tf_position_error=max((np.linalg.norm(np.asarray(tf_at[k][:3])-fused_at[k][:3]) for k in shared),default=float("inf"))
    tf_orientation_error=max((min(np.linalg.norm(np.asarray(tf_at[k][3:])-fused_at[k][3:]),
        np.linalg.norm(np.asarray(tf_at[k][3:])+fused_at[k][3:])) for k in shared),default=float("inf"))
    details["tf_odometry_consistency"]=dict(matched_stamps=len(shared),
        max_position_difference_m=float(tf_position_error),max_quaternion_difference=float(tf_orientation_error))
    checks["tf_matches_published_pose"]=bool(len(shared)>5 and tf_position_error<1e-7 and tf_orientation_error<1e-7)
    visual_raw={"vins":"vins_raw","learned":"learned_raw","roma_external":"roma_raw"}.get(x.visual_source)
    visual_checks={"requested_source":x.visual_source,
        "raw_pose_observed":bool(visual_raw and counts[visual_raw]>1),
        "guard_accepted":counts["vision_accepted"]>0,
        "fusion_active_observed":any(s.get("operating_mode")=="lidar_visual" for s in states)}
    visual_checks=summarize_visual_evidence(dict(visual_checks=visual_checks,
        measurements=list(measurements),counts=dict(counts),states=list(states)))
    report={"visual_checks":visual_checks,"measurements":list(measurements),
        "raw_poses":list(raw_poses),"observed_frames":sorted(frames),"observer_duration_sec":time.monotonic()-started,"checks":checks,"counts":dict(counts),"details":details,"tf_edges":list(edges),
            "tf_publishers":[{"name":p.node_name,"namespace":p.node_namespace} for p in publishers],
            "states":list(states),"poses":list(pose),"static_tf_edges":list(static_edges)}
    p=Path(x.report);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(report,indent=2))
    print(json.dumps({"checks":checks,"counts":dict(counts),"details":details},indent=2),flush=True)
    n.destroy_node();rclpy.shutdown()
    if not all(checks.values()):raise SystemExit(1)


if __name__=="__main__":main()
