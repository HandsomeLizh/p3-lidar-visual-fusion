#!/usr/bin/env python3
"""Synthetic fault injection into the actual multi-process guard/EKF/map stack."""
from pathlib import Path
import copy,json,os,signal,subprocess,time
import numpy as np
import yaml
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image,PointCloud,PointCloud2
from geometry_msgs.msg import Point32
from nav_msgs.msg import Odometry
from std_msgs.msg import Header,String
from std_srvs.srv import Trigger
from tf2_msgs.msg import TFMessage
from cv_bridge import CvBridge
from t3_lidar_visual_fusion.ros_utils import xyz_cloud,stamp_sec
ROOT=Path(__file__).resolve().parents[1]


def main():
    out=ROOT/"results"/time.strftime("fault_test_%Y%m%d_%H%M%S");out.mkdir()
    cfg=yaml.safe_load((ROOT/"config/simulation.yaml").read_text())
    cfg.update({"base_from_lidar":np.eye(4).tolist(),"base_from_imu":np.eye(4).tolist(),
                "visual_source":"vins","visual_odometry_topic":"/fusion/vins_raw",
                "adaptive_source_selection":False,"require_vins_features":True,
                "source_wall_timeout":1.,"map_publish_period":.5,"map_window":16.,
                "max_elevation_points":3000,"pose_tolerance":.12})
    cfg["vision_gate"].update({"max_gap":.4,"max_step":1.,"recovery_frames":4})
    profile=out/"profile.yaml";profile.write_text(yaml.safe_dump(cfg))
    log=(out/"pipeline.log").open("w")
    pipeline=subprocess.Popen(["ros2","launch",str(ROOT/"scripts/fusion.launch.py"),
         "external_estimates:=true","profile:="+str(profile),"output_dir:="+str(out)],
         stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    rclpy.init();n=Node("synthetic_fault_source")
    lio=n.create_publisher(Odometry,"/fusion/lio_raw",10)
    vis=n.create_publisher(Odometry,"/fusion/vins_raw",10)
    images=[n.create_publisher(Image,t,3) for t in ["/fusion/left","/fusion/right"]]
    features=n.create_publisher(PointCloud,"/fusion/vins_features",10)
    cloud=n.create_publisher(PointCloud2,"/fusion/lidar",3)
    outputs=[];tf_edges=set();states=[];maps=[];current={}
    n.create_subscription(Odometry,"/T3/semantic/current_pose",lambda m:outputs.append((stamp_sec(m),m.pose.pose.position.x)),100)
    n.create_subscription(TFMessage,"/tf",lambda m:tf_edges.update((t.header.frame_id,t.child_frame_id) for t in m.transforms),100)
    def status(m):
        d=json.loads(m.data);current.update(d);states.append(dict(d,wall=time.time()))
    n.create_subscription(String,"/fusion/status",status,100)
    n.create_subscription(String,"/fusion/map_status",lambda m:maps.append(dict(json.loads(m.data),wall=time.time())),100)
    rng=np.random.default_rng(17);texture=rng.integers(10,240,(240,320),dtype=np.uint8);bridge=CvBridge()
    def spin(duration):
        until=time.monotonic()+duration
        while time.monotonic()<until:rclpy.spin_once(n,timeout_sec=.005)
    try:
        deadline=time.monotonic()+25
        while not current and time.monotonic()<deadline:
            if pipeline.poll() is not None:raise RuntimeError((out/"pipeline.log").read_text()[-4000:])
            spin(.1)
        if not current:raise RuntimeError("Test pipeline did not become ready")
        start=n.get_clock().now().nanoseconds/1e9
        for i in range(120):
            began=time.monotonic()
            h=Header();h.stamp=n.get_clock().now().to_msg();h.frame_id="odom"
            stamp=h.stamp.sec+h.stamp.nanosec*1e-9;x=.15*(stamp-start)
            for pub in images:
                m=bridge.cv2_to_imgmsg(np.zeros_like(texture) if 35<=i<65 else texture,encoding="mono8")
                m.header=copy.deepcopy(h);pub.publish(m)
            feat=PointCloud();feat.header=copy.deepcopy(h)
            feat.points=[Point32(x=float(k),y=0.,z=1.) for k in range(50)]
            features.publish(feat);spin(.015)
            od=Odometry();od.header=copy.deepcopy(h);od.child_frame_id="base_link"
            od.pose.pose.orientation.w=1.;od.pose.pose.position.x=x;lio.publish(od)
            vo=copy.deepcopy(od);vo.pose.pose.position.x=x+(1000. if i>=65 else 0.)
            if i==25:vo.pose.pose.position.x+=10.
            vis.publish(vo)
            if i%2==0:
                xx,yy=np.meshgrid(np.linspace(-4,4,45),np.linspace(-4,4,45))
                points=np.column_stack([xx.ravel()-x,yy.ravel(),(.1*np.sin(xx)).ravel()])
                cloud.publish(xyz_cloud(points,h))
            spin(max(.001,.1-(time.monotonic()-began)))
        spin(.5)
        client=n.create_client(Trigger,"/T3/mapping/save")
        assert client.wait_for_service(timeout_sec=5)
        future=client.call_async(Trigger.Request())
        deadline=time.monotonic()+20
        while not future.done() and time.monotonic()<deadline:spin(.05)
        saved=bool(future.done() and future.result().success)
        (out/"pose_samples.json").write_text(json.dumps(outputs))
        errors=[abs(x-.15*(stamp-start)) for stamp,x in outputs]
        # Subtract physically expected motion, allowing variable callback cadence.
        max_jump=max([abs((b[1]-a[1])-.15*(b[0]-a[0])) for a,b in zip(outputs,outputs[1:])],default=1e9)
        dark=[s for s in states if s["vision_reason"]=="underexposed"]
        map_before=next((s for s in maps if s["wall"]>start+4.0),None)
        map_after=next((s for s in maps if s["wall"]>start+6.0),None)
        checks={
          "has_filtered_pose":len(outputs)>=90,
          "blackout_rejects_vision":len(dark)>=2 and dark[-1]["vins_accepted"]==dark[0]["vins_accepted"],
          "blackout_keeps_odometry":len(dark)>=2 and dark[-1]["filtered"]>dark[0]["filtered"],
          "blackout_keeps_map":bool(map_before and map_after and map_after["mapped_scans"]>map_before["mapped_scans"]),
          "vision_recovers_after_1000m_reset":bool(dark and current["vins_accepted"]>dark[-1]["vins_accepted"]+10),
          "no_pose_jump":max_jump<.1,
          "position_accuracy":max(errors,default=1e9)<.15,
          "single_main_tf_edge":tf_edges=={("odom","base_link")} and len(n.get_publishers_info_by_topic("/tf"))==1,
          "lidar_checkpoint_saved":saved and (out/"_internal/lidar_voxels.sqlite").stat().st_size>1000,
          "elevation_saved":(out/"local_elevation_latest.npz").exists()}
        result={"synthetic_test":True,"checks":checks,"max_position_error_m":max(errors,default=None),
                "max_motion_residual_m":max_jump,"pose_count":len(outputs),"counters":current,
                "map":maps[-1] if maps else {},"tf_edges":list(tf_edges),"start_stamp":start}
        (out/"report.json").write_text(json.dumps(result,indent=2))
        (out/"gate_trace.json").write_text(json.dumps(states,indent=2))
        print(json.dumps(result,indent=2),flush=True)
        if not all(checks.values()):raise AssertionError("Fault injection checks failed")
    finally:
        if pipeline.poll() is None:
            os.kill(pipeline.pid,signal.SIGINT)
            try:pipeline.wait(timeout=25)
            except subprocess.TimeoutExpired:os.killpg(pipeline.pid,signal.SIGTERM);pipeline.wait(timeout=5)
        n.destroy_node()
        if rclpy.ok():rclpy.shutdown()
        log.close()


if __name__=="__main__":main()
