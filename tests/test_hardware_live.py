#!/usr/bin/env python3
"""Bounded passive real-sensor integration check in a separate ROS domain.

Never sends vehicle commands. Only processes started by this test are stopped.
An input driver already running in the sensor domain is reused.
"""
import argparse, collections, json, os, signal, subprocess, time
from pathlib import Path
import numpy as np
import rclpy
from rclpy.context import Context
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image,Imu,PointCloud2
from nav_msgs.msg import Odometry
from std_msgs.msg import String
from grid_map_msgs.msg import GridMap

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--seconds',type=float,default=65.);parser.add_argument('--output',required=True)
    args=parser.parse_args();out=Path(args.output).resolve();out.mkdir(parents=True,exist_ok=False)
    domain=int(os.environ['ROS_DOMAIN_ID'])
    if domain in (10,19,57):raise ValueError('Choose a separate test domain, such as 73')
    source_context=Context();rclpy.init(context=source_context,domain_id=19)
    source=Node('p3_camera_precheck',context=source_context)
    source_executor=SingleThreadedExecutor(context=source_context);source_executor.add_node(source)
    deadline=time.monotonic()+2
    while time.monotonic()<deadline:source_executor.spin_once(timeout_sec=.1)
    cameras=[source.count_publishers(t)>0 for t in ['/Car/T5/Cam_Left/image_mono/mapping','/Car/T5/Cam_Right/image_mono/mapping']]
    legacy=any(source.count_publishers(t) for t in ['/Car/T5/Cam_Left/image_raw/color','/Car/T5/Cam_Right/image_raw/color'])
    source_executor.shutdown();source.destroy_node();source_context.shutdown()
    if any(cameras) and not all(cameras):raise RuntimeError('Only one camera publisher exists')
    if not any(cameras) and legacy:raise RuntimeError('Existing camera driver lacks compact mapping topics')
    camera_present=all(cameras)
    rclpy.init();node=Node('p3_hardware_live_observer');counts=collections.Counter();last={};poses=[];ages=collections.defaultdict(list)
    def status(msg,key):
        counts[key]+=1
        try:last[key]=json.loads(msg.data)
        except ValueError:last[key]={'invalid_json':True}
    def receive(msg,key):
        counts[key]+=1
        t=msg.header.stamp.sec+msg.header.stamp.nanosec*1e-9
        if len(ages[key])<4000:ages[key].append(time.time()-t)
        if key=='fused':
            p=msg.pose.pose.position;poses.append([t,p.x,p.y,p.z])
    for key,typ,topic in [('imu',Imu,'/fusion/imu'),('cloud',PointCloud2,'/fusion/lidar'),
            ('deskewed',PointCloud2,'/fusion/lidar_deskewed'),('left',Image,'/fusion/left'),('right',Image,'/fusion/right'),
            ('fused',Odometry,'/T3/semantic/current_pose')]:
        node.create_subscription(typ,topic,lambda m,k=key:receive(m,k),qos_profile_sensor_data)
    node.create_subscription(GridMap,'/Car/T3/mapping/grid_map',lambda m:counts.update(['gridmap']),qos_profile_sensor_data)
    for key,topic in [('imu_status','/fusion/imu_status'),('fusion','/fusion/status'),('hardware','/fusion/hardware_status'),('mapping','/fusion/map_status')]:
        node.create_subscription(String,topic,lambda m,k=key:status(m,k),10)
    processes=[]
    def launch(name,command):
        log=(out/(name+'.log')).open('w')
        proc=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        processes.append((name,proc,log));return proc
    try:
        if not camera_present:launch('camera',['bash',str(ROOT/'scripts/start_hardware_camera.sh')])
        pipeline=launch('pipeline',['ros2','launch',str(ROOT/'scripts/fusion.launch.py'),
            'profile:='+str(ROOT/'config/hardware104.yaml'),'output_dir:='+str(out)])
        deadline=time.monotonic()+args.seconds
        while time.monotonic()<deadline:
            rclpy.spin_once(node,timeout_sec=.05)
            if pipeline.poll() is not None:break
        lidar=[]
        if (out/'lidar_metrics.jsonl').exists():
            for line in (out/'lidar_metrics.jsonl').read_text().splitlines():
                try:lidar.append(json.loads(line))
                except ValueError:pass
        passed=(counts['deskewed']>=10 and counts['fused']>=10 and counts['gridmap']>=3 and
                any(r.get('imu',{}).get('mode')=='imu' for r in lidar) and pipeline.poll() is None)
        report=dict(passed=passed,domain=domain,counts=dict(counts),last=last,
            vehicle_commands_published=0,replay_used=False,camera_reused=camera_present,
            message_age_sec={k:{'median':float(np.median(v)),'p95':float(np.quantile(v,.95))} for k,v in ages.items() if v},
            lidar_frames=len(lidar),lidar_valid=sum(r['valid_update'] for r in lidar),
            lidar_imu_scans=sum(r.get('imu',{}).get('mode')=='imu' for r in lidar),
            lidar_processing_median_sec=float(np.median([r['processing_sec'] for r in lidar])) if lidar else None,
            pose_displacement_m=float(np.linalg.norm(np.asarray(poses[-1][1:])-np.asarray(poses[0][1:]))) if poses else None,
            note='Passive interface/static sample check, not moving trajectory ATE or a no-jump guarantee')
        (out/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2),flush=True)
    finally:
        for name,proc,log in reversed(processes):
            if proc.poll() is None:
                os.kill(proc.pid,signal.SIGINT)
                try:proc.wait(timeout=25)
                except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGTERM);proc.wait(timeout=10)
            log.close()
        node.destroy_node();rclpy.shutdown()
    return 0 if passed else 1


if __name__=='__main__':raise SystemExit(main())
