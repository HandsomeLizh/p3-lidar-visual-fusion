#!/usr/bin/env python3
"""Real ROS relay regression: flat-ground visual use, blackout, epoch recovery.

The synthetic filtered echo only supplies a common gauge for this guard test;
it is not a benchmark of EKF or real-scene localization accuracy.
"""
import copy
import json
import sys
import time
from pathlib import Path
import numpy as np
import yaml
import rclpy
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src/t3_lidar_visual_fusion'))
from t3_lidar_visual_fusion.adaptive_guard import AdaptiveGuard

output=ROOT/'results/adaptive_guard_arrival_regression_20260916'
output.mkdir(exist_ok=True)
profile=yaml.safe_load((ROOT/'config/selected_bag_adaptive.yaml').read_text())
profile['vision_gate']['recovery_frames']=3
profile['visual_wall_timeout_sec']=.8
profile['source_wall_timeout']=.8
profile['visual_reference_wait']=.04
path=output/'profile.yaml';path.write_text(yaml.safe_dump(profile))
rclpy.init(args=['--ros-args','-p','profile_path:='+str(path),'-p','output_dir:='+str(output)])
guard=AdaptiveGuard();driver=Node('adaptive_guard_regression_driver')
executor=SingleThreadedExecutor();executor.add_node(guard);executor.add_node(driver)
raw_lio=driver.create_publisher(Odometry,'/fusion/lio_raw',30)
raw_visual=driver.create_publisher(Odometry,'/fusion/learned_raw',30)
images=[driver.create_publisher(Image,t,3) for t in ['/fusion/left','/fusion/right']]
filtered=driver.create_publisher(Odometry,'/fusion/ekf',30)
poses=[];sources=[]
def relay(msg,source):
    sources.append((source,msg.pose.covariance[0]))
    filtered.publish(msg)
driver.create_subscription(Odometry,'/lio/odom',lambda m:relay(m,'lidar'),30)
driver.create_subscription(Odometry,'/fusion/vision_odom_guarded',lambda m:relay(m,'vision'),30)
driver.create_subscription(Odometry,'/T3/semantic/current_pose',lambda m:poses.append(m.pose.pose.position.x),30)
def drain(seconds):
    until=time.monotonic()+seconds
    while time.monotonic()<until:executor.spin_once(timeout_sec=.005)
def image(stamp,black):
    msg=Image();msg.header.stamp=stamp;msg.height=192;msg.width=256;msg.encoding='mono8';msg.step=256
    y,x=np.indices((192,256));a=np.zeros((192,256),np.uint8) if black else np.where(((x//12+y//12)%2)==0,50,200).astype(np.uint8)
    msg.data=a.tobytes();return msg
def odom(stamp,x,cov,epoch='odom'):
    msg=Odometry();msg.header.stamp=stamp;msg.header.frame_id=epoch;msg.child_frame_id='base_link'
    msg.pose.pose.position.x=x;msg.pose.pose.orientation.w=1.
    msg.pose.covariance=np.diag(np.asarray(cov,dtype=float)).reshape(-1).tolist();return msg
def stage(start,n,black,lidar_good,epoch,visual_offset=0.):
    for i in range(start,start+n):
        stamp=driver.get_clock().now().to_msg()
        for pub in images:pub.publish(image(stamp,black))
        drain(.025)
        raw_lio.publish(odom(stamp,i*.02,([.04]*3+[.01]*3) if lidar_good else [1000.]*6))
        drain(.015)
        if not black:raw_visual.publish(odom(stamp,visual_offset+i*.02,[.01]*3+[.005]*3,epoch))
        drain(.06)
drain(.5)
try:
    stage(0,18,False,True,'learned_epoch_1')
    assert guard.visual_constraints>=8 and guard.lidar_constraints>=8,(guard.counters,guard.adaptive_rejections)
    # Healthy visual input arrives first; force the guard to inspect it
    # before the same-stamp LiDAR callback rather than skip agreement checks.
    before_constraints=guard.visual_constraints
    stamp=driver.get_clock().now().to_msg()
    for pub in images:pub.publish(image(stamp,False))
    raw_visual.publish(odom(stamp,.35,[.01]*3+[.005]*3,'learned_epoch_1'))
    drain(.012);guard.process_visual()
    assert guard.visual_constraints==before_constraints
    raw_lio.publish(odom(stamp,.35,[.04]*3+[.01]*3));drain(.08)
    assert guard.visual_constraints==before_constraints+1
    first_alignment=guard.gate.alignment.copy()
    stage(18,18,True,True,'learned_epoch_1')
    assert not guard.visual_usable and guard.lidar_usable
    stage(36,18,False,True,'learned_epoch_1')
    assert guard.visual_usable and np.array_equal(first_alignment,guard.gate.alignment)
    assert abs(poses[-1]-53*.02)<1e-6,poses[-1]
    stage(54,18,False,True,'learned_epoch_9',-700.)
    assert guard.visual_usable and guard.visual_anchors==2
    assert abs(poses[-1]-71*.02)<1e-6,poses[-1]
    drain(.1);before=len(poses)
    stage(72,12,True,False,'learned_epoch_9',-700.)
    assert not any(guard.active_sources())
    assert len(poses)==before,(before,len(poses))
    stage(84,12,False,False,'learned_epoch_10',-900.)
    drain(.1)
    assert not guard.visual_usable
    assert guard.adaptive_rejections['unbridged_visual_epoch']>0
    assert len(poses)==before,(before,len(poses))
    stage(96,18,False,True,'learned_epoch_10',-900.)
    assert guard.visual_usable and guard.lidar_usable
    assert abs(poses[-1]-113*.02)<1e-6,poses[-1]
    assert all(0. <= x <= 113*.02+1e-6 for x in poses)
    result=dict(passed=True,visual_waits_for_same_stamp_lidar=True,same_epoch_alignment_preserved=True,
        reset_aligned_to_current_lidar=True,unobserved_reset_rejected=True,
        resumed_position_m=poses[-1],expected_position_m=113*.02,
        visual_anchors=guard.visual_anchors,lidar_constraints=guard.lidar_constraints,
        qualification='Synthetic ROS guard/transport test, not EKF accuracy')
    (output/'verification.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
finally:
    executor.remove_node(guard);executor.remove_node(driver);guard.destroy_node();driver.destroy_node();rclpy.shutdown()
