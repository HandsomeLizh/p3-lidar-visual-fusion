"""Inject low-covariance jumps; qualified pose and TF must reject together."""
import copy
import json
import os
import sys
import time
from pathlib import Path
import numpy as np
import yaml
import rclpy
from nav_msgs.msg import Odometry
from tf2_msgs.msg import TFMessage
from rclpy.executors import SingleThreadedExecutor
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src/t3_lidar_visual_fusion'))
sys.path.insert(0,str(ROOT/'scripts'))
from t3_lidar_visual_fusion.adaptive_guard import AdaptiveGuard
from generate_config import generate

assert os.environ.get('ROS_DOMAIN_ID')=='68'
out=ROOT/'results/lidar_drift_20260916/output_regression';out.mkdir(exist_ok=True)
p=yaml.safe_load((ROOT/'config/simulation_live.yaml').read_text())
p.update(visual_source='none',source_wall_timeout=30.)
p['telemetry_motion']['enabled']=False;p['stationary']['enabled']=False
profile=out/'profile.yaml';profile.write_text(yaml.safe_dump(p))
generate(profile,out/'config')
assert not yaml.safe_load((out/'config/ekf.yaml').read_text())['ekf_filter_node']['ros__parameters']['publish_tf']
rclpy.init(args=['--ros-args','-p','profile_path:='+str(profile)])
guard=AdaptiveGuard();driver=rclpy.create_node('jump_output_test')
ex=SingleThreadedExecutor();ex.add_node(guard);ex.add_node(driver)
poses=[];tfs=[];lio=[]
driver.create_subscription(Odometry,'/T3/semantic/current_pose',poses.append,100)
driver.create_subscription(TFMessage,'/tf',lambda m:tfs.extend(t for t in m.transforms if t.child_frame_id=='base_link'),100)
driver.create_subscription(Odometry,'/lio/odom',lio.append,100)
filtered=driver.create_publisher(Odometry,'/fusion/ekf',30)
raw=driver.create_publisher(Odometry,'/fusion/lio_raw',30)
def spin(seconds=.12):
    until=time.monotonic()+seconds
    while time.monotonic()<until:ex.spin_once(timeout_sec=.005)
def message(t,x):
    m=Odometry();ns=round(t*1e9);m.header.stamp.sec,m.header.stamp.nanosec=divmod(ns,1000000000)
    m.header.frame_id='odom';m.child_frame_id='base_link';m.pose.pose.orientation.w=1.
    m.pose.pose.position.x=float(x);m.pose.covariance=np.diag([.01]*6).ravel().tolist();return m
try:
    spin(.8)
    raw.publish(message(100.,0.));spin();assert guard.lidar_usable
    filtered.publish(message(100.,0.));spin();assert len(poses)==len(tfs)==1
    filtered.publish(message(109.469,21.6648));spin()
    assert len(poses)==len(tfs)==1 and guard.filter_quality['reason']=='output_discontinuity'
    filtered.publish(message(101.31,4.572));spin()
    assert len(poses)==len(tfs)==1 and guard.filter_quality['reason']=='output_discontinuity'
    raw.publish(message(109.469,21.6648));spin()
    assert len(lio)==1 and guard.lidar_gate.reason=='recovery_discontinuity'
    for t,x in [(110.,.1),(111.,.2),(112.,.3)]:raw.publish(message(t,x));spin()
    assert guard.lidar_usable
    filtered.publish(message(112.,.3));spin();assert len(poses)==len(tfs)==2
    filtered.publish(message(112.,.5));spin();assert len(poses)==len(tfs)==3
    filtered.publish(message(112.,.9));spin()
    assert len(poses)==len(tfs)==3 and guard.filter_quality['reason']=='same_stamp_discontinuity'
    for m,t in zip(poses,tfs):
        assert m.header==t.header
        assert m.pose.pose.position.x==t.transform.translation.x
    guard.ekf_measurement_time_only=True
    before=len(poses)
    filtered.publish(message(114.,.51));spin()
    assert len(poses)==before and guard.last_filter_stamp==112.
    assert guard.ekf_predictions_ignored>=1
    raw.publish(message(113.,.51));spin()
    filtered.publish(message(113.,.51));spin()
    assert len(poses)==len(tfs)==before+1 and guard.last_filter_stamp==113.
    result=dict(passed=True,large_jump_blocked_in_pose_and_tf=True,same_stamp_cumulative_jump_blocked=True,
        long_gap_lidar_jump_blocked=True,nearby_recovery=True,qualified_outputs=len(poses),ekf_main_tf_disabled=True,
        idle_prediction_blocked=True,delayed_measurement_after_prediction_published=True)
    (out/'verification.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
finally:
    ex.shutdown();guard.destroy_node();driver.destroy_node();rclpy.shutdown()
