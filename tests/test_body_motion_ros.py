#!/usr/bin/env python3
"""Actual guard: relative motion can resume after an unobservable epoch reset."""
import json
import os
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


def main():
    assert os.environ['ROS_DOMAIN_ID']=='59'
    output=ROOT/'results/body_motion_guard_regression_20260916';output.mkdir(exist_ok=False)
    profile=yaml.safe_load((ROOT/'config/fusion_motion_candidate.yaml').read_text())
    assert profile['visual_constraint_mode']=='body_twist'
    profile['vision_gate']['recovery_frames']=3
    profile['visual_reference_wait']=2.
    path=output/'profile.yaml';path.write_text(yaml.safe_dump(profile))
    rclpy.init(args=['--ros-args','-p','profile_path:='+str(path),'-p','output_dir:='+str(output)])
    guard=AdaptiveGuard();driver=Node('body_motion_regression_driver')
    executor=SingleThreadedExecutor();executor.add_node(guard);executor.add_node(driver)
    lidar=driver.create_publisher(Odometry,'/fusion/lio_raw',30)
    visual=driver.create_publisher(Odometry,'/fusion/learned_raw',30)
    image_pubs=[driver.create_publisher(Image,name,3) for name in ['/fusion/left','/fusion/right']]
    outputs=[];lidar_outputs=[]
    driver.create_subscription(Odometry,'/fusion/vision_odom_guarded',outputs.append,30)
    driver.create_subscription(Odometry,'/lio/odom',lidar_outputs.append,30)
    def drain(seconds):
        until=time.monotonic()+seconds
        while time.monotonic()<until:executor.spin_once(timeout_sec=.004)
    def odom(stamp,position,cov,epoch):
        msg=Odometry();msg.header.stamp=stamp;msg.header.frame_id=epoch;msg.child_frame_id='base_link'
        msg.pose.pose.position.x=position;msg.pose.pose.orientation.w=1.
        msg.pose.covariance=np.diag(cov).ravel().tolist()
        return msg
    start=driver.get_clock().now().nanoseconds/1e9
    def frame(epoch,offset=0.,black=False,partial=False,anchor=False,jump=0.):
        stamp=driver.get_clock().now().to_msg()
        elapsed=stamp.sec+stamp.nanosec*1e-9-start
        msg=Image();msg.header.stamp=stamp;msg.height=192;msg.width=256;msg.encoding='mono8';msg.step=256
        y,x=np.indices((192,256))
        array=np.zeros((192,256),np.uint8) if black else np.where((x//12+y//12)%2,50,200).astype(np.uint8)
        msg.data=array.tobytes()
        for pub in image_pubs:pub.publish(msg)
        drain(.02)
        cov=[1000.,1000.,.01,.01,.01,1.] if partial else [.04]*3+[.01]*3
        lidar.publish(odom(stamp,.2*elapsed,cov,'odom'));drain(.02)
        if not black:
            cov=[1e6]*6 if anchor else (np.asarray([.01]*3+[.005]*3)/.6**2).tolist()
            visual.publish(odom(stamp,offset+.2*elapsed+jump,cov,epoch))
        drain(.075)
    try:
        drain(.6)
        frame('learned_epoch_1',anchor=True)
        for _ in range(10):frame('learned_epoch_1')
        assert len(outputs)>=7
        before=len(outputs)
        for _ in range(5):frame('learned_epoch_1',black=True)
        assert len(outputs)==before and guard.lidar_usable
        frame('learned_epoch_2',offset=-700.,partial=True,anchor=True)
        for _ in range(10):frame('learned_epoch_2',offset=-700.,partial=True)
        assert len(outputs)>=before+7 and guard.visual_usable
        assert not guard.lidar_covariance_reliable
        before_jump=len(outputs)
        frame('learned_epoch_2',offset=-700.,partial=True,jump=500.)
        assert len(outputs)==before_jump
        for _ in range(5):frame('learned_epoch_2',offset=-700.,partial=True)
        assert len(outputs)>before_jump
        values=[]
        for msg in outputs:
            assert msg.header.frame_id in ('learned_epoch_1','learned_epoch_2')
            assert min(np.diag(np.array(msg.pose.covariance).reshape(6,6)))>=1e5
            vector=[msg.twist.twist.linear.x,msg.twist.twist.linear.y,msg.twist.twist.linear.z,
                    msg.twist.twist.angular.x,msg.twist.twist.angular.y,msg.twist.twist.angular.z]
            np.testing.assert_allclose(vector,[.2,0.,0.,0.,0.,0.],atol=1e-5)
            assert np.linalg.eigvalsh(np.asarray(msg.twist.covariance).reshape(6,6)).min()>0
            values.append(vector)
        result=dict(passed=True,motion_constraints=len(outputs),lidar_constraints=len(lidar_outputs),
                    blackout_closes_visual=True,new_epoch_without_absolute_lidar_reference_recovers=True,
                    pose_jump_does_not_poison_recovery=True,absolute_visual_pose_inactive=True,
                    max_velocity_error_mps=max(abs(v[0]-.2) for v in values),
                    qualification='Actual ROS guard and transport with synthetic motion; not a scene-accuracy test')
        (output/'verification.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
    finally:
        executor.remove_node(guard);executor.remove_node(driver)
        guard.destroy_node();driver.destroy_node();rclpy.shutdown()


if __name__=='__main__':main()
