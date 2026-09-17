"""A slower, co-timed EKF can recover from visual output without backward TF."""
import json
import os
from pathlib import Path
import tempfile
import time
import numpy as np
import rclpy
from nav_msgs.msg import Odometry
from rclpy.executors import SingleThreadedExecutor
from tf2_msgs.msg import TFMessage
import yaml
from t3_lidar_visual_fusion.adaptive_guard import AdaptiveGuard

ROOT=Path(__file__).resolve().parents[1]


def main():
    assert os.environ['ROS_DOMAIN_ID']=='81' and os.environ['ROS_LOCALHOST_ONLY']=='1'
    with tempfile.TemporaryDirectory() as tmp:
        cfg=yaml.safe_load((ROOT/'config/hardware104.yaml').read_text())
        cfg['stationary']['enabled']=False
        profile=Path(tmp)/'profile.yaml';profile.write_text(yaml.safe_dump(cfg))
        rclpy.init(args=['--ros-args','-p','profile_path:='+str(profile)])
        guard=AdaptiveGuard();driver=rclpy.create_node('delayed_source_fixture')
        ex=SingleThreadedExecutor();ex.add_node(guard);ex.add_node(driver)
        poses=[];tfs=[]
        driver.create_subscription(Odometry,'/T3/semantic/current_pose',poses.append,30)
        driver.create_subscription(TFMessage,'/tf',lambda m:tfs.extend(m.transforms),30)
        def spin(seconds=.04):
            until=time.monotonic()+seconds
            while time.monotonic()<until:ex.spin_once(timeout_sec=.001)
        def msg(t,x):
            m=Odometry();m.header.stamp.sec,m.header.stamp.nanosec=divmod(round(t*1e9),10**9)
            m.header.frame_id='odom';m.child_frame_id='base_link';m.pose.pose.orientation.w=1.
            m.pose.pose.position.x=float(x)
            m.pose.covariance=np.diag([.0025]*3+[.0004]*3).ravel().tolist();return m
        try:
            spin(.5);t=time.time()-2.
            guard.lio(msg(t,0.))
            guard.visual_usable=True;guard.gate.enabled=True;guard.last_visual_usable_wall=time.monotonic()
            guard.visual_candidate=msg(t+.8,.08)
            guard.filtered(guard.visual_candidate,source='visual');spin()
            before=guard.last_filter_stamp;assert len(poses)==len(tfs)==1
            guard.ekf_candidate=None
            guard.filtered(msg(t,50.));assert guard.ekf_candidate is None
            guard.filtered(msg(t,0.));spin()
            assert guard.ekf_candidate is not None
            assert guard.last_filter_stamp==before and len(poses)==len(tfs)==1
            assert not guard.select_visual_output(),'Fast visual output latched out the recovered EKF'
            guard.lio(msg(t+1.,.1));guard.filtered(msg(t+1.,.1));spin()
            assert guard.output_source=='ekf' and len(poses)==len(tfs)==2
            guard.lidar_usable=False
            assert guard.select_visual_output(),'A later LiDAR outage must still release visual fallback'
            print(json.dumps(dict(passed=True,older_ekf_proves_recovery=True,
                backward_pose_tf_blocked=True,bad_older_ekf_rejected=True,
                later_lidar_outage_preserves_visual_fallback=True)))
        finally:
            ex.shutdown();guard.destroy_node();driver.destroy_node();rclpy.try_shutdown()


if __name__=='__main__':main()
