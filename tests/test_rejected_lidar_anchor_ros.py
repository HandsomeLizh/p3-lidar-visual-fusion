"""A rejected LiDAR/EKF frame must not reanchor independently tracked vision."""
import json,os,tempfile,time
from pathlib import Path
import numpy as np
import rclpy
from rclpy.executors import SingleThreadedExecutor
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image
from rosgraph_msgs.msg import Clock
import yaml
from t3_lidar_visual_fusion.adaptive_guard import AdaptiveGuard
from t3_lidar_visual_fusion.ros_utils import transform_from_pose

ROOT=Path(__file__).resolve().parents[1]


def main():
    assert os.environ.get('ROS_DOMAIN_ID')=='86' and os.environ.get('ROS_LOCALHOST_ONLY')=='1'
    with tempfile.TemporaryDirectory() as folder:
        cfg=yaml.safe_load((ROOT/'config/simulation_live.yaml').read_text())
        cfg['stationary']['enabled']=False;cfg['telemetry_motion']['enabled']=False
        cfg['vision_gate']['recovery_frames']=2
        profile=Path(folder)/'profile.yaml';profile.write_text(yaml.safe_dump(cfg))
        rclpy.init(args=['--ros-args','-p','profile_path:='+str(profile),'-p','use_sim_time:=true'])
        guard=AdaptiveGuard();driver=rclpy.create_node('rejected_anchor_fixture')
        ex=SingleThreadedExecutor();ex.add_node(guard);ex.add_node(driver)
        pubs={topic:driver.create_publisher(cls,topic,20) for topic,cls in [
            ('/fusion/lio_raw',Odometry),('/fusion/learned_raw',Odometry),('/fusion/ekf',Odometry),
            ('/fusion/left',Image),('/fusion/right',Image),('/clock',Clock)]}
        poses=[];driver.create_subscription(Odometry,'/T3/semantic/current_pose',poses.append,50)
        def drain(duration=.03):
            end=time.monotonic()+duration
            while time.monotonic()<end:ex.spin_once(timeout_sec=.001)
        def odom(t,x,variance=.001,frame='odom'):
            m=Odometry();ns=round(t*1e9);m.header.stamp.sec,m.header.stamp.nanosec=divmod(ns,10**9)
            m.header.frame_id=frame;m.child_frame_id='base_link';m.pose.pose.orientation.w=1.
            m.pose.pose.position.x=x;m.pose.covariance=(np.eye(6)*variance).ravel().tolist();return m
        def image(t):
            m=Image();m.header.stamp=odom(t,0.).header.stamp;m.height=192;m.width=256;m.step=256;m.encoding='mono8'
            yy,xx=np.indices((192,256));m.data=np.where((xx//12+yy//12)%2,50,200).astype(np.uint8).tobytes()
            for topic in ['/fusion/left','/fusion/right']:pubs[topic].publish(m)
        try:
            drain(.5)
            for i in range(12):
                t=1000.+i;x=i*.01
                pubs['/clock'].publish(Clock(clock=odom(t+.01,0.).header.stamp));drain()
                image(t);pubs['/fusion/lio_raw'].publish(odom(t,x));drain()
                pubs['/fusion/learned_raw'].publish(odom(t,x,frame='learned_epoch_1'));drain()
                pubs['/fusion/ekf'].publish(odom(t,x));drain()
            assert poses and guard.visual_continuity.reference is not None
            anchor_before=guard.visual_continuity.reference[1].copy()
            # Reproduce the observed state: the backend remains labelled usable,
            # the formal output has refused its new frame, and same-epoch visual
            # tracking continues. Isolate the unqualified-reference boundary.
            guard.lidar_reference_at=lambda stamp: transform_from_pose(odom(stamp,4.51).pose.pose)
            guard.lidar_reference_covariance_at=lambda stamp: np.eye(6)*.001
            guard.consistency.reset()
            guard.output_qualified=False;guard.filter_quality={'reason':'output_discontinuity'}
            guard.ekf_candidate=None
            count=len(poses)
            t=1012.;pubs['/clock'].publish(Clock(clock=odom(t+.01,0.).header.stamp));drain()
            guard.last_lio=(t,guard.lidar_reference_at(t))
            image(t);drain()
            pubs['/fusion/learned_raw'].publish(odom(t,.12,frame='learned_epoch_1'));drain(.1)
            np.testing.assert_allclose(guard.visual_continuity.reference[1],anchor_before,atol=1e-12,
                                       err_msg='Unaccepted raw LiDAR replaced the qualified visual anchor')
            assert len(poses)>count and guard.output_qualified,guard.filter_quality
            np.testing.assert_allclose(poses[-1].pose.pose.position.x,.12,atol=1e-8)
            print(json.dumps(dict(passed=True,rejected_lidar_frame_did_not_reanchor=True,
                                  visual_output_continues=True,output_x=poses[-1].pose.pose.position.x)))
        finally:
            ex.shutdown();guard.destroy_node();driver.destroy_node();rclpy.shutdown()


if __name__=='__main__':main()
