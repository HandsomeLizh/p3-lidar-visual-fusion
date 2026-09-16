"""Qualified visual pose must outlive failed LiDAR and a diverged EKF."""
from collections import deque
import argparse
import json
import os
from pathlib import Path
import tempfile
import time
import numpy as np
import rclpy
from rclpy.executors import SingleThreadedExecutor
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image
from rosgraph_msgs.msg import Clock
from tf2_msgs.msg import TFMessage
from std_msgs.msg import String
import yaml
from t3_lidar_visual_fusion.adaptive_guard import AdaptiveGuard

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--healthy-frames',type=int,default=12)
    parser.add_argument('--preference',choices=['balanced','visual']);args=parser.parse_args()
    assert os.environ['ROS_DOMAIN_ID']=='73'
    out=ROOT/'results/submap_recovery_20260916';out.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=ROOT/'build',prefix='visual_output_') as temporary:
        cfg=yaml.safe_load((ROOT/'config/simulation_live.yaml').read_text())
        if args.preference:cfg['pose_source_preference']=args.preference
        cfg['visual_continuity']={'enabled':True};cfg['stationary']['enabled']=False
        cfg['telemetry_motion']['enabled']=False;cfg['vision_gate']['recovery_frames']=3
        profile=Path(temporary)/'profile.yaml';profile.write_text(yaml.safe_dump(cfg))
        rclpy.init(args=['--ros-args','-p','profile_path:='+str(profile),'-p','use_sim_time:=true'])
        guard=AdaptiveGuard();driver=rclpy.create_node('isolated_visual_output_test')
        ex=SingleThreadedExecutor();ex.add_node(guard);ex.add_node(driver)
        pubs={topic:driver.create_publisher(cls,topic,30) for topic,cls in [
            ('/fusion/lio_raw',Odometry),('/fusion/learned_raw',Odometry),('/fusion/ekf',Odometry),
            ('/fusion/left',Image),('/fusion/right',Image),('/clock',Clock)]}
        poses=[];tfs=[];seeds=deque(maxlen=2)
        driver.create_subscription(Odometry,'/T3/semantic/current_pose',poses.append,100)
        driver.create_subscription(TFMessage,'/tf',lambda m:tfs.extend(t for t in m.transforms if t.child_frame_id=='base_link'),100)
        driver.create_subscription(Odometry,'/fusion/visual_continuous',seeds.append,10)
        def drain(seconds=.015):
            end=time.monotonic()+seconds
            while time.monotonic()<end:ex.spin_once(timeout_sec=.001)
        def message(stamp,x,variance,epoch='odom'):
            m=Odometry();ns=round(stamp*1e9);m.header.stamp.sec,m.header.stamp.nanosec=divmod(ns,1000000000)
            m.header.frame_id=epoch;m.child_frame_id='base_link';m.pose.pose.orientation.w=1.;m.pose.pose.position.x=float(x)
            m.pose.covariance=(np.eye(6)*variance).ravel().tolist();return m
        frame_index=0
        def frame(lidar_ok=True,black=False,epoch='learned_epoch_1',offset=0.):
            nonlocal frame_index
            t=1000.+frame_index*1.3;x=frame_index*.01;frame_index+=1
            stamp=message(t,0,.01).header.stamp
            pubs['/clock'].publish(Clock(clock=message(t+.01,0,.01).header.stamp));drain()
            im=Image();im.header.stamp=stamp;im.height=192;im.width=256;im.encoding='mono8';im.step=256
            yy,xx=np.indices((192,256));im.data=(np.zeros((192,256),np.uint8) if black else np.where((xx//12+yy//12)%2,50,200).astype(np.uint8)).tobytes()
            for topic in ['/fusion/left','/fusion/right']:pubs[topic].publish(im)
            drain();pubs['/fusion/lio_raw'].publish(message(t,x,.001 if lidar_ok else 1e6));drain()
            pubs['/fusion/learned_raw'].publish(message(t,x+offset,1e6 if frame_index==1 else .001,epoch));drain()
            pubs['/fusion/ekf'].publish(message(t,x if lidar_ok else 500.,.001 if lidar_ok else 1000.));drain()
            return t,x
        try:
            drain(.6)
            for _ in range(args.healthy_frames):frame()
            assert poses and guard.visual_continuity.reference is not None
            start=len(poses)
            for _ in range(150):last_t,last_x=frame(lidar_ok=False)
            assert len(poses)-start>=148,(len(poses)-start,guard.filter_quality,guard.visual_continuity.status())
            assert guard.output_source=='visual' and guard.output_qualified and seeds
            np.testing.assert_allclose(poses[-1].pose.pose.position.x,last_x,atol=1e-8)
            assert abs(poses[-1].header.stamp.sec+poses[-1].header.stamp.nanosec*1e-9-last_t)<1e-8
            assert len(tfs)==len(poses)
            visual_outputs=len(poses)-start
            before=len(poses)
            for _ in range(5):frame(lidar_ok=False,black=True)
            assert len(poses)==before,'Black images still produced qualified visual poses'
            for _ in range(8):frame(lidar_ok=False,epoch='learned_epoch_2',offset=-700.)
            assert len(poses)==before,'Unanchored new visual epoch was joined to old map'
            for _ in range(8):frame(lidar_ok=True,epoch='learned_epoch_2',offset=-700.)
            assert len(poses)>before and guard.output_qualified,'Healthy LiDAR could not take over'
            guard.registration_quality(String(data=json.dumps(dict(submap_id=1,
                bridge_position_variance=.05,bridge_rotation_variance=.02))))
            for _ in range(5):frame(lidar_ok=True,epoch='learned_epoch_2',offset=-700.)
            covariance=np.asarray(poses[-1].pose.covariance).reshape(6,6)
            assert np.linalg.eigvalsh(covariance[:3,:3])[0]>=.05-1e-12
            assert np.linalg.eigvalsh(covariance[3:,3:])[0]>=.02-1e-12
            guard.registration_quality(String(data='{}'))
            guard.registration_quality(String(data='[]'))
            np.testing.assert_allclose(guard.bridge_variance,[.05,.02])
            result=dict(passed=True,failed_lidar_frames=150,visual_outputs_during_lidar_failure=visual_outputs,
                        pose_source_preference=guard.pose_source_preference,
                        initial_healthy_lidar_frames=args.healthy_frames,
                        diverged_ekf_did_not_override_visual=True,blackout_stops_visual=True,
                        new_epoch_requires_reference=True,lidar_recovers=True,
                        submap_global_uncertainty_preserved=True,
                        scope='ROS domain 73; synthetic frontend and EKF messages, real guard/TF transport')
            name='visual_output_verification.json' if args.healthy_frames==12 else 'visual_startup_verification.json'
            if args.preference:name=args.preference+'_'+name
            (out/name).write_text(json.dumps(result,indent=2));print(json.dumps(result))
        finally:
            ex.shutdown();guard.destroy_node();driver.destroy_node();rclpy.try_shutdown()


if __name__=='__main__':main()
