"""Isolated ROS domains: velocity-only relay, EKF arc motion, bad data and dropout."""
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest

import numpy as np
import rclpy
from rclpy.context import Context
from rclpy.executors import SingleThreadedExecutor
from rclpy.signals import SignalHandlerOptions
from geometry_msgs.msg import TwistWithCovarianceStamped
from nav_msgs.msg import Odometry
from scipy.spatial.transform import Rotation
import yaml
from ament_index_python.packages import get_package_prefix

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from generate_config import generate


class TelemetryRosTests(unittest.TestCase):
    def test_independent_receiver_and_velocity_prediction(self):
        self.run_case(False)

    def test_world_velocity_rotated_by_same_packet_attitude(self):
        self.run_case(True)

    def run_case(self, world):
        self.assertEqual(os.environ['ROS_DOMAIN_ID'],'61')
        out=ROOT/('results/velocity_input_20260916/world_ros_transport' if world else 'results/velocity_input_20260916/ros_transport')
        out.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=ROOT/'build',prefix='telemetry_ros_') as directory:
            tmp=Path(directory)
            cfg=yaml.safe_load((ROOT/'config/simulation_live.yaml').read_text())
            # Synthetic FLU wire velocities pass through the legacy decoder's
            # Y (linear) and X/Z (angular) sign changes before entering ROS.
            cfg['telemetry_motion'].update(source_domain=62,enabled=True,fuse_velocity=True,
                calibration_confirmed=True,angular_scale=math.pi/180,
                input_encoding='legacy_driver_from_flu_wire',
                velocity_frame='world' if world else 'body',wire_frame='world' if world else 'body_flu',
                world_frame_alignment_confirmed=world,
                world_from_feedback_rotation=np.eye(3).tolist(),
                orientation_child_from_base_rotation=np.eye(3).tolist(),
                base_from_feedback_rotation=np.eye(3).tolist())
            profile=tmp/'profile.yaml';profile.write_text(yaml.safe_dump(cfg))
            generate(profile,tmp/'config')
            ekf_cfg=tmp/'config/ekf.yaml'
            env=dict(os.environ,ROS_DOMAIN_ID='61',ROS_LOCALHOST_ONLY='1')
            rclpy.init(args=[])
            target=rclpy.create_node('telemetry_test_target');twists=[];poses=[]
            target.create_subscription(TwistWithCovarianceStamped,'/fusion/telemetry_twist',twists.append,100)
            target.create_subscription(Odometry,'/fusion/ekf',poses.append,100)
            context=Context();rclpy.init(args=[],context=context,domain_id=62,signal_handler_options=SignalHandlerOptions.NO)
            source=rclpy.create_node('synthetic_vehicle_source',context=context)
            pub=source.create_publisher(Odometry,'/car/odom',10)
            executors=[SingleThreadedExecutor(),SingleThreadedExecutor(context=context)]
            executors[0].add_node(target);executors[1].add_node(source)
            ekf_binary=str(Path(get_package_prefix('robot_localization'))/'lib/robot_localization/ekf_node')
            commands=[[ekf_binary,'--ros-args','--params-file',str(ekf_cfg),
                '-r','__node:=ekf_filter_node','-r','odometry/filtered:=/fusion/ekf'],
                [sys.executable,'-m','t3_lidar_visual_fusion.telemetry_motion','--ros-args',
                 '-p','profile_path:='+str(profile),'-p','output_dir:='+str(out)]]
            processes=[];logs=[]
            motion_start=None
            def spin(seconds):
                end=time.monotonic()+seconds
                while time.monotonic()<end:
                    for executor in executors:executor.spin_once(timeout_sec=.002)
            def message(v=0.,w=0.,age=0.,frame='base_link'):
                msg=Odometry();ns=source.get_clock().now().nanoseconds-round(age*1e9)
                msg.header.stamp.sec,msg.header.stamp.nanosec=divmod(ns,10**9)
                msg.header.frame_id='odom';msg.child_frame_id=frame
                msg.pose.pose.position.x=100000.;msg.pose.pose.orientation.w=float('nan')
                msg.twist.twist.linear.x=v;msg.twist.twist.angular.z=-w
                if world:
                    # Known world velocity turns with heading; after the same-
                    # packet attitude conversion the body velocity stays +X.
                    yaw=0. if motion_start is None else math.pi/30*(ns/1e9-motion_start)
                    q=Rotation.from_euler('z',yaw).as_quat()
                    msg.pose.pose.orientation.x,msg.pose.pose.orientation.y,msg.pose.pose.orientation.z,msg.pose.pose.orientation.w=map(float,q)
                    msg.twist.twist.linear.x=v*math.cos(yaw)
                    msg.twist.twist.linear.y=-v*math.sin(yaw)  # Legacy decoded world Y.
                return msg
            def drive(seconds,v,w):
                end=time.monotonic()+seconds
                while time.monotonic()<end:pub.publish(message(v,w));spin(.05)
            try:
                for i,command in enumerate(commands):
                    log=(out/f'process_{i}.log').open('w');logs.append(log)
                    processes.append(subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True))
                spin(1.5);drive(1.,0.,0.)
                self.assertGreater(len(twists),5)
                self.assertGreater(len(poses),5)
                p0=poses[-1];motion_start=source.get_clock().now().nanoseconds/1e9
                drive(2.,.1,6.)
                spin(.1);p1=poses[-1]
                duration=p1.header.stamp.sec+p1.header.stamp.nanosec/1e9-motion_start
                omega=math.pi/30
                expected=np.array([.1/omega*math.sin(omega*duration),.1/omega*(1-math.cos(omega*duration))])
                observed=np.array([p1.pose.pose.position.x-p0.pose.pose.position.x,p1.pose.pose.position.y-p0.pose.pose.position.y])
                self.assertLess(np.linalg.norm(observed-expected),.04)
                q=p1.pose.pose.orientation;yaw=Rotation.from_quat([q.x,q.y,q.z,q.w]).as_euler('xyz')[2]
                self.assertLess(abs(yaw-omega*duration),.05)
                self.assertAlmostEqual(twists[-1].twist.twist.angular.z,omega,places=7)
                self.assertEqual(twists[-1].header.frame_id,'base_link')
                self.assertLess(abs(p1.pose.pose.position.x),1.) # UE absolute position was ignored.
                spin(.4);before=len(twists);spin(.5)
                self.assertEqual(len(twists),before,'Old feedback was repeated during dropout')
                for m in [message(age=2.),message(frame='odom'),message(v=float('nan')),message(v=10.)]:
                    pub.publish(m);spin(.1)
                self.assertEqual(len(twists),before,'Invalid feedback reached the estimator')
                drive(.7,0.,0.);self.assertGreater(len(twists),before)
                self.assertTrue(all(p.poll() is None for p in processes))
                result=dict(passed=True,arc_seconds=duration,expected_xy=expected.tolist(),observed_xy=observed.tolist(),
                    yaw=yaw,expected_yaw=omega*duration,twists=len(twists),poses=len(poses),
                    world_frame_test=world,
                    scope='Synthetic feedback in ROS domains 61/62; not a real UE driving accuracy test')
                (out/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
            finally:
                for p in processes:
                    if p.poll() is None:os.killpg(p.pid,signal.SIGINT)
                for p in processes:
                    try:p.wait(timeout=8)
                    except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGTERM);p.wait(timeout=5)
                for log in logs:log.close()
                for executor in executors:executor.shutdown()
                source.destroy_node();target.destroy_node();context.try_shutdown();rclpy.try_shutdown()


if __name__=='__main__':unittest.main()
