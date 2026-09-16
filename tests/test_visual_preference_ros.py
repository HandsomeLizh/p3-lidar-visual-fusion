"""Simulation preference changes the actual pose, with qualified fallback and TF."""
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
import yaml
from t3_lidar_visual_fusion.adaptive_guard import AdaptiveGuard

ROOT=Path(__file__).resolve().parents[1]


def main():
    assert os.environ['ROS_DOMAIN_ID']=='80' and os.environ['ROS_LOCALHOST_ONLY']=='1'
    out=ROOT/'results/visual_reference_recovery_20260916';out.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(dir=ROOT/'build',prefix='visual_preference_') as tmp:
        cfg=yaml.safe_load((ROOT/'config/simulation_live.yaml').read_text())
        assert cfg['pose_source_preference']=='visual'
        cfg['stationary']['enabled']=False;cfg['telemetry_motion']['enabled']=False
        cfg['vision_gate']['recovery_frames']=3
        profile=Path(tmp)/'profile.yaml';profile.write_text(yaml.safe_dump(cfg))
        rclpy.init(args=['--ros-args','-p','profile_path:='+str(profile),'-p','use_sim_time:=true'])
        guard=AdaptiveGuard();driver=rclpy.create_node('isolated_visual_preference_fixture')
        ex=SingleThreadedExecutor();ex.add_node(guard);ex.add_node(driver)
        pubs={t:driver.create_publisher(cls,t,30) for t,cls in [('/fusion/lio_raw',Odometry),
            ('/fusion/learned_raw',Odometry),('/fusion/ekf',Odometry),('/fusion/left',Image),('/fusion/right',Image),('/clock',Clock)]}
        poses=[];tfs=[]
        driver.create_subscription(Odometry,'/T3/semantic/current_pose',poses.append,100)
        driver.create_subscription(TFMessage,'/tf',lambda m:tfs.extend(t for t in m.transforms if t.child_frame_id=='base_link'),100)
        def drain(seconds=.02):
            until=time.monotonic()+seconds
            while time.monotonic()<until:ex.spin_once(timeout_sec=.001)
        def odom(t,x,cov,epoch='odom'):
            m=Odometry();ns=round(t*1e9);m.header.stamp.sec,m.header.stamp.nanosec=divmod(ns,10**9)
            m.header.frame_id=epoch;m.child_frame_id='base_link';m.pose.pose.orientation.w=1.
            m.pose.pose.position.x=x;m.pose.covariance=(np.eye(6)*cov).ravel().tolist();return m
        index=0
        def frame(black=False,lidar=True,visual=True,jump=False):
            nonlocal index
            t=1000.+index*1.3;x=index*.03;lx=x+index*.002+(.008 if index%2 else -.008);index+=1
            pubs['/clock'].publish(Clock(clock=odom(t+.01,0.,.001).header.stamp));drain()
            im=Image();im.header.stamp=odom(t,0.,.001).header.stamp;im.height=192;im.width=256;im.encoding='mono8';im.step=256
            yy,xx=np.indices((192,256));im.data=(np.zeros((192,256),np.uint8) if black else np.where((xx//12+yy//12)%2,50,200).astype(np.uint8)).tobytes()
            for topic in ['/fusion/left','/fusion/right']:pubs[topic].publish(im)
            drain();pubs['/fusion/lio_raw'].publish(odom(t,lx,.0001 if lidar else 1e6));drain()
            if visual:pubs['/fusion/learned_raw'].publish(odom(t,x+(10. if jump else 0.),1e6 if black else .001,'learned_epoch_1'))
            drain();pubs['/fusion/ekf'].publish(odom(t,lx,.0001 if lidar else 1000.));drain()
            return t,x,lx
        try:
            drain(.6)
            for _ in range(12):frame()
            assert guard.output_source=='visual' and guard.output_qualified
            anchor=guard.visual_continuity.reference[0][0];start=len(poses)
            for _ in range(10):t,x,lx=frame()
            assert guard.visual_continuity.reference[0][0]==anchor,'Preferred visual pose was overwritten by LiDAR anchors'
            values=[m.pose.pose.position.x for m in poses[start:]]
            assert len(values)==10,values
            np.testing.assert_allclose(np.diff(values),.03,atol=1e-9)
            assert abs(values[-1]-lx)>.01,'Only the source label changed; pose still copies LiDAR'
            for _ in range(6):t,x,lx=frame(black=True)
            assert guard.output_source=='ekf' and guard.output_qualified
            np.testing.assert_allclose(poses[-1].pose.pose.position.x,lx,atol=1e-9)
            for _ in range(6):frame()
            assert guard.output_source=='visual' and guard.output_qualified
            for _ in range(5):t,x,lx=frame(visual=False)
            assert guard.output_source=='ekf' and guard.output_qualified,'Stale visual candidate suppressed fresh LiDAR/EKF'
            for _ in range(6):frame()
            assert guard.output_source=='visual'
            frame(jump=True)
            assert guard.output_source=='ekf','Visual preference bypassed motion checks'
            before=len(poses)
            for _ in range(4):frame(black=True,lidar=False)
            assert len(poses)==before,'Neither source is valid but a qualified pose was published'
            maximum_step=float(np.max(np.abs(np.diff([m.pose.pose.position.x for m in poses]))))
            times=np.array([m.header.stamp.sec+m.header.stamp.nanosec*1e-9 for m in poses])
            corrections=np.diff([m.pose.pose.position.x for m in poses])-.03*np.diff(times)/1.3
            maximum_correction=float(np.max(np.abs(corrections)))
            assert maximum_correction<.1,maximum_correction
            assert len(tfs)==len(poses)
            for p,t in zip(poses,tfs):
                assert p.header==t.header and p.pose.pose.position.x==t.transform.translation.x
            result=dict(passed=True,preferred_visual_changes_actual_pose=True,stable_visual_anchor=True,
                blackout_falls_back=True,stale_visual_falls_back=True,recovered_visual_returns=True,
                invalid_visual_jump_rejected=True,both_failed_stops_output=True,pose_tf_agree=True,
                maximum_fixture_step_m=maximum_step,maximum_fixture_correction_m=maximum_correction,
                scope='Synthetic qualified frontend/EKF messages and real ROS guard; domain 80; no vehicle or ATE')
            (out/'visual_preference.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
        finally:
            ex.shutdown();guard.destroy_node();driver.destroy_node();rclpy.try_shutdown()


if __name__=='__main__':main()
