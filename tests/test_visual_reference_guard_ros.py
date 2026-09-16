"""Real stereo geometry and ROS guard: reject bad depth, then bridge the same epoch."""
import json
import os
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import numpy as np
import rclpy
from rclpy.executors import SingleThreadedExecutor
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image
from rosgraph_msgs.msg import Clock
from tf2_msgs.msg import TFMessage
import yaml
from test_visual_reference_recovery import Fixture
from t3_lidar_visual_fusion.adaptive_guard import AdaptiveGuard
from t3_lidar_visual_fusion.learned_odometry import LearnedOdometry
from t3_lidar_visual_fusion.ros_utils import set_pose
from t3_lidar_visual_fusion.stereo_geometry import TrackingFailure

ROOT=Path(__file__).resolve().parents[1]


def main():
    assert os.environ['ROS_DOMAIN_ID']=='79' and os.environ['ROS_LOCALHOST_ONLY']=='1'
    out=ROOT/'results/visual_reference_recovery_20260916';out.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(dir=ROOT/'build',prefix='reference_guard_') as tmp:
        cfg=yaml.safe_load((ROOT/'config/hardware104.yaml').read_text())
        cfg['stationary']['enabled']=False;cfg['telemetry_motion']['enabled']=False
        cfg['vision_gate']['recovery_frames']=3
        profile=Path(tmp)/'profile.yaml';profile.write_text(yaml.safe_dump(cfg))
        rclpy.init(args=['--ros-args','-p','profile_path:='+str(profile),'-p','use_sim_time:=true'])
        guard=AdaptiveGuard();driver=rclpy.create_node('isolated_reference_guard_fixture')
        ex=SingleThreadedExecutor();ex.add_node(guard);ex.add_node(driver)
        pubs={topic:driver.create_publisher(cls,topic,30) for topic,cls in [
            ('/fusion/lio_raw',Odometry),('/fusion/learned_raw',Odometry),('/fusion/ekf',Odometry),
            ('/fusion/left',Image),('/fusion/right',Image),('/clock',Clock)]}
        poses=[];tfs=[];records=[];fixture=Fixture()
        driver.create_subscription(Odometry,'/T3/semantic/current_pose',poses.append,100)
        driver.create_subscription(TFMessage,'/tf',lambda m:tfs.extend(t for t in m.transforms if t.child_frame_id=='base_link'),100)
        frontend=SimpleNamespace(tracker=fixture.tracker,cfg=fixture.cfg['learned_visual'],
            failure_streak=0,next_probe=0.,age=lambda left,queued:SimpleNamespace(valid=True,sensor_age_sec=.01),
            increment=lambda name:None,queue=SimpleNamespace(stopped=False),
            pub=pubs['/fusion/learned_raw'],record=records.append)
        def drain(seconds=.02):
            end=time.monotonic()+seconds
            while time.monotonic()<end:ex.spin_once(timeout_sec=.001)
        def message(t,x,variance,epoch='odom'):
            msg=Odometry();ns=round(t*1e9);msg.header.stamp.sec,msg.header.stamp.nanosec=divmod(ns,10**9)
            msg.header.frame_id=epoch;msg.child_frame_id='base_link';msg.pose.pose.orientation.w=1.
            msg.pose.pose.position.x=x;msg.pose.covariance=(np.eye(6)*variance).ravel().tolist();return msg
        index=0
        def frame(lidar=True,bad=False,missing=False):
            nonlocal index
            t=1000.+index*1.0002;x=index*.01;index+=1
            pubs['/clock'].publish(Clock(clock=message(t+.01,0.,.001).header.stamp));drain()
            im=Image();im.header.stamp=message(t,0.,.001).header.stamp
            im.height=192;im.width=256;im.encoding='mono8';im.step=256
            yy,xx=np.indices((192,256));im.data=np.where((xx//12+yy//12)%2,50,200).astype(np.uint8).tobytes()
            for topic in ['/fusion/left','/fusion/right']:pubs[topic].publish(im)
            drain();pubs['/fusion/lio_raw'].publish(message(t,x,.001 if lidar else 1e6));drain()
            try:
                result=fixture.frame(t,x,bad_depth=bad,missing=missing,record_rejection=False)
                msg=message(t,0.,1e6 if result.anchor else .001,'learned_epoch_'+str(result.epoch))
                set_pose(msg.pose.pose,result.base_pose);pubs['/fusion/learned_raw'].publish(msg)
            except TrackingFailure as exc:
                LearnedOdometry.reject(frontend,str(exc),im,time.monotonic(),metrics=exc.metrics)
            drain();pubs['/fusion/ekf'].publish(message(t,x if lidar else 500.,.001 if lidar else 1000.));drain()
            return t,x
        try:
            drain(.6)
            for _ in range(12):frame()
            assert guard.visual_continuity.reference is not None and poses
            latest=guard.last_filter_stamp;before=len(poses)
            pubs['/fusion/ekf'].publish(message(latest-.2,-100.,1000.));drain()
            assert guard.output_qualified and guard.last_filter_stamp==latest
            assert len(poses)==before,'Delayed EKF changed the current valid output'
            before=len(poses);epoch=fixture.tracker.epoch
            for _ in range(6):frame(lidar=False,bad=True)
            assert len(poses)==before,'Rejected stereo depth or diverged EKF produced an output pose'
            assert all(r['reference_retained'] for r in records)
            for _ in range(6):t,x=frame(lidar=False)
            assert fixture.tracker.epoch==epoch and len(poses)>before
            assert guard.output_source=='visual' and guard.output_qualified,guard.visual_continuity.status()
            np.testing.assert_allclose(poses[-1].pose.pose.position.x,x,atol=1e-5)
            recovered_outputs=len(poses)-before;before=len(poses)
            for _ in range(35):frame(lidar=False,missing=True)
            assert len(poses)==before,'Unmatched frames produced a pose'
            for _ in range(8):frame(lidar=False)
            assert fixture.tracker.epoch>epoch and len(poses)==before,'Expired reference bridged without an anchor'
            for _ in range(8):frame(lidar=True)
            assert len(poses)>before and guard.output_qualified
            assert len(tfs)==len(poses)
            result=dict(passed=True,rejected_depth_frames=6,same_epoch_recovered_outputs=recovered_outputs,
                one_hz_with_jitter_accepted=True,late_ekf_does_not_invalidate_latest=True,
                no_pose_during_failed_geometry=True,expired_reference_requires_anchor=True,lidar_recovers=True,
                scope='Hardware104 guard profile; known feature correspondences with real stereo PnP, ROS domain 79; no learned model or vehicle')
            (out/'visual_reference_guard.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
        finally:
            ex.shutdown();guard.destroy_node();driver.destroy_node();rclpy.try_shutdown()


if __name__=='__main__':main()
