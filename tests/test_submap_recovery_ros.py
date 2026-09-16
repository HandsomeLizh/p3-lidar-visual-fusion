"""Compiled VoxelMap recovery in an isolated domain; no UE input/control."""
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import numpy as np
import rclpy
from sensor_msgs.msg import PointCloud2
from nav_msgs.msg import Odometry
from std_msgs.msg import String,Header
from rosgraph_msgs.msg import Clock
from t3_lidar_visual_fusion.ros_utils import xyz_cloud

ROOT=Path(__file__).resolve().parents[1]


def main():
    assert os.environ['ROS_DOMAIN_ID']=='74'
    out=ROOT/'results/submap_recovery_20260916';out.mkdir(exist_ok=True)
    rclpy.init();n=rclpy.create_node('isolated_submap_fixture')
    cloud=n.create_publisher(PointCloud2,'/fusion/lidar',3)
    visual=n.create_publisher(Odometry,'/fusion/visual_continuous',3)
    clock=n.create_publisher(Clock,'/clock',10)
    metrics=[];poses=[];quality=[]
    n.create_subscription(String,'/fusion/lidar_metrics',lambda m:metrics.append(json.loads(m.data)),100)
    n.create_subscription(Odometry,'/fusion/lio_raw',poses.append,100)
    n.create_subscription(String,'/fusion/lidar_quality',lambda m:quality.append(json.loads(m.data)),100)
    process=None;handle=None
    def spin(seconds=.03,predicate=None):
        end=time.monotonic()+seconds
        while time.monotonic()<end:
            rclpy.spin_once(n,timeout_sec=.002)
            if predicate and predicate():return
        if predicate:raise RuntimeError('ROS recovery test timeout')
    def header(t):
        h=Header(frame_id='lidar');ns=round(t*1e9);h.stamp.sec,h.stamp.nanosec=divmod(ns,1000000000);return h
    def launch(label):
        nonlocal process,handle
        metrics.clear();poses.clear();quality.clear()
        handle=(out/(label+'.log')).open('w')
        process=subprocess.Popen([str(ROOT/'install/t3_voxelmap/lib/t3_voxelmap/voxelmap_node'),
            '--ros-args','-p','use_sim_time:=true','-p','visual_recovery_enabled:=true',
            '-p','voxel_size:=2.0','-p','downsample_size:=0.2','-p','max_iterations:=30',
            '-p','visual_seed_wait_sec:=0.15','-p','submap_after_failures:=3','-p','submap_confirmation_scans:=3',
            '-p','threads:=1'],stdout=handle,stderr=subprocess.STDOUT,start_new_session=True)
        spin(10.,lambda:cloud.get_subscription_count()==1)
    def stop():
        nonlocal process,handle
        if handle:
            Path(handle.name).with_suffix('.metrics.json').write_text(json.dumps(metrics,indent=2))
        if process and process.poll() is None:os.killpg(process.pid,signal.SIGINT);process.wait(timeout=15)
        process=None
        if handle:handle.close();handle=None
        spin(.3)
    grid=np.arange(-3.,3.001,.2);a,b=np.meshgrid(grid,grid)
    scene=np.vstack([np.c_[a.ravel(),b.ravel(),np.full(a.size,-1.)],
                     np.c_[np.full(a.size,4.),a.ravel(),b.ravel()],
                     np.c_[a.ravel(),np.full(a.size,4.),b.ravel()]])
    # Replace surrounding surfaces, retaining broad angular coverage. A small
    # cluster far in one corner is a different, weak-geometry failure case.
    distant=-scene+np.array([-2.,-2.,1.])
    frame_index=0
    def frame(points,seed=True,bad=False,position=None):
        nonlocal frame_index
        t=1000.+frame_index;x=frame_index*.02;frame_index+=1
        if position is not None:x=float(position)
        clock.publish(Clock(clock=header(t+.001).stamp));spin()
        if seed:
            m=Odometry();m.header=header(t);m.header.frame_id='odom';m.child_frame_id='base_link'
            m.pose.pose.position.x=x;m.pose.pose.orientation.w=1.
            m.pose.covariance=(np.eye(6)*(1e6 if bad else .0001)).ravel().tolist()
            visual.publish(m);spin()
        before=len(metrics);cloud.publish(xyz_cloud(points-[x,0.,0.],header(t)))
        spin(8.,lambda:len(metrics)>before);spin()
        return metrics[-1],x
    try:
        launch('submap_synthetic')
        for _ in range(5):m,x=frame(scene)
        assert m['valid_update']
        m,x=frame(distant);assert not m['valid_update']
        m,x=frame(scene)
        assert m['valid_update'] and m['visual_seed_used'] and m['submap_id']==0,m
        # No overlap with the old scene. A new map must not publish its seed as
        # a solved registration; several later real registrations are required.
        bootstrap=None;first_recovered=None
        for _ in range(12):
            m,x=frame(distant)
            if m['tracking_reason']=='submap_bootstrap':
                bootstrap=m['frame'];assert not m['valid_update']
            if m['submap_id']==1 and m['valid_update']:
                first_recovered=m['frame'];break
        assert bootstrap is not None and first_recovered is not None,metrics[-8:]
        assert first_recovered-bootstrap>=3
        error=abs(poses[-1].pose.pose.position.x-x)
        assert error<.03,error
        assert quality[-1]['bridge_position_variance']>=.0001
        success=dict(visual_seed_recovers_existing_map=True,bootstrap_frame=bootstrap,
                     recovered_frame=first_recovered,position_error_m=error,
                     retained_global_frame=poses[-1].header.frame_id,submap_id=m['submap_id'])
        stop();frame_index=0;launch('submap_unqualified_visual')
        for _ in range(5):frame(scene)
        for _ in range(8):m,x=frame(distant,bad=True)
        assert not m['valid_update'] and m['recovery_attempts']==0 and m['submap_id']==0,m
        blocked=dict(unqualified_visual_cannot_create_global_submap=True,failed_frames=m['consecutive_failures'])
        stop();frame_index=0;launch('submap_abort')
        for _ in range(5):frame(scene)
        original=dict(metrics[-1]);accepted_x=original['position'][0]
        for _ in range(4):m,x=frame(distant)
        assert m['submap_candidate'],m
        m,x=frame(distant,seed=False)
        assert not m['valid_update'] and not m['submap_candidate'] and m['recovery_aborts']==1,m
        assert m['roots']==original['roots'] and m['map_keyframes']==original['map_keyframes']
        np.testing.assert_allclose(m['position'],original['position'],atol=1e-10)
        # Revisit the last accepted viewpoint to test the restored map itself.
        # This does not assert that an arbitrary later viewpoint must converge.
        for _ in range(3):m,x=frame(scene,position=accepted_x)
        assert m['valid_update'] and m['submap_id']==0,metrics[-6:]
        stop();frame_index=0;launch('submap_weak_geometry')
        for _ in range(5):frame(scene)
        for _ in range(12):m,x=frame(scene+[20.,20.,20.])
        assert m['submap_id']==0 and m['recovery_aborts']>0,m
        result=dict(passed=True,success=success,blocked=blocked,candidate_failure_restores_old_map=True,
                    weak_geometry_does_not_promote_submap=True,
                    scope='ROS domain 74; synthetic geometry and independent visual poses, compiled VoxelMap')
        (out/'submap_verification.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
    finally:
        stop();n.destroy_node();rclpy.try_shutdown()


if __name__=='__main__':main()
