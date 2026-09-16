"""A visual preference must preserve fusion constraints and avoid thick ground."""
import json
import os
from pathlib import Path
from types import SimpleNamespace
import tempfile
import time
import numpy as np
import rclpy
from rclpy.executors import SingleThreadedExecutor
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image
from rosgraph_msgs.msg import Clock
from tf2_msgs.msg import TFMessage
from scipy.spatial.transform import Rotation
import yaml
from t3_lidar_visual_fusion.adaptive_guard import AdaptiveGuard
from t3_lidar_visual_fusion.disk_map import DiskElevationMap
from t3_lidar_visual_fusion.legacy.tiled_semantic_map import TiledSemanticMapManager
from t3_lidar_visual_fusion.terrain_mapper import TerrainMapper
from t3_lidar_visual_fusion.ros_utils import transform_from_pose

ROOT=Path(__file__).resolve().parents[1]


def ground_regression(root, poses, raw_poses):
    xx,yy=np.meshgrid(np.arange(-5.,5.,.2)+.05,np.arange(-5.,5.,.2)+.05)
    ground=np.column_stack([xx.ravel(),yy.ravel(),np.zeros(xx.size)])
    rock=(ground[:,0]>2.)&(ground[:,0]<2.5)&(ground[:,1]>1.)&(ground[:,1]<1.5)
    tops=ground[rock].copy();tops[:,2]=.6
    cloud=np.r_[ground,tops]
    result={}
    for name,transforms in [('corrected',poses),('unconstrained_visual',raw_poses)]:
        # Preserve the original failure fixture, including lifetime height spread.
        grid=(TiledSemanticMapManager(resolution=.2,tile_cells=32) if name=='unconstrained_visual' else
              DiskElevationMap(root/(name+'.sqlite'),resolution=.2,tile_cells=32,max_tiles=8,cache_mib=16))
        try:
            for transform in transforms[::8]:
                grid.update_elevation_only(points_map=cloud@transform[:3,:3].T+transform[:3,3])
            m=grid.extract_window(center_x=0.,center_y=0.,length_x=10.,length_y=10.)
            layers,_=TerrainMapper.terrain_layers(SimpleNamespace(cfg={'map_resolution':.2}),m)
            yy,xx=np.indices(m.elevation.shape)
            x=m.geometry.origin_x+(xx+.5)*.2;y=m.geometry.origin_y+(yy+.5)*.2
            flat=(np.hypot(x,y)>2.)&(np.hypot(x,y)<4.5)&((x-2.25)**2+(y-1.25)**2>.8**2)
            known=flat&np.isfinite(layers['obstacle'])
            black=float(np.mean(layers['obstacle'][known]>.5))
            actual_rock=(x>2.)&(x<2.5)&(y>1.)&(y<1.5)
            result[name]=dict(flat_black_fraction=black,
                rock_black_cells=int(np.count_nonzero(layers['obstacle'][actual_rock]>.5)))
        finally:
            if hasattr(grid,'close'):grid.close()
    assert result['unconstrained_visual']['flat_black_fraction']>.5,result
    assert result['corrected']['flat_black_fraction']<.01,result
    assert result['corrected']['rock_black_cells']>0,result
    return result


def main():
    assert os.environ['ROS_DOMAIN_ID']=='80' and os.environ['ROS_LOCALHOST_ONLY']=='1'
    out=ROOT/'results/map_drift_fix_20260916';out.mkdir(parents=True,exist_ok=True)
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
        poses=[];tfs=[];ground_poses=[];raw_ground_poses=[]
        driver.create_subscription(Odometry,'/T3/semantic/current_pose',poses.append,300)
        driver.create_subscription(TFMessage,'/tf',lambda m:tfs.extend(t for t in m.transforms if t.child_frame_id=='base_link'),300)
        def drain(seconds=.015):
            until=time.monotonic()+seconds
            while time.monotonic()<until:ex.spin_once(timeout_sec=.001)
        def odom(t,x,cov,epoch='odom',z=0.,roll=0.):
            m=Odometry();ns=round(t*1e9);m.header.stamp.sec,m.header.stamp.nanosec=divmod(ns,10**9)
            m.header.frame_id=epoch;m.child_frame_id='base_link'
            q=Rotation.from_euler('x',roll).as_quat()
            m.pose.pose.orientation.x,m.pose.pose.orientation.y,m.pose.pose.orientation.z,m.pose.pose.orientation.w=map(float,q)
            m.pose.pose.position.x=float(x);m.pose.pose.position.z=float(z)
            m.pose.covariance=(np.eye(6)*cov).ravel().tolist();return m
        index=0;x=0.
        def frame(light='normal',lidar=True,visual=True,jump=False,filter_ok=True,moving=False):
            nonlocal index,x
            t=1000.+index*1.3;index+=1
            if moving:x+=.03
            pubs['/clock'].publish(Clock(clock=odom(t+.01,0.,.001).header.stamp));drain()
            im=Image();im.header.stamp=odom(t,0.,.001).header.stamp;im.height=192;im.width=256;im.encoding='mono8';im.step=256
            yy,xx=np.indices((192,256));pixels=np.where((xx//12+yy//12)%2,50,200).astype(np.uint8)
            if light!='normal':pixels.fill(0 if light=='black' else 255)
            im.data=pixels.tobytes()
            for topic in ['/fusion/left','/fusion/right']:pubs[topic].publish(im)
            drain();pubs['/fusion/lio_raw'].publish(odom(t,x,.0001 if lidar else 1e6));drain()
            raw=odom(t,x+(10. if jump else 0.),.001,'learned_epoch_1',z=index*.0025,roll=index*.00015)
            if visual:pubs['/fusion/learned_raw'].publish(raw)
            drain();pubs['/fusion/ekf'].publish(odom(t,x,.0001 if filter_ok and lidar else 1000.));drain()
            return raw
        try:
            drain(.6)
            for _ in range(140):
                raw=frame()
                if index>=8:
                    assert guard.output_source=='ekf' and guard.output_qualified,'Visual preference bypassed healthy fused height/tilt'
                    ground_poses.append(transform_from_pose(poses[-1].pose.pose))
                    raw_ground_poses.append(transform_from_pose(raw.pose.pose))
            ground=ground_regression(Path(tmp),ground_poses,raw_ground_poses)
            for light in ['black','white']:
                for _ in range(6):frame(light=light)
                assert guard.output_source=='ekf' and guard.output_qualified
                assert not guard.visual_usable,'Invalid light was accepted as visual motion'
            for _ in range(6):frame()
            assert guard.visual_usable
            for _ in range(8):frame(lidar=False,moving=True)
            assert guard.output_source=='visual' and guard.output_qualified,'LiDAR outage broke valid visual continuity'
            for _ in range(8):frame()
            assert guard.output_source=='ekf' and guard.output_qualified
            for _ in range(6):frame(filter_ok=False)
            assert guard.output_source=='visual' and guard.output_qualified,'Diverged filter suppressed valid vision'
            for _ in range(6):frame()
            for _ in range(5):frame(visual=False)
            assert guard.output_source=='ekf' and guard.output_qualified,'Stale visual candidate suppressed fresh fusion'
            for _ in range(6):frame()
            frame(jump=True)
            assert guard.output_source=='ekf','Preference bypassed visual motion checks'
            before=len(poses)
            for _ in range(4):frame(light='white',lidar=False)
            assert len(poses)==before,'Both sources invalid but qualified poses were published'
            assert len(tfs)==len(poses)
            for p,t in zip(poses,tfs):
                assert p.header==t.header
                np.testing.assert_allclose([p.pose.pose.position.x,p.pose.pose.position.y,p.pose.pose.position.z],
                    [t.transform.translation.x,t.transform.translation.y,t.transform.translation.z],atol=1e-12)
            result=dict(passed=True,healthy_fusion_retains_height_and_tilt=True,ground_obstacle_regression=ground,
                visual_motion_information_scale=guard.visual_motion_information_scale,
                blackout_and_overexposure_fall_back=True,lidar_outage_visual_continues=True,
                diverged_filter_visual_continues=True,stale_visual_falls_back=True,
                invalid_visual_jump_rejected=True,both_failed_stops_output=True,pose_tf_agree=True,
                scope='Synthetic qualified frontend/EKF messages, real ROS guard and elevation mapper; domain 80; no vehicle or ATE')
            (out/'visual_preference.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
        finally:
            ex.shutdown();guard.destroy_node();driver.destroy_node();rclpy.try_shutdown()


if __name__=='__main__':main()
