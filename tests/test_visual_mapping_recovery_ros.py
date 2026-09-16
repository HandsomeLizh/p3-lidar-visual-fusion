"""Real guard/EKF/VoxelMap/mapper transport; synthetic frontend and geometry.

Domain 76 only, no simulator connection or vehicle-command publisher.
"""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import numpy as np
import rclpy
from rclpy.executors import SingleThreadedExecutor
from sensor_msgs.msg import Image,PointCloud2
from nav_msgs.msg import Odometry
from std_msgs.msg import Header,String
from rosgraph_msgs.msg import Clock
import yaml
from t3_lidar_visual_fusion.adaptive_guard import AdaptiveGuard
from t3_lidar_visual_fusion.terrain_mapper import TerrainMapper
from t3_lidar_visual_fusion.ros_utils import xyz_cloud,stamp_sec

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from generate_config import generate


def main():
    assert os.environ['ROS_DOMAIN_ID']=='76'
    out=ROOT/'results/submap_recovery_20260916';out.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=ROOT/'build',prefix='visual_map_recovery_') as temporary:
        tmp=Path(temporary);cfg=yaml.safe_load((ROOT/'config/simulation_live.yaml').read_text())
        cfg.update(visual_continuity={'enabled':True},base_from_lidar=np.eye(4).tolist(),
                   semantic_topic='',map_window=8.,tile_cells=16,map_publish_period=100.,
                   global_publish_period=100.,mapping_pose_settle_sec=0.)
        cfg['stationary']['enabled']=False;cfg['telemetry_motion']['enabled']=False
        cfg['vision_gate']['recovery_frames']=3
        profile=tmp/'profile.yaml';profile.write_text(yaml.safe_dump(cfg));generate(profile,tmp/'config')
        rclpy.init(args=['--ros-args','-p','profile_path:='+str(profile),
                         '-p','output_dir:='+str(tmp/'map'),'-p','use_sim_time:=true'])
        guard=AdaptiveGuard();mapper=TerrainMapper();driver=rclpy.create_node('isolated_recovery_mapping_fixture')
        executor=SingleThreadedExecutor()
        for node in [guard,mapper,driver]:executor.add_node(node)
        pubs={name:driver.create_publisher(cls,name,30) for name,cls in [
            ('/fusion/lidar',PointCloud2),('/fusion/left',Image),('/fusion/right',Image),
            ('/fusion/learned_raw',Odometry),('/clock',Clock)]}
        metrics=[];poses=[];records=[];processes=[];handles=[]
        driver.create_subscription(String,'/fusion/lidar_metrics',lambda m:metrics.append(json.loads(m.data)),100)
        driver.create_subscription(Odometry,'/T3/semantic/current_pose',poses.append,100)
        def launch(name,args):
            handle=(out/(name+'.log')).open('w');handles.append(handle)
            processes.append(subprocess.Popen(args,stdout=handle,stderr=subprocess.STDOUT,start_new_session=True))
        def drain(seconds=.02,predicate=None):
            deadline=time.monotonic()+seconds
            while time.monotonic()<deadline:
                executor.spin_once(timeout_sec=.001)
                if predicate and predicate():return
            if predicate:raise RuntimeError('Combined recovery test timeout')
        def header(stamp,frame='lidar'):
            h=Header(frame_id=frame);ns=round(stamp*1e9);h.stamp.sec,h.stamp.nanosec=divmod(ns,1000000000);return h
        def set_clock(stamp):
            pubs['/clock'].publish(Clock(clock=header(stamp).stamp));drain()
        grid=np.arange(-3.,3.001,.2);a,b=np.meshgrid(grid,grid)
        scene=np.vstack([np.c_[a.ravel(),b.ravel(),np.full(a.size,-1.)],
                         np.c_[np.full(a.size,4.),a.ravel(),b.ravel()],
                         np.c_[a.ravel(),np.full(a.size,4.),b.ravel()]])
        distant=-scene+[-2.,-2.,1.]
        yy,xx=np.indices((192,256));pixels=np.where((xx//12+yy//12)%2,50,200).astype(np.uint8).tobytes()
        index=0
        def frame(points,label):
            nonlocal index
            t=1000.+index*1.3;x=max(0,index-10)*.01;index+=1
            before=mapper.stats['mapped_scans'];count=len(metrics);set_clock(t+.001)
            im=Image();im.header=header(t,'camera');im.height=192;im.width=256;im.encoding='mono8';im.step=256;im.data=pixels
            pubs['/fusion/left'].publish(im);pubs['/fusion/right'].publish(im);drain()
            pubs['/fusion/lidar'].publish(xyz_cloud(points-[x,0.,0.],header(t)))
            m=Odometry();m.header=header(t,'learned_epoch_1');m.child_frame_id='base_link'
            m.pose.pose.orientation.w=1.;m.pose.pose.position.x=x;m.pose.covariance=(np.eye(6)*.001).ravel().tolist()
            pubs['/fusion/learned_raw'].publish(m)
            drain(8.,lambda:len(metrics)>count)
            for j in range(1,9):set_clock(t+j*.05);drain(.02)
            emitted=[p for p in poses if abs(stamp_sec(p)-t)<1e-7]
            record=dict(stage=label,stamp=t,x=x,lidar=metrics[-1],source=guard.output_source,
                        emitted=len(emitted),mapped=mapper.stats['mapped_scans']-before,
                        qualified=guard.output_qualified,filter_quality=guard.filter_quality,
                        visual=guard.visual_continuity.status())
            if emitted:record['pose_error_m']=abs(emitted[-1].pose.pose.position.x-x)
            records.append(record)
            return record
        try:
            launch('combined_voxelmap',[str(ROOT/'install/t3_voxelmap/lib/t3_voxelmap/voxelmap_node'),
                '--ros-args','-p','use_sim_time:=true','-p','visual_recovery_enabled:=true',
                '-p','voxel_size:=2.0','-p','downsample_size:=0.2','-p','max_iterations:=20',
                '-p','submap_after_failures:=4','-p','visual_seed_wait_sec:=1.0','-p','threads:=1',
                '-p','visual_seed_position_variance:='+str(float(cfg.get('qualified_position_variance',4.))),
                '-p','visual_seed_rotation_variance:='+str(float(cfg.get('qualified_rotation_variance',.5)))])
            launch('combined_ekf',['ros2','run','robot_localization','ekf_node','--ros-args',
                '-r','__node:=ekf_filter_node','--params-file',str(tmp/'config/ekf.yaml'),
                '-p','use_sim_time:=true','-r','odometry/filtered:=/fusion/ekf'])
            drain(12.,lambda:pubs['/fusion/lidar'].get_subscription_count()>=2)
            for _ in range(15):frame(scene,'healthy')
            assert guard.visual_continuity.reference is not None
            failing=[]
            for _ in range(24):failing.append(frame(scene+[20.,20.,20.],'failed_registration'))
            assert all(not r['lidar']['valid_update'] for r in failing)
            assert failing[-1]['lidar']['recovery_aborts']>=1,failing[-1]
            assert all(r['source']=='visual' and r['qualified'] and r['emitted']>=1 for r in failing),failing
            assert all(r['mapped']>=1 for r in failing),failing
            assert max(r['pose_error_m'] for r in failing)<.03,failing
            recovering=[]
            for _ in range(16):recovering.append(frame(distant,'recovering'))
            assert any(r['lidar']['valid_update'] and r['lidar']['submap_id']==1 for r in recovering),recovering
            assert all(r['qualified'] and r['mapped']>=1 for r in recovering),recovering
            result=dict(passed=True,failed_registration_frames=len(failing),
                        visual_outputs_during_failures=sum(r['emitted'] for r in failing),
                        mapped_scans_during_failures=sum(r['mapped'] for r in failing),
                        aborted_candidates=failing[-1]['lidar']['recovery_aborts'],
                        later_submap_recovers=True,maximum_visual_pose_error_m=max(r['pose_error_m'] for r in failing),
                        scope='Domain 76: compiled VoxelMap, robot_localization EKF, guard and mapper; synthetic frontend/geometry. Not an ATE benchmark.')
            (out/'visual_mapping_recovery_verification.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
        finally:
            (out/'visual_mapping_recovery_frames.json').write_text(json.dumps(records,indent=2))
            for p in processes:
                if p.poll() is None:os.killpg(p.pid,signal.SIGINT);p.wait(timeout=15)
            for h in handles:h.close()
            executor.shutdown()
            mapper.dense_writer.close();mapper.grid.close();mapper.delivery.close();mapper.cloud.close();mapper.tum.close()
            for node in [guard,mapper,driver]:node.destroy_node()
            rclpy.try_shutdown()


if __name__=='__main__':main()
