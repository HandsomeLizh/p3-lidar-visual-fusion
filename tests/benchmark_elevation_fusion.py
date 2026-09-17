"""Replay a saved real cloud with controlled height bias; never commands a car.

This measures repeat-scan consistency and runtime, not ground-truth accuracy.
Both modes receive the same recorded geometry and artificial pose perturbation.
"""
import json
import os
from pathlib import Path
import tempfile
import time
import numpy as np
import rclpy
import yaml
from nav_msgs.msg import Odometry
from std_msgs.msg import Header
from t3_lidar_visual_fusion.terrain_mapper import TerrainMapper
from t3_lidar_visual_fusion.ros_utils import xyz_cloud

ROOT=Path(__file__).resolve().parents[1]
SNAPSHOT=ROOT/'results/visual_only_live_20260916_213835/black_cell_audit/live_sensor_snapshot.npz'


def main():
    assert os.environ['ROS_DOMAIN_ID']=='86'
    out=Path(os.environ.get('T3_TEST_RESULTS',ROOT/'results/elevation_fusion_validation_20260917'))
    out.mkdir(parents=True,exist_ok=True)
    with np.load(SNAPSHOT,allow_pickle=False) as data:raw=data['lidar']
    report=dict(input=str(SNAPSHOT),points=len(raw),artificial_pose_bias_m=.2,
        scope='Saved real LiDAR geometry; synthetic height offset; isolated ROS 86; no motion commands',modes={})
    for probabilistic in (False,True):
        label='probabilistic_elevation_only' if probabilistic else 'previous_mean_with_classification'
        with tempfile.TemporaryDirectory(dir=ROOT/'build',prefix='height_benchmark_') as folder:
            cfg=yaml.safe_load((ROOT/'config/simulation_live.yaml').read_text())
            cfg['elevation_fusion']['enabled']=probabilistic
            cfg['publish_terrain_classification']=not probabilistic
            cfg['dynamic_map']['enabled']=False
            cfg.update(map_window=32.,semantic_topic='',map_publish_period=100.,global_publish_period=100.,
                       mapping_pose_settle_sec=0.,stereo_mapping_topic='')
            tmp=Path(folder);profile=tmp/'profile.yaml';profile.write_text(yaml.safe_dump(cfg))
            rclpy.init(args=['--ros-args','-p','profile_path:='+str(profile),'-p','output_dir:='+str(tmp/'map')])
            mapper=TerrainMapper()
            mount=np.asarray(cfg['base_from_lidar']);world=raw@mount[:3,:3].T+mount[:3,3]
            cells=np.unique(np.floor(world[:,:2]/cfg['map_resolution']).astype(np.int64),axis=0)
            update_times=[];publish_times=[]
            def scan(index,dz,partial):
                msg=Odometry();msg.header=Header(frame_id='odom');msg.header.stamp.sec=1000+index
                msg.child_frame_id='base_link';msg.pose.pose.orientation.w=1.;msg.pose.pose.position.z=dz
                msg.pose.covariance=np.diag(cfg['vision_pose_variance']).ravel().tolist()
                mapper.odom(msg)
                lidar=raw[world[:,0]>=0.] if partial else raw
                header=Header(stamp=msg.header.stamp,frame_id='sensors')
                before=mapper.stats['mapped_scans'];mapper.enqueue(xyz_cloud(lidar,header),'lidar',mount)
                mapper.process()
                assert mapper.stats['mapped_scans']==before+1,mapper.stats
                update_times.append(mapper.stats['last_map_update_sec'])
                started=time.monotonic();mapper.publish();publish_times.append(time.monotonic()-started)
            try:
                for i in range(6):scan(i,0.,False)
                baseline=mapper.grid.height_priors(cells)[0]
                for i in range(6,14):scan(i,.2,True)
                latest=mapper.grid.height_priors(cells)[0]
                rescanned=np.isfinite(baseline)&np.isfinite(latest)&(cells[:,0]>=0)
                retained=np.isfinite(baseline)&np.isfinite(latest)&(cells[:,0]<0)
                residual=latest[rescanned]-baseline[rescanned]
                tile_stats=mapper.grid.memory_stats()
                result=dict(mapped_scans=mapper.stats['mapped_scans'],overlap_cells=int(rescanned.sum()),
                    repeated_height_change_median_m=float(np.median(residual)),
                    repeated_height_change_abs_p95_m=float(np.percentile(abs(residual),95)),
                    repeated_height_change_abs_max_m=float(abs(residual).max()),
                    unobserved_history_max_change_m=float(abs(latest[retained]-baseline[retained]).max()),
                    update_median_sec=float(np.median(update_times)),update_p95_sec=float(np.percentile(update_times,95)),
                    local_publication_median_sec=float(np.median(publish_times)),
                    local_publication_p95_sec=float(np.percentile(publish_times,95)),
                    stored_cloud_points=mapper.cloud.count,tile_array_mib=tile_stats['tile_array_mib'],
                    tile_cache_limit_mib=cfg['tile_cache_mib'],height_bias=mapper.height_bias.stats)
                report['modes'][label]=result
                if probabilistic:
                    assert result['repeated_height_change_abs_p95_m']<.001,result
                    assert result['unobserved_history_max_change_m']<.000001,result
                print(label,json.dumps(result),flush=True)
            finally:
                mapper.dense_writer.close();mapper.grid.close();mapper.delivery.close();mapper.cloud.close();mapper.tum.close()
                mapper.destroy_node();rclpy.try_shutdown()
    (out/'recorded_cloud_benchmark.json').write_text(json.dumps(report,indent=2))


if __name__=='__main__':main()
