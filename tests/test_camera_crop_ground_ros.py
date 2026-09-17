"""Recorded camera-cropped LiDAR regression in an isolated ROS domain."""
import argparse
import json
import os
import time
from pathlib import Path
import numpy as np
import rclpy
import yaml
from nav_msgs.msg import Odometry
from std_msgs.msg import Header
from t3_lidar_visual_fusion.terrain_mapper import TerrainMapper
from t3_lidar_visual_fusion.ground_clearance import GroundClearance
from t3_lidar_visual_fusion.ros_utils import xyz_cloud, set_pose


class PreviousCropFirst(GroundClearance):
    def select(self, points, sensor_origin, stamp=None, **_kwargs):
        return super().select(points, sensor_origin, stamp=stamp)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    assert os.environ['ROS_DOMAIN_ID']=='97' and os.environ['ROS_LOCALHOST_ONLY']=='1'
    args.output.mkdir(parents=True,exist_ok=False)
    data=np.load(args.data/'observations.npz')
    ids=sorted(int(k[4:]) for k in data.files if k.startswith('raw_'))
    assert len(ids)>=3
    reports={};clouds={};grids={}
    for mode in ('previous','fixed'):
        out=args.output/mode;out.mkdir()
        cfg=yaml.safe_load((args.data/'profile.yaml').read_text())
        cfg.update(mapping_pose_settle_sec=0.,instantaneous_cloud=True,
                   deskew={'enabled':False},map_publish_period=1000.,global_publish_period=1000.)
        cfg['stereo_mapping']['enabled']=False
        profile=out/'profile.yaml';profile.write_text(yaml.safe_dump(cfg))
        rclpy.init(args=['--ros-args','-p','profile_path:='+str(profile),'-p','output_dir:='+str(out)])
        mapper=TerrainMapper()
        if mode=='previous':mapper.ground_clearance=PreviousCropFirst(**cfg['ground_clearance'])
        timings=[];frames=[]
        try:
            for i in ids:
                stamp=float(data['stamp_'+str(i)][0]);pose=data['pose_'+str(i)]
                h=Header(frame_id='odom');h.stamp.sec,h.stamp.nanosec=divmod(round(stamp*1e9),10**9)
                msg=Odometry();msg.header=h;msg.child_frame_id='base_link';set_pose(msg.pose.pose,pose)
                # Isolate the crop/ground regression from changing localization
                # quality: identical, explicit test covariance in both runs.
                msg.pose.covariance=(np.eye(6)*1e-5).ravel().tolist();mapper.odom(msg)
                h=Header(frame_id='lidar',stamp=h.stamp)
                raw=data['raw_'+str(i)];cloud=xyz_cloud(raw,h);before=bytes(cloud.data)
                start=time.perf_counter();mapper.enqueue(cloud,'lidar',cfg['base_from_lidar']);mapper.process()
                timings.append(time.perf_counter()-start)
                assert bytes(cloud.data)==before
                assert mapper.stats['dropped_scans']==0,mapper.stats
                frames.append(dict(ground=mapper.stats['ground_clearance'],fusion=mapper.stats['elevation_fusion']))
            mapper.publish();mapper.publish_global();mapper.save();mapper.report_status()
            clouds[mode]=np.array(mapper.cloud.connection.execute('SELECT x,y,z FROM voxels ORDER BY x,y,z').fetchall())
            window=mapper.grid.extract_window(center_x=pose[0,3],center_y=pose[1,3],length_x=64.,length_y=64.)
            grids[mode]=window.elevation.copy()
            reports[mode]=dict(known_cells=int(np.isfinite(window.elevation).sum()),
                stored_points=len(clouds[mode]),median_process_ms=float(np.median(timings)*1000),
                p95_process_ms=float(np.percentile(timings,95)*1000),frames=frames)
        finally:
            mapper.dense_writer.close();mapper.grid.close();mapper.delivery.close();mapper.cloud.close()
            mapper.tum.close();mapper.destroy_node();rclpy.try_shutdown()
    assert reports['previous']['known_cells']==0,reports['previous']['known_cells']
    assert reports['fixed']['known_cells']>400,reports['fixed']['known_cells']
    np.testing.assert_equal(clouds['previous'],clouds['fixed'])
    np.savez_compressed(args.output/'comparison.npz',previous=grids['previous'],fixed=grids['fixed'])
    report=dict(passed=True,scope='Recorded geometry and poses, identical fixed covariance; not live localization accuracy.',
        frames=len(ids),modes=reports,view_cloud_unchanged=True,vehicle_commands_published=0)
    (args.output/'report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k!='modes'}))
    print(json.dumps({k:{n:v for n,v in d.items() if n!='frames'} for k,d in reports.items()}))


if __name__=='__main__':main()
