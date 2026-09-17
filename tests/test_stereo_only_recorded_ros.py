"""Actual parked-vehicle image pairs into a stereo-only mapper, isolated CPU run."""
import argparse,json,os,tempfile,time
from pathlib import Path
import cv2,numpy as np,rclpy,yaml
from std_msgs.msg import Header,String
from nav_msgs.msg import Odometry
from t3_lidar_visual_fusion.learned_matching import LearnedMatcher
from t3_lidar_visual_fusion.learned_tracker import LearnedStereoTracker
from t3_lidar_visual_fusion.stereo_mapping import sparse_map_points
from t3_lidar_visual_fusion.terrain_mapper import TerrainMapper
from t3_lidar_visual_fusion.ros_utils import xyz_cloud

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser();p.add_argument('--samples',type=Path,required=True)
    p.add_argument('--asset-root',type=Path,required=True);p.add_argument('--frames',type=int,default=10)
    args=p.parse_args()
    assert os.environ['ROS_DOMAIN_ID']=='98' and os.environ['ROS_LOCALHOST_ONLY']=='1'
    cfg=yaml.safe_load((ROOT/'config/hardware104.yaml').read_text())
    cfg['learned_visual']['device']='cpu';cfg['learned_visual']['cpu_threads']=2
    cfg.update(mapping_pose_settle_sec=0.,map_publish_period=1000.,global_publish_period=1000.)
    cv2.setNumThreads(1)
    tracker=LearnedStereoTracker(cfg,LearnedMatcher(args.asset_root,cfg['learned_visual']))
    ns=json.loads((args.samples/'report.json').read_text())['stamps_ns'][:args.frames]
    records=[]
    with tempfile.TemporaryDirectory(dir=ROOT/'build') as temporary:
        out=Path(temporary);profile=out/'profile.yaml';profile.write_text(yaml.safe_dump(cfg))
        rclpy.init(args=['--ros-args','-p','profile_path:='+str(profile),'-p','output_dir:='+str(out/'map')])
        mapper=TerrainMapper()
        try:
            for i,stamp_ns in enumerate(ns):
                left=cv2.imread(str(args.samples/f'{i:02d}_0.png'),0);right=cv2.imread(str(args.samples/f'{i:02d}_1.png'),0)
                assert left is not None and right is not None
                cv2.setRNGSeed(0);result=tracker.process(stamp_ns*1e-9,left,right)
                if result.anchor:continue
                points,variance=sparse_map_points(tracker.geometry,result.map_points,cfg['stereo_mapping'])
                h=Header(frame_id='camera_left_optical');h.stamp.sec,h.stamp.nanosec=divmod(int(stamp_ns),10**9)
                # The recording is parked. A fixed, known fixture pose isolates
                # geometric admission from the localization accuracy experiment.
                pose=Odometry();pose.header.stamp=h.stamp;pose.header.frame_id='odom';pose.child_frame_id='base_link'
                pose.pose.pose.orientation.w=1.;pose.pose.covariance=(np.eye(6)*.0001).ravel().tolist();mapper.odom(pose)
                mapper.fusion_status(String(data=json.dumps(dict(vision_enabled=True,localization_valid=True))))
                start=time.perf_counter();mapper.enqueue_stereo(xyz_cloud(points,h,variance,'position_variance'));mapper.process()
                record=dict(index=i,admitted_depth_points=len(points),cloud_voxels=mapper.cloud.count,
                    stereo_cells=mapper.stats['stereo_cells'],terrain_points=mapper.stats.get('stereo_terrain_points',0),
                    ground=mapper.stats.get('stereo_ground_clearance'),mapping_sec=time.perf_counter()-start)
                records.append(record);print(json.dumps(record),flush=True)
            mapper.publish();mapper.publish_global();mapper.save()
            assert mapper.stats['lidar_scans']==0 and mapper.stats['tof_scans']==0
            assert mapper.cloud.count>0 and mapper.stats['stereo_cells']>0,mapper.stats
            report=dict(passed=True,frames=len(records),cloud_voxels=mapper.cloud.count,stats=mapper.stats,records=records,
                median_mapping_sec=float(np.median([r['mapping_sec'] for r in records])),
                scope='Real parked stereo images, CPU frontend, fixed fixture pose; no live nodes or moving accuracy claim')
            target=ROOT/'results/hardware104_stereo_only_20260917/recorded.json'
            target.parent.mkdir(parents=True,exist_ok=True);target.write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
        finally:
            mapper.dense_writer.close();mapper.grid.close();mapper.delivery.close();mapper.cloud.close();mapper.tum.close()
            mapper.destroy_node();rclpy.try_shutdown()


if __name__=='__main__':main()
