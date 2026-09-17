"""Mixed backlog must preserve stereo grid evidence and fresh range scans."""
import json,os,tempfile,time
from pathlib import Path
import numpy as np,rclpy,yaml
from nav_msgs.msg import Odometry
from std_msgs.msg import Header,String
from grid_map_msgs.msg import GridMap
from t3_lidar_visual_fusion.terrain_mapper import TerrainMapper
from t3_lidar_visual_fusion.ros_utils import xyz_cloud

ROOT=Path(__file__).resolve().parents[1]
def main():
    assert os.environ['ROS_DOMAIN_ID']=='88' and os.environ['ROS_LOCALHOST_ONLY']=='1'
    with tempfile.TemporaryDirectory(dir=ROOT/'build') as tmp:
        root=Path(tmp);cfg=yaml.safe_load((ROOT/'config/hardware104.yaml').read_text())
        cfg['stereo_mapping']['enabled']=True  # Mixed-source queue fixture.
        cfg['mapping_source']='range'
        cfg.update(base_from_lidar=np.eye(4).tolist(),base_from_camera_left=np.eye(4).tolist(),
            ground_clearance={'enabled':False},self_filter={'enabled':False},
            elevation_fusion={'enabled':False},map_output='terrain',
            mapping_pose_settle_sec=0.,map_window=8.,tile_cells=16,map_publish_period=1000.,global_publish_period=1000.)
        path=root/'profile.yaml';path.write_text(yaml.safe_dump(cfg))
        rclpy.init(args=['--ros-args','-p','profile_path:='+str(path),'-p','output_dir:='+str(root/'map')])
        mapper=TerrainMapper();maps={'local':[],'global':[]}
        for key,topic in [('local','/Car/T3/mapping/grid_map'),('global','/Car/T3/mapping/global_grid_map')]:
            mapper.create_subscription(GridMap,topic,lambda m,k=key:maps[k].append(m),10)
        def header(t,frame):
            h=Header(frame_id=frame);h.stamp.sec,h.stamp.nanosec=divmod(round(t*1e9),10**9);return h
        def pose(t,variance=.00001):
            m=Odometry();m.header=header(t,'odom');m.child_frame_id='base_link';m.pose.pose.orientation.w=1.
            m.pose.covariance=(np.eye(6)*variance).ravel().tolist();mapper.odom(m)
        stereo=np.array([[.05,.05,.5],[.25,.05,1.2],[.45,.05,.5]])
        def visual(t):
            mapper.fusion_status(String(data=json.dumps(dict(vision_enabled=True,localization_valid=True))))
            mapper.enqueue_stereo(xyz_cloud(stereo,header(t,'camera_left_optical'),[.001]*3,'position_variance'))
        try:
            pose(.1,.002);pose(.1,.004)
            assert len(mapper.pose_quality)==len(mapper.poses.samples)==1
            assert mapper.pose_quality_at(.1)[1]==.004
            pose(.5,.008)
            assert mapper.poses.at(.3,.12,.6) is not None
            assert np.allclose(mapper.pose_quality_at(.3)[3],np.eye(6)*.006)
            assert mapper.pose_quality_at(.61) is not None
            assert mapper.pose_quality_at(.63) is None
            pose(1.2,.004)
            assert mapper.pose_quality_at(.85) is None
            mapper.poses.samples.clear();mapper.pose_quality.clear()
            for t in [1.,1.1,1.2,1.3,1.4,1.5]:pose(t)
            for t in [1.,1.1,1.2,1.3]:mapper.enqueue(xyz_cloud([[2.1,2.1,0.]],header(t,'lidar')),'lidar',np.eye(4))
            assert len(mapper.pending)==4
            visual(1.3);mapper.process()
            assert mapper.stats['stereo_scans']==1 and mapper.stats['lidar_scans']==0
            mapper.process();assert mapper.stats['lidar_scans']==1
            assert mapper.last_header.stamp.nanosec==300000000
            assert mapper.stats['superseded_scans']==3
            visual(1.4);mapper.process()
            assert mapper.stats['stereo_cells']==3,mapper.stats
            mapper.publish();mapper.publish_global()
            until=time.monotonic()+.6
            while time.monotonic()<until:rclpy.spin_once(mapper,timeout_sec=.01)
            for key in maps:
                assert maps[key],key
                msg=maps[key][-1]
                layers={name:np.asarray(msg.data[i].data) for i,name in enumerate(msg.layers)}
                assert all(name in layers for name in ('elevation','elevation_variance','obstacle'))
                assert np.count_nonzero(np.isclose(layers['elevation'],1.2))==1
                assert np.count_nonzero(layers['obstacle']>.5)>=1
                assert np.isnan(layers['elevation']).any()
            mapper.save()
            result={'passed':True,'stereo_not_starved_by_full_lidar_queue':True,
                'pose_covariance_revision_and_interpolation':True,
                'newest_settled_lidar_selected':True,'stereo_cells':mapper.stats['stereo_cells'],
                'local_and_global_grid_include_stereo_height_variance_obstacles':True,'unknown_preserved':True}
            out=ROOT/'results/hardware104_mapping_latency_20260916';out.mkdir(exist_ok=True)
            (out/'queue_grid_verification.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
        finally:
            mapper.dense_writer.close(flush_revision=mapper.last_dense_revision if mapper.last_dense_revision>=0 else None)
            mapper.grid.close();mapper.delivery.close();mapper.cloud.close();mapper.tum.close();mapper.destroy_node();rclpy.shutdown()
if __name__=='__main__':main()
