"""Isolated mapper: calibrated body rejection, timed points and LiDAR-only height."""
import json,os,tempfile,time
from pathlib import Path
import numpy as np,rclpy,yaml
from scipy.spatial.transform import Rotation
from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud2
from grid_map_msgs.msg import GridMap
from std_msgs.msg import Header
from t3_lidar_visual_fusion.terrain_mapper import TerrainMapper
from t3_lidar_visual_fusion.ros_utils import xyz_cloud,cloud_arrays,set_pose,stamp_sec

ROOT=Path(__file__).resolve().parents[1]


def main():
    assert os.environ['ROS_DOMAIN_ID']=='96' and os.environ['ROS_LOCALHOST_ONLY']=='1'
    with tempfile.TemporaryDirectory(dir=ROOT/'build') as temporary:
        out=Path(temporary);cfg=yaml.safe_load((ROOT/'config/hardware104.yaml').read_text())
        cfg['mapping_source']='range';cfg['stereo_mapping']['enabled']=False
        cfg['mapping_camera_view']={'enabled':False}
        cfg.update(mapping_pose_settle_sec=0.,ground_clearance={'enabled':False},
            instantaneous_cloud=False,deskew={'enabled':False},cloud_motion_compensated=False,
            tile_cells=16,map_publish_period=1000.,global_publish_period=1000.)
        profile=out/'profile.yaml';profile.write_text(yaml.safe_dump(cfg))
        rclpy.init(args=['--ros-args','-p','profile_path:='+str(profile),'-p','output_dir:='+str(out/'map')])
        mapper=TerrainMapper();received={'cloud':[],'local':[],'global':[]}
        for key,kind,topic in [('cloud',PointCloud2,'/T3/mapping/lidar_map'),
                ('local',GridMap,'/Car/T3/mapping/grid_map'),('global',GridMap,'/T3/mapping/global_grid_map')]:
            mapper.create_subscription(kind,topic,lambda m,k=key:received[k].append(m),10)
        lidar=np.array(cfg['base_from_lidar'])
        rotation=Rotation.from_euler('z',.9).as_matrix()
        def header(t,frame):
            h=Header(frame_id=frame);h.stamp.sec,h.stamp.nanosec=divmod(round(t*1e9),10**9);return h
        def pose(t):
            transform=np.eye(4);transform[:3,:3]=rotation
            transform[:3,3]=[5.+(t-100.)*2.,-3.,.2]
            m=Odometry();m.header=header(t,'odom');m.child_frame_id='base_link'
            set_pose(m.pose.pose,transform);m.pose.covariance=(np.eye(6)*.00001).ravel().tolist()
            mapper.odom(m);return transform
        def height_at(x,y):
            center=(np.floor(np.array([x,y])/.2)+.5)*.2+.001
            return float(mapper.grid.extract_window(center_x=center[0],center_y=center[1],
                length_x=.2,length_y=.2).elevation[0,0])
        try:
            # One all-body cell must remain unknown; another cell contains a
            # genuine measured ground point beneath the discarded self return.
            base=np.array([[-.61,.11,.5],[-.81,-.31,.75],[-.61,.11,-.4],
                [1.51,.11,.5],[-.11,1.31,.5],[-1.71,.11,.5],[-.81,.51,1.25]])
            sensor=(base-lidar[:3,3])@lidar[:3,:3]
            # Bad range/NaN points are interleaved with valid timed samples.
            sensor=np.insert(sensor,2,[0.,0.,0.],axis=0)
            sensor=np.r_[sensor,[[np.nan,0.,0.]]]
            times=np.arange(len(sensor))*.01
            pose(100.);pose(100.08)
            msg=xyz_cloud(sensor,header(100.,'lidar'),times,'time');before=bytes(msg.data)
            mapper.enqueue(msg,'lidar',lidar);mapper.process()
            assert bytes(msg.data)==before,'Raw/localization cloud was changed'
            assert mapper.stats['dropped_scans']==0,mapper.stats
            assert mapper.stats['self_filter']['sources']['lidar']['last_removed_points']==2
            expected=base[2:]@rotation.T+[5.,-3.,.2]
            expected[:,0]+=2.*times[3:8]
            stored=np.array(mapper.cloud.connection.execute('SELECT x,y,z FROM voxels').fetchall())
            assert len(stored)==len(expected),(stored,expected)
            for p in expected:assert np.linalg.norm(stored-p,axis=1).min()<2e-6,(stored,p)
            assert abs(height_at(*expected[0,:2])-expected[0,2])<2e-6
            absent=base[1]@rotation.T+[5.02,-3.,.2]
            assert np.isnan(height_at(*absent[:2])),'Discarded body became observed terrain'
            mapper.publish();mapper.publish_global()
            until=time.monotonic()+.6
            while time.monotonic()<until:rclpy.spin_once(mapper,timeout_sec=.01)
            assert all(received.values()),received.keys()
            assert len(cloud_arrays(received['cloud'][-1]))==len(expected)
            assert set(received['local'][-1].layers)=={'elevation','elevation_variance','height_range','roughness','observation_count'}
            # An empty filtered scan consumes its queue without changing the map
            # revision/timestamp or counting as a localization error.
            count=mapper.cloud.count;revision=mapper.grid.update_id;stamp=stamp_sec(mapper.last_header)
            pose(101.);pose(101.08)
            only_self=(base[:2]-lidar[:3,3])@lidar[:3,:3]
            mapper.enqueue(xyz_cloud(only_self,header(101.,'lidar'),[0.,.01],'time'),'lidar',lidar)
            mapper.process()
            assert not mapper.pending and mapper.stats['dropped_scans']==0
            assert mapper.cloud.count==count and mapper.grid.update_id==revision
            assert stamp_sec(mapper.last_header)==stamp
            # The explicit range-only fixture disables stereo map input while
            # keeping learned stereo odometry selected for localization.
            assert not cfg['stereo_mapping']['enabled'] and cfg['visual_source']=='learned'
            assert mapper.stereo_cloud_pub is None
            assert '/fusion/stereo_points' not in {s.topic_name for s in mapper.input_node.subscriptions}
            stereo=mapper.create_publisher(PointCloud2,'/fusion/stereo_points',1)
            for t in [102.,102.2]:
                pose(t);stereo.publish(xyz_cloud([[0.,0.,2.]],header(t,'camera_left_optical'),[.001],'position_variance'))
                rclpy.spin_once(mapper,timeout_sec=.05);mapper.process()
            assert mapper.grid.update_id==revision and mapper.stats['stereo_cells']==0
            report=dict(passed=True,body_frame_after_rotation=True,raw_input_unchanged=True,
                timed_point_alignment=True,measured_ground_and_external_obstacles_retained=True,
                discarded_cells_stay_unknown=True,empty_filtered_scan_safe=True,stereo_mapping_disabled=True,
                pointcloud_and_both_grid_topics_published=True,stats=mapper.stats,
                vehicle_commands_published=0)
            result=ROOT/'results/hardware104_self_filter_20260917/ros.json'
            result.parent.mkdir(parents=True,exist_ok=True);result.write_text(json.dumps(report,indent=2));print(json.dumps(report))
        finally:
            mapper.dense_writer.close();mapper.grid.close();mapper.delivery.close();mapper.cloud.close();mapper.tum.close()
            mapper.destroy_node();rclpy.try_shutdown()


if __name__=='__main__':main()
