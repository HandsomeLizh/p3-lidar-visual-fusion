"""Stereo-only persistent geometry; unrelated range returns never enter either map."""
import json,os,tempfile,time
from pathlib import Path
import numpy as np,rclpy,yaml
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import PointCloud2
from nav_msgs.msg import Odometry
from std_msgs.msg import Header,String
from grid_map_msgs.msg import GridMap
from rclpy.qos import QoSProfile,ReliabilityPolicy,DurabilityPolicy
from t3_lidar_visual_fusion.terrain_mapper import TerrainMapper
from t3_lidar_visual_fusion.ros_utils import xyz_cloud,cloud_arrays,set_pose

ROOT=Path(__file__).resolve().parents[1]


def main():
    assert os.environ['ROS_DOMAIN_ID']=='97' and os.environ['ROS_LOCALHOST_ONLY']=='1'
    with tempfile.TemporaryDirectory(dir=ROOT/'build') as directory:
        out=Path(directory);cfg=yaml.safe_load((ROOT/'config/hardware104.yaml').read_text())
        assert cfg['mapping_source']=='stereo' and cfg['stereo_mapping']['enabled']
        cfg.update(mapping_pose_settle_sec=0.,tile_cells=16,map_publish_period=1000.,global_publish_period=1000.)
        profile=out/'profile.yaml';profile.write_text(yaml.safe_dump(cfg))
        rclpy.init(args=['--ros-args','-p','profile_path:='+str(profile),'-p','output_dir:='+str(out/'map')])
        mapper=TerrainMapper();received={k:[] for k in ['cloud','local','global','lidar_cloud']}
        qos=QoSProfile(depth=1,reliability=ReliabilityPolicy.RELIABLE,durability=DurabilityPolicy.TRANSIENT_LOCAL)
        for key,kind,topic in [('cloud',PointCloud2,'/T3/mapping/stereo_map'),
            ('lidar_cloud',PointCloud2,'/T3/mapping/lidar_map'),('local',GridMap,'/Car/T3/mapping/grid_map'),
            ('global',GridMap,'/Car/T3/mapping/global_grid_map')]:
            mapper.create_subscription(kind,topic,lambda m,k=key:received[k].append(m),qos)
        camera=np.array(cfg['base_from_camera_left'])
        first=np.eye(4);first[:3,:3]=Rotation.from_euler('z',.6).as_matrix();first[:3,3]=[5.,3.,.2]
        second=first.copy();second[:3,:3]=Rotation.from_euler('z',.9).as_matrix();second[0,3]+=2.
        xx,yy=np.meshgrid(np.arange(1.11,3.92,.2),np.arange(-1.09,1.12,.2))
        ground=np.c_[xx.ravel(),yy.ravel(),(.06*xx+.02*yy-.45).ravel()]
        # This high surface remains in 3D, but must not replace ground height.
        overhead=np.array([[2.91,.21,2.6]])
        body=np.array([[.45,.11,.3]])
        base=np.r_[ground,overhead,body]
        optical=(base-camera[:3,3])@camera[:3,:3]
        def header(t,frame):
            h=Header(frame_id=frame);h.stamp.sec,h.stamp.nanosec=divmod(round(t*1e9),10**9);return h
        def feed(t,pose=first,healthy=True,variance=.00001):
            odom=Odometry();odom.header=header(t,'odom');odom.child_frame_id='base_link'
            set_pose(odom.pose.pose,pose);odom.pose.covariance=(np.eye(6)*variance).ravel().tolist();mapper.odom(odom)
            mapper.fusion_status(String(data=json.dumps(dict(vision_enabled=healthy,localization_valid=True))))
            mapper.enqueue_stereo(xyz_cloud(optical,header(t,'camera_left_optical'),np.full(len(optical),.001),'position_variance'))
            mapper.process()
        def drain():
            until=time.monotonic()+.3
            while time.monotonic()<until:rclpy.spin_once(mapper,timeout_sec=.01)
        try:
            drain();feed(100.);first_count=mapper.cloud.count
            assert first_count>100 and mapper.stats['stereo_cells']==0
            feed(100.2);assert mapper.stats['stereo_cells']>50,mapper.stats
            assert mapper.stats['self_filter']['sources']['stereo']['last_removed_points']==1
            stored=np.array(mapper.cloud.connection.execute('SELECT x,y,z FROM voxels').fetchall())
            expected=np.r_[ground,overhead]@first[:3,:3].T+first[:3,3]
            assert len(stored)==len(expected),(len(stored),len(expected))
            for point in stored:assert np.linalg.norm(expected-point,axis=1).min()<2e-6
            assert stored[:,2].max()>2.7
            mapper.publish();mapper.publish_global();drain()
            assert received['cloud'] and received['local'] and received['global']
            assert not received['lidar_cloud']
            assert cloud_arrays(received['cloud'][-1])[:,2].max()>2.7
            local=received['local'][-1];height=np.array(local.data[local.layers.index('elevation')].data)
            assert np.isfinite(height).sum()>50 and np.nanmax(height)<.2
            # A deliberately conflicting range surface cannot add or overwrite
            # either geometry output, even if a caller bypasses subscriptions.
            revision=mapper.grid.update_id;count=mapper.cloud.count
            assert cfg['mapping_lidar_topic'] not in {s.topic_name for s in mapper.input_node.subscriptions}
            mapper.enqueue(xyz_cloud([[1.,2.,30.]],header(100.3,'lidar')),'lidar',cfg['base_from_lidar'])
            mapper.enqueue(xyz_cloud([[1.,2.,30.]],header(100.3,'tof')),'tof',np.eye(4))
            mapper.process();assert mapper.cloud.count==count and mapper.grid.update_id==revision
            assert mapper.stats['lidar_scans']==mapper.stats['tof_scans']==0
            feed(101.,second);feed(101.2,second)
            assert mapper.cloud.count>count
            count=mapper.cloud.count;revision=mapper.grid.update_id
            feed(102.,second,False);feed(102.2,second,True,.5)
            assert mapper.cloud.count==count and mapper.grid.update_id==revision
            assert mapper.stats['stereo_rejected']==2
            mapper.publish();mapper.publish_global();mapper.save();drain()
            assert not received['lidar_cloud']
            metadata=json.loads((out/'map/map_metadata.json').read_text())
            assert metadata['mapping_source']=='stereo' and metadata['cloud_topic']=='/T3/mapping/stereo_map'
            report=dict(passed=True,stereo_only_subscription_and_publication=True,
                range_and_tof_cannot_enter_maps=True,actual_triangulated_xyz_preserved=True,
                cloud_accumulates_with_fused_pose=True,grid_from_confirmed_stereo_only=True,
                overhead_only_in_3d=True,body_filtered=True,poor_quality_pauses_map=True,
                first_cloud_points=first_count,final_cloud_points=count,stats=mapper.stats,
                vehicle_commands_published=0)
        finally:
            mapper.dense_writer.close();mapper.grid.close();mapper.delivery.close();mapper.cloud.close();mapper.tum.close()
            mapper.destroy_node()
        # Switching source on an existing map is rejected before opening stores.
        cfg['mapping_source']='range';profile.write_text(yaml.safe_dump(cfg))
        try:TerrainMapper()
        except ValueError as exc:assert 'source changed' in str(exc)
        else:raise AssertionError('Mixed map provenance accepted')
        report['source_change_requires_new_map']=True
        rclpy.try_shutdown()
        path=ROOT/'results/hardware104_stereo_only_20260917/ros.json'
        path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(report,indent=2));print(json.dumps(report))


if __name__=='__main__':main()
