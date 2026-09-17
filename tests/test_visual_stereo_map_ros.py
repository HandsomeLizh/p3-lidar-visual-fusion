"""Exercise mixed mapping sources under visual body poses in isolated ROS 84."""
import argparse
import json
import os
from pathlib import Path
import tempfile
import time
import numpy as np
import yaml
import rclpy
from rclpy.executors import SingleThreadedExecutor
from nav_msgs.msg import Odometry
from grid_map_msgs.msg import GridMap
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Header
from scipy.spatial.transform import Rotation
from t3_lidar_visual_fusion.terrain_mapper import TerrainMapper
from t3_lidar_visual_fusion.ros_utils import xyz_cloud,set_pose

ROOT=Path(__file__).resolve().parents[1]


def main():
    assert os.environ['ROS_DOMAIN_ID']=='84'
    parser=argparse.ArgumentParser();parser.add_argument('--stereo-only',action='store_true')
    parser.add_argument('--lidar-camera-fov',action='store_true')
    parser.add_argument('--soft-display',action='store_true')
    parser.add_argument('--continuous-map',action='store_true')
    args=parser.parse_args()
    if args.continuous_map:args.soft_display=True
    if args.soft_display:args.lidar_camera_fov=True
    assert not (args.stereo_only and args.lidar_camera_fov)
    with tempfile.TemporaryDirectory(dir=ROOT/'build',prefix='stereo_map_') as folder:
        tmp=Path(folder)
        cfg=yaml.safe_load((ROOT/'config/simulation_live.yaml').read_text())
        cfg.update(stereo_mapping_topic='/fusion/stereo_points',semantic_topic='',
            mapping_sources=['stereo'] if args.stereo_only else ['lidar'] if args.lidar_camera_fov else ['lidar','stereo'],
            lidar_mapping_crop=dict(enabled=args.lidar_camera_fov),
            mapping_pose_settle_sec=0.,mapping_wait_timeout=.2,
            map_publish_period=100.,global_publish_period=100.)
        cfg['dynamic_map']['enabled']=False
        if args.soft_display:
            cfg['lidar_mapping_crop'].update(match_stereo_depth_limit=False,
                full_density_range_m=10.,range_feather_m=5.,image_feather_fraction=.15,sampling_cell_m=.1)
        if args.continuous_map:
            cfg['lidar_mapping_crop']['continuity']=dict(enabled=True,sector_deg=2.,
                max_gap_m=.25,feather_m=.6,seed_range_m=5.)
        profile=tmp/'profile.yaml';profile.write_text(yaml.safe_dump(cfg))
        rclpy.init(args=['--ros-args','-p','profile_path:='+str(profile),'-p','output_dir:='+str(tmp/'map')])
        mapper=TerrainMapper();driver=rclpy.create_node('visual_stereo_map_fixture')
        executor=SingleThreadedExecutor();executor.add_node(mapper);executor.add_node(driver)
        pose_pub=driver.create_publisher(Odometry,'/T3/semantic/current_pose',10)
        lidar_pub=driver.create_publisher(PointCloud2,'/fusion/lidar',2)
        stereo_pub=driver.create_publisher(PointCloud2,'/fusion/stereo_points',2)
        published_grids=[]
        driver.create_subscription(GridMap,'/Car/T3/mapping/grid_map',published_grids.append,2)
        def drain(seconds):
            end=time.monotonic()+seconds
            while time.monotonic()<end:executor.spin_once(timeout_sec=.003)
        def published_height(message,points):
            layer=message.data[message.layers.index('elevation')]
            height=np.asarray(layer.data).reshape(*(d.size for d in layer.layout.dim))[::-1,::-1]
            origin=np.array([message.info.pose.position.x-message.info.length_x/2,
                             message.info.pose.position.y-message.info.length_y/2])
            cells=np.floor((points[:,:2]-origin)/message.info.resolution).astype(int)
            return height[cells[:,1],cells[:,0]]
        try:
            drain(.5)
            # Separate spatial cells prove that both inputs contribute to storage.
            lidar_world=np.array([[2.04,.04,.02],[2.44,.04,.02],[2.84,.04,.02]])
            if args.continuous_map:
                lidar_world=np.array([[x,.04,.02] for x in 2.14+np.arange(21)*.15])
                connected_world=lidar_world[:-1].copy()
                lidar_world=np.r_[lidar_world,[[8.14,.04,.02]]]
            if args.soft_display:
                lidar_world=np.r_[lidar_world,[[x,y,.02] for x in np.arange(11.05,14.76,.4)
                                               for y in [-1.45,-.45,.55,1.55]]]
            stereo_world=np.array([[3.04,1.04,.12],[3.44,1.04,.12],[3.84,1.04,.12]])
            excluded_world=np.array([[-3.,0.,.02],[2.,10.,5.],[30.,0.,.02]])
            mount=np.asarray(cfg['base_from_lidar'])
            for x,angle in [(0.,0.),(.5,.2)]:
                stamp=driver.get_clock().now().to_msg()
                body=np.eye(4);body[:3,:3]=Rotation.from_euler('z',angle).as_matrix();body[0,3]=x
                odom=Odometry(header=Header(stamp=stamp,frame_id='map'),child_frame_id='base_link')
                set_pose(odom.pose.pose,body)
                pose_pub.publish(odom)
                world_from_lidar=body@mount
                lidar_input=np.r_[lidar_world,excluded_world] if args.lidar_camera_fov else lidar_world
                lidar=(lidar_input-world_from_lidar[:3,3])@world_from_lidar[:3,:3]
                stereo=(stereo_world-body[:3,3])@body[:3,:3]
                lidar_pub.publish(xyz_cloud(lidar,Header(stamp=stamp,frame_id='lidar')))
                stereo_pub.publish(xyz_cloud(stereo,Header(stamp=stamp,frame_id='base_link')))
                drain(.4)
            assert mapper.stats['lidar_scans']==(0 if args.stereo_only else 2),mapper.stats
            assert mapper.stats['stereo_scans']==(0 if args.lidar_camera_fov else 2),mapper.stats
            actual=mapper.cloud.preview()
            stored=np.asarray(mapper.cloud.connection.execute('SELECT x,y,z FROM voxels').fetchall())
            expected_points=stereo_world if args.stereo_only else lidar_world if args.lidar_camera_fov else np.r_[lidar_world,stereo_world]
            if args.continuous_map:expected_points=connected_world
            for expected in expected_points:
                check=stored if args.soft_display else actual
                assert np.min(np.linalg.norm(check-expected,axis=1))<1e-5,(expected,check)
            if args.lidar_camera_fov:
                # A point exactly on a fine-voxel boundary can occupy adjacent
                # bins after float32 ROS transport; compare measured geometry.
                assert len(np.unique(np.round(stored,5),axis=0))==len(expected_points),(stored,mapper.stats)
                if args.soft_display:
                    assert 3<len(actual)<len(lidar_world),(actual,mapper.stats)
                    # Cosmetic sampling never removes eligible height evidence.
                    cells=np.floor(expected_points[:,:2]/cfg['map_resolution']).astype(int)
                    priors=mapper.grid.height_priors(cells)
                    assert np.isfinite(priors[0]).all(),priors
                    assert mapper.stats['lidar_crop_last']['thinning_scope']=='display_only'
                    if args.continuous_map:
                        np.testing.assert_allclose(priors[0],.02,atol=1e-5)
                        far=lidar_world[lidar_world[:,0]>10.]
                        far_priors=mapper.grid.height_priors(np.floor(far[:,:2]/cfg['map_resolution']).astype(int))
                        assert np.isnan(far_priors[0]).all(),far_priors
                        assert np.max(stored[:,0])<5.1 and np.max(actual[:,0])<5.1
                        assert mapper.stats['lidar_crop_last']['continuity']['scope']=='mapping_and_display'
                        mapper.publish();drain(.15)
                        assert published_grids
                        msg=published_grids[-1];assert msg.info.length_x==64. and msg.info.length_y==64.
                        assert np.isnan(published_height(msg,far)).all(), 'P4 received disconnected heights'
                else:assert len(actual)==len(lidar_world),(actual,mapper.stats)
                for excluded in np.r_[excluded_world,stereo_world]:
                    assert np.min(np.linalg.norm(actual-excluded,axis=1))>.5,(excluded,actual)
                assert mapper.stats['lidar_crop_last']['input_points']==len(lidar_world)+3,mapper.stats
                assert mapper.stats['lidar_crop_last']['kept_points']==len(expected_points),mapper.stats
            if args.continuous_map:
                # This region was excluded, but a later nearby connected scan
                # must populate the same world cells: no permanent blacklist.
                target=np.array([[8.14,.04,.02]])
                assert np.isnan(published_height(published_grids[-1],target)).all()
                revisit=np.array([[x,.04,.02] for x in 8.14+np.arange(19)*.15])
                stamp=driver.get_clock().now().to_msg()
                body=np.eye(4);body[0,3]=5.
                odom=Odometry(header=Header(stamp=stamp,frame_id='map'),child_frame_id='base_link')
                set_pose(odom.pose.pose,body);pose_pub.publish(odom)
                world_from_lidar=body@mount
                lidar=(revisit-world_from_lidar[:3,3])@world_from_lidar[:3,:3]
                lidar_pub.publish(xyz_cloud(lidar,Header(stamp=stamp,frame_id='lidar')))
                drain(.4);assert mapper.stats['lidar_scans']==3,mapper.stats
                mapper.publish();drain(.15)
                np.testing.assert_allclose(published_height(published_grids[-1],target),.02,atol=1e-5)
                np.testing.assert_allclose(published_height(published_grids[-1],connected_world),.02,atol=1e-5)
                assert np.min(np.linalg.norm(mapper.cloud.preview()-target[0],axis=1))<1e-5
            if args.stereo_only:
                for expected in lidar_world:
                    assert np.min(np.linalg.norm(actual-expected,axis=1))>.5,actual
                revision=mapper.grid.update_id
                # Localization may continue on LiDAR, but even a high LiDAR
                # obstacle must not enter storage when stereo stops publishing.
                stamp=driver.get_clock().now().to_msg()
                odom.header.stamp=stamp;pose_pub.publish(odom)
                lidar_pub.publish(xyz_cloud(lidar_world+[0.,0.,4.],Header(stamp=stamp,frame_id='lidar')))
                drain(.3)
                assert mapper.grid.update_id==revision,'LiDAR changed the stereo-only grid'
                np.testing.assert_array_equal(mapper.cloud.preview(),actual)
            before=mapper.stats['mapped_scans']
            stamp=driver.get_clock().now().to_msg()
            stereo_pub.publish(xyz_cloud(stereo_world,Header(stamp=stamp,frame_id='camera_left_optical')))
            drain(.1)
            assert mapper.stats['mapped_scans']==before,'Wrong frame was integrated'
            # Neither stream can create a map observation without a matching pose.
            stamp=driver.get_clock().now().to_msg();stamp.sec+=10
            stereo_pub.publish(xyz_cloud(stereo_world+100.,Header(stamp=stamp,frame_id='base_link')))
            lidar_pub.publish(xyz_cloud(lidar_world+100.,Header(stamp=stamp,frame_id='lidar')))
            drain(.6)
            assert mapper.stats['mapped_scans']==before and not mapper.pending,mapper.stats
            print(json.dumps(dict(passed=True,geometry_after_body_motion=True,
                missing_pose_rejected=True,wrong_frame_rejected=True,
                mapping_sources=mapper.mapping_sources,lidar_excluded=args.stereo_only,
                lidar_camera_fov=args.lidar_camera_fov,continuous_map=args.continuous_map,
                later_connected_region_admitted=args.continuous_map,stats=mapper.stats)))
        finally:
            executor.shutdown();mapper.dense_writer.close();mapper.grid.close();mapper.delivery.close()
            mapper.cloud.close();mapper.tum.close();mapper.destroy_node();driver.destroy_node();rclpy.try_shutdown()


if __name__=='__main__':main()
