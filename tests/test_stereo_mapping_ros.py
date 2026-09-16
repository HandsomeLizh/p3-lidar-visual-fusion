"""Domain 78: actual mapper/GridMap transport, synthetic calibrated sparse points.

No CUDA initialization, drivers, control messages or motion. Checks stereo-only
geometry after a valid pose exists, then quality rejection and LiDAR replacement.
"""
import json,os,sys,tempfile,time
from pathlib import Path
import numpy as np,yaml,rclpy
from rclpy.executors import SingleThreadedExecutor
from sensor_msgs.msg import PointCloud2
from nav_msgs.msg import Odometry
from std_msgs.msg import Header,String
from grid_map_msgs.msg import GridMap
from t3_lidar_visual_fusion.terrain_mapper import TerrainMapper
from t3_lidar_visual_fusion.ros_utils import xyz_cloud

ROOT=Path(__file__).resolve().parents[1]


def main():
    assert os.environ['ROS_DOMAIN_ID']=='78'
    with tempfile.TemporaryDirectory(dir=ROOT/'build',prefix='stereo_map_') as directory:
        cfg=yaml.safe_load((ROOT/'config/hardware104.yaml').read_text())
        mount=np.eye(4);mount[:3,:3]=[[0,0,1],[-1,0,0],[0,-1,0]];mount[2,3]=.8
        cfg.update(base_from_camera_left=mount.tolist(),base_from_lidar=np.eye(4).tolist(),
            instantaneous_cloud=True,deskew={'enabled':False},mapping_lidar_topic='/fusion/lidar',
            mapping_pose_settle_sec=0.,map_window=4.,tile_cells=8,map_publish_period=100.,global_publish_period=100.)
        path=Path(directory);profile=path/'profile.yaml';profile.write_text(yaml.safe_dump(cfg))
        rclpy.init(args=['--ros-args','-p','profile_path:='+str(profile),'-p','output_dir:='+str(path/'map')])
        os.environ['T3_VISUAL_RUNTIME']=str(path/'visual')
        sys.path.insert(0,str(ROOT/'visual'))
        from visual_monitor import Monitor
        monitor=Monitor()
        mapper=TerrainMapper();driver=rclpy.create_node('isolated_stereo_mapping_fixture')
        ex=SingleThreadedExecutor();ex.add_node(mapper);ex.add_node(driver);ex.add_node(monitor)
        pubs={topic:driver.create_publisher(cls,topic,5) for topic,cls in [
            ('/fusion/stereo_points',PointCloud2),('/fusion/lidar',PointCloud2),
            ('/T3/semantic/current_pose',Odometry),('/fusion/status',String)]}
        maps=[];clouds=[]
        driver.create_subscription(GridMap,'/Car/T3/mapping/grid_map',maps.append,5)
        driver.create_subscription(PointCloud2,'/T3/mapping/stereo_map',clouds.append,5)
        def drain(seconds=.15):
            end=time.monotonic()+seconds
            while time.monotonic()<end:ex.spin_once(timeout_sec=.005)
        def header(t,frame):
            h=Header(frame_id=frame);h.stamp.sec,h.stamp.nanosec=divmod(round(t*1e9),10**9);return h
        def frame(t,healthy=True,pose_variance=.0001,wrong_frame=False):
            pose=Odometry();pose.header=header(t,'odom');pose.child_frame_id='base_link';pose.pose.pose.orientation.w=1.
            pose.pose.covariance=(np.eye(6)*pose_variance).ravel().tolist()
            pubs['/T3/semantic/current_pose'].publish(pose)
            pubs['/fusion/status'].publish(String(data=json.dumps(dict(vision_enabled=healthy,localization_valid=True))))
            drain(.04)
            points=np.array([[.05,.6,1.25],[.25,.6,1.25]])
            h=header(t,'wrong' if wrong_frame else 'camera_left_optical')
            pubs['/fusion/stereo_points'].publish(xyz_cloud(points,h,[.001,.001],'position_variance'));drain(.2)
        try:
            drain(.5);frame(1000.);assert mapper.stats['stereo_cells']==0
            frame(1000.5);assert mapper.stats['stereo_cells']==2,mapper.stats
            assert mapper.stats['lidar_scans']==0
            mapper.publish();drain()
            assert maps and clouds and clouds[-1].width==2
            assert monitor.display_points==2
            assert 'elevation_variance' in maps[-1].layers
            count=mapper.stats['stereo_cells']
            frame(1001.,False);frame(1001.5,pose_variance=1.);frame(1002.,wrong_frame=True)
            assert mapper.stats['stereo_cells']==count and mapper.stats['stereo_rejected']>=3
            pose=Odometry();pose.header=header(1002.5,'odom');pose.pose.pose.orientation.w=1.
            pose.pose.covariance=(np.eye(6)*.0001).ravel().tolist();pubs['/T3/semantic/current_pose'].publish(pose);drain(.04)
            pubs['/fusion/lidar'].publish(xyz_cloud([[1.25,-.05,.6]],header(1002.5,'lidar')));drain(.3)
            frame(1003.);frame(1003.5)
            tile=mapper.grid.tiles.get((0,-1));row,col=tile.xy_to_single_index(1.25,-.05)
            assert abs(tile.elevation_mean[row,col]-.6)<1e-5
            assert not tile.stereo_owned[row,col]
            assert len(mapper.grid.stereo_preview)==1
            mapper.publish();mapper.publish_global();drain(.3)
            assert monitor.display_points==2 and monitor.grid is not None
            mapper.grid.checkpoint();mapper.delivery.checkpoint(mapper.grid)
            result=dict(passed=True,stereo_adds_geometry_without_lidar=True,grid_and_preview_published=True,
                invalid_visual_pose_or_frame_rejected=True,lidar_replaces_stereo=True,stereo_cannot_overwrite_lidar=True,
                combined_display_receives_both_sources=True,global_grid_published=True,
                max_pending_cells=cfg['stereo_mapping']['pending_cells'],vehicle_commands_published=0,
                scope='Synthetic points and formal poses, real mapper and ROS transport; not a moving accuracy test')
            output=ROOT/'results/hardware104_deployment/stereo_mapping_ros.json';output.write_text(json.dumps(result,indent=2))
            print(json.dumps(result))
        finally:
            ex.shutdown();mapper.dense_writer.close();mapper.grid.close();mapper.delivery.close();mapper.cloud.close();mapper.tum.close()
            mapper.destroy_node();monitor.destroy_node();driver.destroy_node();rclpy.try_shutdown()


if __name__=='__main__':main()
