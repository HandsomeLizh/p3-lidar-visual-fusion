"""Real ROS messages prove body display cannot contaminate the persistent map."""
import copy
import json
import os
from pathlib import Path
import tempfile
import time

import numpy as np
import rclpy
from rclpy.executors import SingleThreadedExecutor
from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud2
from grid_map_msgs.msg import GridMap
from std_msgs.msg import Header
import yaml
from t3_lidar_visual_fusion.sensor_adapter import SensorAdapter
from t3_lidar_visual_fusion.terrain_mapper import TerrainMapper
from t3_lidar_visual_fusion.ros_utils import xyz_cloud,cloud_arrays

ROOT=Path(__file__).resolve().parents[1]


def main():
    assert os.environ.get('ROS_DOMAIN_ID')=='85' and os.environ.get('ROS_LOCALHOST_ONLY')=='1'
    with tempfile.TemporaryDirectory() as folder:
        out=Path(folder)
        cfg=yaml.safe_load((ROOT/'config/simulation_lidar_camera_fov.yaml').read_text())
        cfg.update(visual_source='none',lidar_mapping_crop={'enabled':False},
                   mapping_pose_settle_sec=0.,map_publish_period=100.,global_publish_period=100.)
        cfg['dynamic_map']['enabled']=False
        profile=out/'profile.yaml';profile.write_text(yaml.safe_dump(cfg))
        rclpy.init(args=['--ros-args','-p','profile_path:='+str(profile),'-p','output_dir:='+str(out/'map')])
        adapter=SensorAdapter();mapper=TerrainMapper();observer=rclpy.create_node('vehicle_display_fixture')
        executor=SingleThreadedExecutor()
        for node in [adapter,mapper,observer]:executor.add_node(node)
        previews=[];normalized=[];grids=[]
        observer.create_subscription(PointCloud2,'/T3/demo/vehicle_returns',previews.append,2)
        observer.create_subscription(PointCloud2,'/fusion/lidar',normalized.append,2)
        observer.create_subscription(GridMap,'/Car/T3/mapping/grid_map',grids.append,2)
        def drain(duration=.3):
            end=time.monotonic()+duration
            while time.monotonic()<end:executor.spin_once(timeout_sec=.003)
        mount=np.asarray(cfg['base_from_lidar'])
        x,y=np.meshgrid(np.arange(3.04,4.44,.1),np.arange(2.04,3.44,.1))
        environment=np.c_[x.ravel(),y.ravel(),np.full(x.size,-.27)]
        body=np.array([[-.40,0.,.30],[.55,0.,.65]])
        # The second body point is <0.5m from the LiDAR: it must still be shown.
        assert np.linalg.norm(body[1]-mount[:3,3])<cfg['min_range']
        try:
            drain(.6)
            for index,returns in enumerate([body,np.repeat(body,13000,axis=0),np.empty((0,3))]):
                stamp=observer.get_clock().now().to_msg()
                pose=Odometry(header=Header(stamp=stamp,frame_id='odom'),child_frame_id='base_link')
                pose.pose.pose.position.x=float(index)*1.5;pose.pose.pose.orientation.w=1.
                pose.pose.covariance=(np.eye(6)*.0001).ravel().tolist()
                mapper.odom(pose)
                base=np.r_[environment,returns]
                sensor=(base-mount[:3,3])@mount[:3,:3]
                adapter.cloud(xyz_cloud(sensor,Header(stamp=stamp,frame_id=cfg['lidar_input_frame'])))
                drain(.4)
                shown=cloud_arrays(previews[-1]);kept=cloud_arrays(normalized[-1])
                assert previews[-1].header.frame_id=='base_link'
                assert previews[-1].header.stamp==stamp,'Body preview must retain acquisition time'
                assert len(shown)==min(len(returns),cfg['vehicle_returns']['max_display_points'])
                assert len(kept)==len(environment),'Body leaked into localization input'
                if len(shown):assert adapter.vehicle_returns.mask(shown).all()
                stored=np.asarray(mapper.cloud.connection.execute('select x,y,z from voxels').fetchall())
                assert len(stored) and np.max(stored[:,2])<-.2,'Body leaked into map storage'
            assert len(cloud_arrays(previews[-1]))==0,'Empty frame failed to clear body display'
            mapper.publish();drain(.2)
            assert grids
            heights=np.asarray(grids[-1].data[grids[-1].layers.index('elevation')].data)
            assert np.isfinite(heights).any() and np.nanmax(heights)<-.2,'Body leaked into P4 heights'
            print(json.dumps(dict(passed=True,display_before_range_crop=True,display_point_cap=12000,
                empty_frame_clears=True,no_body_in_localization=True,no_body_in_persistent_map=True,
                no_body_in_p4_elevation=True,map_window_m=[grids[-1].info.length_x,grids[-1].info.length_y])))
        finally:
            mapper.dense_writer.close();mapper.grid.close();mapper.cloud.close();mapper.delivery.close();mapper.tum.close()
            for node in [adapter,mapper,observer]:executor.remove_node(node);node.destroy_node()
            executor.shutdown();rclpy.shutdown()


if __name__=='__main__':main()
