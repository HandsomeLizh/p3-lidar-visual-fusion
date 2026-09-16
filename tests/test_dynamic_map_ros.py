"""Isolated ROS verification of obstacle addition, qualified removal and messages."""
from collections import deque
import json
import os
from pathlib import Path
import tempfile
import time
import numpy as np
import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.qos import QoSProfile,DurabilityPolicy,ReliabilityPolicy
from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Header
from grid_map_msgs.msg import GridMap
import yaml
from t3_lidar_visual_fusion.terrain_mapper import TerrainMapper
from t3_lidar_visual_fusion.ros_utils import xyz_cloud

ROOT=Path(__file__).resolve().parents[1]


def main():
    assert os.environ['ROS_DOMAIN_ID']=='72'
    with tempfile.TemporaryDirectory(dir=ROOT/'build',prefix='dynamic_map_') as temporary:
        tmp=Path(temporary);cfg=yaml.safe_load((ROOT/'config/simulation_live.yaml').read_text())
        sensor=np.eye(4);sensor[2,3]=1.
        cfg.update(map_window=8.,tile_cells=16,semantic_topic='',map_publish_period=100.,
                   global_publish_period=100.,mapping_pose_settle_sec=0.,base_from_lidar=sensor.tolist())
        profile=tmp/'profile.yaml';profile.write_text(yaml.safe_dump(cfg))
        rclpy.init(args=['--ros-args','-p','profile_path:='+str(profile),'-p','output_dir:='+str(tmp/'map')])
        mapper=TerrainMapper();driver=rclpy.create_node('isolated_dynamic_map_fixture')
        executor=SingleThreadedExecutor();executor.add_node(mapper);executor.add_node(driver)
        odom_pub=driver.create_publisher(Odometry,'/T3/semantic/current_pose',10)
        cloud_pub=driver.create_publisher(PointCloud2,'/fusion/lidar',10)
        retained=QoSProfile(depth=1,reliability=ReliabilityPolicy.RELIABLE,durability=DurabilityPolicy.TRANSIENT_LOCAL)
        messages={k:deque(maxlen=1) for k in ['local','global']}
        for key,topic in [('local','/Car/T3/mapping/grid_map'),('global','/T3/mapping/global_grid_map')]:
            driver.create_subscription(GridMap,topic,messages[key].append,retained)
        def drain(seconds):
            end=time.monotonic()+seconds
            while time.monotonic()<end:executor.spin_once(timeout_sec=.003)
        def scan(points,variance=.0001):
            before=mapper.stats['mapped_scans']
            stamp=driver.get_clock().now().to_msg()
            pose=Odometry();pose.header=Header(stamp=stamp,frame_id='map');pose.child_frame_id='base_link'
            pose.pose.pose.orientation.w=1.;pose.pose.covariance=(np.eye(6)*variance).ravel().tolist()
            odom_pub.publish(pose);cloud_pub.publish(xyz_cloud(points,Header(stamp=stamp,frame_id='sensors')))
            deadline=time.monotonic()+3.
            while mapper.stats['mapped_scans']==before and time.monotonic()<deadline:drain(.01)
            assert mapper.stats['mapped_scans']==before+1,mapper.stats
        def publish():
            mapper.publish();mapper.publish_global();drain(.15)
        def value(key,layer,x=2.02,y=.02):
            msg=messages[key][-1];g=msg.info;r=g.resolution
            nx,ny=round(g.length_x/r),round(g.length_y/r)
            a=np.asarray(msg.data[msg.layers.index(layer)].data).reshape(ny,nx)[::-1,::-1]
            ox=g.pose.position.x-g.length_x/2;oy=g.pose.position.y-g.length_y/2
            return a[int(np.floor((y-oy)/r)),int(np.floor((x-ox)/r))]
        try:
            drain(.5)
            ground=np.array([[2.02+x,.02+y,0.] for x in [-.4,-.2,0.,.2,.4] for y in [-.4,-.2,0.,.2,.4]])
            obstacle=np.array([[2.02,.02,1.]])
            unseen=np.array([[30.02,10.02,2.]])
            scan(np.r_[ground,obstacle,unseen]-sensor[:3,3]);publish()
            assert all(value(k,'obstacle')==1. for k in messages)
            free=np.array([[4.04,.04,0.],[4.04,.4,0.],[4.04,-.4,0.],[4.04,0.,.4]])
            scan(free);scan(free)
            assert mapper.cleanup.votes
            scan(free,variance=1.)
            assert not mapper.cleanup.votes,'Uncertain pose kept clearance votes'
            for _ in range(2):
                scan(free)
                assert mapper.stats['cleanup']['removed_total']==0
            scan(free);publish()
            assert mapper.stats['cleanup']['removed_total']==1,mapper.stats
            for key in messages:
                assert value(key,'obstacle')==0.,key
                assert value(key,'elevation')==0.,key
            assert np.any(np.all(np.isclose(mapper.cloud.preview(),unseen),axis=1))
            assert not np.any(np.all(np.isclose(mapper.cloud.preview(),obstacle),axis=1))
            assert np.isnan(value('global','elevation',12.,8.)),'Unobserved cell became known'
            assert mapper.stats['dropped_scans']==0
            result=dict(passed=True,mapped_scans=mapper.stats['mapped_scans'],
                        removed_total=mapper.stats['cleanup']['removed_total'],
                        local_and_global_updated=True,unobserved_preserved=True,
                        uncertain_pose_resets_confirmation=True,scope='Isolated ROS domain 72')
            out=ROOT/'results/map_visual_20260916';out.mkdir(exist_ok=True)
            (out/'dynamic_map_verification.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
        finally:
            executor.shutdown();mapper.dense_writer.close();mapper.grid.close();mapper.delivery.close();mapper.cloud.close();mapper.tum.close()
            mapper.destroy_node();driver.destroy_node();rclpy.try_shutdown()


if __name__=='__main__':main()
