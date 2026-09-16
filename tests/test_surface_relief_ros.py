"""Exercise real mapper callbacks and local/global ROS output during height drift."""
import json
import os
from pathlib import Path
import tempfile
import time
from collections import deque
import numpy as np
import yaml
import rclpy
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from rclpy.executors import SingleThreadedExecutor
from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Header
from grid_map_msgs.msg import GridMap
from t3_lidar_visual_fusion.terrain_mapper import TerrainMapper
from t3_lidar_visual_fusion.ros_utils import xyz_cloud

ROOT = Path(__file__).resolve().parents[1]


def main():
    assert os.environ['ROS_DOMAIN_ID'] == '81'
    with tempfile.TemporaryDirectory(dir=ROOT/'build', prefix='surface_ros_') as td:
        tmp = Path(td)
        cfg = yaml.safe_load((ROOT/'config/simulation_live.yaml').read_text())
        sensor = np.eye(4); sensor[2, 3] = 1.
        cfg.update(map_window=8., tile_cells=16, semantic_topic='',
                   map_publish_period=100., global_publish_period=100.,
                   mapping_pose_settle_sec=0., base_from_lidar=sensor.tolist())
        cfg['dynamic_map']['enabled'] = False
        profile = tmp/'profile.yaml'; profile.write_text(yaml.safe_dump(cfg))
        rclpy.init(args=['--ros-args', '-p', 'profile_path:='+str(profile), '-p', 'output_dir:='+str(tmp/'map')])
        mapper = TerrainMapper(); driver = rclpy.create_node('surface_relief_fixture')
        executor = SingleThreadedExecutor(); executor.add_node(mapper); executor.add_node(driver)
        odom = driver.create_publisher(Odometry, '/T3/semantic/current_pose', 10)
        clouds = driver.create_publisher(PointCloud2, '/fusion/lidar', 10)
        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        messages = {key: deque(maxlen=1) for key in ['local', 'global']}
        for key, topic in [('local', '/Car/T3/mapping/grid_map'), ('global', '/T3/mapping/global_grid_map')]:
            driver.create_subscription(GridMap, topic, messages[key].append, qos)
        def drain(seconds):
            end = time.monotonic()+seconds
            while time.monotonic()<end: executor.spin_once(timeout_sec=.003)
        def scan(world_points, x, dz):
            before = mapper.stats['mapped_scans']
            stamp = driver.get_clock().now().to_msg()
            p = Odometry(); p.header = Header(stamp=stamp, frame_id='map'); p.child_frame_id='base_link'
            p.pose.pose.orientation.w=1.; p.pose.pose.position.x=float(x); p.pose.pose.position.z=float(dz)
            p.pose.covariance=(np.eye(6)*.0001).ravel().tolist()
            odom.publish(p)
            clouds.publish(xyz_cloud(world_points-[x,0.,1.], Header(stamp=stamp,frame_id='sensors')))
            end=time.monotonic()+4.
            while mapper.stats['mapped_scans']==before and time.monotonic()<end: drain(.01)
            assert mapper.stats['mapped_scans']==before+1, mapper.stats
        def publish():
            mapper.publish(); mapper.publish_global(); drain(.2)
        def layer(key, name):
            m=messages[key][-1]; nx=round(m.info.length_x/m.info.resolution); ny=round(m.info.length_y/m.info.resolution)
            a=np.asarray(m.data[m.layers.index(name)].data).reshape(ny,nx)[::-1,::-1]
            return a, m.info
        def value(key, name, x, y):
            a,g=layer(key,name); ox=g.pose.position.x-g.length_x/2; oy=g.pose.position.y-g.length_y/2
            return a[int(np.floor((y-oy)/g.resolution)),int(np.floor((x-ox)/g.resolution))]
        try:
            drain(.5)
            xx,yy=np.meshgrid(np.arange(-3.98,4.,.2),np.arange(-3.98,4.,.2))
            ground=np.column_stack([xx.ravel(),yy.ravel(),np.zeros(xx.size)])
            rocks=np.array([[2.02,.02,.8],[2.22,.02,.8],[2.02,.22,.8],[2.22,.22,.8]])
            started=time.monotonic()
            for index,dz in enumerate(np.linspace(0.,.36,13)):
                scan(np.r_[ground,rocks], index*.1, dz)
                publish()
                for key in messages:
                    assert value(key,'obstacle',-.38,.42)==0., (index,key)
                    assert value(key,'obstacle',2.02,.02)==1., (index,key)
            retained=value('global','elevation',-.38,.42)
            scan(ground+[30.,0.,0.],30.,.36); publish()
            assert value('global','elevation',-.38,.42)==retained
            assert value('global','obstacle',-.38,.42)==0.
            assert value('global','obstacle',2.02,.02)==1.
            assert np.isnan(value('global','elevation',15.,0.)), 'Unseen gap was filled'
            # Retained points from different times must not restore false relief.
            mapper.grid.rebuild_cells([[-2,2]],mapper.cloud);mapper.dirty=True;publish()
            assert value('global','obstacle',-.38,.42)==0.
            result=dict(passed=True,mapped_scans=mapper.stats['mapped_scans'],
                        injected_height_drift_m=.36,local_global_agree=True,
                        real_rocks_preserved=True,history_retained_after_30m_motion=True,
                        unobserved_gap_preserved=True,elapsed_sec=time.monotonic()-started,
                        scope='Isolated ROS 81; synthetic motion and height drift; no vehicle commands')
            target=Path(os.environ.get('T3_TEST_RESULTS',ROOT/'results/surface_relief'))
            target.mkdir(parents=True,exist_ok=True)
            (target/'surface_relief_ros.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
        finally:
            executor.shutdown();mapper.dense_writer.close();mapper.grid.close();mapper.delivery.close();mapper.cloud.close();mapper.tum.close()
            mapper.destroy_node();driver.destroy_node();rclpy.try_shutdown()


if __name__=='__main__':main()
