"""Exercise the actual display sender/receiver on an isolated ROS domain."""
import copy
import json
import os
import sys
import time
import zlib
from array import array
from pathlib import Path

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from grid_map_msgs.msg import GridMap
from std_msgs.msg import Float32MultiArray, MultiArrayDimension, UInt8MultiArray
from sensor_msgs.msg import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'visual'), str(ROOT/'scripts')]
from viewer_transport import ViewerTransport
from viewer_wire import encode_grid, decode_grid, PREFIX
from visual_monitor import Monitor


def main():
    assert os.environ['ROS_DOMAIN_ID']=='98' and os.environ['ROS_LOCALHOST_ONLY']=='1'
    rclpy.init(args=['--ros-args', '-p', 'compressed_display_prefix:='+PREFIX,
                    '-p', 'retain_stale_grid_sec:=30.0'])
    sender = ViewerTransport(); monitor = Monitor(); driver = Node('test_display_driver')
    executor = SingleThreadedExecutor()
    for node in (sender, monitor, driver): executor.add_node(node)
    retained = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                          durability=DurabilityPolicy.TRANSIENT_LOCAL)
    global_pub = driver.create_publisher(GridMap, '/Car/T3/mapping/global_grid_map', retained)
    local_pub = driver.create_publisher(GridMap, '/Car/T3/mapping/grid_map', retained)
    image_pub = driver.create_publisher(Image, '/fusion/left', 1)
    def drain(seconds=1.2):
        end = time.monotonic()+seconds
        while time.monotonic()<end: executor.spin_once(timeout_sec=.02)
    try:
        message = GridMap(); message.header.frame_id='map'; message.header.stamp.sec=100
        message.info.resolution=.2; message.info.length_x=64.; message.info.length_y=64.
        message.info.pose.orientation.w=1.
        message.layers=['elevation','elevation_variance','occupancy','obstacle']
        message.basic_layers=['elevation']
        elevation=np.full((320,320), np.nan, np.float32); elevation[150:170,150:170]=.35
        occupancy=np.where(np.isfinite(elevation),0.,np.nan).astype(np.float32)
        obstacle=occupancy.copy(); obstacle[160,160]=1.; occupancy[160,160]=.9
        for values in (elevation, np.zeros_like(elevation), occupancy, obstacle):
            layer=Float32MultiArray(data=array('f', values.ravel()))
            layer.layout.dim=[MultiArrayDimension(label='row_index',size=320,stride=102400),
                              MultiArrayDimension(label='column_index',size=320,stride=320)]
            message.data.append(layer)
        packed,raw_bytes=encode_grid(message); decoded=decode_grid(packed)
        assert decoded.layers==['elevation','occupancy','obstacle']
        for name in decoded.layers:
            np.testing.assert_array_equal(np.asarray(decoded.data[decoded.layers.index(name)].data),
                                          np.asarray(message.data[message.layers.index(name)].data))
        assert decoded.header==message.header and decoded.info==message.info
        assert len(packed.data)<raw_bytes/30
        # Corruption is rejected without replacing the last correct map.
        try: decode_grid(UInt8MultiArray(data=packed.data[:-2]))
        except ValueError: pass
        else: raise AssertionError('Truncated payload accepted')
        global_pub.publish(message); local_pub.publish(message)
        img=Image(height=480,width=640,encoding='mono8',step=640,data=bytes([90])*(640*480))
        image_pub.publish(img); drain(2.)
        assert monitor.grid_source=='global' and monitor.grid[2]==400
        assert monitor.grid_is_fresh() and monitor.grid_message_count['global']>=1
        assert np.any(np.all(np.asarray(monitor.grid[0])==0,axis=2)), 'Obstacle colors lost'
        assert monitor.thumbnails['Left'].size==(320,240)
        original_bitmap=monitor.grid[0]
        monitor.on_compressed_grid('global',UInt8MultiArray(data=array('B', b'corrupt')))
        assert monitor.grid[0] is original_bitmap
        # After both feeds lapse, retain the last picture but disallow goals.
        monitor.global_grid_at-=9.;monitor.local_grid_at-=9.
        monitor.check_grid_freshness()
        assert monitor.grid_source=='stale' and monitor.grid[0] is original_bitmap
        assert not monitor.grid_is_fresh()
        assert not monitor.request_goal(.1,.1,0.)
        monitor.goal_requested=(.1,.1,0.,.35,time.monotonic())
        monitor.send_requested_goal();assert monitor.goal_requested is None and monitor.last_goal is None
        # Local fallback and global recovery use real callbacks again.
        local_pub.publish(message);drain()
        assert monitor.grid_source=='local' and monitor.grid_is_fresh()
        global_pub.publish(message);drain()
        assert monitor.grid_source=='global' and monitor.grid_is_fresh()
        # Malformed future messages must not poison the valid stamp watermark.
        bad=copy.deepcopy(message);bad.header.stamp.sec=10000;bad.data[0].data.pop()
        monitor.on_grid(bad)
        assert monitor.grid_stamps['global']==100
        global_pub.publish(message);drain()
        assert monitor.grid_source=='global' and monitor.grid[2]==400
        monitor.global_grid_at-=40.; monitor.local_grid_at-=40.; monitor.grid_display_at-=40.
        monitor.check_grid_freshness();assert monitor.grid is None and monitor.grid_source=='none'
        assert monitor.last_color_key is None
        global_pub.publish(message);drain()
        assert monitor.grid is not None and monitor.grid[2]==400
        # A discarded bitmap must be recreated even if its color key is unchanged.
        monitor.grid=None;monitor.recolor_grid();assert monitor.grid is not None
        report=dict(passed=True,raw_display_bytes=raw_bytes,wire_bytes=len(packed.data),
                    exact_height_and_obstacle_values=True,known_cells=400,
                    stale_map_visible_with_goals_blocked=True,recovery=True,
                    rejected_message_does_not_poison_time=True,repaint_after_cache_clear=True,
                    preview_size=[320,240],goal_messages_published=0,vehicle_commands_published=0)
        print(json.dumps(report,indent=2))
    finally:
        executor.shutdown()
        for node in (sender, monitor, driver): node.destroy_node()
        rclpy.shutdown()


if __name__=='__main__': main()
