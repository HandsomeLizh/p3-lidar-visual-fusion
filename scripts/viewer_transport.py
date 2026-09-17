#!/usr/bin/env python3
"""Small display-only feeds; leave estimation and planner topics untouched."""
import argparse
import io
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image as PILImage
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from sensor_msgs.msg import Image, CompressedImage
from grid_map_msgs.msg import GridMap
from std_msgs.msg import UInt8MultiArray

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'visual'))
from viewer_wire import PREFIX, encode_grid


class ViewerTransport(Node):
    def __init__(self, status_file=None):
        super().__init__('viewer_transport', namespace='/viewer104_transport')
        self.status_file = status_file
        self.stats = dict(ready=True, protocol=1, pid=os.getpid())
        self.status_at = 0.
        self.image_at = {}
        self.pending = {}
        retained = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                              durability=DurabilityPolicy.TRANSIENT_LOCAL)
        live = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE)
        self.grid_pubs = {}
        for key, topic in [('global', '/Car/T3/mapping/global_grid_map'),
                           ('local', '/Car/T3/mapping/grid_map')]:
            self.grid_pubs[key] = self.create_publisher(UInt8MultiArray, PREFIX+'/'+key+'_grid_zlib', retained)
            self.create_subscription(GridMap, topic, lambda msg, k=key: self.pending.__setitem__(k, msg), retained)
        self.image_pubs = {}
        for side in ('left', 'right'):
            self.image_pubs[side] = self.create_publisher(CompressedImage, PREFIX+'/'+side+'/compressed', live)
            self.create_subscription(Image, '/fusion/'+side, lambda msg, s=side: self.preview(s, msg), live)
        # At most one pending message per source; never build a catch-up queue.
        self.create_timer(.1, self.flush)
        self.get_logger().info('Compressed display transport ready')

    def preview(self, side, msg):
        now = time.monotonic()
        if now - self.image_at.get(side, -float('inf')) < .5:
            return
        self.image_at[side] = now
        try:
            channels = {'mono8': 1, 'rgb8': 3, 'bgr8': 3, 'rgba8': 4, 'bgra8': 4}.get(msg.encoding)
            if channels is None or len(msg.data) > 24_000_000:
                return
            pixels = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.step)
            pixels = pixels[:, :msg.width*channels].reshape(msg.height, msg.width, channels)
            pixels = pixels[:, :, 0] if channels == 1 else pixels[:, :, :3]
            if msg.encoding.startswith('bgr'):
                pixels = pixels[:, :, ::-1]
            bitmap = PILImage.fromarray(pixels)
            bitmap.thumbnail((320, 240))
            stream = io.BytesIO(); bitmap.save(stream, format='JPEG', quality=70)
            data = stream.getvalue()
            self.image_pubs[side].publish(CompressedImage(header=msg.header, format='jpeg', data=data))
            self.stats[side] = dict(input_bytes=len(msg.data), wire_bytes=len(data), receipt=time.time())
        except (ValueError, TypeError, OSError) as error:
            self.get_logger().warning('Preview rejected: '+str(error), throttle_duration_sec=5.)

    def flush(self):
        for key in list(self.pending):
            message = self.pending.pop(key)
            started = time.monotonic()
            try:
                packed, raw_size = encode_grid(message)
                self.grid_pubs[key].publish(packed)
                self.stats[key] = dict(input_array_bytes=sum(len(x.data)*4 for x in message.data),
                    display_cdr_bytes=raw_size, wire_bytes=len(packed.data),
                    encode_sec=time.monotonic()-started, layers=list(message.layers),
                    stamp=message.header.stamp.sec+message.header.stamp.nanosec*1e-9,
                    messages=self.stats.get(key, {}).get('messages', 0)+1, receipt=time.time())
            except (ValueError, IndexError, TypeError) as error:
                self.get_logger().warning('Display grid rejected: '+str(error), throttle_duration_sec=5.)
        if self.status_file and time.monotonic()-self.status_at>=1.:
            self.status_at=time.monotonic()
            try:
                self.stats['updated_at']=time.time()
                tmp = self.status_file.with_suffix('.tmp')
                tmp.write_text(json.dumps(self.stats, indent=2)); tmp.replace(self.status_file)
            except OSError as error:
                self.get_logger().warning('Display status could not be saved: '+str(error), throttle_duration_sec=5.)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-state', type=Path)
    args = parser.parse_args()
    state = json.loads(args.run_state.read_text()) if args.run_state else None
    status = Path(state['output'])/'viewer_transport_status.json' if state else None
    rclpy.init()
    node = ViewerTransport(status)
    if state:
        from control import alive
        pipeline = state['processes']['pipeline']
        node.create_timer(1., lambda: rclpy.shutdown() if not alive(pipeline) and rclpy.ok() else None)
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok(): rclpy.shutdown()


if __name__ == '__main__':
    main()
