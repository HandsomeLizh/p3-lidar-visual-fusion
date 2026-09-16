#!/usr/bin/env python3
"""Passive P3 comparison recorder. Ground truth is never published."""
import argparse
import json
from pathlib import Path
import time
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud2
from grid_map_msgs.msg import GridMap
from std_msgs.msg import Header, String


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    rclpy.init()
    node = Node('p3_comparison_observer')
    counts, timings, receipts, frames = {}, [], [], set()
    details = {}
    live = QoSProfile(depth=200, reliability=ReliabilityPolicy.RELIABLE)
    retained = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                         durability=DurabilityPolicy.TRANSIENT_LOCAL)
    trajectory = (args.output / 'observed_odometry.tum').open('w', buffering=1)

    def pose(m):
        counts['poses'] = counts.get('poses', 0) + 1
        p, q = m.pose.pose.position, m.pose.pose.orientation
        t = m.header.stamp.sec + 1e-9 * m.header.stamp.nanosec
        trajectory.write(' '.join(format(v, '.12g') for v in
                         (t, p.x, p.y, p.z, q.x, q.y, q.z, q.w)) + '\n')
        frames.add((m.header.frame_id, m.child_frame_id))

    def timing(m):
        record = json.loads(m.data)
        record['observer_receipt_monotonic_sec'] = time.monotonic()
        timings.append(record)
        with (args.output / 'p3_timing.jsonl').open('a') as stream:
            stream.write(json.dumps(record) + '\n')

    def grid(m):
        import numpy as np
        counts['grid'] = counts.get('grid', 0) + 1
        details['grid_layers'] = list(m.layers)
        if 'elevation' in m.layers:
            a = np.asarray(m.data[list(m.layers).index('elevation')].data)
            details['finite_height_cells'] = int(np.isfinite(a).sum())

    def cloud(m):
        counts['cloud'] = counts.get('cloud', 0) + 1
        details['preview_points'] = m.width * m.height

    node.create_subscription(Odometry, '/T3/semantic/current_pose', pose, live)
    node.create_subscription(String, '/Car/T3/metrics/frame_timing', timing, live)
    node.create_subscription(Header, '/Car/T3/debug/input_processed',
        lambda m: receipts.append(m.stamp.sec + m.stamp.nanosec * 1e-9), live)
    node.create_subscription(GridMap, '/T3/mapping/global_grid_map', grid, retained)
    node.create_subscription(PointCloud2, '/T3/mapping/lidar_map', cloud, retained)
    (args.output / 'observer.ready').write_text('ready\n')
    started = time.monotonic()
    while not (args.output / 'observer.stop').exists() and time.monotonic()-started < 1800:
        rclpy.spin_once(node, timeout_sec=.1)
    trajectory.close()
    (args.output / 'p3_observation.json').write_text(json.dumps(dict(
        counts=counts, details=details, pose_frames=sorted(frames), receipts=receipts,
        timings=timings, duration_sec=time.monotonic()-started), indent=2))
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
