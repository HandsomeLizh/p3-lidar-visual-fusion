#!/usr/bin/env python3
"""Replay recorded live clouds through the unchanged adapter/backend, varying only epoch/clock.

Diagnostic only: does not start a vehicle, visual frontend, fusion, or map publisher.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import signal
import sqlite3
import subprocess
import time

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]


def write(path, data):
    path.write_text(json.dumps(data, indent=2) + '\n')


def clouds(run):
    import rclpy.serialization
    from sensor_msgs.msg import PointCloud2
    bag = run / 'live_input_bag'
    metadata = yaml.safe_load((bag / 'metadata.yaml').read_text())['rosbag2_bagfile_information']
    for part in metadata['relative_file_paths']:
        with sqlite3.connect('file:' + str(bag / part) + '?mode=ro', uri=True) as database:
            rows = database.execute("SELECT m.data FROM messages m JOIN topics t ON t.id=m.topic_id "
                                    "WHERE t.type='sensor_msgs/msg/PointCloud2' ORDER BY m.timestamp,m.id")
            for (data,) in rows:
                yield rclpy.serialization.deserialize_message(data, PointCloud2)


def replay(run, out, mode, overrides=None):
    import rclpy
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.node import Node
    from nav_msgs.msg import Odometry
    from rosgraph_msgs.msg import Clock
    from std_msgs.msg import String
    from t3_lidar_visual_fusion.sensor_adapter import SensorAdapter
    os.environ['ROS_DOMAIN_ID'] = '58'
    os.environ['ROS_LOCALHOST_ONLY'] = '1'
    case = out / mode
    case.mkdir()
    params = yaml.safe_load((run / 'voxelmap_ros_parameters.yaml').read_text())
    settings = params['/**']['ros__parameters']
    settings.update(timing_path=str(case / 'lidar_metrics.jsonl'), use_sim_time=mode == 'rebased')
    settings.update(overrides or {})
    (case / 'parameters.yaml').write_text(yaml.safe_dump(params))
    log = (case / 'backend.log').open('w')
    command = [str(ROOT / 'install/t3_voxelmap/lib/t3_voxelmap/voxelmap_node'),
               '--ros-args', '--params-file', str(case / 'parameters.yaml')]
    process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
    rclpy.init(args=['--ros-args', '-p', 'profile_path:=' + str(run / 'profile.yaml'),
                    '-p', 'normalized_camera_input:=true'])
    adapter = SensorAdapter()
    observer = Node('live_lidar_timestamp_observer')
    executor = SingleThreadedExecutor()
    executor.add_node(adapter)
    executor.add_node(observer)
    results = []
    quality = []

    def received(message):
        p, q = message.pose.pose.position, message.pose.pose.orientation
        results.append(dict(stamp_ns=message.header.stamp.sec * 10**9 + message.header.stamp.nanosec,
                            pose=[p.x, p.y, p.z, q.x, q.y, q.z, q.w],
                            covariance=list(message.pose.covariance)))

    observer.create_subscription(Odometry, '/fusion/lio_raw', received, 10)
    observer.create_subscription(String, '/fusion/lidar_quality', lambda m: quality.append(json.loads(m.data)), 10)
    clock = observer.create_publisher(Clock, '/clock', 10)
    started = time.monotonic()
    count = 0
    first = None
    try:
        while adapter.cloud_pub.get_subscription_count() < 1 or observer.count_publishers('/fusion/lio_raw') < 1:
            executor.spin_once(timeout_sec=.05)
            if process.poll() is not None or time.monotonic() - started > 10:
                raise RuntimeError('Replay backend did not become ready')
        for message in clouds(run):
            raw = message.header.stamp.sec * 10**9 + message.header.stamp.nanosec
            if first is None:
                first = raw
            stamp = 10**12 + raw - first if mode == 'rebased' else raw
            message.header.stamp.sec, message.header.stamp.nanosec = divmod(stamp, 10**9)
            tick = Clock()
            tick.clock = copy.deepcopy(message.header.stamp)
            clock.publish(tick)
            adapter.cloud(message)
            count += 1
            deadline = time.monotonic() + 5
            while len(results) < count:
                executor.spin_once(timeout_sec=.01)
                if time.monotonic() > deadline or process.poll() is not None:
                    raise RuntimeError('Missing replay pose at frame ' + str(count))
            if results[-1]['stamp_ns'] != stamp:
                raise RuntimeError('Replay output/header mismatch')
            results[-1]['original_stamp_ns'] = raw
        final_deadline = time.monotonic() + .2
        while len(quality) < count and time.monotonic() < final_deadline:
            executor.spin_once(timeout_sec=.01)
        write(case / 'poses.json', results)
        write(case / 'quality.json', quality)
        print(json.dumps(dict(mode=mode, frames=count, elapsed_sec=time.monotonic()-started)), flush=True)
    finally:
        process.send_signal(signal.SIGINT)
        try:
            process.wait(timeout=8)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        log.close()
        executor.shutdown()
        adapter.destroy_node()
        observer.destroy_node()
        rclpy.shutdown()
    return results


def difference(a, b):
    x, y = np.asarray(a), np.asarray(b)
    positions = np.linalg.norm(x[:, :3]-y[:, :3], axis=1)
    # Absolute quaternion dot handles equivalent q/-q representations.
    dots = np.clip(np.abs(np.sum(x[:, 3:]*y[:, 3:], axis=1)), 0, 1)
    angles = 2*np.arccos(dots)
    return dict(frames=len(x), position_rmse_m=float(np.sqrt(np.mean(positions**2))),
                max_position_difference_m=float(positions.max()),
                max_rotation_difference_deg=float(np.degrees(angles.max())))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    args = parser.parse_args()
    run = args.run.resolve()
    if not run.is_relative_to(ROOT / 'results'):
        raise ValueError('Choose an owned test result')
    out = run / 'timestamp_replay'
    out.mkdir(exist_ok=False)
    online = json.loads((run / 'verification.json').read_text())['raw_poses']
    online = [x for x in online if x['source'] == 'lio_raw']
    original = replay(run, out, 'original')
    rebased = replay(run, out, 'rebased')
    if len(online) != len(original) or len(original) != len(rebased):
        raise RuntimeError('Online and replay frame counts differ')
    for x, y in zip(online, original):
        if abs(x['stamp_sec']-y['original_stamp_ns']/1e9) > 1e-6:
            raise RuntimeError('Online/replay timestamp ordering differs')
    result = dict(
        online_vs_original_replay=difference([x['pose'] for x in online], [x['pose'] for x in original]),
        original_vs_rebased_replay=difference([x['pose'] for x in original], [x['pose'] for x in rebased]),
        backend_sha256=hashlib.sha256((ROOT/'install/t3_voxelmap/lib/t3_voxelmap/voxelmap_node').read_bytes()).hexdigest(),
        qualification=['Exact recorded clouds and existing SensorAdapter; unchanged backend executable and geometry parameters.',
                       'Original headers + wall clock compared with epoch 1000 + simulation clock; preserved acquisition intervals.',
                       'Each replay waits for the pose before publishing the next cloud; replay wall rate is accelerated.',
                       'This isolates the LiDAR frontend, not full fusion or camera/reference exposure synchronization.'])
    write(out / 'comparison.json', result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
