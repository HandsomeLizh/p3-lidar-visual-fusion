#!/usr/bin/env python3
"""Receive live sensors in domain 10 and forward them to mapping in domain 57.

Vehicle control belongs to the operator. This node has no command publishers.
"""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import signal
import sys
import time

import rclpy
from rclpy.context import Context
from rclpy.executors import SingleThreadedExecutor
from rclpy.qos import QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from rclpy.signals import SignalHandlerOptions
from sensor_msgs.msg import Image, Imu, PointCloud2
from nav_msgs.msg import Odometry
from std_msgs.msg import String
import yaml

from control import alive, identity, save as save_run_state
from stereo_transport import StereoNormalizer

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--record-input', action='store_true', help='Optionally record forwarded sensor data; disabled by default')
    args = parser.parse_args()
    out = args.output.resolve()
    state = json.loads((ROOT/'run_state.json').read_text())
    if not out.is_relative_to(ROOT/'results') or state['output'] != str(out):
        raise ValueError('Choose the active owned run')
    if state['domain'] != '57' or state['localhost_only'] != '0':
        raise ValueError('Live relay requires domain 57 with network discovery')
    if not alive(state['processes'].get('pipeline')):
        raise RuntimeError('Mapping pipeline is not running')
    profile = yaml.safe_load((out/'profile.yaml').read_text())
    if not profile.get('normalized_camera_input'):
        raise ValueError('Live profile must enable normalized_camera_input')
    normalizer = StereoNormalizer(profile)
    os.environ['ROS_LOCALHOST_ONLY'] = '0'
    contexts = [Context(), Context()]
    nodes, executors = [], []
    for context, domain, name in zip(contexts, [10, 57], ['fusion_live_sensor_source', 'fusion_live_sensor_transfer']):
        rclpy.init(context=context, domain_id=domain, signal_handler_options=SignalHandlerOptions.NO)
        node = rclpy.create_node(name, context=context)
        executor = SingleThreadedExecutor(context=context)
        executor.add_node(node)
        nodes.append(node); executors.append(executor)
    source, target = nodes
    topics = {'left': '/fusion/left', 'right': '/fusion/right',
              'lidar': profile['lidar_topic'], 'imu': profile['imu_topic']}
    types = {'left': Image, 'right': Image, 'lidar': PointCloud2, 'imu': Imu}
    publishers = {key: target.create_publisher(types[key], topic, 100 if key == 'imu' else 3)
                  for key, topic in topics.items()}
    counts, last_stamp, last_received = Counter(), {}, {}
    status = {'mapping': {}, 'fusion': {}}
    writer = None
    if args.record_input:
        import rosbag2_py
        from rclpy.serialization import serialize_message
        writer = rosbag2_py.SequentialWriter()
        writer.open(rosbag2_py.StorageOptions(uri=str(out/'live_input_bag'), storage_id='sqlite3'),
                    rosbag2_py.ConverterOptions('', ''))
        for key, topic in topics.items():
            writer.create_topic(rosbag2_py.TopicMetadata(name=topic, serialization_format='cdr',
                type='sensor_msgs/msg/'+types[key].__name__))

    def receive(key, message):
        counts[key+'_received'] += 1
        now = time.monotonic()
        last_received[key] = now
        ns = message.header.stamp.sec*10**9+message.header.stamp.nanosec
        if ns <= last_stamp.get(key, -1):
            counts[key+'_old_or_duplicate'] += 1
            return
        try:
            if key in ('left', 'right'):
                age = time.time()-ns/1e9
                if age > profile['visual_max_age_sec']-.5 or age < -profile.get('visual_future_tolerance_sec', .1):
                    counts[key+'_stale'] += 1
                    return
                message = normalizer.convert(message, 0 if key == 'left' else 1)
            elif key == 'lidar' and len(message.data) > profile.get('max_cloud_bytes', 16000000):
                raise ValueError('Cloud exceeds configured byte limit')
            publishers[key].publish(message)
            last_stamp[key] = ns
            counts[key+'_forwarded'] += 1
            if writer is not None:
                writer.write(topics[key], serialize_message(message), time.time_ns())
        except Exception as error:
            counts[key+'_rejected'] += 1
            source.get_logger().warning(f'{key}: {error}', throttle_duration_sec=5)

    for key in ('left', 'right', 'lidar', 'imu'):
        reliability = profile.get('lidar_input_reliability' if key == 'lidar' else 'image_input_reliability', 'reliable')
        qos = qos_profile_sensor_data if key == 'imu' else QoSProfile(depth=2,
            reliability=ReliabilityPolicy.RELIABLE if reliability == 'reliable' else ReliabilityPolicy.BEST_EFFORT)
        source.create_subscription(types[key], profile[key+'_topic'], lambda m, k=key: receive(k, m), qos)
    target.create_subscription(Odometry, '/T3/semantic/current_pose',
        lambda _: counts.update(['fused_messages']), 10)
    for key, topic in [('mapping', '/fusion/map_status'), ('fusion', '/fusion/status')]:
        target.create_subscription(String, topic, lambda m, k=key: status.__setitem__(k, json.loads(m.data)), 3)
    stopping = [False]
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stopping.__setitem__(0, True))
    state['processes']['live_relay'] = dict(pid=os.getpid(), identity=identity(os.getpid()),
        command=[sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]], log=str(out/'live_relay.log'))
    save_run_state(state)
    began, last_report = time.monotonic(), 0.
    stop_reason = 'operator_stop'

    def report(phase):
        now = time.monotonic()
        data = dict(kind='manual_live_mapping', status=phase, source_domain=10, mapping_domain=57,
            vehicle_commands_published=0, replay_used=False, counts=dict(counts),
            elapsed_sec=now-began, sensor_idle_seconds={key: now-value for key, value in last_received.items()},
            last_forwarded_stamp_ns=dict(last_stamp), last_map_status=status['mapping'],
            last_fusion_status=status['fusion'], stop_reason=stop_reason if phase == 'stopped' else None,
            recorded_input_bag=str(out/'live_input_bag') if args.record_input else None)
        temporary = out/'live_session_status.json.tmp'
        temporary.write_text(json.dumps(data, indent=2)+'\n')
        temporary.replace(out/'live_session_status.json')

    print('MANUAL LIVE: sensor relay ready; vehicle remains under operator control.', flush=True)
    try:
        while not stopping[0] and not (out/'live.stop').exists():
            for executor in executors:
                executor.spin_once(timeout_sec=.005)
            if time.monotonic()-last_report >= 2:
                if not alive(state['processes']['pipeline']):
                    stop_reason = 'mapping_pipeline_stopped'
                    break
                report('running')
                print(json.dumps(dict(counts=counts, mapped_scans=status['mapping'].get('mapped_scans', 0))), flush=True)
                last_report = time.monotonic()
    finally:
        report('stopped')
        writer = None
        for executor in executors:
            executor.shutdown()
        for node in nodes:
            node.destroy_node()
        for context in contexts:
            context.shutdown()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
