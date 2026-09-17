#!/usr/bin/env python3
"""Receive live sensors in domain 10 and forward them to mapping in domain 57.

Vehicle control belongs to the operator. This node has no command publishers.
"""
import argparse
import json
import os
from pathlib import Path
import signal
import sys
import time
import threading

import rclpy
from rclpy.context import Context
from rclpy.executors import SingleThreadedExecutor, ExternalShutdownException
from rclpy.qos import QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from rclpy.signals import SignalHandlerOptions
from sensor_msgs.msg import Image, Imu, PointCloud2
from nav_msgs.msg import Odometry
from std_msgs.msg import String
import yaml

from control import alive, identity, save as save_run_state
from stereo_transport import StereoNormalizer
from live_transport import SensorInbox

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
    inbox = SensorInbox(profile)
    worker_state = {'busy_since': None, 'group': None, 'max_forward_sec': 0.0,
                    'recording_error': None}
    metadata = [None]
    status = {'mapping': {}, 'fusion': {}}
    writer = None
    recording_path = None
    if args.record_input:
        import rosbag2_py
        from rclpy.serialization import serialize_message
        recording_path = out/'live_input_bag'
        if recording_path.exists():
            recording_path = out/f'live_input_bag_resume_{time.time_ns()}'
        writer = rosbag2_py.SequentialWriter()
        writer.open(rosbag2_py.StorageOptions(uri=str(recording_path), storage_id='sqlite3'),
                    rosbag2_py.ConverterOptions('', ''))
        for key, topic in topics.items():
            writer.create_topic(rosbag2_py.TopicMetadata(name=topic, serialization_format='cdr',
                type='sensor_msgs/msg/'+types[key].__name__))

    def forward_groups():
        nonlocal writer
        while True:
            group = inbox.take()
            if group is None:
                return
            started = time.monotonic()
            worker_state.update(busy_since=started, group=[s.key for s in group])
            try:
                if not all(inbox.fresh(s) for s in group):
                    for s in group:
                        inbox.count(s.key+'_stale_at_forward')
                    continue
                # Validate and normalize BOTH images before publishing either one.
                messages = [normalizer.convert(s.message, 0 if s.key == 'left' else 1)
                            if s.key in ('left', 'right') else s.message for s in group]
                if not all(inbox.fresh(s) for s in group):
                    for s in group:
                        inbox.count(s.key+'_stale_after_convert')
                    continue
                delivered = []
                for sample, message in zip(group, messages):
                    if inbox.closed or not inbox.fresh(sample):
                        inbox.count(sample.key+'_stale_at_publish')
                        break
                    publishers[sample.key].publish(message)
                    inbox.forwarded(sample)
                    delivered.append((sample, message, time.time_ns()))
                # Disk recording cannot block the receive executor or split stereo
                # publication. A disk error disables recording, not live sensors.
                if writer is not None:
                    try:
                        for sample, message, received_ns in delivered:
                            writer.write(topics[sample.key], serialize_message(message), received_ns)
                    except Exception as error:
                        worker_state['recording_error'] = str(error)
                        inbox.count('recording_errors')
                        writer = None
                        source.get_logger().error(f'Input recording disabled: {error}')
            except Exception as error:
                for sample in group:
                    inbox.count(sample.key+'_rejected')
                source.get_logger().warning(f'Live forwarding: {error}', throttle_duration_sec=5)
            finally:
                worker_state['max_forward_sec'] = max(worker_state['max_forward_sec'], time.monotonic()-started)
                worker_state.update(busy_since=None, group=None)

    def receive_meta(message):
        if len(message.data) > 1024*1024:
            inbox.count('meta_oversized')
            return
        try:
            metadata[0] = json.loads(message.data)
            inbox.count('meta_received')
        except (ValueError, TypeError):
            inbox.count('meta_invalid')

    for key in ('left', 'right', 'lidar', 'imu'):
        reliability = profile.get('lidar_input_reliability' if key == 'lidar' else 'image_input_reliability', 'reliable')
        qos = qos_profile_sensor_data if key == 'imu' else QoSProfile(depth=2,
            reliability=ReliabilityPolicy.RELIABLE if reliability == 'reliable' else ReliabilityPolicy.BEST_EFFORT)
        source.create_subscription(types[key], profile[key+'_topic'], lambda m, k=key: inbox.offer(k, m), qos)
    source.create_subscription(String, '/sensor/meta', receive_meta, 1)
    target.create_subscription(Odometry, '/T3/semantic/current_pose',
        lambda _: inbox.count('fused_messages'), 10)
    def receive_status(message, key):
        try:
            status[key] = json.loads(message.data)
        except (ValueError, TypeError):
            inbox.count('invalid_status')
    for key, topic in [('mapping', '/fusion/map_status'), ('fusion', '/fusion/status')]:
        target.create_subscription(String, topic, lambda m, k=key: receive_status(m, k), 3)
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
        snapshot = inbox.snapshot()
        transport = dict(worker_state)
        transport['busy_seconds'] = 0.0 if transport['busy_since'] is None else now-transport['busy_since']
        transport['degraded'] = transport['busy_seconds'] > max(inbox.max_age.values())
        data = dict(kind='manual_live_mapping', status=phase, source_domain=10, mapping_domain=57,
            vehicle_commands_published=0, replay_used=False, **snapshot,
            elapsed_sec=now-began, transport=transport, last_map_status=status['mapping'],
            last_fusion_status=status['fusion'], stop_reason=stop_reason if phase == 'stopped' else None,
            recorded_input_bag=str(recording_path) if recording_path is not None else None)
        temporary = out/'live_session_status.json.tmp'
        temporary.write_text(json.dumps(data, indent=2)+'\n')
        temporary.replace(out/'live_session_status.json')
        if metadata[0] is not None:
            temporary = out/'latest_sensor_meta.json.tmp'
            temporary.write_text(json.dumps(metadata[0], indent=2)+'\n')
            temporary.replace(out/'latest_sensor_meta.json')

    print('MANUAL LIVE: sensor relay ready; vehicle remains under operator control.', flush=True)
    def receive_loop():
        try:
            executors[0].spin()
        except ExternalShutdownException:
            pass
        except Exception as error:
            inbox.count('receive_executor_errors')
            print(f'Receive executor failed: {error}', flush=True)
            stopping[0] = True
    receiver = threading.Thread(target=receive_loop, name='sensor_receive', daemon=True)
    worker = threading.Thread(target=forward_groups, name='sensor_forward', daemon=True)
    receiver.start()
    worker.start()
    try:
        while not stopping[0] and not (out/'live.stop').exists():
            executors[1].spin_once(timeout_sec=.02)
            if time.monotonic()-last_report >= 2:
                if not alive(state['processes']['pipeline']):
                    stop_reason = 'mapping_pipeline_stopped'
                    break
                report('running')
                print(json.dumps(dict(counts=inbox.snapshot()['counts'], mapped_scans=status['mapping'].get('mapped_scans', 0))), flush=True)
                last_report = time.monotonic()
    finally:
        inbox.close()
        worker.join(timeout=2.0)
        for executor in executors:
            executor.shutdown()
        for context in contexts:
            context.shutdown()
        receiver.join(timeout=2.0)
        worker.join(timeout=3.0)
        if worker.is_alive():
            stop_reason = 'forward_worker_shutdown_timeout'
            report('failed')
            # Do not destroy publishers concurrently with a blocked DDS write.
            raise RuntimeError('DDS forward worker did not stop after context shutdown')
        writer = None
        report('stopped')
        for node in nodes:
            node.destroy_node()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
