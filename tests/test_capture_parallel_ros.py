#!/usr/bin/env python3
"""Fake UE + real capture + real relay; isolated localhost domains 89/90 only."""
import argparse
from collections import defaultdict
from datetime import datetime
import json
import os
from pathlib import Path
import signal
import shlex
import socket
import struct
import subprocess
import sys
import threading
import time

import cv2
import numpy as np
import rclpy
from rclpy.context import Context
from rclpy.executors import SingleThreadedExecutor
from sensor_msgs.msg import Image, PointCloud2
from std_msgs.msg import String
import yaml

ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / 'results/capture_parallel_20260917'
MAGIC, DATA, HEAD = 0x4c494441, 4, 0xeb9aeb9a


def exact(sock, count):
    value = b''
    while len(value) < count:
        part = sock.recv(count-len(value))
        if not part:
            raise EOFError()
        value += part
    return value


def packet(batch, sequence, name, payload, sensor, camera, flags=0):
    header = struct.pack('<6I4B', HEAD, batch, 3, sequence, len(payload), 0,
                         sensor, camera, flags, 0)
    contents = header + name.encode('utf-16-le').ljust(128, b'\0') + payload
    return struct.pack('<4I', MAGIC, DATA, len(contents), MAGIC ^ DATA ^ len(contents)) + contents


class FakeUE:
    def __init__(self, faults=False, full=False):
        self.server = socket.socket()
        self.server.bind(('127.0.0.1', 0))
        self.server.listen(2)
        self.server.settimeout(.2)
        self.port = self.server.getsockname()[1]
        self.stop = threading.Event()
        self.requests = []
        self.errors = []
        self.faults = faults
        self.full = full
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

    def run(self):
        batch = 0
        while not self.stop.is_set():
            try:
                conn, _ = self.server.accept()
            except socket.timeout:
                continue
            conn.settimeout(2.)
            with conn:
                try:
                    while not self.stop.is_set():
                        magic, command, size, check = struct.unpack('<4I', exact(conn, 16))
                        exact(conn, size)
                        assert magic == MAGIC and check == (magic ^ command ^ size)
                        if command != 1:
                            continue
                        batch += 1
                        self.requests.append(dict(batch=batch, time=time.monotonic()))
                        if self.faults and batch == 1:
                            # Invalid allocation size must be rejected without allocating it.
                            huge = 512*1024*1024
                            conn.sendall(struct.pack('<4I', MAGIC, DATA, huge, MAGIC ^ DATA ^ huge))
                            break
                        image = np.full((2048, 2448, 3) if self.full else (120, 160, 3), batch % 251, np.uint8)
                        png = cv2.imencode('.png', image)[1].tobytes()
                        left = png[:35] if self.faults and batch == 2 else png
                        for camera, contents in [(0, left), (1, png)]:
                            wire = packet(batch, camera, f'rgb_{camera}.png', contents, 1, camera)
                            # Exercise TCP stream fragmentation, not one recv per message.
                            conn.sendall(wire[:11]); conn.sendall(wire[11:])
                        if self.faults and batch == 3:
                            break  # incomplete stereo+LiDAR batch is discarded on reconnect
                        if self.full:
                            points = np.zeros((131072, 3), dtype='<f4')
                            points[:, 0] = batch*100.
                            cloud = struct.pack('<I', len(points)) + points.tobytes()
                        else:
                            cloud = struct.pack('<I6f', 2, batch*100., 0., 0., batch*100., 100., 0.)
                        conn.sendall(packet(batch, 2, 'cloud.bin', cloud, 4, 0))
                        meta = json.dumps(dict(timestamp=datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f'),
                                               test_batch_id=batch, test_only=True)).encode()
                        conn.sendall(packet(batch, 3, 'meta.json', meta, 255, 255, 1))
                except (EOFError, BrokenPipeError, ConnectionResetError, socket.timeout):
                    pass
                except Exception as error:
                    self.errors.append(repr(error))

    def close(self):
        self.stop.set()
        self.thread.join(3.)
        self.server.close()


def launch_relay(folder, full=False):
    fixture = folder / 'relay_fixture'
    output = fixture / 'results/active'
    output.mkdir(parents=True)
    profile = dict(normalized_camera_input=True, input_image_size=[2448, 2048] if full else [160, 120], output_image_size=[640, 536] if full else [80, 60],
        left_topic='/test/left/image_raw', right_topic='/test/right/image_raw', lidar_topic='/test/points',
        imu_topic='/test/imu', visual_max_age_sec=4., stereo_max_skew=.01)
    (output/'profile.yaml').write_text(yaml.safe_dump(profile))
    pid = os.getpid()
    identity = Path(f'/proc/{pid}/stat').read_text().split(') ')[1].split()[19]
    state = dict(output=str(output), domain='57', localhost_only='0',
                 processes=dict(pipeline=dict(pid=pid, identity=identity)))
    (fixture/'run_state.json').write_text(json.dumps(state))
    # Only test harness overrides fixed domains/run-state destination. No production
    # state is read or written, and both contexts are restricted to localhost.
    script = f'''
import os,sys,json
from pathlib import Path
sys.path.insert(0,{str(ROOT/'scripts')!r})
import live_sensor_relay as relay
relay.ROOT = Path({str(fixture)!r})
relay.save_run_state = lambda s: (relay.ROOT/'run_state.json').write_text(json.dumps(s))
original = relay.rclpy.init
def isolated_init(*a, **kw):
    kw['domain_id'] = {{10:89,57:90}}[kw['domain_id']]
    os.environ['ROS_LOCALHOST_ONLY'] = '1'
    return original(*a, **kw)
relay.rclpy.init = isolated_init
sys.argv = ['relay','--output',{str(output)!r},'--record-input']
raise SystemExit(relay.main())
'''
    runner = folder/'relay_harness.py'
    runner.write_text(script)
    log = open(folder/'relay.log', 'w')
    return subprocess.Popen([sys.executable, str(runner)], stdout=log, stderr=subprocess.STDOUT), output, log


def run_case(binary, case):
    folder = RESULT / f'{case}_{time.time_ns()}'
    folder.mkdir(parents=True)
    env = dict(os.environ, ROS_DOMAIN_ID='89', ROS_LOCALHOST_ONLY='1',
               RMW_IMPLEMENTATION='rmw_cyclonedds_cpp')
    os.environ.update({k: env[k] for k in ('ROS_DOMAIN_ID', 'ROS_LOCALHOST_ONLY', 'RMW_IMPLEMENTATION')})
    nodes, executors, contexts = [], [], []
    received = {'raw': defaultdict(list), 'relay': defaultdict(list)}
    statuses, metas = [], []
    for domain, label in [(89, 'raw'), (90, 'relay')]:
        context = Context()
        rclpy.init(context=context, domain_id=domain)
        node = rclpy.create_node(f'capture_parallel_test_{label}', context=context)
        executor = SingleThreadedExecutor(context=context)
        executor.add_node(node)
        def observe(message, key, label=label):
            stamp = message.header.stamp.sec*10**9+message.header.stamp.nanosec
            marker = int(message.data[0]) if key != 'lidar' else round(struct.unpack_from('<f', message.data)[0])
            received[label][key].append(dict(stamp=stamp, marker=marker, age=time.time()-stamp/1e9))
        for key, topic in [('left', '/test/left/image_raw' if domain == 89 else '/fusion/left'),
                           ('right', '/test/right/image_raw' if domain == 89 else '/fusion/right'),
                           ('lidar', '/test/points')]:
            node.create_subscription(PointCloud2 if key == 'lidar' else Image, topic,
                lambda m, k=key, cb=observe: cb(m, k), 10)
        if domain == 89:
            node.create_subscription(String, '/sensor/batch_status', lambda m: statuses.append(json.loads(m.data)), 10)
            node.create_subscription(String, '/sensor/meta', lambda m: metas.append(json.loads(m.data)), 10)
        nodes.append(node); executors.append(executor); contexts.append(context)
    relay, relay_output, relay_log = launch_relay(folder, full=case == 'full')
    server = FakeUE(faults=case == 'faults', full=case == 'full')
    count = 230 if case == 'blocked' else 12
    params = {'tcp_host':'127.0.0.1', 'tcp_port':server.port, 'batch_count':count,
        'capture_freq':2. if case == 'full' else 10., 'batch_interval':0., 'timeout':2., 'save_data':False,
        'save_dir':str(folder/'scratch'), 'timestamp_mode':'meta_relative',
        'parallel_publication':case != 'serial', 'auto_start':True, 'publish_thumbnail':False,
        'output_topics':'[rgb_0:=/test/left/image_raw,rgb_1:=/test/right/image_raw,cpulidar_0:=/test/points]'}
    if case == 'blocked':
        env['LD_PRELOAD'] = str(RESULT/'publish_delay.so')
        env['P3_TEST_WRITE_DELAY_SEC'] = '18'
    command = [str(binary), '--ros-args']
    for key, value in params.items():
        command += ['-p', f'{key}:={str(value).lower() if isinstance(value, bool) else value}']
    cap_log = open(folder/'capture.log', 'w')
    capture = None
    errors = []
    rss_samples = []
    try:
        deadline = time.monotonic()+1.
        while time.monotonic() < deadline:
            for executor in executors: executor.spin_once(timeout_sec=.01)
        capture = subprocess.Popen(command, env=env, stdout=cap_log, stderr=subprocess.STDOUT)
        deadline = time.monotonic()+(31 if case == 'blocked' else 16)
        finished_at = None
        while time.monotonic() < deadline:
            for executor in executors: executor.spin_once(timeout_sec=.005)
            if capture.poll() is not None:
                raise AssertionError(f'Capture exited early {capture.returncode}')
            if relay.poll() is not None:
                raise AssertionError(f'Relay exited early {relay.returncode}')
            try:
                values = Path(f'/proc/{capture.pid}/status').read_text().splitlines()
                rss_samples.append(int(next(s for s in values if s.startswith('VmRSS:')).split()[1])/1024.)
            except (OSError, StopIteration):
                pass
            if any(s.get('status') == 'finished' for s in statuses):
                finished_at = finished_at or time.monotonic()
                if time.monotonic()-finished_at > 1.: break
        assert finished_at, 'capture did not finish finite fake sequence'
        assert not server.errors, server.errors
        assert metas and received['raw']['lidar'] and received['relay']['lidar'], 'missing real ROS data'
        for label in received:
            for key, values in received[label].items():
                stamps = [v['stamp'] for v in values]
                assert all(b>a for a,b in zip(stamps, stamps[1:])), (label, key, 'nonmonotonic')
            left = {v['stamp']:v['marker'] for v in received[label]['left']}
            right = {v['stamp']:v['marker'] for v in received[label]['right']}
            clouds = {v['stamp']:v['marker'] for v in received[label]['lidar']}
            assert left.keys() & right.keys() & clouds.keys(), label
            for stamp in left.keys() & right.keys():
                assert left[stamp] == right[stamp], 'cross-batch stereo contamination'
            for stamp in left.keys() & clouds.keys():
                assert left[stamp] == clouds[stamp] % 251, 'cloud/image stamp mismatch'
        if case == 'blocked':
            first = server.requests[0]['time']
            during_stall = sum(2 < r['time']-first < 17 for r in server.requests)
            assert during_stall > 30, f'producer blocked: {during_stall}'
            assert max(s.get('transport', {}).get('replaced_batches', 0) for s in statuses) > 30
            assert max(v['age'] for v in received['raw']['left']) > 15., 'stall not injected'
            assert max(v['age'] for k in received['relay'] for v in received['relay'][k]) < 3.6, 'stale batch leaked to mapper'
            assert received['relay']['left'][-1]['marker'] > 190, 'fresh data did not recover'
        if case == 'faults':
            assert min(v['marker'] for v in received['raw']['left']) >= 4, 'partial/bad batch published'
    except Exception as error:
        errors.append(repr(error))
    finally:
        for process in (capture, relay):
            if process and process.poll() is None:
                process.send_signal(signal.SIGINT)
                try: process.wait(timeout=7)
                except subprocess.TimeoutExpired:
                    process.kill(); process.wait(); errors.append('shutdown timeout')
            if process and process.returncode != 0: errors.append(f'process exit {process.returncode}')
        cap_log.close(); relay_log.close(); server.close()
        for executor in executors: executor.shutdown()
        for node in nodes: node.destroy_node()
        for context in contexts: context.shutdown()
    result = dict(case=case, passed=not errors, errors=errors, requests=len(server.requests),
        capture_peak_rss_mib=max(rss_samples, default=0.),
        raw_counts={k:len(v) for k,v in received['raw'].items()},
        relay_counts={k:len(v) for k,v in received['relay'].items()},
        raw_max_age={k:max(vv['age'] for vv in v) for k,v in received['raw'].items()},
        relay_max_age={k:max(vv['age'] for vv in v) for k,v in received['relay'].items()},
        transport=[s for s in statuses if 'transport' in s],
        relay_status=json.loads((relay_output/'live_session_status.json').read_text())
            if (relay_output/'live_session_status.json').exists() else None)
    (folder/'result.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(dict(case=case, passed=not errors, errors=errors, path=str(folder))), flush=True)
    if errors: raise AssertionError(errors)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path, default=ROOT/'install/capture_transport/lib/sensor_capture_ros2/sensor_capture_node')
    parser.add_argument('--case', choices=['normal','serial','blocked','faults','full'], default='normal')
    args = parser.parse_args()
    if args.case == 'blocked':
        RESULT.mkdir(parents=True, exist_ok=True)
        candidates = [args.binary.parent/'CMakeFiles/sensor_capture_node.dir/flags.make',
                      ROOT/'build/capture_transport/CMakeFiles/sensor_capture_node.dir/flags.make',
                      ROOT/'build/capture_parallel_test/CMakeFiles/sensor_capture_node.dir/flags.make']
        flags = next(p for p in candidates if p.exists()).read_text()
        includes = next(line.split('=',1)[1] for line in flags.splitlines() if line.startswith('CXX_INCLUDES'))
        subprocess.run(['g++','-shared','-fPIC','-std=c++17','-pthread',*shlex.split(includes),
            str(ROOT/'tests/capture_publish_delay.cpp'),'-ldl','-o',str(RESULT/'publish_delay.so')], check=True)
    run_case(args.binary.resolve(), args.case)
