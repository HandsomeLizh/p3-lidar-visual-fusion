"""Exercise bounded LiDAR/vision waiting with real ROS delivery; no vehicle input."""
import argparse
import copy
import json
import os
from pathlib import Path
import signal
import subprocess
import time

import numpy as np
import rclpy
from nav_msgs.msg import Odometry
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Header, String
from t3_lidar_visual_fusion.ros_utils import xyz_cloud


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--binary', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--wait', type=float, required=True)
    ap.add_argument('--expect-fixed', action='store_true')
    args = ap.parse_args()
    assert os.environ.get('ROS_DOMAIN_ID') == '88'
    assert os.environ.get('ROS_LOCALHOST_ONLY') == '1'
    args.out.mkdir(parents=True, exist_ok=True)
    rclpy.init()
    node = rclpy.create_node('isolated_delayed_visual_test')
    clouds = node.create_publisher(PointCloud2, '/fusion/lidar', 20)
    visuals = node.create_publisher(Odometry, '/fusion/learned_raw', 20)
    clock = node.create_publisher(Clock, '/clock', 20)
    metrics = []
    node.create_subscription(String, '/fusion/lidar_metrics',
                             lambda m: metrics.append(json.loads(m.data)), 100)
    x, y = np.meshgrid(np.arange(-4., 4.001, .15), np.arange(-4., 4.001, .15))
    ground = np.c_[x.ravel(), y.ravel(), np.full(x.size, -1.)]
    process = None
    handle = None

    def spin(seconds, predicate=None):
        until = time.monotonic() + seconds
        while time.monotonic() < until:
            rclpy.spin_once(node, timeout_sec=.002)
            if predicate and predicate():
                return
        if predicate:
            raise RuntimeError('Isolated estimator did not respond')

    def header(index):
        msg = Header(frame_id='lidar')
        ns = round((1000. + .4 * index) * 1e9)
        msg.stamp.sec, msg.stamp.nanosec = divmod(ns, 10**9)
        return msg

    def send_cloud(index):
        h = header(index)
        clock.publish(Clock(clock=copy.deepcopy(h.stamp)))
        clouds.publish(xyz_cloud(ground - [.02 * index, 0., 0.], h))

    def send_visual(index, invalid=False):
        msg = Odometry()
        msg.header = header(index)
        msg.header.frame_id = 'learned_epoch_1'
        msg.child_frame_id = 'base_link'
        msg.pose.pose.orientation.w = 1.
        msg.pose.pose.position.x = .02 * index
        msg.pose.covariance = (np.eye(6) * (1e6 if invalid else .0001)).ravel().tolist()
        visuals.publish(msg)

    def stop():
        nonlocal process, handle
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signal.SIGINT)
            process.wait(timeout=10)
        process = None
        if handle is not None:
            handle.close()
            handle = None
        spin(2., lambda: clouds.get_subscription_count() == 0)

    def case(name, events, expected):
        nonlocal process, handle
        metrics.clear()
        handle = (args.out / (name + '.log')).open('w')
        process = subprocess.Popen([
            str(args.binary), '--ros-args', '-p', 'use_sim_time:=true',
            '-p', 'voxel_size:=1.0', '-p', 'downsample_size:=0.1',
            '-p', 'max_iterations:=30', '-p', 'tracking_max_speed:=0.3',
            '-p', 'degeneracy_projection_enabled:=true',
            '-p', 'independent_visual_topic:=/fusion/learned_raw',
            '-p', 'independent_visual_wait_sec:=' + str(args.wait), '-p', 'threads:=1'],
            stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
        spin(10., lambda: clouds.get_subscription_count() == 1 and
             visuals.get_subscription_count() == 1)
        send_visual(0)
        spin(.03)
        send_cloud(0)
        spin(3., lambda: len(metrics) == 1)
        assert metrics[0]['valid_update'], metrics
        began = time.monotonic()
        for when, kind, index in sorted(events):
            spin(max(0., began + when - time.monotonic()))
            if kind == 'cloud':
                send_cloud(index)
            else:
                send_visual(index, invalid=kind == 'invalid')
        spin(args.wait + .5)
        (args.out / (name + '.metrics.json')).write_text(json.dumps(metrics, indent=2))
        frames = {round((m['stamp_sec'] - 1000.) / .4): m for m in metrics}
        result = dict(expected=expected, received=sorted(frames),
                      accepted=[i for i, m in frames.items() if m['valid_update']],
                      rejected={i: m['tracking_reason'] for i, m in frames.items()
                                if not m['valid_update']},
                      queue_peak=max(m.get('pending_peak', 0) for m in metrics),
                      dropped=metrics[-1]['pending_dropped'],
                      wait_sec=[m.get('reference_wait_sec') for m in metrics],
                      maximum_position_error_m=max(abs(m['position'][0] - .02 * i)
                                                   for i, m in frames.items() if m['valid_update']))
        if name == 'missing_and_invalid':
            result['rejections_hold_pose_and_map'] = all(
                not frames[i]['valid_update'] and
                frames[i]['map_keyframes'] == frames[0]['map_keyframes'] and
                np.allclose(frames[i]['position'], frames[0]['position'], atol=1e-10)
                for i in (1, 2) if i in frames) and all(i in frames for i in (1, 2))
        else:
            result['all_expected_accepted'] = all(i in frames and frames[i]['valid_update'] for i in expected)
        stop()
        print(name, json.dumps(result), flush=True)
        return result

    def cadence(count, period, delay):
        events = []
        for i in range(1, count + 1):
            t = .05 + (i - 1) * period
            events.extend([(t, 'cloud', i), (t + delay, 'visual', i)])
        return events

    results = {}
    try:
        results['ready'] = case('ready', cadence(4, .35, .04), list(range(5)))
        results['late'] = case('late', cadence(5, 1.1, .70), list(range(6)))
        results['overlap'] = case('overlap', cadence(6, .50, .70), list(range(7)))
        # At this arrival rate three frames can precede the first reference.
        # The bounded queue deliberately drops intermediate work and keeps the
        # latest frame; asking it to retain every frame would defeat the cap.
        results['overload'] = case('overload', cadence(6, .35, .70), [0, 1, 6])
        results['burst'] = case('burst', cadence(10, .02, .70), [0, 1, 10])
        events = [(.05, 'cloud', 1), (.25, 'cloud', 2), (.30, 'invalid', 2),
                  (1.5, 'cloud', 3), (1.55, 'visual', 3)]
        results['missing_and_invalid'] = case('missing_and_invalid', events, [0, 3])
        report = dict(binary=str(args.binary), wait_limit_sec=args.wait, cases=results,
                      scope='Synthetic flat terrain and asynchronous visual delivery in localhost ROS domain 88')
        (args.out / 'report.json').write_text(json.dumps(report, indent=2))
        if args.expect_fixed:
            for name in ('ready', 'late', 'overlap', 'overload', 'burst'):
                assert results[name]['all_expected_accepted'], results[name]
                assert results[name]['maximum_position_error_m'] < .005, results[name]
                assert results[name]['queue_peak'] <= 2, results[name]
            assert results['burst']['dropped'] > 0, results['burst']
            assert results['missing_and_invalid']['rejections_hold_pose_and_map']
            assert 3 in results['missing_and_invalid']['accepted']
            # Raising an upper wait bound does not impose that delay on ready data.
            assert max(results['ready']['wait_sec']) < .3, results['ready']
            assert max(results['late']['wait_sec']) < 1.2, results['late']
        print('PASS' if args.expect_fixed else 'BASELINE_RECORDED', flush=True)
    finally:
        if process is not None:
            stop()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
