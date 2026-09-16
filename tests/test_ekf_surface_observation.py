#!/usr/bin/env python3
"""Replay recorded guarded constraints through actual ROS EKFs; no truth input.

Fast replay diagnoses pose component gating. It is not runtime timing evidence.
"""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import numpy as np
import yaml
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from rosgraph_msgs.msg import Clock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src/t3_lidar_visual_fusion'))
from t3_lidar_visual_fusion.body_motion import BodyMotion
from t3_lidar_visual_fusion.ros_utils import transform_from_pose


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--frontend-confidence-ablation', action='store_true',
                        help='Diagnostic only: replace guard selection with frontend-valid, confidence >= .45 constraints')
    args = parser.parse_args()
    assert os.environ.get('ROS_DOMAIN_ID') == '58'
    source, output = Path(args.source).resolve(), Path(args.output).resolve()
    assert output.is_relative_to(ROOT/'results')
    output.mkdir(exist_ok=False)
    record = json.loads((source/'verification.json').read_text())
    raw = {(m['source'], round(m['stamp_sec']*1e9)): m for m in record['raw_poses']}
    guarded = [m for m in record['measurements'] if m['source'] in ('lio_guarded', 'vision_accepted')]
    if not all('pose' in m and 'pose_covariance' in m for m in guarded):
        # Legacy verifier did not retain guarded transforms. Reconstruction is
        # valid only for one visual epoch, initially aligned to identity.
        epochs = {m['frame'] for m in record['raw_poses'] if m['source'] == 'learned_raw'}
        assert len(epochs) == 1, epochs
        first = next(m for m in record['raw_poses'] if m['source'] == 'learned_raw')
        assert np.allclose(first['pose'], [0, 0, 0, 0, 0, 0, 1], atol=1e-10)
    rows = {}
    for m in guarded:
        ns = round(m['stamp_sec']*1e9)
        if 'pose' in m and 'pose_covariance' in m:
            value = {'pose': m['pose'], 'cov': m['pose_covariance']}
        else:
            original = raw[('lio_raw' if m['source'] == 'lio_guarded' else 'learned_raw', ns)]
            covariance = np.array(original['pose_covariance']).reshape(6, 6)
            np.fill_diagonal(covariance, m['pose_covariance_diagonal'])
            value = {'pose': original['pose'], 'cov': covariance.ravel().tolist()}
        rows.setdefault(ns, {})[m['source']] = value
    if args.frontend_confidence_ablation:
        assert len({m['frame'] for m in record['raw_poses'] if m['source']=='learned_raw'}) == 1
        tracked = {round(m['sensor_stamp_sec']*1e9) for m in map(json.loads, (source/'learned_metrics.jsonl').read_text().splitlines()) if m.get('tracking_valid')}
        profile = yaml.safe_load((source/'profile.yaml').read_text())
        floor = np.asarray(profile['vision_pose_variance'])
        consecutive = 0
        for ns, values in sorted(rows.items()):
            values.pop('vision_accepted', None)
            original = raw.get(('learned_raw', ns))
            if original is None or ns not in tracked:
                consecutive = 0; continue
            covariance = np.asarray(original['pose_covariance']).reshape(6, 6)
            score = min(1., float(np.sqrt(np.min(floor/np.diag(covariance)))))
            if score < .45:
                consecutive = 0; continue
            consecutive += 1
            if consecutive >= 4:
                values['vision_accepted'] = {'pose': original['pose'], 'cov': original['pose_covariance']}
    modes = ['six_pose', 'six_differential', 'body_twist']
    motion_adapter = BodyMotion(max_gap=6.)
    for ns, measurements in sorted(rows.items()):
        value = measurements.get('vision_accepted')
        if value is None: continue
        msg = Odometry(); p, q = msg.pose.pose.position, msg.pose.pose.orientation
        p.x,p.y,p.z,q.x,q.y,q.z,q.w = map(float, value['pose'])
        result = motion_adapter.update(ns/1e9, 'fixed_epoch', transform_from_pose(msg.pose.pose), value['cov'])
        if result is not None:
            value['body_twist'], value['twist_cov'] = result[0].tolist(), result[1].ravel().tolist()
    rclpy.init()
    node = Node('surface_observation_probe')
    clock_pub = node.create_publisher(Clock, '/clock', 10)
    observed = {mode: {} for mode in modes}
    pubs = {mode: {key: node.create_publisher(Odometry, '/surface_probe/'+mode+'/'+key, 100)
                   for key in ('lio_guarded', 'vision_accepted')} for mode in modes}
    def receive(mode, msg):
        p, q = msg.pose.pose.position, msg.pose.pose.orientation
        ns = msg.header.stamp.sec*1000000000+msg.header.stamp.nanosec
        observed[mode][ns] = {'pose': [p.x, p.y, p.z, q.x, q.y, q.z, q.w],
                              'cov': list(msg.pose.covariance)}
    for mode in modes:
        node.create_subscription(Odometry, '/surface_probe/'+mode,
                                 lambda m, name=mode: receive(name, m), 100)
    processes, handles = [], []
    def spin(seconds):
        deadline = time.monotonic()+seconds
        while time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=.002)
    def tick(ns):
        msg = Clock(); msg.clock.sec, msg.clock.nanosec = divmod(ns, 1000000000)
        clock_pub.publish(msg)
    try:
        for mode in modes:
            cfg = yaml.safe_load((source/'config/ekf.yaml').read_text())['ekf_filter_node']['ros__parameters']
            cfg.update(use_sim_time=True, publish_tf=False,
                       odom0='/surface_probe/'+mode+'/lio_guarded',
                       odom1='/surface_probe/'+mode+'/vision_accepted',
                       debug=True, debug_out_file=str(output/(mode+'_debug.txt')))
            if mode in ('ground_pose', 'ground_differential'):
                cfg['odom1_config'] = [True, True, False, False, False, True]+[False]*9
            if mode.endswith('_differential'):
                cfg['odom1_differential'] = True
            if mode == 'body_twist':
                cfg['odom1_differential'] = False
                cfg['odom1_config'] = [False]*6+[True]*6+[False]*3
            if mode == 'lidar_only':
                cfg = {k: v for k, v in cfg.items() if not k.startswith('odom1')}
            path = output/(mode+'.yaml')
            path.write_text(yaml.safe_dump({mode: {'ros__parameters': cfg}}))
            handle = (output/(mode+'.log')).open('w'); handles.append(handle)
            processes.append(subprocess.Popen([
                str(ROOT/'deps/root/opt/ros/humble/lib/robot_localization/ekf_node'),
                '--ros-args', '-r', '__node:='+mode, '-r', 'odometry/filtered:=/surface_probe/'+mode,
                '--params-file', str(path)], stdout=handle, stderr=subprocess.STDOUT, start_new_session=True))
        deadline = time.monotonic()+15
        while any(pubs[mode]['lio_guarded'].get_subscription_count() != 1 for mode in modes):
            if time.monotonic() > deadline: raise TimeoutError('EKF discovery')
            spin(.02)
        for i in range(60):
            tick(999000000000+i*10000000); spin(.015)
        for frame, (ns, measurements) in enumerate(sorted(rows.items())):
            # Keep recorded source ordering and allow each same-stamp correction
            # to be processed before advancing to the next acquisition time.
            for step in range(180):
                tick(ns+step*10000000)
                for j, (key, value) in enumerate(measurements.items()):
                    if step != 2+12*j: continue
                    msg = Odometry()
                    msg.header.stamp.sec, msg.header.stamp.nanosec = divmod(ns, 1000000000)
                    msg.header.frame_id = 'odom'; msg.child_frame_id = 'base_link'
                    p, q = msg.pose.pose.position, msg.pose.pose.orientation
                    p.x, p.y, p.z, q.x, q.y, q.z, q.w = map(float, value['pose'])
                    msg.pose.covariance = value['cov']
                    for mode in modes:
                        if mode == 'body_twist' and key == 'vision_accepted':
                            if 'body_twist' not in value: continue
                            v, w = msg.twist.twist.linear, msg.twist.twist.angular
                            v.x,v.y,v.z,w.x,w.y,w.z = map(float, value['body_twist'])
                            msg.twist.covariance = value['twist_cov']
                        if mode != 'lidar_only' or key == 'lio_guarded': pubs[mode][key].publish(msg)
                spin(.012)
                if step >= 44 and all(any(abs(t-ns) < 3 for t in observed[mode]) for mode in modes):
                    break
            else:
                raise TimeoutError('No current measurement-time output: '+str(ns))
            if frame % 25 == 0: print('frame', frame, flush=True)
        spin(.3)
        (output/'observed.json').write_text(json.dumps(observed))
        result = {'source': str(source), 'frames': len(rows),
                  'frontend_confidence_ablation': args.frontend_confidence_ablation,
                  'qualification': 'Actual EKFs with the same recorded guarded measurements. Accelerated diagnostic only, no truth input and no timing claim.',
                  'modes': {}}
        for mode in modes:
            samples = []
            for ns in sorted(rows):
                closest = min(observed[mode], key=lambda t: abs(t-ns))
                assert abs(closest-ns) < 3, (mode, ns, closest)
                value = observed[mode][closest]
                cov = np.asarray(value['cov']).reshape(6, 6)
                position_var = float(np.linalg.eigvalsh(cov[:3, :3]).max())
                rotation_var = float(np.linalg.eigvalsh(cov[3:, 3:]).max())
                samples.append({'stamp_sec': ns/1e9, 'pose': value['pose'],
                                'position_max_variance': position_var, 'rotation_max_variance': rotation_var,
                                'qualified': position_var <= 4. and rotation_var <= .5})
            result['modes'][mode] = {'qualified_frames': sum(s['qualified'] for s in samples),
                                    'max_position_variance': max(s['position_max_variance'] for s in samples),
                                    'samples': samples}
            with (output/(mode+'.tum')).open('w') as stream:
                for s in samples: stream.write(' '.join(map(str, [s['stamp_sec']]+s['pose']))+'\n')
        (output/'result.json').write_text(json.dumps(result, indent=2))
        print(json.dumps({k: {n: v for n, v in d.items() if n != 'samples'} for k, d in result['modes'].items()}, indent=2), flush=True)
    finally:
        for process in processes:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGINT)
                try: process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL); process.wait(timeout=5)
        for handle in handles: handle.close()
        node.destroy_node(); rclpy.shutdown()


if __name__ == '__main__': main()
