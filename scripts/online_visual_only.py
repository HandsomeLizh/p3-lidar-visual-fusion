#!/usr/bin/env python3
"""Visual-only localization with LiDAR mapping and the P3 UI.

Reference poses are recorded and scored here, never supplied to the tracker.
The default domain 57 and odom-frame outputs match the original P3 interfaces.
LiDAR is used for mapping only. P4 goal sending is enabled explicitly with
--allow-goals after the P4 feedback frame has been aligned to this session.
"""
import argparse
from collections import Counter, deque
import copy
import csv
import json
import hashlib
import os
from pathlib import Path
import signal
import subprocess
import time

import numpy as np
import rclpy
from rclpy.context import Context
from rclpy.executors import SingleThreadedExecutor
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from rclpy.signals import SignalHandlerOptions
from geometry_msgs.msg import PoseStamped, TransformStamped
from nav_msgs.msg import Odometry, Path as RosPath
from sensor_msgs.msg import Image, PointCloud2
from std_msgs.msg import String
from tf2_ros import TransformBroadcaster, StaticTransformBroadcaster
import yaml

from control import alive, identity
from stereo_transport import StereoNormalizer
from t3_lidar_visual_fusion.learned_odometry import LearnedOdometry
from t3_lidar_visual_fusion.sensor_adapter import SensorAdapter
from t3_lidar_visual_fusion.ros_utils import stamp_sec

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / 'visual_only_live_state.json'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--no-rviz', action='store_true')
    parser.add_argument('--allow-goals', action='store_true',
        help='Enable P4 goal submission in the existing P3 window')
    parser.add_argument('--domain', type=int, default=57)
    parser.add_argument('--backend', choices=['xfeat_lighterglue','roma'], default='xfeat_lighterglue')
    args = parser.parse_args()
    if not 0 <= args.domain <= 232 or args.domain == 10:
        raise ValueError('Output domain must be 0..232 and separate from input domain 10')
    if STATE.exists() and alive(json.loads(STATE.read_text()).get('process')):
        raise RuntimeError('A visual-only live trial is already running: ' + str(STATE))
    out = (args.output or ROOT / 'results' / time.strftime('visual_only_live_%Y%m%d_%H%M%S')).resolve()
    if not out.is_relative_to(ROOT / 'results'):
        raise ValueError('Output must be under this fusion project results directory')
    out.mkdir(exist_ok=False)
    cfg = yaml.safe_load((ROOT / 'config/simulation_live.yaml').read_text())
    cfg.update(use_imu=False, imu_mode='off', adaptive_source_selection=False,
        mapping_pose_settle_sec=0.,
        map_pending_scans=8, map_window=32., semantic_topic='', tof_sources=[],
        localization_mode='visual_only', lidar_backend='none')
    mapping_sources = cfg.get('mapping_sources',['lidar'] + (['stereo'] if cfg.get('stereo_mapping_topic') else []))
    # Keep the previously validated conservative clearance policy.
    cfg['dynamic_map']['enabled'] = False
    cfg['learned_visual']['image_stationary_enabled'] = True
    if args.backend == 'roma':
        original = ROOT.parent / 'roma_t3_algorithm_bundle_20260825'
        source = original / 'third_party/RoMa'
        cfg.update(output_image_size=[816,682], normalized_image_encoding='bgr8',
                   visual_max_age_sec=8., mapping_wait_timeout=10., pose_max_gap=6.)
        cfg['learned_visual'].update(backend='roma', max_image_side=1024,
            max_normalized_image_bytes=2000000, max_processing_hz=2.,
            roma_source_root=str(source),
            roma_source_sha256=hashlib.sha256((source/'roma_vo.py').read_bytes()).hexdigest(),
            roma_checkpoint_dir=str(original/'models/torch/hub/checkpoints'),
            roma_keyframe_limit=12, roma_min_available_mib=8192, roma_stationary_max_gap_sec=6.,
            roma_min_pnp_inliers=50, roma_min_pnp_ratio=.25, roma_min_pnp_coverage=.15)
    profile = out / 'profile.yaml'
    profile.write_text(yaml.safe_dump(cfg, sort_keys=False))
    os.environ['ROS_LOCALHOST_ONLY'] = '0'
    rclpy.init(domain_id=args.domain, signal_handler_options=SignalHandlerOptions.NO, args=[
        '--ros-args', '-p', 'profile_path:=' + str(profile),
        '-p', 'workspace_root:=' + str(ROOT), '-p', 'output_dir:=' + str(out),
        '-p', 'use_sim_time:=false', '-p', 'normalized_camera_input:=true'])
    source_context = Context()
    rclpy.init(context=source_context, domain_id=10, args=[], signal_handler_options=SignalHandlerOptions.NO)
    source = rclpy.create_node('visual_only_live_sensor_input', context=source_context)
    tracker = LearnedOdometry()
    adapter = SensorAdapter()
    view = rclpy.create_node('visual_only_live_view')
    target_executor = SingleThreadedExecutor()
    source_executor = SingleThreadedExecutor(context=source_context)
    target_executor.add_node(tracker)
    target_executor.add_node(adapter)
    target_executor.add_node(view)
    source_executor.add_node(source)
    normalizer = StereoNormalizer(cfg)
    images = [view.create_publisher(Image, t, 2) for t in ['/fusion/left', '/fusion/right']]
    path_pub = view.create_publisher(RosPath, '/visual_test/path', 1)
    pose_pub = view.create_publisher(Odometry, '/visual_test/odometry', 3)
    retained = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.TRANSIENT_LOCAL)
    map_path_pub = view.create_publisher(RosPath, '/T3/semantic/trajectory', retained)
    map_pose_pub = view.create_publisher(Odometry, '/T3/semantic/current_pose', 3)
    car_pose_pub = view.create_publisher(Odometry, '/Car/T3/localization/odometry', 3)
    filtered_pub = view.create_publisher(Odometry, '/odometry/filtered', 3)
    car_path_pub = view.create_publisher(RosPath, '/Car/T3/debug/trajectory', retained)
    status_pub = view.create_publisher(String, '/visual_test/status', 1)
    health_pub = view.create_publisher(String, '/fusion/status', 1)
    tf = TransformBroadcaster(view)
    static_tf = StaticTransformBroadcaster(view)
    map_odom = TransformStamped(child_frame_id='odom')
    map_odom.header.frame_id = 'map'
    map_odom.header.stamp = view.get_clock().now().to_msg()
    map_odom.transform.rotation.w = 1.
    static_tf.sendTransform(map_odom)
    counts = Counter()
    last_received = {}
    last_images = [-1., -1.]
    reference = deque(maxlen=4000)
    pending = deque(maxlen=100)
    samples = deque(maxlen=10000)
    poses = deque(maxlen=5000)
    epoch = [None]
    epochs = set()
    last_pose_stamp = [-1.]
    last_qualified_wall = [None]
    visual_valid = [False]
    mapping_paused = [False]
    rejected_epoch = [None]
    raw_file = (out / 'visual_raw.jsonl').open('w', buffering=1)
    ref_file = (out / 'reference.csv').open('w', newline='', buffering=1)
    ref_writer = csv.writer(ref_file)
    ref_writer.writerow(['stamp_sec', 'x', 'y', 'z', 'qx', 'qy', 'qz', 'qw', 'received_wall_sec'])

    def image_input(side, msg):
        now = time.time()
        last_received['left' if side == 0 else 'right'] = now
        counts['images_received_' + str(side)] += 1
        stamp = stamp_sec(msg)
        if stamp <= last_images[side]:
            counts['duplicate_images'] += 1
            return
        if now - stamp > cfg['visual_max_age_sec'] - .5 or stamp - now > .1:
            counts['stale_images'] += 1
            return
        try:
            images[side].publish(normalizer.convert(msg, side))
            last_images[side] = stamp
            counts['images_forwarded_' + str(side)] += 1
        except Exception as exc:
            counts['image_rejected'] += 1
            source.get_logger().warning(str(exc), throttle_duration_sec=5.)

    def lidar_input(msg):
        last_received['lidar'] = time.time()
        counts['lidar_received'] += 1
        age = time.time() - stamp_sec(msg)
        if age > cfg['visual_max_age_sec'] or age < -.1:
            counts['stale_lidar'] += 1
            return
        adapter.cloud(msg)

    def reference_input(msg):
        p, q = msg.pose.position, msg.pose.orientation
        row = [stamp_sec(msg), p.x, p.y, p.z, q.x, q.y, q.z, q.w]
        if not np.isfinite(row).all():
            return
        reference.append((row[0], np.array(row[1:4])))
        ref_writer.writerow(row + [time.time()])
        last_received['reference'] = time.time()
        counts['reference_messages'] += 1

    def visual_input(msg):
        stamp = stamp_sec(msg)
        p, q = msg.pose.pose.position, msg.pose.pose.orientation
        values = [p.x, p.y, p.z, q.x, q.y, q.z, q.w]
        covariance = np.array(msg.pose.covariance).reshape(6, 6)
        finite = np.isfinite(values).all() and np.isfinite(covariance).all()
        valid = bool(finite and np.linalg.eigvalsh(covariance).min() >= 0.
                     and np.linalg.eigvalsh(covariance[:3, :3]).max() <= cfg['qualified_position_variance']
                     and np.linalg.eigvalsh(covariance[3:, 3:]).max() <= cfg['qualified_rotation_variance'])
        counts['visual_messages'] += 1
        raw_file.write(json.dumps(dict(stamp_sec=stamp, frame=msg.header.frame_id,
            pose=values, qualified=valid, pose_covariance_diagonal=covariance.diagonal().tolist(),
            received_wall_sec=time.time())) + '\n')
        # A reset is a new coordinate system, not a zero-motion observation.
        # Retain the current map and require a new session for the new origin.
        if epoch[0] is not None and epoch[0] != msg.header.frame_id:
            mapping_paused[0] = True
            rejected_epoch[0] = msg.header.frame_id
        if mapping_paused[0]:
            visual_valid[0] = False
            counts['epoch_reset_poses_blocked'] += 1
            return
        if not valid or stamp <= last_pose_stamp[0]:
            visual_valid[0] = False
            counts['unqualified_visual_messages'] += 1
            return
        if epoch[0] != msg.header.frame_id:
            poses.clear()
            samples.clear()
            pending.clear()
            epoch[0] = msg.header.frame_id
            epochs.add(epoch[0])
        last_pose_stamp[0] = stamp
        last_qualified_wall[0] = time.time()
        visual_valid[0] = True
        m = copy.deepcopy(msg)
        m.header.frame_id = 'odom'
        pose_pub.publish(m)
        map_pose_pub.publish(m)
        car_pose_pub.publish(m)
        filtered_pub.publish(m)
        pose = PoseStamped(header=m.header, pose=m.pose.pose)
        poses.append(pose)
        path = RosPath(header=m.header, poses=list(poses))
        path_pub.publish(path)
        map_path_pub.publish(path)
        car_path_pub.publish(path)
        t = TransformStamped(header=m.header, child_frame_id='base_link')
        t.transform.translation.x, t.transform.translation.y, t.transform.translation.z = values[:3]
        t.transform.rotation = m.pose.pose.orientation
        tf.sendTransform(t)
        pending.append((stamp, np.array(values[:3]), time.time()))
        counts['qualified_visual_poses'] += 1

    reliable = QoSProfile(depth=2, reliability=ReliabilityPolicy.RELIABLE)
    for side, topic in enumerate([cfg['left_topic'], cfg['right_topic']]):
        source.create_subscription(Image, topic, lambda m, s=side: image_input(s, m), reliable)
    source.create_subscription(PointCloud2, cfg['lidar_topic'], lidar_input, reliable)
    source.create_subscription(PoseStamped, '/car/pose', reference_input,
        QoSProfile(depth=30, reliability=ReliabilityPolicy.BEST_EFFORT))
    view.create_subscription(Odometry, cfg['visual_odometry_topic'], visual_input, 30)
    stopped = [False]
    for sig in [signal.SIGINT, signal.SIGTERM]:
        signal.signal(sig, lambda *_: stopped.__setitem__(0, True))
    children = {}
    state = dict(output=str(out), domain=args.domain, source_domain=10,
        process=dict(pid=os.getpid(), identity=identity(os.getpid())),
        mode='visual_localization_' + '_'.join(mapping_sources) + '_mapping', processes={},
        mapping_sources=mapping_sources,
        visual_backend=args.backend,
        reference_used_for_localization=False, vehicle_commands_published=0,
        p4_goal_submission_enabled=args.allow_goals)
    STATE.write_text(json.dumps(state, indent=2))
    child_env = dict(os.environ, ROS_DOMAIN_ID=str(args.domain), DISPLAY=os.environ.get('DISPLAY', ':0'),
            XAUTHORITY=os.environ.get('XAUTHORITY', '/run/user/1000/gdm/Xauthority'), QT_QPA_PLATFORM='xcb')
    # OpenCV's bundled Qt plugin cannot be loaded by system RViz Qt.
    for key in ('QT_QPA_PLATFORM_PLUGIN_PATH', 'QT_QPA_FONTDIR'):
        child_env.pop(key, None)
    runtime = out / 'visual_runtime'
    runtime.mkdir()
    child_env['T3_VISUAL_RUNTIME'] = str(runtime)
    child_env['T3_VISUAL_ALLOW_GOALS'] = '1' if args.allow_goals else '0'
    try:
        previous = json.loads((ROOT / 'run_state.json').read_text())
        settings = Path(previous['output']) / 'visual_runtime/view_settings.json'
        (runtime / 'view_settings.json').write_text(settings.read_text())
    except (OSError, ValueError, KeyError):
        pass

    def spawn(name, command):
        with (out / (name + '.log')).open('w') as log:
            proc = subprocess.Popen(command, env=child_env, cwd=ROOT,
                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        children[name] = proc
        state['processes'][name] = dict(pid=proc.pid, identity=identity(proc.pid), command=command)
        STATE.write_text(json.dumps(state, indent=2))

    def report():
        now = time.time()
        with tracker.lock:
            stationary = dict(tracker.last.get('stationary', {}))
        while pending:
            stamp, point, received = pending[0]
            nearest = min(reference, key=lambda r: abs(r[0] - stamp)) if reference else None
            if nearest is not None and abs(nearest[0] - stamp) <= .25:
                samples.append((stamp, point, nearest[1], abs(nearest[0] - stamp)))
                pending.popleft()
            elif now - received > 3.:
                counts['no_reference_match'] += 1
                pending.popleft()
            else:
                break
        data = dict(state, counts=dict(counts), epoch=epoch[0], epoch_count=len(epochs),
            stationary=stationary,
            rejected_epoch=rejected_epoch[0], mapping_paused_after_epoch_reset=mapping_paused[0],
            sensor_idle_sec={k: now - v for k, v in last_received.items()},
            qualified_pose_age_sec=None if last_qualified_wall[0] is None else now - last_qualified_wall[0],
            status='tracking' if visual_valid[0] and last_qualified_wall[0] is not None
                and now - last_qualified_wall[0] < cfg['visual_max_age_sec']
                and now - last_pose_stamp[0] < cfg['visual_max_age_sec'] else
                'visual_origin_reset_restart_required' if mapping_paused[0] else 'waiting_for_qualified_visual_pose',
            reference_association='nearest header timestamp within 0.25 s; exposure synchronization not independently calibrated')
        if len(samples) >= 4:
            estimate = np.array([x[1] for x in samples])
            truth = np.array([x[2] for x in samples])
            a, b = estimate.mean(axis=0), truth.mean(axis=0)
            u, _, vt = np.linalg.svd((estimate - a).T @ (truth - b))
            d = np.eye(3)
            d[2, 2] = np.linalg.det(vt.T @ u.T)
            rotation = vt.T @ d @ u.T
            errors = np.linalg.norm((estimate - a) @ rotation.T + b - truth, axis=1)
            travel = float(np.linalg.norm(np.diff(truth, axis=0), axis=1).sum())
            data['current_epoch_accuracy'] = dict(samples=len(samples), reference_sampled_travel_m=travel,
                informative_motion=travel >= 1., ate_rmse_m=float(np.sqrt(np.mean(errors ** 2))),
                endpoint_after_alignment_m=float(errors[-1]), maximum_error_m=float(errors.max()),
                max_reference_skew_sec=max(x[3] for x in samples),
                method='one rigid position alignment for the current epoch, fixed scale; reset epochs are never joined')
        tmp = out / 'online_status.tmp'
        tmp.write_text(json.dumps(data, indent=2))
        tmp.replace(out / 'online_status.json')
        status_pub.publish(String(data=json.dumps(data)))
        health_pub.publish(String(data=json.dumps(dict(localization_valid=data['status']=='tracking',
            localization_reason=data['status'], output_source='visual',
            pose_source_preference='visual_only', stationary=stationary, mapping_sources=mapping_sources,
            visual_backend=args.backend,
            lidar_used_for_localization=False, reference_used_for_localization=False))))
        return data

    previous = 0.
    try:
        spawn('mapper', [str(ROOT / 'install/t3_lidar_visual_fusion/lib/t3_lidar_visual_fusion/terrain_mapper'),
            '--ros-args', '-p', 'profile_path:=' + str(profile), '-p', 'output_dir:=' + str(out),
            '-p', 'use_sim_time:=false'])
        if not args.no_rviz:
            spawn('monitor', ['bash', str(ROOT / 'scripts/p3_visuals.sh'), 'monitor', '-p', 'use_sim_time:=false'])
            spawn('window', ['bash', str(ROOT / 'scripts/p3_visuals.sh'), 'window', str(runtime),
                '--ros-args', '-r', '__node:=fusion_visual_only_window'])
        print('READY: visual localization; ' + ' + '.join(mapping_sources) + ' mapping; original P3 UI/interfaces; domain '
              + str(args.domain) + '; ' + str(out), flush=True)
        while not stopped[0] and not (out / 'stop').exists():
            source_executor.spin_once(timeout_sec=.003)
            target_executor.spin_once(timeout_sec=.003)
            if time.monotonic() - previous >= 1.:
                failed = [(name, proc.returncode) for name, proc in children.items() if proc.poll() is not None]
                if failed:
                    raise RuntimeError('Child process stopped: ' + repr(failed))
                report()
                previous = time.monotonic()
    finally:
        report()
        for proc in children.values():
            if proc.poll() is None:
                proc.send_signal(signal.SIGINT)
        tracker.close()
        for proc in children.values():
            try:
                proc.wait(timeout=10.)
            except subprocess.TimeoutExpired:
                proc.terminate()
        source_executor.shutdown()
        target_executor.shutdown()
        for node in [source, view, tracker, adapter]:
            node.destroy_node()
        source_context.shutdown()
        rclpy.try_shutdown()
        raw_file.close()
        ref_file.close()


if __name__ == '__main__':
    main()
