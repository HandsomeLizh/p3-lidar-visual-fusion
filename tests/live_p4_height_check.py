"""Supervise P4 RViz-goal navigation and compare repeatedly observed heights.

No movement without --execute. P4 plans and controls; this script only sends
selected goals, records evidence, and parks/cancels at completion or failure.
Run with both the P3 and P4 installed environments loaded, while UE is active.
"""
import argparse
import json
import math
import os
from pathlib import Path
import signal
import time

import numpy as np
import rclpy
from rclpy.context import Context
from rclpy.executors import SingleThreadedExecutor
from rclpy.qos import QoSProfile, DurabilityPolicy, qos_profile_sensor_data
from rclpy.signals import SignalHandlerOptions
from action_msgs.msg import GoalStatusArray
from action_msgs.srv import CancelGoal
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Odometry, OccupancyGrid, Path as RosPath
from std_msgs.msg import String
from diagnostic_msgs.msg import DiagnosticArray
from grid_map_msgs.msg import GridMap
from lunar_planning_msgs.action import NavigateToPose
from lunar_planning_msgs.msg import TrackingStatus
from rosidl_runtime_py.convert import message_to_ordereddict

ROOT = Path(__file__).resolve().parents[1]
ACTION = '/Car/T4/navigation/navigate_to_pose'


def grid_arrays(message):
    result = {}
    for key, data in zip(message.layers, message.data):
        ny, nx = (d.size for d in data.layout.dim)
        result[key] = np.asarray(data.data).reshape(ny, nx)[::-1, ::-1].copy()
    if message.outer_start_index or message.inner_start_index:
        raise RuntimeError('Circular GridMap requires an explicit decoder')
    result['origin'] = np.array([message.info.pose.position.x-message.info.length_x/2,
                                 message.info.pose.position.y-message.info.length_y/2])
    result['resolution'] = np.array(message.info.resolution)
    return result


def compare(before, after):
    a, b = before['elevation'], after['elevation']
    rr, cc = np.indices(a.shape)
    res = float(before['resolution'])
    delta = np.rint((before['origin']-after['origin'])/res).astype(int)
    br, bc = rr+delta[1], cc+delta[0]
    valid = (br >= 0) & (br < b.shape[0]) & (bc >= 0) & (bc < b.shape[1]) & np.isfinite(a)
    xy = before['origin'] + np.stack([cc+.5, rr+.5], axis=-1)*res
    valid &= (np.linalg.norm(xy-before['pose'][:2], axis=-1) < 8.) & (before['observation_count'] >= 3)
    rr, cc, br, bc = rr[valid], cc[valid], br[valid], bc[valid]
    valid = np.isfinite(b[br, bc]) & (after['observation_count'][br, bc] > before['observation_count'][rr, cc])
    diff = b[br[valid], bc[valid]]-a[rr[valid], cc[valid]]
    if not len(diff):
        return dict(reobserved_cells=0)
    return dict(reobserved_cells=len(diff), median_m=float(np.median(diff)),
                abs_p95_m=float(np.percentile(abs(diff), 95)), abs_max_m=float(abs(diff).max()),
                changed_over_3cm=int((abs(diff) > .03).sum()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--face-offset', action='store_true',
                        help='Align nonzero goal yaw with its initial body offset (reverse offsets keep forward-facing yaw)')
    parser.add_argument('--offsets', default='[[-2.0,0.6],[0,0],[-2.0,-0.6],[0,0]]',
                        help='Goal XY offsets in the initial body axes; all final yaws match the initial yaw')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    offsets = np.asarray(json.loads(args.offsets), dtype=float)
    if offsets.ndim != 2 or offsets.shape[1] != 2 or not np.isfinite(offsets).all() or np.linalg.norm(offsets, axis=1).max() > 5:
        raise ValueError('Use finite XY offsets within 5 m')
    out = args.output.resolve()
    if not out.is_relative_to(ROOT/'results'):
        raise ValueError('Output must stay in this project results directory')
    out.mkdir(parents=True, exist_ok=False)
    os.environ['ROS_LOCALHOST_ONLY'] = '0'
    contexts, nodes, executors = [], [], []
    for domain in (57, 10):
        ctx = Context()
        rclpy.init(context=ctx, domain_id=domain, signal_handler_options=SignalHandlerOptions.NO)
        node = rclpy.create_node('p3_p4_height_supervisor_'+str(domain), context=ctx)
        executor = SingleThreadedExecutor(context=ctx); executor.add_node(node)
        contexts.append(ctx); nodes.append(node); executors.append(executor)
    p4, car = nodes
    last, seen, counts = {}, {}, {}
    total_travel = [0.]
    def xyz(message):
        p = message.pose.pose.position
        return np.array([p.x, p.y, p.z])
    def yaw(message):
        q = message.pose.pose.orientation
        return math.atan2(2*(q.w*q.z+q.x*q.y), 1-2*(q.y*q.y+q.z*q.z))
    def receive(key, value):
        if key == 'feedback_pose' and key in last:
            total_travel[0] += float(np.linalg.norm(xyz(value)-xyz(last[key])))
        last[key] = value; seen[key] = time.monotonic(); counts[key] = counts.get(key, 0)+1
    retained = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
    for key, typ, topic, qos in [
        ('map', GridMap, '/Car/T3/mapping/grid_map', retained),
        ('fine_map', OccupancyGrid, '/Car/T4/mapping/local_fine_map', retained),
        ('coarse_map', OccupancyGrid, '/Car/T4/mapping/exploration_map', retained),
        ('visual', Odometry, '/Car/T3/localization/odometry', qos_profile_sensor_data),
        ('feedback_pose', Odometry, '/P4/input/odometry', qos_profile_sensor_data),
        ('path', RosPath, '/Car/T4/planning/local_path', retained),
        ('global_path', RosPath, '/Car/T4/planning/global_route', retained),
        ('command', Twist, '/P4/debug/cmd_vel', 10),
        ('tracking', TrackingStatus, '/Car/T4/control/tracking_status', 10),
        ('action', GoalStatusArray, ACTION+'/_action/status', retained)]:
        p4.create_subscription(typ, topic, lambda msg, k=key: receive(k, msg), qos)
    p4.create_subscription(DiagnosticArray, '/Car/T4/planning/diagnostics',
                          lambda msg: receive('diagnostics', message_to_ordereddict(msg)), 10)
    p4.create_subscription(NavigateToPose.Impl.FeedbackMessage, ACTION+'/_action/feedback',
                          lambda msg: receive('action_feedback', message_to_ordereddict(msg)), 10)
    for key, topic in [('relay', '/P4/control/relay_status'), ('health', '/fusion/status'),
                       ('map_status', '/fusion/map_status'), ('feedback', '/P4/input/feedback_status')]:
        p4.create_subscription(String, topic, lambda msg, k=key: receive(k, json.loads(msg.data)), 10)
    goal_pub = p4.create_publisher(PoseStamped, '/Car/T4/rviz_goal', 10)
    park_pub = car.create_publisher(String, '/car/set_mode', 10)
    zero_pub = car.create_publisher(Twist, '/car/cmd_vel', 10)
    cancel = p4.create_client(CancelGoal, ACTION+'/_action/cancel_goal')
    get_result = p4.create_client(NavigateToPose.Impl.GetResultService, ACTION+'/_action/get_result')
    result = dict(execute_requested=args.execute, goals=[], command_sent=False,
                  mapper_localization='pure_visual', map_sources=['lidar', 'stereo'],
                  p4_control_feedback='existing P4 aligned UE telemetry; not used by P3 mapping')
    trace = (out/'supervision.jsonl').open('w', buffering=1)
    active_id = [None]
    def spin(seconds):
        until = time.monotonic()+seconds
        while time.monotonic() < until:
            for executor in executors:
                executor.spin_once(timeout_sec=.003)
    def guard():
        for key, age in [('visual', 5.), ('feedback_pose', .7), ('map', 7.), ('health', 3.), ('relay', 3.)]:
            if key not in seen or time.monotonic()-seen[key] > age:
                raise RuntimeError('Stale input: '+key)
        if last['health'].get('localization_reason') == 'visual_origin_reset_restart_required':
            raise RuntimeError('Visual origin changed')
        visual_stamp = last['visual'].header.stamp
        if time.time()-(visual_stamp.sec+visual_stamp.nanosec*1e-9) > 5.:
            raise RuntimeError('Visual measurement over 5 s old')
        if last.get('feedback', {}).get('reason') != 'READY':
            raise RuntimeError('P4 feedback is not ready')
    def snapshot(name):
        arrays = grid_arrays(last['map']); arrays['pose'] = xyz(last['visual'])
        np.savez_compressed(out/(name+'.npz'), **arrays)
        for key in ('fine_map', 'coarse_map'):
            if key in last:
                m = last[key]
                np.savez_compressed(out/(name+'_p4_'+key+'.npz'),
                    occupancy=np.asarray(m.data).reshape(m.info.height,m.info.width),
                    origin=np.array([m.info.origin.position.x,m.info.origin.position.y]),
                    resolution=np.array(m.info.resolution))
        return arrays
    def wait_ready(seconds=20.):
        deadline = time.monotonic()+seconds
        while True:
            spin(.1)
            try:
                guard()
                stamp = last['visual'].header.stamp
                if time.time()-(stamp.sec+stamp.nanosec*1e-9) <= 3.5:
                    return
            except RuntimeError as exc:
                if 'origin changed' in str(exc):
                    raise
            if time.monotonic() >= deadline:
                raise RuntimeError('Fresh P3/P4 inputs did not become ready while parked')
    def patch_info(arrays, point):
        a = arrays['elevation']; rr, cc = np.indices(a.shape)
        xy = arrays['origin']+np.stack([cc+.5, rr+.5], axis=-1)*float(arrays['resolution'])
        values = a[np.linalg.norm(xy-point, axis=-1) < .8]
        finite = values[np.isfinite(values)]
        return dict(known_fraction=float(len(finite)/max(1, len(values))),
                    height_p95_span_m=float(np.percentile(finite,95)-np.percentile(finite,5)) if len(finite) else None)
    def park():
        if active_id[0] is not None and cancel.service_is_ready():
            request = CancelGoal.Request(); request.goal_info.goal_id = active_id[0]
            cancel.call_async(request)
        for _ in range(8):
            park_pub.publish(String(data='park')); zero_pub.publish(Twist()); spin(.1)
    def interrupted(*unused):
        raise KeyboardInterrupt('Supervised test interrupted')
    signal.signal(signal.SIGINT, interrupted); signal.signal(signal.SIGTERM, interrupted)
    try:
        spin(10.); wait_ready()
        if last['relay'].get('path') is not None or last['relay'].get('state') != 'PARKED':
            raise RuntimeError('P4 already has an active motion task')
        baseline = snapshot('before'); initial_pose = xyz(last['feedback_pose']); initial_yaw = yaw(last['feedback_pose'])
        rotation = np.array([[math.cos(initial_yaw), -math.sin(initial_yaw)], [math.sin(initial_yaw), math.cos(initial_yaw)]])
        targets = offsets@rotation.T+initial_pose[:2]
        result['targets'] = [dict(xy=p.tolist(), **patch_info(baseline, p)) for p in targets]
        result['initial_pose'] = initial_pose.tolist(); result['initial_yaw_rad'] = initial_yaw
        result['map_layers'] = list(last['map'].layers)
        result['p4_fine_map_received'] = 'fine_map' in last
        print('PREFLIGHT '+json.dumps(result), flush=True)
        if not args.execute:
            result['outcome'] = 'observation_only'; return
        if goal_pub.get_subscription_count() < 1:
            raise RuntimeError('No P4 RViz goal subscriber')
        for index, point in enumerate(targets):
            wait_ready()
            quality = patch_info(grid_arrays(last['map']), point)
            if quality['known_fraction'] < .8 or quality['height_p95_span_m'] > .12:
                raise RuntimeError('Goal footprint is not sufficiently observed and flat: '+json.dumps(quality))
            previous_ids = {bytes(s.goal_info.goal_id.uuid) for s in last.get('action', GoalStatusArray()).status_list}
            goal = PoseStamped(); goal.header.frame_id = 'map'; goal.header.stamp = p4.get_clock().now().to_msg()
            goal.pose.position.x, goal.pose.position.y = map(float, point)
            desired_yaw = initial_yaw
            if args.face_offset and np.linalg.norm(offsets[index]) > .1:
                direction = 1. if offsets[index,0] >= 0 else -1.
                desired_yaw += math.atan2(direction*offsets[index,1], direction*offsets[index,0])
            goal.pose.orientation.z = math.sin(desired_yaw/2); goal.pose.orientation.w = math.cos(desired_yaw/2)
            phase = dict(index=index+1, target=point.tolist(), initial_position=xyz(last['feedback_pose']).tolist(),
                         target_yaw_rad=desired_yaw, started_wall=time.time(), observed_yaws=[], path_messages_before=counts.get('path',0))
            result['goals'].append(phase)
            began = time.monotonic(); initial_travel = total_travel[0]; result['command_sent'] = True
            goal_pub.publish(goal); print('GOAL '+json.dumps(phase), flush=True)
            saved_path_count = counts.get('path', 0)
            waiting_for_map_since = None
            while True:
                spin(.1); guard()
                if counts.get('path', 0) != saved_path_count:
                    saved_path_count = counts['path']
                    message = last['path']
                    (out/('goal_'+str(index+1)+'_path_'+str(saved_path_count)+'.json')).write_text(json.dumps(
                        dict(frame=message.header.frame_id, poses=[dict(x=p.pose.position.x, y=p.pose.position.y,
                            qz=p.pose.orientation.z, qw=p.pose.orientation.w) for p in message.poses]), indent=2))
                if total_travel[0]-initial_travel > 10. or np.linalg.norm(xyz(last['feedback_pose'])-initial_pose) > 6.:
                    raise RuntimeError('Supervised distance bound exceeded')
                phase['observed_yaws'].append(yaw(last['feedback_pose']))
                status = None
                for item in last.get('action', GoalStatusArray()).status_list:
                    if bytes(item.goal_info.goal_id.uuid) not in previous_ids:
                        if active_id[0] is not None and bytes(active_id[0].uuid) != bytes(item.goal_info.goal_id.uuid):
                            raise RuntimeError('Another goal appeared during the supervised test')
                        active_id[0] = item.goal_info.goal_id; status = item.status
                record = dict(wall=time.time(), goal=index+1, position=xyz(last['feedback_pose']).tolist(),
                              visual=xyz(last['visual']).tolist(), yaw_rad=yaw(last['feedback_pose']),
                              travel_m=total_travel[0], relay=last.get('relay'),
                              height_bias=last.get('map_status',{}).get('height_bias'), action_status=status)
                tracking = last.get('tracking')
                if tracking is not None:
                    record['tracking'] = dict(state=tracking.state, reason=tracking.reason,
                                              heading_error_rad=tracking.heading_error_rad,
                                              cross_track_m=tracking.cross_track_m, progress_m=tracking.progress_m)
                if 'command' in last:
                    record['command'] = [last['command'].linear.x, last['command'].angular.z]
                trace.write(json.dumps(record)+'\n')
                feedback = last.get('action_feedback', {})
                if active_id[0] is not None and feedback.get('goal_id', {}).get('uuid') == list(active_id[0].uuid):
                    reason = feedback['feedback'].get('reason_code')
                    if reason == 'WAITING_FOR_MAP':
                        waiting_for_map_since = waiting_for_map_since or time.monotonic()
                        if time.monotonic()-waiting_for_map_since > 15.:
                            raise RuntimeError('P4 waiting for map without a path for over 15 s')
                    else:
                        waiting_for_map_since = None
                if (tracking is not None and active_id[0] is not None and
                        bytes(tracking.session_id.uuid) == bytes(active_id[0].uuid) and tracking.state == tracking.FAILED):
                    phase['controller_failure'] = record['tracking']
                    raise RuntimeError('P4 controller failed: '+tracking.reason)
                if status in (4, 5, 6):
                    request = NavigateToPose.Impl.GetResultService.Request(); request.goal_id = active_id[0]
                    future = get_result.call_async(request); until = time.monotonic()+5.
                    while not future.done() and time.monotonic() < until:
                        spin(.05)
                    if not future.done():
                        raise RuntimeError('P4 terminal result unavailable')
                    response = future.result(); message = response.result
                    phase['result'] = dict(status=response.status, outcome=message.outcome, reason=message.reason_code)
                    active_id[0] = None
                    if response.status != 4 or message.outcome != 0:
                        raise RuntimeError('P4 goal did not complete: '+json.dumps(phase['result']))
                    break
                if time.monotonic()-began > 180.:
                    raise RuntimeError('P4 goal timed out in the supervised test')
            phase.update(seconds=time.monotonic()-began, distance_m=total_travel[0]-initial_travel,
                         endpoint=xyz(last['feedback_pose']).tolist(),
                         endpoint_distance_m=float(np.linalg.norm(xyz(last['feedback_pose'])[:2]-point)),
                         path_messages=counts.get('path',0)-phase['path_messages_before'])
            phase['yaw_span_deg'] = float(np.ptp(np.rad2deg(np.unwrap(phase.pop('observed_yaws')))))
            park(); spin(8.); wait_ready()
            phase['height_change'] = compare(baseline, snapshot('goal_'+str(index+1)))
            print('COMPLETED '+json.dumps(phase), flush=True)
        result['returned_height_change'] = compare(baseline, snapshot('returned'))
        result['outcome'] = 'completed'
    except BaseException as exc:
        result.update(outcome='stopped', error=type(exc).__name__+': '+str(exc))
        print(result['error'], flush=True)
    finally:
        if result['command_sent']:
            park(); before = xyz(last['feedback_pose']); spin(2.)
            result['park_displacement_m'] = float(np.linalg.norm(xyz(last['feedback_pose'])-before))
            try:
                result['final_height_change'] = compare(baseline, snapshot('final'))
            except (KeyError, RuntimeError):
                pass
        for phase in result['goals']:
            if 'observed_yaws' in phase:
                angles = phase.pop('observed_yaws')
                phase['yaw_span_deg'] = float(np.ptp(np.rad2deg(np.unwrap(angles)))) if angles else 0.
        result['distance_m'] = total_travel[0]; result['counts'] = counts
        result['relay_after'] = last.get('relay'); result['map_status'] = last.get('map_status')
        result['diagnostics_after'] = last.get('diagnostics')
        result['action_feedback_after'] = last.get('action_feedback')
        tracking = last.get('tracking')
        if tracking is not None:
            result['tracking_after'] = dict(state=tracking.state, reason=tracking.reason,
                                            heading_error_rad=tracking.heading_error_rad,
                                            cross_track_m=tracking.cross_track_m, progress_m=tracking.progress_m)
        (out/'result.json').write_text(json.dumps(result, indent=2)); trace.close()
        print(json.dumps(result), flush=True)
        for executor, node, context in zip(executors, nodes, contexts):
            executor.shutdown(); node.destroy_node(); context.try_shutdown()


if __name__ == '__main__':
    main()
