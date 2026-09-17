#!/usr/bin/env python3
"""Offline UE twist consistency audit. Never connects, publishes or calibrates.

Position and attitude are grading references, not localization inputs. Input
velocity fields must be the saved legacy driver's decoded values, NOT raw meta.
All candidate integrations assume m/s and degrees/s and receiver time. These
assumptions are tested, not certified by a successful basis change.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

C = np.array([[-1., 0., 0.], [0., 0., 1.], [0., 1., 0.]])


def stats(values):
    a = np.asarray(values, dtype=float)
    if not a.size:
        return {"n": 0}
    return dict(n=int(a.size), median=float(np.median(a)),
                p95=float(np.percentile(a, 95)), maximum=float(np.max(a)))


def load_record(path):
    path = Path(path)
    extra = {}
    if path.suffix == '.npz':
        with np.load(path, allow_pickle=False) as a:
            d = {k: np.asarray(a[k]) for k in ('t', 'p', 'q', 'v', 'w')}
    elif path.suffix == '.jsonl':
        rows = [json.loads(s) for s in path.read_text().splitlines() if s.strip()]
        rows = [r for r in rows if r['topic'] == '/car/odom']
        d = dict(t=[r['stamp'] for r in rows], p=[r['pose'][:3] for r in rows],
                 q=[r['pose'][3:] for r in rows], v=[r['velocity'][:3] for r in rows],
                 w=[r['velocity'][3:] for r in rows])
    else:
        data = json.loads(path.read_text())
        rows = data if isinstance(data, list) else data.get('telemetry')
        if not isinstance(rows, list) or not rows:
            raise ValueError('Expected continuous decoded telemetry; one raw metadata batch cannot be audited as a trajectory')
        def fields(keys):
            return [[r['message'][k] for k in keys] for r in rows]
        d = dict(t=[r['receive_monotonic'] for r in rows], p=fields(['x', 'y', 'z']),
                 q=fields(['qx', 'qy', 'qz', 'qw']), v=fields(['vx', 'vy', 'vz']),
                 w=fields(['wx', 'wy', 'wz']))
        header = np.array([r['message']['header']['stamp']['sec'] +
                           r['message']['header']['stamp']['nanosec'] * 1e-9 for r in rows])
        wall = np.array([r['receive_wall'] for r in rows])
        extra = dict(header_duration_sec=float(header[-1] - header[0]),
                     receiver_to_subscriber_delay_sec=stats(wall - header))
    d = {k: np.asarray(v, dtype=float) for k, v in d.items()}
    if len(d['t']) < 3 or any(len(v) != len(d['t']) for v in d.values()):
        raise ValueError('Need at least three consistent samples')
    if not np.isfinite(np.c_[d['t'], d['p'], d['q'], d['v'], d['w']]).all():
        raise ValueError('Nonfinite input')
    if np.any(np.diff(d['t']) <= 0):
        raise ValueError('Nonmonotonic time: split reset epochs rather than sorting them')
    if np.any(np.abs(np.linalg.norm(d['q'], axis=1) - 1.) > .01):
        raise ValueError('Malformed reference quaternion')
    d['t'] -= d['t'][0]
    return d, extra


def interpolate(t, values, at):
    return np.stack([np.interp(at, t, values[:, i]) for i in range(values.shape[1])], axis=-1)


def integral(values, t):
    return np.sum((values[1:] + values[:-1]) * .5 * np.diff(t)[:, None], axis=0)


def propagate(t, v, w, initial_rotation):
    rotation = initial_rotation.copy()
    position = np.zeros((len(t), 3))
    for i, dt in enumerate(np.diff(t)):
        angle = .5 * (w[i] + w[i+1]) * dt
        half = Rotation.from_rotvec(angle * .5).as_matrix()
        position[i+1] = position[i] + rotation @ half @ (.5 * (v[i] + v[i+1])) * dt
        rotation = rotation @ Rotation.from_rotvec(angle).as_matrix()
    return position, rotation


def score_trajectory(prediction, reference, t):
    error = np.linalg.norm(prediction - reference, axis=1)
    mse = np.sum(.5 * (error[1:]**2 + error[:-1]**2) * np.diff(t)) / (t[-1] - t[0])
    return dict(end_displacement_m=prediction[-1].tolist(),
                endpoint_error_m=float(error[-1]), max_error_m=float(np.max(error)),
                time_weighted_rmse_m=float(np.sqrt(mse)))


def audit(d, extra=None):
    t, p = d['t'], d['p'] - d['p'][0]
    rb = Rotation.from_matrix(Rotation.from_quat(d['q']).as_matrix() @ C.T)
    matrices = rb.as_matrix()
    attitude = Slerp(t, rb)
    v = d['v'] @ C.T
    w = np.deg2rad(d['w'] @ C.T)
    transform = lambda a, b: np.einsum('nij,nj->ni', a, b)
    world_velocity = transform(matrices, v)
    states = np.c_[d['p'], d['q'], d['v'], d['w']]
    report = dict(samples=len(t), duration_sec=float(t[-1]),
                  receiver_interval_sec=stats(np.diff(t)),
                  adjacent_identical_pose_and_twist_fraction=float(np.mean(np.all(np.diff(states, axis=0) == 0., axis=1))),
                  reference_net_displacement_m=float(np.linalg.norm(p[-1])),
                  reference_sampled_path_m=float(np.sum(np.linalg.norm(np.diff(p, axis=0), axis=1))),
                  reported_speed_integral_m=float(integral(np.linalg.norm(v, axis=1)[:, None], t)[0]),
                  timing=extra or {})
    # Small pose updates may be delivered in bursts. A millimetre update over
    # a sub-ms RECEIVE interval is not evidence of a physical teleport. Report
    # bunching, but reject only large discontinuities as reset candidates.
    step_distance = np.linalg.norm(np.diff(p, axis=0), axis=1)
    step_speed = step_distance / np.diff(t)
    report['receive_timing_diagnostic'] = dict(
        minimum_interval_sec=float(np.min(np.diff(t))),
        sub_5ms_position_updates=int(np.sum((np.diff(t) < .005) & (step_distance > .001))),
        apparent_step_speed_over_4mps=int(np.sum(step_speed > 4.)),
        note='Receiver-time derivatives can spike on bunched packets; not physical speed or a proven teleport')
    if np.max(np.diff(t)) <= .55 and not np.any((step_speed > 4.) & (step_distance > .5)):
        position, _ = propagate(t, v, w, matrices[0])
        aligned = np.vstack([np.zeros(3), np.cumsum(.5*(world_velocity[1:]+world_velocity[:-1])*np.diff(t)[:, None], axis=0)])
        report['free_twist_integration'] = score_trajectory(position, p, t)
        report['reference_attitude_each_sample_diagnostic_only'] = score_trajectory(aligned, p, t)
    else:
        report['whole_record_integration_omitted'] = 'receive gap >0.55s or reference step >0.5m at >4m/s; use qualified windows'
    candidates = {
        'declared_actor_body': world_velocity,
        'decoded_velocity_as_world': d['v'],
        'inverse_decoded_attitude_diagnostic': transform(np.swapaxes(Rotation.from_quat(d['q']).as_matrix(), 1, 2), d['v']),
    }
    # Deliberately NOT a production correction. Uses grading attitude and tests
    # the previously observed twice-heading pattern; speed scale remains 1.
    yaw = rb.as_euler('xyz')[:, 2]
    rz = Rotation.from_rotvec(np.c_[yaw*0., yaw*0., 2.*yaw]).as_matrix()
    candidates['twice_heading_diagnostic_only'] = transform(matrices, transform(rz, v))
    report['windows'] = {}
    for duration in (.5, 1., 2., 10.):
        angular_errors, yaw_errors, speeds, yaw_ratios = [], [], [], []
        linear = {k: dict(direction_error_deg=[], endpoint_error_m=[]) for k in candidates}
        static_count = false_quiet = turning = gaps = jumps = yaw_same = 0
        for start in np.arange(.05, t[-1] - duration, duration):
            end = start + duration
            lo = max(0, np.searchsorted(t, start, side='right') - 1)
            hi = min(len(t), np.searchsorted(t, end, side='left') + 1)
            if np.max(np.diff(t[lo:hi])) > .55:
                gaps += 1
                continue
            at = np.r_[start, t[(t > start) & (t < end)], end]
            reference_p = interpolate(t, p, at)
            delta = reference_p[-1] - reference_p[0]
            distances = np.linalg.norm(np.diff(reference_p, axis=0), axis=1)
            if np.any((distances > .5) & (distances / np.diff(at) > 4.)):
                jumps += 1
                continue
            rotations = attitude([start, end]).as_matrix()
            wa, va = interpolate(t, w, at), interpolate(t, v, at)
            _, predicted_r = propagate(at, va, wa, rotations[0])
            ref_angle = Rotation.from_matrix(rotations[0].T @ rotations[1]).as_rotvec()
            pred_angle = Rotation.from_matrix(rotations[0].T @ predicted_r).as_rotvec()
            ref_speed = np.linalg.norm(delta) / duration
            if abs(ref_angle[2]) / duration > np.deg2rad(1.):
                turning += 1
                angular_errors.append(np.rad2deg(Rotation.from_matrix(predicted_r.T @ rotations[1]).magnitude()))
                yaw_errors.append(np.rad2deg(abs(pred_angle[2] - ref_angle[2])))
                yaw_same += int(pred_angle[2] * ref_angle[2] > 0.)
                raw_angle = integral(wa, at)[2]
                if abs(raw_angle) > 1e-8:
                    yaw_ratios.append(ref_angle[2] / raw_angle)
            if ref_speed > .02:
                measured_distance = integral(np.linalg.norm(va, axis=1)[:, None], at)[0]
                if measured_distance > 1e-8:
                    speeds.append(np.linalg.norm(delta) / measured_distance)
                for name, candidate in candidates.items():
                    pred = integral(interpolate(t, candidate, at), at)
                    linear[name]['endpoint_error_m'].append(np.linalg.norm(pred - delta))
                    denom = np.linalg.norm(pred) * np.linalg.norm(delta)
                    if denom > 1e-12:
                        linear[name]['direction_error_deg'].append(np.rad2deg(np.arccos(np.clip(pred @ delta / denom, -1., 1.))))
            # Illustrative agreement check only. Low feedback alone must never
            # force zero motion: unchanged payload is not evidence of freshness.
            feedback_quiet = np.max(np.linalg.norm(va, axis=1)) < .015 and np.max(np.linalg.norm(wa, axis=1)) < np.deg2rad(.5)
            if feedback_quiet:
                static_count += 1
                false_quiet += int(ref_speed > .02 or np.linalg.norm(ref_angle) / duration > np.deg2rad(1.))
        report['windows'][str(duration)] = dict(
            turning_windows=turning, gyro_rotation_error_deg=stats(angular_errors),
            gyro_yaw_increment_error_deg=stats(yaw_errors), yaw_same_sign_windows=yaw_same,
            reference_over_reported_yaw_integral=stats(yaw_ratios),
            reference_displacement_over_reported_speed_integral=stats(speeds),
            linear_hypotheses={k: {metric: stats(a) for metric, a in values.items()} for k, values in linear.items()},
            feedback_quiet_windows=static_count, quiet_but_reference_moving_windows=false_quiet,
            skipped_gap_windows=gaps, skipped_reference_jump_windows=jumps)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--record', action='append', required=True, help='NAME=PATH (decoded legacy telemetry only)')
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    result = {'assumptions': 'Fixed C, m/s, degree/s, receiver time; no fitted scale or offset. Reference pose is grading-only. Raw metadata is a separate unverified route.', 'records': {}}
    for record in args.record:
        name, path = record.split('=', 1)
        if name in result['records']:
            raise ValueError('Duplicate record name')
        data, extra = load_record(path)
        result['records'][name] = audit(data, extra)
        print(name, 'done', flush=True)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
