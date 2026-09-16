#!/usr/bin/env python3
"""Evaluate actual live data without presenting it as bag playback."""
import argparse
import json
import math
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
P3 = ROOT.parent / 'roma_t3_algorithm_bundle_20260825'
sys.path.insert(0, str(P3 / 'workspace/scripts'))
from evaluate_trajectory_accuracy import _load_tum, _rigid_alignment, _path_length
from assess_bag_run import quantiles


def read(path):
    return json.loads(path.read_text())


def records(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    out = args.output.resolve()
    assert out.is_relative_to(ROOT / 'results')
    session, verify = read(out / 'live_session_status.json'), read(out / 'verification.json')
    mapping = read(out / 'map_statistics.json')
    inputs, outputs = records(out / 'live_inputs.jsonl'), records(out / 'live_outputs.jsonl')
    forwarded = [x for x in inputs if x['kind'] == 'forwarded']
    reference = np.loadtxt(out / 'live_reference.csv', delimiter=',', skiprows=1)
    start, stop = session['motion']['started_wall'], session['motion']['finished_wall']

    def score(t, xyz):
        t, xyz = np.asarray(t), np.asarray(xyz)
        right = np.searchsorted(reference[:, 0], t)
        clipped = np.clip(right, 1, len(reference)-1)
        mask = ((t >= start) & (t <= stop) & (right > 0) & (right < len(reference)) &
                (reference[clipped, 0]-reference[clipped-1, 0] <= .25))
        t, xyz = t[mask], xyz[mask]
        if len(t) < 4:
            return None, None
        truth = np.column_stack([np.interp(t, reference[:, 0], reference[:, i]) for i in (1, 2, 3)])
        rotation, translation, scale = _rigid_alignment(xyz, truth)
        aligned = xyz @ rotation.T + translation
        errors = np.linalg.norm(aligned-truth, axis=1)
        middle = reference[(reference[:, 0] > t[0]) & (reference[:, 0] < t[-1]), 1:4]
        travel = _path_length(np.vstack([truth[:1], middle, truth[-1:]]))
        result = dict(samples=len(t), first_stamp_sec=float(t[0]), last_stamp_sec=float(t[-1]),
                      scored_reference_travel_m=travel, ate_m=float(np.sqrt(np.mean(errors**2))),
                      ate_over_travel_percent=100*float(np.sqrt(np.mean(errors**2)))/travel,
                      max_error_m=float(errors.max()), p95_error_m=float(np.percentile(errors, 95)),
                      estimated_sample_path_m=_path_length(xyz), scale_diagnostic_not_applied=float(scale))
        return result, (t, aligned, truth, errors)

    t, xyz, _ = _load_tum(out / 'trajectory_map.tum')
    accuracy, curve = score(t, xyz)
    assert accuracy is not None
    raw_scores = {}
    for source in ('lio_raw', 'learned_raw'):
        raw = [x for x in verify['raw_poses'] if x['source'] == source]
        epochs = sorted(set(x['frame'] for x in raw))
        raw_scores[source] = dict(epoch_count=len(epochs), full_trajectory_ate_allowed=len(epochs)==1, epochs={})
        for epoch in epochs:
            rows = [x for x in raw if x['frame'] == epoch and max(x['pose_covariance'][i*7] for i in range(6)) < 1e5]
            if rows:
                result, _ = score([x['stamp_sec'] for x in rows], [x['pose'][:3] for x in rows])
                if result:raw_scores[source]['epochs'][epoch] = result
    receipt_latency, header_latency = [], []
    for frame in forwarded:
        matches = [x for x in outputs if abs(x['sensor_stamp_sec']-frame['sensor_stamp_sec']) < .08
                   and x['received_monotonic_sec'] >= frame['publish_start_monotonic_sec']]
        if matches:
            last = max(matches, key=lambda x: x['received_monotonic_sec'])
            receipt_latency.append(last['received_monotonic_sec']-frame['publish_start_monotonic_sec'])
            header_latency.append(last['header_age_sec'])
    lidar, visual = records(out / 'lidar_metrics.jsonl'), records(out / 'learned_metrics.jsonl')
    resources = records(out / 'resources.jsonl')
    fusion = read(out / 'fusion_status.json')
    result = dict(
        kind='actual_live_simulation', accepted=False if session['status']!='completed' else None,
        session_completed=session['status']=='completed',
        stop_reason=session.get('error'), motion=session['motion'], counts=session['counts'],
        source_period_sec=quantiles(np.diff([x['sensor_stamp_sec'] for x in forwarded])),
        mapping=mapping, interface_checks=verify['checks'], topic_message_counts=verify['counts'],
        fused_motion_interval_accuracy=accuracy, raw_source_diagnostics=raw_scores,
        timing=dict(source_header_to_transfer_sec=quantiles([x['header_age_sec'] for x in forwarded]),
                    transfer_to_last_qualified_pose_sec=quantiles(receipt_latency),
                    sensor_header_to_last_qualified_pose_sec=quantiles(header_latency),
                    lidar_processing_sec=quantiles([x['processing_sec'] for x in lidar]),
                    visual_processing_sec=quantiles([x['processing_sec'] for x in visual])),
        memory=dict(peak_owned_tree_rss_mib=max(x['tree_rss_mib'] for x in resources),
                    pressure_observed=any(x['memory_warning'] for x in resources)),
        visual_constraints=fusion.get('visual_constraints'), rejections=fusion.get('adaptive_rejections'),
        park_verification=read(out / 'park_verification.json') if (out / 'park_verification.json').exists() else None,
        qualifications=[
            'Measured from live sensor_capture_node and lunar_car_node, not rosbag playback.',
            'Fixed-scale SE3 positional diagnostic on motion-only qualified outputs; no fitted time shift or scale.',
            'Reference is only recorded and used for motion-stop supervision; it is not fed to either estimator.',
            'Exposure-to-reference synchronization is not independently calibrated.',
            'Visual epochs are scored separately and never stitched into a full-trajectory ATE.',
            'Topic-level checks passing does not imply a correct map or accepted online test.',
            'Existing P3 RoMa was running concurrently; memory is this owned pipeline, transfer and RViz only.',
        ])
    (out / 'live_assessment.json').write_text(json.dumps(result, indent=2) + '\n')
    ts, aligned, truth, errors = curve
    np.savetxt(out / 'live_accuracy_samples.csv', np.column_stack([ts, aligned, truth, errors]),
               delimiter=',', header='stamp_sec,estimate_x,estimate_y,estimate_z,reference_x,reference_y,reference_z,error_m', comments='')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    origin = truth[0]
    axes[0].plot(truth[:, 0]-origin[0], truth[:, 1]-origin[1], label='Live simulator reference')
    axes[0].plot(aligned[:, 0]-origin[0], aligned[:, 1]-origin[1], label='Fusion (rigid alignment, scale = 1)')
    axes[0].set(xlabel='X / m', ylabel='Y / m', title='Online test: failed localization' if result['accepted'] is False else 'Live trajectory diagnostic')
    axes[0].axis('equal');axes[0].legend(fontsize=8)
    axes[1].plot(ts-start, errors)
    axes[1].set(xlabel='Time after motion start / s', ylabel='Position error / m', title=f'ATE {accuracy["ate_m"]:.2f} m')
    fig.savefig(out / 'live_accuracy.png', dpi=150);plt.close(fig)
    print(json.dumps({k:result[k] for k in ['accepted','stop_reason','counts','fused_motion_interval_accuracy','timing','memory']}, indent=2))


if __name__ == '__main__':
    main()
