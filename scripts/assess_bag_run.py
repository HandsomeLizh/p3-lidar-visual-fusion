#!/usr/bin/env python3
"""Assess this workspace's playback using P3's fixed-scale trajectory metrics."""
import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path
import numpy as np
from visual_evidence import summarize_visual_evidence

ROOT = Path(__file__).resolve().parents[1]
P3 = Path("/home/yanfa/P3/roma_t3_algorithm_bundle_20260825")
sys.path.insert(0, str(P3 / "workspace/scripts"))
from evaluate_trajectory_accuracy import _load_tum, _rigid_alignment, _statistics, _path_length

def quantiles(values):
    a = np.asarray(values, dtype=float)
    a = a[np.isfinite(a)]
    if not len(a):
        return None
    return dict(count=int(len(a)), mean=float(a.mean()), p50=float(np.median(a)),
                p95=float(np.percentile(a, 95)), p99=float(np.percentile(a, 99)),
                max=float(a.max()))

def read_json(path):
    return json.loads(path.read_text()) if path.exists() else {}

def clean(value):
    if isinstance(value, dict):
        return {k:clean(v) for k,v in value.items()}
    if isinstance(value, list):
        return [clean(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value

def assess(output, reference_path=None, trajectory_path=None, reference_max_gap=.25):
    output = output.resolve()
    if not output.is_relative_to(ROOT / "results"):
        raise ValueError("Only assess this workspace's run directory")
    if not math.isfinite(reference_max_gap) or not 0 < reference_max_gap <= 1.:
        raise ValueError("Reference interpolation limit must be in (0, 1] seconds")
    replay = read_json(output / "replay.json")
    if not replay:
        raise ValueError("Completed playback report required")
    frame_zero = replay["first_raw_header_ns"]
    bag = Path(replay["bag"])
    if reference_path is not None:
        reference_path = Path(reference_path)
        reference_zero = frame_zero  # Explicit CSV must use the replay's rebased clock.
        qualification = "Explicit reference CSV on the replay clock."
    elif bag.name == "20260824_235238":
        reference_path = ROOT/"test_data/20260824_235238/reference.csv"
        reference_zero = read_json(ROOT/"test_data/20260824_235238/full_index.json")["frames"][0]["record_ns"]
        qualification = "Approximate bag receive-time association."
    elif bag.parent.name == "short_20260915_190151":
        reference_path = ROOT/"test_data/short_20260915_190151/reference.csv"
        selection = read_json(reference_path.parent/"selection.json")
        reference_zero = selection["first_raw_header_ns"]
        qualification = selection["reference"]["qualification"]
    else:
        raise ValueError("Unknown reference: pass --reference with rebased timestamps")
    reference = np.loadtxt(reference_path, delimiter=",", skiprows=1)
    reference[:, 0] += (reference_zero-frame_zero)/1e9
    trajectory_path=Path(trajectory_path) if trajectory_path is not None else output/"trajectory_map.tum"
    times, positions, _ = _load_tum(trajectory_path)
    # At most one sample per 0.2 s, so output frequency cannot dominate score.
    _, keep = np.unique(np.floor((times-times[0])/.2).astype(np.int64), return_index=True)
    available = len(times)
    times, positions = times[keep], positions[keep]
    right = np.searchsorted(reference[:,0], times)
    selected = (right>0) & (right<len(reference))
    rr = np.clip(right, 1, len(reference)-1)
    bracketing_gaps=reference[rr,0]-reference[rr-1,0]
    selected &= bracketing_gaps <= reference_max_gap
    reference_sampling=dict(max_allowed_gap_sec=reference_max_gap,
        scored_bracketing_gap_sec=quantiles(bracketing_gaps[selected]),
        rejected_output_samples=int((~selected).sum()),
        policy='No reference extrapolation; linear interpolation only between observed reference samples')
    times, positions = times[selected], positions[selected]
    if len(times)<10:
        raise ValueError("Fewer than ten time-associated estimated poses")
    truth = np.column_stack([np.interp(times, reference[:,0], reference[:,axis]) for axis in (1,2,3)])
    rotation, translation, scale = _rigid_alignment(positions, truth)
    aligned = positions @ rotation.T + translation
    errors = np.linalg.norm(aligned-truth, axis=1)
    distance = _path_length(truth)
    def reference_interval_length(begin, end):
        # Integrate the reference stream, not the estimator's chosen output rate.
        middle=reference[(reference[:,0]>begin)&(reference[:,0]<end),1:4]
        endpoints=np.asarray([[np.interp(t,reference[:,0],reference[:,axis])
                                for axis in (1,2,3)] for t in (begin,end)])
        return _path_length(np.vstack([endpoints[:1],middle,endpoints[1:]]))
    reference_travel=reference_interval_length(float(times[0]),float(times[-1]))
    future = np.searchsorted(times, times+10.)
    anchor = np.flatnonzero((future<len(times)) & (times+10.<=times[-1]))
    anchor = anchor[np.abs(times[np.clip(future[anchor],0,len(times)-1)]-times[anchor]-10.) <= .25]
    displacement = [float(np.linalg.norm((aligned[future[i]]-aligned[i]) -
                                         (truth[future[i]]-truth[i]))) for i in anchor]
    verification = read_json(output / "verification.json") or read_json(output / "interface_verification.json")
    measured = verification.get("measurements", [])
    with (output / "replay_frames.csv").open() as stream:
        published = list(csv.DictReader(stream))
    input_begin=float(published[0]["sensor_stamp_sec"])
    input_end=float(published[-1]["sensor_stamp_sec"])
    input_reference_travel=reference_interval_length(input_begin,input_end)
    gaps=np.diff(times)
    intervals=sorted((max(input_begin,float(t)-.25),min(input_end,float(t)+.25)) for t in times
                     if input_begin-.25<=t<=input_end+.25)
    covered=0.;right_edge=input_begin
    for left,right in intervals:
        covered+=max(0.,right-max(left,right_edge))
        right_edge=max(right_edge,right)
    coverage=dict(qualified_output_time_fraction=covered/(input_end-input_begin),
        qualified_output_coverage_definition="Union of +/- 0.25 s around scored qualified poses, clipped to the played input interval",
        input_reference_path_m=input_reference_travel,
        input_duration_sec=input_end-input_begin,
        evaluated_first_stamp_sec=float(times[0]),evaluated_last_stamp_sec=float(times[-1]),
        output_span_fraction=float((times[-1]-times[0])/(input_end-input_begin)),
        max_qualified_pose_gap_sec=float(gaps.max()) if len(gaps) else None,
        gaps_over_two_seconds=int((gaps>2.).sum()),
        qualification="ATE is scored only where qualified output exists. Input path includes initialization; ratio uses reference travel over the evaluated interval.")
    latencies = {}
    for source in ("lio_guarded","vision_accepted"):
        values = []
        for row in published:
            t, wall = float(row["sensor_stamp_sec"]), float(row["publish_start_monotonic_sec"])
            candidates = [m["received_monotonic_sec"]-wall for m in measured
                if m["source"]==source and abs(m["stamp_sec"]-t)<=.08
                and m["received_monotonic_sec"]>=wall]
            if candidates:
                values.append(min(candidates))
        latencies[source] = quantiles(values)
    learned_records = []
    for path in sorted(output.glob("learned_metrics.jsonl*"), reverse=True):
        with path.open() as stream:
            for line in stream:
                try:learned_records.append(json.loads(line))
                except ValueError:continue
    timing = {}
    for group, records in (("all_attempts",learned_records),
                           ("tracked_frames",[r for r in learned_records if r.get("tracking_valid")]),
                           ("after_first_two_attempts",learned_records[2:])):
        timing[group] = {key:quantiles([r[key] for r in records if isinstance(r.get(key),(int,float))])
            for key in ("processing_sec","extract_sec","stereo_match_sec","temporal_match_sec",
                        "geometry_sec","sensor_age_sec","wall_since_input_sec")}
    def source_accuracy(rows):
        if len(rows)<4:return None
        a=np.asarray(rows,dtype=float)
        a=a[np.isfinite(a).all(axis=1)]
        a=a[(a[:,0]>=reference[0,0])&(a[:,0]<=reference[-1,0])]
        if len(a)<4:return None
        order=np.argsort(a[:,0]);a=a[order]
        ref_xyz=np.column_stack([np.interp(a[:,0],reference[:,0],reference[:,i]) for i in (1,2,3)])
        R,t,scale=_rigid_alignment(a[:,1:4],ref_xyz)
        errors=np.linalg.norm(a[:,1:4]@R.T+t-ref_xyz,axis=1)
        travel=reference_interval_length(float(a[0,0]),float(a[-1,0]))
        return dict(samples=len(a),duration_sec=float(a[-1,0]-a[0,0]),
            reference_travel_m=travel,
            ate_over_reference_travel_percent=100*float(np.sqrt(np.mean(errors**2)))/travel if travel>0 else None,
            estimated_path_m=_path_length(a[:,1:4]),reference_path_m=_path_length(ref_xyz),
            fixed_scale_position_error_m=_statistics(errors),scale_diagnostic_not_applied=float(scale))
    raw_diagnostics={}
    raw=verification.get("raw_poses",[])
    for source in ("lio_raw","learned_raw","vins_raw","roma_raw"):
        groups=sorted({row["frame"] for row in raw if row["source"]==source})
        entries=[]
        for frame in groups:
            selected=[row for row in raw if row["source"]==source and row["frame"]==frame]
            diagnostic=source_accuracy([[row["stamp_sec"],*row["pose"][:3]] for row in selected])
            if diagnostic is not None:entries.append(dict(frame=frame,**diagnostic))
        if entries:raw_diagnostics[source]=entries
    lidar_records=[]
    if (output/"lidar_metrics.jsonl").exists():
        for line in (output/"lidar_metrics.jsonl").read_text().splitlines():
            try:lidar_records.append(json.loads(line))
            except ValueError:continue
    lidar_timing={group:{key:quantiles([r[key] for r in rows if isinstance(r.get(key),(int,float))])
                 for key in ("processing_sec","matching_sec","solve_sec","map_update_sec","peak_rss_mib")}
                 for group,rows in (("all_frames",lidar_records),("after_first_frame",lidar_records[1:]))}
    resources = read_json(output/"resources_latest.json")
    fusion = read_json(output/"fusion_status.json")
    learned = read_json(output/"learned_metrics.json")
    start, end = float(published[0]["publish_start_monotonic_sec"]),float(published[-1]["publish_end_monotonic_sec"])
    sample_seconds = max(end-start, 1e-9)
    report = dict(
        trajectory_source=str(trajectory_path),
        bag=replay["bag"],frames_played=replay["frames"],
        sensor_duration_sec=float(published[-1]["sensor_stamp_sec"])-float(published[0]["sensor_stamp_sec"]),
        replay_rate=replay["rate"],input_publication_wall_span_sec=sample_seconds,
        input_hz=(len(published)-1)/sample_seconds,
        accuracy=dict(matched_samples=len(times),available_output_poses=available,
            reference_sampling=reference_sampling,
            evaluated_duration_sec=float(times[-1]-times[0]),reference_path_m=distance,
            reference_travel_m=reference_travel,
            ate_over_reference_travel_percent=100*float(np.sqrt(np.mean(errors**2)))/reference_travel if reference_travel>0 else None,
            ratio_definition="100 * translational ATE RMSE / dense reference travel over the evaluated interval",
            coverage=coverage,
            estimate_path_m=_path_length(positions),fixed_scale_se3_position_error_m=_statistics(errors),
            endpoint_error_after_alignment_m=float(errors[-1]),
            ten_second_displacement_error_m=_statistics(displacement),
            rmse_over_reference_path_percent=100*float(np.sqrt(np.mean(errors**2)))/distance if distance>0 else None,
            scale_diagnostic_not_applied=float(scale),
            reference_spatial_singular_values_m=np.linalg.svd(truth-truth.mean(axis=0),compute_uv=False).tolist(),
            qualification=qualification+f" Reference interpolation is limited to {reference_max_gap:g} s gaps; increasing this bound reduces timing certainty."+" P3 fixed-scale SE(3) positional diagnostic; no fitted time offset or scale correction. Reference body quaternion convention and exposure synchronization are not certified."),
        input_publish_to_guarded_measurement_receipt_sec=latencies,
        timing=timing,learned_metrics=learned,lidar_timing=lidar_timing,
        lidar_last_metrics=lidar_records[-1] if lidar_records else None,
        adaptive_source_summary={key:fusion.get(key) for key in ("fusion_strategy","operating_mode",
            "visual_constraints","visual_pose_constraints","visual_motion_constraints","visual_constraint_mode",
            "lidar_pose_constraints","visual_anchors","lidar_anchors","adaptive_rejections")},
        raw_source_diagnostics=raw_diagnostics,
        source_counters=dict(lio=fusion.get("lio"),visual_received=fusion.get("visual_received"),
                             visual_accepted=fusion.get("visual_accepted")),
        visual_checks=summarize_visual_evidence(verification),
        observer_reported_visual_checks=verification.get("visual_checks"),
        interface_checks=verification.get("checks"),
        memory=dict(peak_process_tree_rss_mib=resources.get("tree_rss_peak_mib"),
                    warning=resources.get("memory_warning")),
        map_statistics=read_json(output/"map_statistics.json"),
        notes=["Frame processing cost is separate from this bag's sparse acquisition rate.",
               "Pose publication frequency and message age do not prove that every prediction uses a new sensor observation.",
               "No reference surface is available to certify elevation-map geometric accuracy.",
               "Guard-topic activity and a reported mode do not prove EKF acceptance; inspect non-anchor measurement evidence and final trajectory accuracy."],
        evaluation_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        p3_accuracy_implementation=str(P3/"workspace/scripts/evaluate_trajectory_accuracy.py"),
        p3_accuracy_sha256=hashlib.sha256((P3/"workspace/scripts/evaluate_trajectory_accuracy.py").read_bytes()).hexdigest())
    with (output/"trajectory_accuracy_samples.csv").open("w",newline="") as stream:
        writer=csv.writer(stream)
        writer.writerow(["replay_stamp_sec","estimate_x","estimate_y","estimate_z","reference_x","reference_y","reference_z","position_error_m"])
        writer.writerows(np.column_stack([times,aligned,truth,errors]))
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(1,2,figsize=(11,4.2),layout="constrained")
        origin=truth[0]
        axes[0].plot(truth[:,0]-origin[0],truth[:,1]-origin[1],label="Simulator reference")
        axes[0].plot(aligned[:,0]-origin[0],aligned[:,1]-origin[1],label="Estimate: rigid alignment, scale=1")
        axes[0].set(xlabel="X offset / m",ylabel="Y offset / m")
        axes[0].set_aspect("equal");axes[0].legend(fontsize=8)
        axes[1].plot(times-times[0],errors)
        axes[1].set(xlabel="Time / s",ylabel="Position error / m",
                    title=f"RMSE {np.sqrt(np.mean(errors**2)):.3f} m")
        fig.savefig(output/"trajectory_accuracy.png",dpi=140);plt.close(fig)
    except ImportError:
        report["plot_unavailable"]="matplotlib not installed"
    report=clean(report)
    (output/"assessment.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    print(json.dumps({k:report[k] for k in ("frames_played","sensor_duration_sec","accuracy",
        "input_publish_to_guarded_measurement_receipt_sec","interface_checks","memory")},indent=2))
    return report

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("output",type=Path)
    parser.add_argument("--reference",type=Path)
    parser.add_argument("--trajectory",type=Path)
    parser.add_argument("--reference-max-gap",type=float,default=.25)
    args=parser.parse_args()
    assess(args.output,args.reference,args.trajectory,args.reference_max_gap)
