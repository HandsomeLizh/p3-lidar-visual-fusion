#!/usr/bin/env python3
"""Read-only short-bag visual continuity diagnostic; no ROS output or truth input."""
import json,time,sys
from pathlib import Path
from collections import Counter
import numpy as np
import cv2,yaml
from cv_bridge import CvBridge
from replay_bag import batches,TOPICS
from t3_lidar_visual_fusion.learned_matching import LearnedMatcher
from t3_lidar_visual_fusion.learned_tracker import LearnedStereoTracker
from t3_lidar_visual_fusion.image_quality import assess_image
from t3_lidar_visual_fusion.stereo_geometry import TrackingFailure
ROOT=Path(__file__).resolve().parents[1]
BAG=Path("/home/yanfa/P3/roma_t3_algorithm_bundle_20260825/recordings/short_20260915_190151/bag")
out=ROOT/"test_data/short_20260915_190151/visual_continuity.json"
profile=yaml.safe_load((ROOT/"config/selected_bag_xfeat.yaml").read_text())
matcher=LearnedMatcher(ROOT,profile["learned_visual"])
tracker=LearnedStereoTracker(profile,matcher)
bridge=CvBridge();cv2.setNumThreads(1)
records=[];first=None
for index,(raw,group) in enumerate(batches(BAG,0)):
    if first is None:first=raw
    stamp=1000.+(raw-first)/1e9
    pair=[cv2.resize(bridge.imgmsg_to_cv2(group[t],desired_encoding="mono8"),
          tuple(profile["output_image_size"]),interpolation=cv2.INTER_AREA) for t in TOPICS[:2]]
    start=time.perf_counter()
    try:
        quality=[assess_image(x,texture_required=False) for x in pair]
        if not all(x.valid for x in quality):raise TrackingFailure(next(x.reason for x in quality if not x.valid))
        result=tracker.process(stamp,*pair)
        record=dict(frame=index,stamp_sec=stamp,anchor=result.anchor,epoch=result.epoch,
            pose=result.base_pose.tolist(),metrics=result.metrics)
    except (TrackingFailure,ValueError,cv2.error) as error:
        tracker.reset()
        record=dict(frame=index,stamp_sec=stamp,rejected=str(error))
    record["observed_processing_sec"]=time.perf_counter()-start
    records.append(record)
    if index%10==0:print(json.dumps(dict(frame=index,reason=record.get("rejected",record.get("metrics",{}).get("reason")))),flush=True)
counts=Counter(x.get("rejected",x.get("metrics",{}).get("reason")) for x in records)
sys.path.insert(0,"/home/yanfa/P3/roma_t3_algorithm_bundle_20260825/workspace/scripts")
from evaluate_trajectory_accuracy import _rigid_alignment,_statistics,_path_length
reference=np.loadtxt(ROOT/"test_data/short_20260915_190151/reference.csv",delimiter=",",skiprows=1)
epochs=[]
for epoch in sorted({x["epoch"] for x in records if "epoch" in x}):
    rows=[x for x in records if x.get("epoch")==epoch]
    if len(rows)<4:continue
    stamps=np.array([x["stamp_sec"] for x in rows])
    xyz=np.array([x["pose"] for x in rows])[:,:3,3]
    truth=np.column_stack([np.interp(stamps,reference[:,0],reference[:,i]) for i in [1,2,3]])
    R,t,scale=_rigid_alignment(xyz,truth)
    error=np.linalg.norm(xyz@R.T+t-truth,axis=1)
    epochs.append(dict(epoch=epoch,frames=len(rows),duration_sec=float(stamps[-1]-stamps[0]),
        reference_path_m=_path_length(truth),estimate_path_m=_path_length(xyz),
        fixed_scale_position_error_m=_statistics(error),scale_diagnostic_not_applied=float(scale)))
report=dict(bag=str(BAG),sensor_inputs=["stereo"],imu_input=False,counts=dict(counts),epochs=epochs,records=records,
    performance_note="Offline continuity diagnostic during concurrent build; timing is not an isolated or full ROS pipeline benchmark.",
    reference_note="Reference only used after all tracking completed; approximate telemetry association; each uninterrupted epoch aligned separately without scale correction.")
out.write_text(json.dumps(report,indent=2,allow_nan=False))
print(json.dumps({k:report[k] for k in ["counts","epochs","performance_note"]},indent=2),flush=True)
