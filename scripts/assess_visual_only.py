#!/usr/bin/env python3
"""Assess pure visual continuity without aligning restarted epochs as one path."""
import argparse,json,sys
from pathlib import Path
import numpy as np
from assess_bag_run import assess,_rigid_alignment,_statistics,_path_length,quantiles,clean
ROOT=Path(__file__).resolve().parents[1]
def main():
 a=argparse.ArgumentParser();a.add_argument("output",type=Path);x=a.parse_args()
 out=x.output.resolve();replay=json.loads((out/"replay.json").read_text())
 raw=[json.loads(line) for line in (out/"visual_raw.jsonl").read_text().splitlines()]
 verification=json.loads((out/"verification.json").read_text())
 reference=np.loadtxt(ROOT/"test_data/20260824_235238/reference.csv",delimiter=",",skiprows=1)
 zero=json.loads((ROOT/"test_data/20260824_235238/full_index.json").read_text())["frames"][0]["record_ns"]
 reference[:,0]+=(zero-replay["first_raw_header_ns"])/1e9
 groups={}
 for row in raw:groups.setdefault(row["frame"],[]).append(row)
 epochs=[]
 for name,rows in groups.items():
  t=np.array([row["stamp_sec"] for row in rows]);p=np.array([row["pose"][:3] for row in rows])
  result=dict(frame=name,poses=len(rows),start_sec=float(t[0]),end_sec=float(t[-1]),duration_sec=float(t[-1]-t[0]))
  if len(rows)>=4 and t[-1]>t[0]:
   truth=np.column_stack([np.interp(t,reference[:,0],reference[:,i]) for i in (1,2,3)])
   rot,trans,scale=_rigid_alignment(p,truth);error=np.linalg.norm(p@rot.T+trans-truth,axis=1)
   middle=reference[(reference[:,0]>t[0])&(reference[:,0]<t[-1]),1:4]
   travel=_path_length(np.vstack([truth[:1],middle,truth[-1:]]))
   result.update(ate_rmse_m=float(np.sqrt(np.mean(error**2))),reference_travel_m=travel,
    ate_over_reference_travel_percent=100*float(np.sqrt(np.mean(error**2)))/travel if travel>0 else None,
    position_errors=_statistics(error),scale_diagnostic_not_applied=float(scale))
  epochs.append(result)
 records=[json.loads(line) for line in (out/"learned_metrics.jsonl").read_text().splitlines()]
 tracked=[d for d in records if d.get("tracking_valid")]
 reasons={}
 for d in records:reasons[d["reason"]]=reasons.get(d["reason"],0)+1
 timings={group:{k:quantiles([d[k] for d in data if isinstance(d.get(k),(int,float))])
  for k in ["processing_sec","sensor_age_sec","wall_since_input_sec"]}
  for group,data in [("all_attempts",records),("tracked",tracked),("after_first_two",records[2:])]}
 report=dict(frames_played=replay["frames"],poses=len(raw),epochs=epochs,epoch_count=len(epochs),
  full_trajectory_available=len(epochs)==1,counts_by_reason=reasons,timing=timings,
  no_lidar_or_imu_observed=verification["no_lidar_or_imu_observed"],
  memory=json.loads((out/"resources_latest.json").read_text()),
  qualification="Different epochs have independent origins. Per-epoch alignment is diagnostic only; it is never a whole-trajectory ATE.")
 if len(epochs)==1:
  with (out/"trajectory_map.tum").open("w") as f:
   for row in raw:f.write(" ".join(format(v,".15g") for v in [row["stamp_sec"],*row["pose"]])+"\n")
  report["whole_trajectory_accuracy"]=assess(out)["accuracy"]
 report=clean(report)
 (out/"visual_assessment.json").write_text(json.dumps(report,indent=2,allow_nan=False))
 print(json.dumps({k:report[k] for k in ["frames_played","poses","epoch_count","full_trajectory_available","counts_by_reason","no_lidar_or_imu_observed"]},indent=2))
if __name__=="__main__":main()
