#!/usr/bin/env python3
"""Compare complete trajectories on the same bag timestamps; no fitted scale."""
from pathlib import Path
import csv,json
import numpy as np
from assess_bag_run import _rigid_alignment,_statistics,_path_length
ROOT=Path(__file__).resolve().parents[1]
def main():
 source=ROOT/"results/long_voxelmap_80m_20260915"
 p3=ROOT/"results/p3_roma_long_80m_20260915"
 visual=ROOT/"results/xfeat_compact_long_80m_20260915"
 a=json.loads((source/"assessment.json").read_text())
 p=json.loads((p3/"comparison.json").read_text())
 v=json.loads((visual/"visual_assessment.json").read_text())
 raw=json.loads((source/"verification.json").read_text())["raw_poses"]
 estimates={"VoxelMap":np.asarray([[x["stamp_sec"],*x["pose"]] for x in raw if x["source"]=="lio_raw"]),
            "P3 RoMa":np.loadtxt(p3/"observed_path.tum")}
 if v["full_trajectory_available"]:estimates["XFeat + LighterGlue"]=np.loadtxt(visual/"trajectory_map.tum")
 reference=np.loadtxt(ROOT/"test_data/20260824_235238/reference.csv",delimiter=",",skiprows=1)
 replay=json.loads((source/"replay.json").read_text())
 zero=json.loads((ROOT/"test_data/20260824_235238/full_index.json").read_text())["frames"][0]["record_ns"]
 reference[:,0]+=(zero-replay["first_raw_header_ns"])/1e9
 result={};aligned={};errors={}
 for name,e in estimates.items():
  t=e[:,0];truth=np.column_stack([np.interp(t,reference[:,0],reference[:,i]) for i in (1,2,3)])
  rotation,translation,scale=_rigid_alignment(e[:,1:4],truth)
  aligned[name]=e[:,1:4]@rotation.T+translation;errors[name]=np.linalg.norm(aligned[name]-truth,axis=1)
  middle=reference[(reference[:,0]>t[0])&(reference[:,0]<t[-1]),1:4]
  distance=_path_length(np.vstack([truth[:1],middle,truth[-1:]]))
  result[name]=dict(samples=len(e),duration_sec=float(t[-1]-t[0]),reference_travel_m=distance,
   ate_rmse_m=float(np.sqrt(np.mean(errors[name]**2))),
   ate_over_reference_travel_percent=100*float(np.sqrt(np.mean(errors[name]**2)))/distance,
   errors=_statistics(errors[name]),estimated_path_m=_path_length(e[:,1:4]),
   scale_diagnostic_not_applied=float(scale))
 costs={"VoxelMap":a["lidar_timing"]["after_first_frame"]["processing_sec"],
        "P3 RoMa":p["p3_processing"]["pose_processing_sec"],
        "XFeat + LighterGlue":v["timing"]["after_first_two"]["processing_sec"]}
 medians={k:c.get("p50",c.get("median")) for k,c in costs.items()}
 report=dict(dataset=replay["bag"],methods=result,processing_seconds=costs,
  compact_visual=v,raw_visual_report=str(ROOT/"results/xfeat_only_long_80m_20260915/visual_assessment.json"),
  broken_fusion_baseline=a["accuracy"],
  qualification="Fixed-scale SE3 positional diagnostic on approximate bag-receive timestamps. Different memory/pipeline scopes and shared host load; no general ranking.")
 (ROOT/"results/localizer_comparison_20260915.json").write_text(json.dumps(report,indent=2))
 import matplotlib
 matplotlib.use("Agg")
 import matplotlib.pyplot as plt
 colors={"VoxelMap":"#009E73","P3 RoMa":"#D55E00","XFeat + LighterGlue":"#0072B2"}
 fig,axes=plt.subplots(1,3,figsize=(13.5,4.1),layout="constrained")
 first=next(iter(estimates.values()));t=first[:,0]
 truth=np.column_stack([np.interp(t,reference[:,0],reference[:,i]) for i in (1,2,3)])
 origin=truth[0]
 axes[0].plot(truth[:,0]-origin[0],truth[:,1]-origin[1],color="#444444",lw=2.4,label="Reference")
 for name,e in estimates.items():
  points=aligned[name]-origin
  axes[0].plot(points[:,0],points[:,1],label=name,color=colors[name],lw=1.2)
  axes[1].plot(e[:,0]-1000,errors[name]*100,label=name,color=colors[name],lw=1.25)
 axes[0].set(xlabel="X offset (m)",ylabel="Y offset (m)",title="Same 79.4 m trajectory");axes[0].axis("equal")
 axes[0].legend(fontsize=7)
 axes[1].set(xlabel="Sensor time (s)",ylabel="Position error (cm)",title="One rigid alignment; scale fixed at 1")
 names=list(result);values=[1000*medians[k] for k in names]
 bars=axes[2].bar(range(len(names)),values,color=[colors[k] for k in names],width=.65)
 axes[2].set_xticks(range(len(names)),["VoxelMap","P3 RoMa","XFeat +\nLighterGlue"][:len(names)])
 axes[2].set(ylabel="Median processing (ms)",title="Localization processing after warm-up")
 axes[2].set_ylim(0,max(values)*1.2)
 for b,value in zip(bars,values):axes[2].text(b.get_x()+b.get_width()/2,value+max(values)*.025,f"{value:.0f}",ha="center",fontsize=9)
 for ax in axes:ax.grid(alpha=.15)
 fig.savefig(ROOT/"results/localizer_comparison_20260915.png",dpi=170);plt.close(fig)
 with (ROOT/"results/localizer_comparison_20260915.csv").open("w",newline="") as f:
  writer=csv.writer(f);writer.writerow(["method","samples","reference_travel_m","ate_rmse_m","ate_over_reference_travel_percent","median_processing_sec"])
  for name,d in result.items():writer.writerow([name,d["samples"],d["reference_travel_m"],d["ate_rmse_m"],d["ate_over_reference_travel_percent"],medians[name]])
 print(json.dumps({"methods":result,"median_processing_sec":medians},indent=2))
if __name__=="__main__":main()
