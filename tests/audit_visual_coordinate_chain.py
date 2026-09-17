"""Read-only comparison of saved XFeat/RoMa poses and UE reference positions.

Matches identical visual timestamps, keeps motion samples to avoid stationary
weighting, and reports raw height separately from a rigidly aligned 3D ATE.
No reference data or fitted corrections are used by localization or mapping.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation


def nearest(times, stamps):
    ix=np.searchsorted(times,stamps).clip(1,len(times)-1)
    return np.where(abs(times[ix-1]-stamps)<abs(times[ix]-stamps),ix-1,ix)


def evaluate(poses,truth):
    start_rotation=Rotation.from_quat(poses[0,4:8]).as_matrix()
    local=(poses[:,1:4]-poses[0,1:4])@start_rotation
    estimate=local-local.mean(0);reference=truth-truth.mean(0)
    u,s,vt=np.linalg.svd(estimate.T@reference)
    d=np.eye(3);d[2,2]=np.linalg.det(vt.T@u.T)
    rotation=vt.T@d@u.T
    error=estimate@rotation.T-reference
    # Fit only for diagnosis: a constant tilt cannot explain an arbitrary
    # time-varying height error. Do not install this fit as calibration.
    target=local[:,2]-(truth[:,2]-truth[0,2])
    design=np.c_[local[:,:2],np.ones(len(local))]
    plane=np.linalg.lstsq(design,target,rcond=None)[0]
    remaining=target-design@plane
    return dict(samples=len(local),ate_se3_no_scale_rmse_m=float(np.sqrt(np.mean(np.sum(error**2,axis=1)))),
        endpoint_after_alignment_m=float(np.linalg.norm(error[-1])),
        output_frame_z_change_m=float(poses[-1,3]-poses[0,3]),
        initial_body_frame_z_change_m=float(local[-1,2]),
        reference_z_change_m=float(truth[-1,2]-truth[0,2]),
        diagnostic_constant_tilt_plane=plane.tolist(),
        residual_after_constant_tilt_fit_p95_m=float(np.percentile(abs(remaining),95)))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('visual_run',type=Path)
    parser.add_argument('roma_tum',type=Path)
    parser.add_argument('output',type=Path)
    args=parser.parse_args()
    records=[json.loads(l) for l in (args.visual_run/'visual_raw.jsonl').read_text().splitlines()]
    records=[r for r in records if r['qualified']]
    xfeat=np.asarray([[r['stamp_sec']]+r['pose'] for r in records])
    roma=np.loadtxt(args.roma_tum)
    truth=np.genfromtxt(args.visual_run/'reference.csv',delimiter=',',skip_header=1)
    ri=nearest(roma[:,0],xfeat[:,0]);ti=nearest(truth[:,0],xfeat[:,0])
    mask=(abs(roma[ri,0]-xfeat[:,0])<.001)&(abs(truth[ti,0]-xfeat[:,0])<=.25)
    xfeat,roma,truth=xfeat[mask],roma[ri[mask]],truth[ti[mask]]
    selected=[0]
    for i in range(1,len(truth)):
        if np.linalg.norm(truth[i,1:4]-truth[selected[-1],1:4])>=.05:
            selected.append(i)
    # Include the last sample to retain the actual endpoint during a stop.
    if selected[-1]!=len(truth)-1:selected.append(len(truth)-1)
    xfeat,roma,truth=xfeat[selected],roma[selected],truth[selected]
    report=dict(method='Identical image header times; UE nearest <=0.25 s; spatial sampling >=0.05 m; SE3 without scale.',
        old_roma_freshly_initialized=False,reference_used_for_localization=False,
        calibration_modified=False,frame_alignment_warning='Initial body frame is not independently verified gravity-up.',
        start_stamp=float(truth[0,0]),end_stamp=float(truth[-1,0]),
        reference_sampled_travel_m=float(np.linalg.norm(np.diff(truth[:,1:4],axis=0),axis=1).sum()),
        xfeat=evaluate(xfeat,truth[:,1:4]),roma_existing=evaluate(roma,truth[:,1:4]))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
