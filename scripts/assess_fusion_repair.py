#!/usr/bin/env python3
"""Compare one complete clean run and one predeclared photometric intervention."""
from pathlib import Path
import csv
import json
import numpy as np
from assess_bag_run import _rigid_alignment, _statistics

ROOT=Path(__file__).resolve().parents[1]


def read_json(path): return json.loads(path.read_text())


def main():
    clean=ROOT/'results/long_fusion_repaired_clean_verified_20260916'
    fault=ROOT/'results/long_fusion_repaired_perturbed_verified_20260916'
    baseline=ROOT/'results/long_voxelmap_80m_20260915'
    outputs={'clean':clean,'perturbed':fault}
    manifests=[read_json(p/'runtime_manifest.json') for p in outputs.values()]
    assert manifests[0]==manifests[1], 'Clean and perturbed runs must use identical built code'
    travel=read_json(clean/'assessment.json')['accuracy']['reference_travel_m']
    source=read_json(clean/'verification.json')['raw_poses']
    raw_lidar=np.asarray([[p['stamp_sec'],*p['pose']] for p in source if p['source']=='lio_raw'])
    trajectories={'VoxelMap':raw_lidar,**{k:np.loadtxt(p/'trajectory_map.tum') for k,p in outputs.items()}}
    reference=np.loadtxt(ROOT/'test_data/20260824_235238/reference.csv',delimiter=',',skiprows=1)
    replay=read_json(clean/'replay.json')
    first=read_json(ROOT/'test_data/20260824_235238/full_index.json')['frames'][0]['record_ns']
    reference[:,0]+=(first-replay['first_raw_header_ns'])/1e9
    times=raw_lidar[:,0]
    assert len(times)==174 and np.all(np.diff(times)>0)
    truth=np.column_stack([np.interp(times,reference[:,0],reference[:,i]) for i in (1,2,3)])
    errors={};aligned={};scores={}
    for name,trajectory in trajectories.items():
        assert len(trajectory)==174 and np.max(np.abs(trajectory[:,0]-times))<1e-7,(name,len(trajectory))
        rotation,translation,scale=_rigid_alignment(trajectory[:,1:4],truth)
        aligned[name]=trajectory[:,1:4]@rotation.T+translation
        errors[name]=np.linalg.norm(aligned[name]-truth,axis=1)
        scores[name]=dict(position_error_m=_statistics(errors[name]),
            ate_over_reference_travel_percent=100*np.sqrt(np.mean(errors[name]**2))/travel,
            scale_diagnostic_not_applied=float(scale),poses=174)
    # Intervals use the one whole-run alignment; never re-align each segment.
    schedule=read_json(fault/'replay.json')['perturbations']
    labels=[]
    for i in range(174):
        labels.append(next((b['kind'] for b in schedule if b['start']<=i<b['end']),'clean'))
    intervals=[]
    observed={}
    for name,path in outputs.items():
        verification=read_json(path/'verification.json')
        measured={kind:set() for kind in ['lio_guarded','vision_accepted','fused']}
        for m in verification['measurements']:
            if m['source'] in measured:
                i=int(np.argmin(abs(times-m['stamp_sec'])))
                if abs(times[i]-m['stamp_sec'])<1e-7:
                    measured[m['source']].add(i)
        observed[name]=measured
    for kind in dict.fromkeys(labels):
        indices=[i for i,label in enumerate(labels) if label==kind]
        intervals.append(dict(kind=kind,frames=len(indices),
            clean_error_m=_statistics(errors['clean'][indices]),
            perturbed_error_m=_statistics(errors['perturbed'][indices]),
            lidar_frames=len(set(indices)&observed['perturbed']['lio_guarded']),
            visual_constraint_frames=len(set(indices)&observed['perturbed']['vision_accepted']),
            fused_frames=len(set(indices)&observed['perturbed']['fused'])))
    recovery=[]
    accepted=observed['perturbed']['vision_accepted']
    for block in schedule:
        if block['kind']=='grayscale_control': continue
        next_fault=min([b['start'] for b in schedule if b['start']>=block['end'] and b['kind']!='grayscale_control']+[174])
        later=next((i for i in sorted(accepted) if block['end']<=i<next_fault),None)
        recovery.append(dict(kind=block['kind'],first_restored_frame=block['end'],
            first_accepted_visual_frame=later,
            recovery_sensor_seconds=None if later is None else float(times[later]-times[block['end']]),
            clear_interval_end_frame=next_fault,
            note='Includes frontend recovery, quality checks and sparse acquisition; not neural inference latency'))
    old_source=read_json(baseline/'verification.json')['raw_poses']
    old_lidar=np.asarray([[p['stamp_sec'],*p['pose']] for p in old_source if p['source']=='lio_raw'])
    fault_source=read_json(fault/'verification.json')['raw_poses']
    fault_lidar=np.asarray([[p['stamp_sec'],*p['pose']] for p in fault_source if p['source']=='lio_raw'])
    assert old_lidar.shape==raw_lidar.shape==fault_lidar.shape
    runtime={}
    for name,path in outputs.items():
        a=read_json(path/'assessment.json')
        runtime[name]={k:a.get(k) for k in ['accuracy','interface_checks','memory','map_statistics',
            'lidar_timing','timing','adaptive_source_summary','source_counters','input_publish_to_guarded_measurement_receipt_sec']}
        runtime[name]['tf_odometry_consistency']=read_json(path/'verification.json')['details'].get('tf_odometry_consistency')
        with (path/'replay_frames.csv').open() as stream:
            published={round(float(row['sensor_stamp_sec'])*1e9):float(row['publish_start_monotonic_sec']) for row in csv.DictReader(stream)}
        latest={}
        for m in read_json(path/'verification.json')['measurements']:
            if m['source']=='fused':latest[round(m['stamp_sec']*1e9)]=m['received_monotonic_sec']
        latency=[wall-published[stamp] for stamp,wall in latest.items() if stamp in published]
        runtime[name]['last_qualified_pose_receipt_latency_sec']=_statistics(np.asarray(latency))
    report=dict(dataset=replay['bag'],frames=174,sensor_duration_sec=float(times[-1]-times[0]),
        reference_travel_m=travel,methods=scores,runtime_code_identical=True,
        old_failed_fusion=read_json(baseline/'assessment.json')['accuracy'],
        intervals=intervals,recovery=recovery,runtime=runtime,
        max_lidar_position_change_from_old_backend_m=float(np.max(np.linalg.norm(raw_lidar[:,1:4]-old_lidar[:,1:4],axis=1))),
        max_lidar_position_change_with_image_interference_m=float(np.max(np.linalg.norm(raw_lidar[:,1:4]-fault_lidar[:,1:4],axis=1))),
        schedule=schedule,
        qualification='Single simulation trajectory, approximate receive-time association, one SE3 position alignment per full trajectory, fixed scale, no truth supplied to estimators. Controlled pixel perturbations do not reproduce every physical lighting effect.')
    target=ROOT/'results/fusion_repair_comparison_20260916'
    target.with_suffix('.json').write_text(json.dumps(report,indent=2))
    with target.with_suffix('.csv').open('w',newline='') as stream:
        writer=csv.writer(stream);writer.writerow(['frame','stamp_sec','condition','voxelmap_error_m','clean_fusion_error_m','perturbed_fusion_error_m','perturbed_visual_constraint'])
        for i in range(174):writer.writerow([i,times[i],labels[i],errors['VoxelMap'][i],errors['clean'][i],errors['perturbed'][i],int(i in accepted)])
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(3,1,figsize=(11.5,8.8),layout='constrained',gridspec_kw={'height_ratios':[2.1,2.1,1]})
    elapsed=times-times[0];colors={'VoxelMap':'#009E73','clean':'#0072B2','perturbed':'#D55E00'}
    names={'VoxelMap':'Raw VoxelMap','clean':'Repaired fusion: original images','perturbed':'Repaired fusion: perturbed images'}
    for name in trajectories:
        axes[0].plot(aligned[name][:,0]-truth[0,0],aligned[name][:,1]-truth[0,1],label=names[name],color=colors[name],lw=1.2)
        axes[1].plot(elapsed,errors[name]*100,label=names[name],color=colors[name],lw=1.2)
    axes[0].plot(truth[:,0]-truth[0,0],truth[:,1]-truth[0,1],color='#555555',ls='--',label='Reference',lw=1.)
    axes[0].set(xlabel='X offset (m)',ylabel='Y offset (m)',title='Same 174 inputs / 79.4 m; one fixed-scale rigid alignment per trajectory')
    axes[0].set_aspect('equal',adjustable='datalim');axes[0].legend(fontsize=8,ncol=2)
    axes[1].set(ylabel='Position error (cm)',xlabel='Sensor time (s)',title='Full-run alignment retained inside every intervention interval')
    for block in schedule:
        begin=elapsed[block['start']];end=elapsed[block['end']] if block['end']<174 else elapsed[-1]
        for ax in axes[1:]:ax.axvspan(begin,end,color='#E69F00',alpha=.12)
        axes[2].text((begin+end)/2,1.35,block['kind'].replace('_','\n'),ha='center',va='bottom',fontsize=7)
    for key,y,color,label in [('lio_guarded',1,'#009E73','LiDAR'),('vision_accepted',0,'#0072B2','Vision')]:
        indices=sorted(observed['perturbed'][key]);axes[2].scatter(elapsed[indices],[y]*len(indices),s=10,color=color)
    axes[2].set(yticks=[0,1],yticklabels=['Visual constraints','LiDAR constraints'],ylim=(-.5,2.3),xlabel='Sensor time (s)',title='Guarded constraints sent to EKF during the perturbed run')
    for ax in axes:ax.grid(alpha=.15)
    fig.savefig(target.with_suffix('.png'),dpi=165);plt.close(fig)
    print(json.dumps(dict(methods=scores,intervals=intervals,recovery=recovery),indent=2))


if __name__=='__main__':main()
