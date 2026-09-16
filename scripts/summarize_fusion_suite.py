#!/usr/bin/env python3
"""Compare components on identical reference-associated times; retain failures."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from assess_bag_run import _rigid_alignment, _statistics, quantiles


def read(path):
    return json.loads(path.read_text()) if path.exists() else {}


def source_on_times(verification, source, times, truth, travel):
    records=[m for m in verification.get('raw_poses',[]) if m['source']==source]
    epochs=sorted({m['frame'] for m in records})
    if len(epochs)!=1:
        return dict(full_scored_interval=False,epochs=epochs,reason='Source has zero or multiple independent epochs; no stitched full-trajectory ATE')
    selected=[]
    for stamp in times:
        row=min(records,key=lambda r:abs(r['stamp_sec']-stamp))
        if abs(row['stamp_sec']-stamp)>1e-5:
            return dict(full_scored_interval=False,epochs=epochs,reason='Source does not cover all scored fused timestamps')
        selected.append(row['pose'][:3])
    positions=np.asarray(selected)
    rotation,translation,scale=_rigid_alignment(positions,truth)
    errors=np.linalg.norm(positions@rotation.T+translation-truth,axis=1)
    stats=_statistics(errors)
    return dict(full_scored_interval=True,epochs=epochs,samples=len(times),ate_m=stats['rmse'],
                ate_over_travel_percent=100*stats['rmse']/travel,max_error_m=stats['max'],
                p95_error_m=stats['p95'],scale_diagnostic_not_applied=float(scale),errors_m=errors.tolist())


def main():
    parser=argparse.ArgumentParser();parser.add_argument('output',type=Path)
    args=parser.parse_args();root=args.output.resolve()
    if not root.is_relative_to(ROOT/'results'):raise ValueError('Results must belong to this workspace')
    definition=read(root/'suite_definition.json')['definition'];cases=[]
    labels={'repair_clean':'Terrain A','repair_perturbed':'Terrain A + image faults',
            'short_low_contrast':'Short / low contrast','next_day_route':'Next day / flat',
            'heldout_terrain':'Disjoint terrain'}
    for case in definition['cases']:
        output=root/case['name'];assessment=read(output/'assessment.json')
        verification=read(output/'verification.json')
        mapped=read(output/'map_statistics.json')
        row=dict(name=case['name'],label=labels.get(case['name'],case['name']),output=str(output),
                 passed=(output/'benchmark_complete.json').exists(),frames=case['frames'],
                 map_statistics=mapped,checks=verification.get('checks'),
                 error=read(output/'benchmark_error.json'),methods={})
        if assessment:
            samples=np.loadtxt(output/'trajectory_accuracy_samples.csv',delimiter=',',skiprows=1)
            times,truth=samples[:,0],samples[:,4:7]
            accuracy=assessment['accuracy'];travel=accuracy['reference_travel_m'];stat=accuracy['fixed_scale_se3_position_error_m']
            row.update(travel_m=travel,duration_sec=assessment['sensor_duration_sec'],reference_sampling=accuracy['reference_sampling'],
                available_output_poses=accuracy['available_output_poses'],scored_samples=len(times),
                scored_interval_sec=[float(times[0]),float(times[-1])],
                full_input_span=accuracy['coverage']['output_span_fraction']>.999,
                input_travel_m=accuracy['coverage']['input_reference_path_m'])
            row['methods']['fused']=dict(full_scored_interval=True,samples=len(times),ate_m=stat['rmse'],
                ate_over_travel_percent=accuracy['ate_over_reference_travel_percent'],
                max_error_m=stat['max'],p95_error_m=stat['p95'],errors_m=samples[:,-1].tolist())
            for method,source in [('lidar','lio_raw'),('visual','learned_raw')]:
                row['methods'][method]=source_on_times(verification,source,times,truth,travel)
            with (output/'replay_frames.csv').open() as stream:published=list(csv.DictReader(stream))
            final_latency=[];first_latency=[]
            for sample in published:
                stamp,wall=float(sample['sensor_stamp_sec']),float(sample['publish_start_monotonic_sec'])
                delays=[m['received_monotonic_sec']-wall for m in verification.get('measurements',[])
                        if m['source']=='fused' and abs(m['stamp_sec']-stamp)<1e-5 and m['received_monotonic_sec']>=wall]
                if delays:first_latency.append(min(delays));final_latency.append(max(delays))
            row['timing']=dict(lidar=assessment.get('lidar_timing',{}).get('after_first_frame',{}).get('processing_sec'),
                visual=assessment.get('timing',{}).get('tracked_frames',{}).get('processing_sec'),
                first_qualified_pose_receipt_sec=quantiles(first_latency),
                last_same_stamp_pose_receipt_sec=quantiles(final_latency),
                qualification='Pose latency starts when this player begins publishing an input group, not at camera exposure. Processing medians exclude initialization as named.')
            row['peak_process_tree_rss_mib']=assessment['memory']['peak_process_tree_rss_mib']
            row['visual_checks']=assessment.get('visual_checks')
            row['times_sec']=times.tolist()
        cases.append(row)
    result=dict(cases=cases,all_passed=all(c['passed'] for c in cases),
                qualification='LiDAR and visual component ATE use exactly the fused reference-associated timestamps, fixed scale, independent SE(3) alignment. Reference is used only after playback. Multiple visual epochs are not joined for scoring.',
                suite_definition=read(root/'suite_definition.json'))
    (root/'comparison.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    normal=[next(c for c in cases if c['name']==n) for n in ['repair_clean','short_low_contrast','next_day_route','heldout_terrain']]
    fig,ax=plt.subplots(figsize=(10,4.8),layout='constrained')
    positions=np.arange(len(normal));colors={'lidar':'#83939e','visual':'#329678','fused':'#266dd3'}
    for offset,method in zip([-.25,0,.25],['lidar','visual','fused']):
        values=[c['methods'].get(method,{}).get('ate_over_travel_percent',np.nan) if c['passed'] else np.nan for c in normal]
        bars=ax.bar(positions+offset,values,.23,label={'lidar':'LiDAR component','visual':'Visual component','fused':'Fusion'}[method],color=colors[method])
        for bar,value in zip(bars,values):
            if np.isfinite(value):ax.annotate(f'{value:.3f}%',(bar.get_x()+bar.get_width()/2,value),xytext=(0,3),textcoords='offset points',ha='center',fontsize=8)
    ax.set_xticks(positions,[c['label']+f"\n{c.get('travel_m',0):.2f} m" for c in normal])
    ax.set_ylabel('Translational ATE / reference travel (%)')
    ax.set_title('Same timestamps, scale fixed at 1; full scored intervals only')
    ax.legend(ncol=3,loc='upper right');ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    ax.margins(y=.25);fig.savefig(root/'trajectory_comparison.png',dpi=160);plt.close(fig)
    clean=next(c for c in cases if c['name']=='repair_clean');fault=next(c for c in cases if c['name']=='repair_perturbed')
    if clean['methods'] and fault['methods']:
        fig,axes=plt.subplots(2,1,figsize=(11,6),sharex=True,layout='constrained',gridspec_kw={'height_ratios':[2,1]})
        for case,color in [(clean,'#266dd3'),(fault,'#e0792c')]:
            t=np.asarray(case['times_sec']);axes[0].plot(t-t[0],case['methods']['fused']['errors_m'],label=case['label'],color=color)
        verification=read(Path(fault['output'])/'verification.json')
        with (Path(fault['output'])/'replay_frames.csv').open() as stream:published=list(csv.DictReader(stream))
        stamps=np.array([float(r['sensor_stamp_sec']) for r in published]);base=stamps[0]
        constraints=[m['stamp_sec']-base for m in verification['measurements'] if m['source']=='vision_accepted']
        axes[1].scatter(constraints,np.ones(len(constraints)),s=12,color='#329678',label='Visual body motion delivered')
        report=[]
        blocks=read(Path(fault['output'])/'replay.json')['perturbations']
        for block in blocks:
            indices=range(block['start'],min(block['end'],len(stamps)))
            selected=stamps[list(indices)]
            accepted=sum(any(abs(t-(stamp-base))<1e-5 for t in constraints) for stamp in selected)
            fused=sum(any(abs(t-stamp)<1e-5 for t in fault['times_sec']) for stamp in selected)
            record=dict(**block,input_frames=len(selected),visual_motion_frames=accepted,qualified_pose_frames=fused)
            if block['kind']!='grayscale_control' and block['end']<len(stamps):
                next_fault=min([b['start'] for b in blocks if b['start']>=block['end'] and b['kind']!='grayscale_control']+[len(stamps)])
                recovered=[i for i in range(block['end'],next_fault) if any(abs(t-(stamps[i]-base))<1e-5 for t in constraints)]
                record['first_visual_motion_frame_after_fault']=recovered[0] if recovered else None
                record['recovery_sensor_seconds']=float(stamps[recovered[0]]-stamps[block['end']]) if recovered else None
            report.append(record)
            begin=stamps[block['start']]-base;end=(stamps[block['end']]-base if block['end']<len(stamps) else stamps[-1]-base)
            for ax in axes:ax.axvspan(begin,end,alpha=.12,color='#b45d23')
            axes[1].text((begin+end)/2,.5,block['kind'].replace('_','\n'),ha='center',va='center',fontsize=8)
        axes[0].set_ylabel('Fused position error (m)');axes[0].legend();axes[0].grid(alpha=.2)
        axes[1].set(xlabel='Sensor time from start (s)',ylim=(0,1.35),yticks=[]);axes[1].legend(loc='lower right',fontsize=8)
        fig.savefig(root/'image_fault_comparison.png',dpi=160);plt.close(fig)
        result['image_fault_blocks']=report
        (root/'comparison.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    print(json.dumps([{k:c.get(k) for k in ['name','passed','travel_m','available_output_poses','scored_samples','peak_process_tree_rss_mib']} for c in cases],indent=2),flush=True)


if __name__=='__main__':main()
