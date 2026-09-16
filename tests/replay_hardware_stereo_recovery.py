"""Actual synchronized parked rover images: CUDA tracking and blackout recovery.

Does not open cameras, publish ROS messages, or use motion ground truth.
"""
import argparse,importlib.util,json,sys,time
from pathlib import Path
import cv2,numpy as np,yaml
from t3_lidar_visual_fusion.learned_matching import LearnedMatcher
from t3_lidar_visual_fusion.learned_tracker import LearnedStereoTracker
from t3_lidar_visual_fusion.image_quality import assess_image
from t3_lidar_visual_fusion.stereo_mapping import sparse_map_points

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    source=ROOT/'results/hardware104_recovery_20260916'
    spec=importlib.util.spec_from_file_location('t3_lidar_visual_fusion.baseline_tracker',source/'baseline/learned_tracker.py')
    old=importlib.util.module_from_spec(spec);sys.modules[spec.name]=old;spec.loader.exec_module(old)
    cfg=yaml.safe_load((ROOT/'config/hardware104.yaml').read_text());cv2.setNumThreads(1)
    samples=source/'camera_sample';times=json.loads((samples/'report.json').read_text())['stamps_ns']
    pairs=[(t*1e-9,cv2.imread(str(samples/f'{i:02d}_0.png'),0),cv2.imread(str(samples/f'{i:02d}_1.png'),0)) for i,t in enumerate(times)]
    backend=LearnedMatcher(ROOT,cfg['learned_visual'])
    features=backend.extract(pairs[0][1]);backend.match(features,features)
    report={}
    for noise in ('clean','blackout','overexposure'):
        for label,klass in [('baseline',old.LearnedStereoTracker),('candidate',LearnedStereoTracker)]:
            tracker=klass(cfg,backend);rows=[]
            for i,(t,a,b) in enumerate(pairs):
                if noise!='clean' and 4<=i<8:
                    a=np.full_like(a,0 if noise=='blackout' else 255);b=np.full_like(b,0 if noise=='blackout' else 255)
                begin=time.perf_counter();row=dict(index=i,stamp=t,valid=False,map_points=0)
                try:
                    quality=[assess_image(image,texture_required=False) for image in (a,b)]
                    if not all(q.valid for q in quality):raise ValueError(next(q.reason for q in quality if not q.valid))
                    cv2.setRNGSeed(0);result=tracker.process(t,a,b)
                    row.update(epoch=result.epoch,valid=not result.anchor,position=result.base_pose[:3,3].tolist(),reason=result.metrics['reason'])
                    if not result.anchor:
                        points,_=sparse_map_points(tracker.geometry,result.map_points,cfg['stereo_mapping']);row['map_points']=len(points)
                except ValueError as exc:
                    if label=='baseline':tracker.reset()
                    else:tracker.reject(t)
                    row.update(epoch=tracker.epoch,reason=str(exc))
                row['processing_sec']=time.perf_counter()-begin;rows.append(row)
            report[label+'_'+noise]=dict(pairs=len(rows),tracked=sum(r['valid'] for r in rows),epochs=sorted({r['epoch'] for r in rows}),
                tracking_median_sec=float(np.median([r['processing_sec'] for r in rows if r['valid']])),
                admitted_points_median=float(np.median([r['map_points'] for r in rows if r['valid']])),rows=rows)
            print(json.dumps({label+'_'+noise:{k:v for k,v in report[label+'_'+noise].items() if k!='rows'}}),flush=True)
    a=report['baseline_clean'];b=report['candidate_clean']
    assert a['tracked']==b['tracked']==len(pairs)-1
    np.testing.assert_allclose([r['position'] for r in a['rows']],[r['position'] for r in b['rows']],atol=1e-5)
    for noise in ('blackout','overexposure'):
        candidate=report['candidate_'+noise];baseline=report['baseline_'+noise]
        assert candidate['epochs']==[1] and baseline['epochs']==[1,2]
        assert candidate['tracked']>baseline['tracked']
        assert all(not r['valid'] and r['map_points']==0 for r in candidate['rows'][4:8])
        assert all(r['valid'] and r['map_points']>0 for r in candidate['rows'][8:])
    report.update(passed=True,device=cfg['learned_visual']['device'],
        scope='Recorded actual parked rover stereo, real learned features, injected blackout/overexposure. No live driver, ROS output or ATE.')
    (args.output/'comparison.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
