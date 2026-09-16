"""Offline recorded vehicle stereo, CPU only; never opens drivers or CUDA.

Checks real-image feature/tracking/depth admission, not motion ATE or GPU speed.
"""
import json,time
from pathlib import Path
import cv2,numpy as np,yaml
from t3_lidar_visual_fusion.learned_matching import LearnedMatcher
from t3_lidar_visual_fusion.learned_tracker import LearnedStereoTracker
from t3_lidar_visual_fusion.stereo_mapping import sparse_map_points

ROOT=Path(__file__).resolve().parents[1]


def main():
    cfg=yaml.safe_load((ROOT/'config/hardware104.yaml').read_text())
    cfg['learned_visual']['device']='cpu';cv2.setNumThreads(1)
    out=ROOT/'results/hardware104_deployment';samples=out/'camera_sample'
    stamps=json.loads((samples/'report.json').read_text())['stamps_ns']
    print('Loading bounded CPU matcher for recorded camera samples',flush=True)
    tracker=LearnedStereoTracker(cfg,LearnedMatcher(ROOT,cfg['learned_visual']))
    records=[]
    for i,stamp in enumerate(stamps):
        result=tracker.process(stamp*1e-9,cv2.imread(str(samples/f'{i:02d}_0.png'),0),
                               cv2.imread(str(samples/f'{i:02d}_1.png'),0))
        if result.anchor:continue
        start=time.perf_counter();points,variance=sparse_map_points(tracker.geometry,result.map_points,cfg['stereo_mapping'])
        records.append(dict(points=len(points),filter_seconds=time.perf_counter()-start,
                            median_position_std_m=float(np.sqrt(np.median(variance))) if len(variance) else None))
    assert len(records)>=8 and all(0<r['points']<=cfg['stereo_mapping']['max_points'] for r in records),records
    report=dict(passed=True,tracked_pairs=len(records),median_admitted_points=float(np.median([r['points'] for r in records])),
        filter_median_sec=float(np.median([r['filter_seconds'] for r in records])),records=records,
        scope='Previously recorded parked vehicle stereo; CPU matcher, no live or CUDA test')
    (out/'stereo_mapping_recorded.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))


if __name__=='__main__':main()
