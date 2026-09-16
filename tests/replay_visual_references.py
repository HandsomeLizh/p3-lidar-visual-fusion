"""Recorded stereo pairs with real learned features; compare bounded reference recovery."""
import argparse
import importlib.util
import json
from pathlib import Path
import sqlite3
import time
import cv2
import numpy as np
import yaml
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from t3_lidar_visual_fusion.learned_matching import LearnedMatcher
from t3_lidar_visual_fusion.learned_tracker import LearnedStereoTracker
from t3_lidar_visual_fusion.image_quality import assess_image


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ['workspace','baseline','bag','profile','output']:parser.add_argument('--'+key,type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    spec=importlib.util.spec_from_file_location('t3_lidar_visual_fusion.baseline_tracker',args.baseline)
    old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
    profile=yaml.safe_load(args.profile.read_text());cv2.setNumThreads(1)
    backend=LearnedMatcher(args.workspace,profile['learned_visual'])
    bridge=CvBridge();frames={}
    meta=yaml.safe_load((args.bag/'metadata.yaml').read_text())['rosbag2_bagfile_information']
    for part in meta['relative_file_paths']:
        with sqlite3.connect('file:'+str(args.bag/part)+'?mode=ro',uri=True) as db:
            for topic,data in db.execute("SELECT t.name,m.data FROM messages m JOIN topics t ON t.id=m.topic_id WHERE t.name IN ('/fusion/left','/fusion/right') ORDER BY m.timestamp,m.id"):
                msg=deserialize_message(data,Image);stamp=msg.header.stamp.sec+msg.header.stamp.nanosec*1e-9
                frames.setdefault(stamp,{})[topic]=bridge.imgmsg_to_cv2(msg,desired_encoding='mono8')
    pairs=[(stamp,data['/fusion/left'],data['/fusion/right']) for stamp,data in sorted(frames.items()) if len(data)==2]
    if len(pairs)<12:raise ValueError('Need at least 12 synchronized recorded pairs')
    warm=backend.extract(pairs[0][1]);backend.match(warm,warm);del warm
    output={}
    for noise in ['clean','blackout_4_frames']:
        for label,klass in [('baseline',old.LearnedStereoTracker),('fixed',LearnedStereoTracker)]:
            tracker=klass(profile,backend);rows=[]
            for index,(stamp,left,right) in enumerate(pairs):
                if noise!='clean' and 4<=index<8:left=np.zeros_like(left);right=np.zeros_like(right)
                start=time.perf_counter()
                try:
                    quality=[assess_image(x,texture_required=False) for x in (left,right)]
                    if not all(q.valid for q in quality):raise ValueError(next(q.reason for q in quality if not q.valid))
                    result=tracker.process(stamp,left,right)
                    row=dict(index=index,stamp=stamp,epoch=result.epoch,valid=not result.anchor,
                        position=result.base_pose[:3,3].tolist(),metrics=result.metrics)
                except ValueError as exc:
                    if label=='baseline':tracker.reset()
                    else:tracker.reject(stamp)
                    row=dict(index=index,stamp=stamp,epoch=tracker.epoch,valid=False,reason=str(exc))
                row['processing_sec']=time.perf_counter()-start;rows.append(row)
            output[label+'_'+noise]=dict(pairs=len(rows),tracked=sum(r['valid'] for r in rows),
                epochs=sorted(set(r['epoch'] for r in rows)),p50_sec=float(np.median([r['processing_sec'] for r in rows])),rows=rows)
            print(json.dumps({label+'_'+noise:{k:v for k,v in output[label+'_'+noise].items() if k!='rows'}}),flush=True)
    output['scope']='Actual recorded stereo and XFeat/LighterGlue; optional four-frame blackout; no ground truth, ROS publication, or vehicle commands.'
    (args.output/'comparison.json').write_text(json.dumps(output,indent=2)+'\n')


if __name__=='__main__':main()
