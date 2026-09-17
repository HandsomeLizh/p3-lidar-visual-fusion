"""Compute independent visual motion from an existing bag; no ROS publishers."""
import argparse,json,sqlite3,time
from pathlib import Path
import cv2
import numpy as np
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import Image
import yaml
from t3_lidar_visual_fusion.learned_matching import LearnedMatcher
from t3_lidar_visual_fusion.learned_tracker import LearnedStereoTracker
from t3_lidar_visual_fusion.image_quality import assess_image


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--workspace',type=Path,required=True)
    parser.add_argument('--run',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--max-frames',type=int,default=0)
    a=parser.parse_args();cfg=yaml.safe_load((a.run/'profile.yaml').read_text())
    cv2.setNumThreads(1)
    tracker=LearnedStereoTracker(cfg,LearnedMatcher(a.workspace,cfg['learned_visual']))
    dbpath=next((a.run/'live_input_bag').glob('*.db3'));db=sqlite3.connect(f'file:{dbpath}?mode=ro',uri=True)
    topics=dict(db.execute('select id,name from topics'));pending={};records=[]
    for topic,data in db.execute('select topic_id,data from messages order by timestamp'):
        name=topics[topic]
        if name not in ['/fusion/left','/fusion/right']:continue
        message=deserialize_message(data,Image);stamp=message.header.stamp.sec+message.header.stamp.nanosec/1e9
        if message.encoding not in ['mono8','8UC1']:raise RuntimeError('Expected recorded normalized mono image')
        pixels=np.ndarray((message.height,message.width),dtype=np.uint8,buffer=message.data,strides=(message.step,1)).copy()
        pair=pending.setdefault(stamp,{})
        pair[name]=pixels
        if len(pair)<2:continue
        try:
            images=[pair[n] for n in ['/fusion/left','/fusion/right']]
            health=[assess_image(image,texture_required=False) for image in images]
            if not all(q.valid for q in health):raise ValueError('Image quality invalid')
            result=tracker.process(stamp,*images)
            q=max(.1,min(min(v.score for v in health),result.metrics.get('pnp_ratio',1.),result.metrics.get('stereo_motion_ratio',1.)))
            variance=np.full(6,1e6) if result.anchor else np.asarray(cfg['vision_pose_variance'])*min(25.,1/q**2)
            records.append(dict(stamp=stamp,epoch='learned_epoch_'+str(result.epoch),pose=result.base_pose.tolist(),
                                covariance=np.diag(variance).ravel().tolist(),valid=not result.anchor,
                                quality=q,metrics=result.metrics))
        except Exception as error:
            records.append(dict(stamp=stamp,valid=False,reason=str(error)))
        pending.pop(stamp)
        if len(records)%20==0:print('visual frames',len(records),flush=True)
        if a.max_frames and len(records)>=a.max_frames:break
    a.out.write_text(json.dumps(records,indent=2));db.close()
    for i in range(max(1,len(records)-12),len(records)):
        if records[i].get('valid') and records[i-1].get('valid'):
            b,c=(np.asarray(records[j]['pose']) for j in [i-1,i])
            print('VISUAL_STEP',i+1,records[i]['stamp'],float(np.linalg.norm(c[:3,3]-b[:3,3])))


if __name__=='__main__':main()
