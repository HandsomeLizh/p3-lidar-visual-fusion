#!/usr/bin/env python3
"""Physical coordinate checks that ATE's free rigid alignment cannot catch."""
from pathlib import Path
import json
import sys
import numpy as np
import yaml
from scipy.spatial.transform import Rotation

ROOT=Path('/home/yanfa/P3/lidar_visual_fusion')
sys.path.insert(0,str(ROOT/'scripts'))
from indexed_bag import groups

profile=yaml.safe_load((ROOT/'config/long_bag_voxelmap.yaml').read_text())
transform=np.array(profile['base_from_lidar'])
old=np.array(yaml.safe_load((ROOT/'archive/simulation_lidar_basis_before_20260916/long_bag_voxelmap.yaml').read_text())['base_from_lidar'])
assert np.linalg.det(transform[:3,:3])>.999
assert transform[0,0]>.99 and transform[1,1]>.99 and transform[2,2]>.99
ray=np.array(profile['base_from_camera_left'])[:3,2]
assert ray[0]>0 and ray[2]<0, 'Camera center ray must point forward and down in ROS body axes'
_,batch=next(groups('/home/yanfa/Env_X/InterFace/bags/20260824_235238',ROOT/'test_data/20260824_235238/first_10min_index.json',limit=1,lidar_only=True))
msg=batch['/Car/T5/OS1/points']
assert not msg.is_bigendian and [(f.name,f.offset) for f in msg.fields]==[('x',0),('y',4),('z',8)]
points=np.ndarray((msg.height,msg.width,3),dtype='<f4',buffer=msg.data,strides=(msg.row_step,msg.point_step,4)).reshape(-1,3)
points=points[np.isfinite(points).all(axis=1)]
ground={}
for name,t in [('inherited',old),('corrected',transform)]:
    body=points@t[:3,:3].T+t[:3,3]
    z=body[(body[:,0]>2)&(body[:,0]<8)&(abs(body[:,1])<3),2]
    ground[name]=float(np.median(z))
assert ground['inherited']>1.
assert -.5<ground['corrected']<-.05, 'Nearby ground must be below the vehicle body origin'
change=transform@np.linalg.inv(old)
checks=[]
for name in ['long_fusion_repaired_clean_verified_20260916','multi_trajectory_20260916/short_low_contrast','multi_trajectory_20260916/next_day_route']:
    raw=json.loads((ROOT/'results'/name/'verification.json').read_text())['raw_poses']
    lidar={round(m['stamp_sec']*1e9):m for m in raw if m['source']=='lio_raw'}
    visual=[m for m in raw if m['source']=='learned_raw' and m['frame']=='learned_epoch_1']
    sample=visual[-1];value=lidar[round(sample['stamp_sec']*1e9)]['pose']
    pose=np.eye(4);pose[:3,:3]=Rotation.from_quat(value[3:]).as_matrix();pose[:3,3]=value[:3]
    corrected=change@pose@np.linalg.inv(change)
    visual_yaw=Rotation.from_quat(sample['pose'][3:]).as_euler('xyz')[2]
    lidar_yaw=Rotation.from_matrix(corrected[:3,:3]).as_euler('xyz')[2]
    assert corrected[1,3]*sample['pose'][1]>0
    assert lidar_yaw*visual_yaw>0
    checks.append(dict(run=name,corrected_lidar_y=float(corrected[1,3]),visual_y=sample['pose'][1],
        corrected_lidar_yaw=float(lidar_yaw),visual_yaw=float(visual_yaw)))
result=dict(passed=True,ground_median_base_z_m=ground,turn_direction_checks=checks,
    qualification='No fitted trajectory alignment, scale or truth. Published ROS input axes and physical ground-side checks; deterministic change of coordinate basis for recorded LiDAR motion.')
(ROOT/'results/simulation_lidar_basis_regression_20260916.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
