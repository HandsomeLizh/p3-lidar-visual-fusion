from pathlib import Path
import json,numpy as np,sys
r=Path('/home/yanfa/P3/lidar_visual_fusion')
o=r/'results/short_xfeat_20260915_211029'
sys.path.insert(0,'/home/yanfa/P3/roma_t3_algorithm_bundle_20260825/workspace/scripts')
from evaluate_trajectory_accuracy import _rigid_alignment,_statistics,_path_length
d=json.loads((o/'verification.json').read_text())
ref=np.loadtxt(r/'test_data/short_20260915_190151/reference.csv',delimiter=',',skiprows=1)
print('CHECKS',d['checks'],d['counts'])
for source in ['lio_raw','learned_raw','fused']:
 a=np.array(d['poses'] if source=='fused' else [[p['stamp_sec'],*p['pose']] for p in d['raw_poses'] if p['source']==source])
 if len(a)<3:continue
 use=(a[:,0]>=ref[0,0])&(a[:,0]<=ref[-1,0]);a=a[use]
 xyz=np.column_stack([np.interp(a[:,0],ref[:,0],ref[:,i]) for i in [1,2,3]])
 R,t,scale=_rigid_alignment(a[:,1:4],xyz)
 e=np.linalg.norm(a[:,1:4]@R.T+t-xyz,axis=1)
 print(source,'n',len(a),'path',_path_length(a[:,1:4]),'refpath',_path_length(xyz),'error',_statistics(e))
 print('FIRST',a[:8,:4].round(4).tolist(),'LAST',a[-8:,:4].round(4).tolist())
