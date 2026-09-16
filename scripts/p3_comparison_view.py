#!/usr/bin/env python3
"""Own saved P3 display; never touch the original demo sessions."""
import argparse
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
P3=ROOT.parent/'roma_t3_algorithm_bundle_20260825'
sys.path.insert(0,str(P3/'integration_demo'))
import democtl

parser=argparse.ArgumentParser()
parser.add_argument('action',nargs='?',choices=['start','stop'],default='start')
parser.add_argument('--output',type=Path,default=ROOT/'results/p3_roma_long_80m_20260915/verified_view')
args=parser.parse_args()
out=args.output.resolve()
if not out.is_relative_to(ROOT/'results') or not out.is_dir():
    raise ValueError('Choose an existing saved view inside this workspace/results')
democtl.STATE=out/'view_state.json'
state=democtl.read_state()
if args.action=='stop':
    for name in ('rviz','visuals','preview'):democtl.stop_service(state,name)
else:
    if any(democtl.alive(x) for x in state['services'].values()):
        raise RuntimeError('This saved view is already open')
    for name in ['trajectory_map.tum','global_grid_map.npz','lidar_map.pcd','pose_processing_time.csv']:
        if not (out/name).is_file():raise ValueError('Saved view is incomplete: '+name)
    state=dict(output=str(out),domain=60,replay=False,services={})
    democtl.save_state(state)
    env=democtl.environment(out,domain=60)
    env['ROS_LOCALHOST_ONLY']='1'
    for key,folder in [('XDG_CACHE_HOME','cache'),('CUDA_CACHE_PATH','cache/cuda'),('TMPDIR','tmp'),('MPLCONFIGDIR','cache/matplotlib')]:
        env[key]=str(out/folder);Path(env[key]).mkdir(parents=True,exist_ok=True)
    democtl.start_service(state,'preview',['/usr/bin/python3',str(P3/'integration_demo/view_saved.py'),'publish','--output',str(out)],env)
    democtl.start_rviz(state,env)
    print('Completed saved map view, isolated ROS domain 60:',out)
