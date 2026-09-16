#!/usr/bin/env python3
"""Sequential, reproducible bag tests; each run owns its processes and artifacts."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from control import alive
from source_manifest import check as check_build

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--definition', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--profile', default=str(ROOT/'config/long_bag_voxelmap.yaml'))
    parser.add_argument('--await-output', help='Adopt one already running benchmark in this suite')
    parser.add_argument('--rviz', action='store_true')
    args=parser.parse_args()
    output=Path(args.output).resolve();profile=Path(args.profile).resolve()
    if not output.is_relative_to(ROOT/'results'): raise ValueError('Output must be in this workspace/results')
    definition=json.loads(Path(args.definition).read_text())
    cases=definition['cases']
    names=[case['name'] for case in cases]
    if len(names)!=len(set(names)) or any(not n.replace('_','').isalnum() for n in names):
        raise ValueError('Case names must be unique letters, digits and underscores')
    output.mkdir(exist_ok=True)
    if (output/'suite_complete.json').exists(): raise ValueError('Completed suite is immutable; choose a new output')
    frozen=dict(profile=str(profile),profile_sha256=hashlib.sha256(profile.read_bytes()).hexdigest(),
                source_sha256=check_build()['source_sha256'],definition=definition)
    (output/'suite_definition.json').write_text(json.dumps(frozen,indent=2))
    adopted=Path(args.await_output).resolve() if args.await_output else None
    if adopted is not None and adopted.parent!=output: raise ValueError('Adopted benchmark must belong to this suite')
    results=[]
    for case in cases:
        run=output/case['name']
        if check_build()['source_sha256']!=frozen['source_sha256'] or hashlib.sha256(profile.read_bytes()).hexdigest()!=frozen['profile_sha256']:
            raise RuntimeError('Code/profile changed during the suite; do not mix implementations')
        if run==adopted:
            deadline=time.monotonic()+1200.
            while True:
                state_path=ROOT/'run_state.json'
                state=json.loads(state_path.read_text()) if state_path.exists() else {}
                active=(state.get('output')==str(run) and any(alive(p) for p in state.get('processes',{}).values()))
                finished=(run/'benchmark_complete.json').exists() or (run/'benchmark_error.json').exists()
                if finished and not active: break
                if time.monotonic()>deadline: raise TimeoutError('Existing benchmark did not finish')
                time.sleep(2.)
        else:
            if run.exists(): raise ValueError('Never overwrite a benchmark: '+str(run))
            command=[sys.executable,str(ROOT/'scripts/run_fusion_repair_benchmark.py'),
                     '--profile',str(profile),'--bag',case['bag'],'--frames',str(case['frames']),
                     '--frame-index',case['frame_index'],'--output',str(run),
                     '--reference-max-gap',str(case.get('reference_max_gap',.25))]
            if case.get('reference'): command+=['--reference',case['reference']]
            if case.get('perturbations'): command+=['--perturbations',case['perturbations']]
            if args.rviz: command+=['--rviz']
            print('START',case['name'],flush=True)
            with (output/(case['name']+'.log')).open('w') as log:
                subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=False)
        passed=(run/'benchmark_complete.json').exists()
        error=json.loads((run/'benchmark_error.json').read_text()) if (run/'benchmark_error.json').exists() else None
        results.append(dict(name=case['name'],output=str(run),passed=passed,error=error))
        (output/'suite_progress.json').write_text(json.dumps(results,indent=2))
        print('FINISHED',case['name'],'passed='+str(passed),flush=True)
    report=dict(completed=True,all_passed=all(case['passed'] for case in results),cases=results,
                qualification='Same frozen code/profile, 1x replay, original inputs retained. Short bag uses explicitly reported 0.5 s reference interpolation; other cases use 0.25 s.')
    (output/'suite_complete.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)


if __name__=='__main__': main()
