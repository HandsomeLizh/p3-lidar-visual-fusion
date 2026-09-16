#!/usr/bin/env python3
"""Run one fixed trajectory, save maps, stop only this workspace's process group."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from control import alive
from source_manifest import check as check_build

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--perturbations', default='')
    parser.add_argument('--frames', type=int, default=174)
    parser.add_argument('--bag', default='/home/yanfa/Env_X/InterFace/bags/20260824_235238')
    parser.add_argument('--frame-index', default=str(ROOT/'test_data/20260824_235238/first_10min_index.json'))
    parser.add_argument('--reference', help='Offline reference CSV already rebased to this replay clock')
    parser.add_argument('--profile', default=str(ROOT/'config/long_bag_voxelmap.yaml'))
    parser.add_argument('--reference-max-gap', type=float, default=.25,
                        help='Maximum observed reference interpolation gap, recorded in the report')
    parser.add_argument('--rviz', action='store_true')
    args = parser.parse_args()
    if not math.isfinite(args.reference_max_gap) or not 0 < args.reference_max_gap <= 1.:
        raise ValueError('Reference interpolation limit must be in (0, 1] seconds')
    output = Path(args.output).resolve()
    if not output.is_relative_to(ROOT/'results') or output.exists():
        raise ValueError('Choose a new output directory inside this workspace/results')
    index = json.loads(Path(args.frame_index).read_text())
    if Path(index['bag']).resolve() != Path(args.bag).resolve():
        raise ValueError('Frame index belongs to another bag')
    if not 1 <= args.frames <= len(index['frames']):
        raise ValueError('Requested frames must fit the validated frame index')
    duration = (index['frames'][args.frames-1]['record_ns']-index['frames'][0]['record_ns'])/1e9
    command = [sys.executable, str(ROOT/'scripts/control.py'), 'start',
        '--profile', str(Path(args.profile).resolve()),
        '--bag', args.bag, '--frame-index', args.frame_index,
        '--frames', str(args.frames), '--rate', '1', '--domain', '57',
        '--verify', '--verify-seconds', str(duration+70), '--output', str(output)]
    if args.perturbations: command += ['--image-perturbations', args.perturbations]
    if args.rviz: command += ['--rviz']
    env = dict(os.environ, ROS_DOMAIN_ID='57', ROS_LOCALHOST_ONLY='1')
    started = False
    try:
        manifest = check_build()
        subprocess.run(command, cwd=ROOT, env=env, check=True, timeout=100)
        started = True
        state = json.loads((ROOT/'run_state.json').read_text())
        if Path(state['output']) != output: raise RuntimeError('Controller ownership changed')
        snapshot = output/'source_snapshot'; snapshot.mkdir()
        (output/'runtime_manifest.json').write_text(json.dumps(manifest, indent=2))
        for rel in ['src/t3_lidar_visual_fusion', 'src/t3_voxelmap', 'scripts', 'config']:
            shutil.copytree(ROOT/rel, snapshot/rel, ignore=shutil.ignore_patterns('__pycache__'))
        deadline = time.monotonic()+duration+150
        while alive(state['processes']['player']):
            if not alive(state['processes']['pipeline']): raise RuntimeError('Pipeline exited during playback')
            if time.monotonic() > deadline: raise TimeoutError('Playback deadline exceeded')
            time.sleep(2.)
        if not (output/'replay.json').exists(): raise RuntimeError('Player exited without replay completion')
        time.sleep(4.)
        drain_deadline=time.monotonic()+40.
        while True:
            saved = subprocess.run(['bash', str(ROOT/'save_map.sh')], cwd=ROOT, env=env,
                                   text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=50)
            with (output/'save_map_result.txt').open('a') as stream:stream.write(saved.stdout)
            if saved.returncode or 'success=True' not in saved.stdout: raise RuntimeError('Map checkpoint failed: '+saved.stdout[-1000:])
            map_status=json.loads((output/'map_statistics.json').read_text())
            if map_status['pending']==0:break
            if time.monotonic()>drain_deadline:raise TimeoutError('Map queue did not drain after EOF')
            time.sleep(.2)
        replay=json.loads((output/'replay.json').read_text())
        if map_status['mapped_scans']!=replay['frames'] or map_status['dropped_scans']:
            raise RuntimeError('Incomplete mapping scan accounting: '+str(map_status))
        if args.rviz:
            screenshot=subprocess.run([sys.executable,str(ROOT/'scripts/capture_rviz.py'),
                '--output',str(output)],cwd=ROOT,env=env,text=True,
                stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=15)
            (output/'screenshot.log').write_text(screenshot.stdout)
        subprocess.run([sys.executable, str(ROOT/'scripts/control.py'), 'stop'], cwd=ROOT, env=env, check=True, timeout=95)
        started = False
        verification = output/'verification.json'
        if not verification.exists(): raise RuntimeError('Interface verifier did not produce a report; inspect verifier.log')
        checks = json.loads(verification.read_text())['checks']
        if not all(checks.values()): raise RuntimeError('Interface verification failed: '+str(checks))
        if args.frames>=10:
            assessment = [sys.executable, str(ROOT/'scripts/assess_bag_run.py'), str(output),
                          '--reference-max-gap', str(args.reference_max_gap)]
            if args.reference: assessment += ['--reference', args.reference]
            subprocess.run(assessment, cwd=ROOT, env=env, check=True, timeout=90)
        else:
            (output/'assessment_skipped.json').write_text(json.dumps(dict(reason="Smoke run has fewer than ten poses; no accuracy claim")))
        (output/'benchmark_complete.json').write_text(json.dumps(dict(completed=True, wall_finished=time.time()), indent=2))
        print('BENCHMARK COMPLETE', output, flush=True)
    except Exception as error:
        if output.exists(): (output/'benchmark_error.json').write_text(json.dumps(dict(error=str(error)), indent=2))
        raise
    finally:
        if started:
            state = json.loads((ROOT/'run_state.json').read_text())
            if Path(state['output']) == output:
                subprocess.run([sys.executable, str(ROOT/'scripts/control.py'), 'stop'], cwd=ROOT, env=env, timeout=95)


if __name__ == '__main__': main()
