#!/usr/bin/env python3
"""Run unchanged P3 algorithms on the frozen 174-frame comparison input.

Own process identities, logs, cache and output live in lidar_visual_fusion.
Does not call democtl or touch its sessions, vehicle controls or state files.
"""
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import signal
import subprocess
import sys
import time
from control import identity, alive
from watch_resources import descendants

ROOT = Path(__file__).resolve().parents[1]
P3 = ROOT.parent / 'roma_t3_algorithm_bundle_20260825'
OUT = ROOT / 'results/xfeat_compact_long_80m_20260915'


def main():
    global OUT
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",type=Path,default=OUT)
    args=parser.parse_args()
    OUT=args.output.resolve()
    if not OUT.is_relative_to(ROOT/"results"):
        raise ValueError("Comparison outputs must stay within this workspace results directory")
    for path in (ROOT/"results").glob("*/process_state.json"):
        previous=json.loads(path.read_text())
        if int(previous.get("domain",-1))==62 and any(alive(v) for v in previous.get("processes",{}).values()):
            raise RuntimeError("An owned comparison is already active in this domain: "+str(path.parent))
    OUT.mkdir(exist_ok=False)
    state = dict(output=str(OUT), domain=62, processes={})
    snapshot = OUT / 'source_snapshot'
    snapshot.mkdir()
    import yaml
    sources = [ROOT/'src/t3_lidar_visual_fusion/t3_lidar_visual_fusion']
    for source in sources:
        shutil.copytree(source,snapshot/source.name,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    profile=yaml.safe_load((ROOT/'config/long_bag_voxelmap.yaml').read_text())
    profile.update(imu_mode="off",use_imu=False,adaptive_source_selection=False)
    (OUT/'profile.yaml').write_text(yaml.safe_dump(profile,sort_keys=False))
    for helper in ["replay_bag.py","indexed_bag.py","stereo_transport.py"]:
        shutil.copy2(ROOT/"scripts"/helper,snapshot/helper)
    manifest={str(p.relative_to(snapshot)):hashlib.sha256(p.read_bytes()).hexdigest()
              for p in snapshot.rglob('*') if p.is_file()}
    (snapshot/'manifest.json').write_text(json.dumps(manifest,indent=2))
    command = 'source ' + shlex.quote(str(ROOT/'scripts/env.sh')) + ' && env -0'
    result = subprocess.run(['bash', '-c', command], check=True, stdout=subprocess.PIPE)
    env = dict(e.split('=', 1) for e in result.stdout.decode().split('\0') if '=' in e)
    env.update(ROS_DOMAIN_ID='62', ROS_LOCALHOST_ONLY='1',
               RMW_IMPLEMENTATION='rmw_cyclonedds_cpp', PYTHONUNBUFFERED='1',
               PYTHONDONTWRITEBYTECODE='1', T3_BUNDLE_ROOT=str(P3),
               TORCH_HOME=str(P3/'models/torch'), T3_OUTPUT_DIR=str(OUT))
    env['PYTHONPATH'] = str(snapshot) + ':' + env.get('PYTHONPATH', '')
    for key, folder in [('ROS_LOG_DIR','ros_logs'), ('XDG_CACHE_HOME','cache'),
                        ('MPLCONFIGDIR','cache/matplotlib'), ('TMPDIR','tmp'),
                        ('CUDA_CACHE_PATH','cache/cuda')]:
        env[key] = str(OUT/folder)
        Path(env[key]).mkdir(parents=True, exist_ok=True)
    processes = {}
    def save():
        (OUT/'process_state.json').write_text(json.dumps(state, indent=2))
    def start(name, argv):
        with (OUT/(name+'.log')).open('w') as stream:
            process = subprocess.Popen(argv, env=env, stdout=stream,
                stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
        processes[name] = process
        state['processes'][name] = dict(pid=process.pid, identity=identity(process.pid), command=argv)
        save()
        return process
    def stop(name):
        entry = state['processes'].get(name)
        if alive(entry):
            os.kill(entry['pid'], signal.SIGINT) if name == 'mapper' else os.killpg(entry['pid'], signal.SIGINT)
            processes[name].wait(timeout=180)
    peak = 0.
    def monitor():
        nonlocal peak
        values = descendants([p.pid for p in processes.values() if p.poll() is None])
        rss = sum(values.values()); peak = max(peak, rss)
        record = dict(wall_time=time.time(), tree_rss_mib=rss, tree_rss_peak_mib=peak,
                      process_rss_mib=values)
        with (OUT/'resources.jsonl').open('a') as f: f.write(json.dumps(record)+'\n')
        (OUT/'resources_latest.json').write_text(json.dumps(record, indent=2))
    try:
        (OUT/'benchmark_environment.json').write_text(json.dumps(dict(
            source_program=str(ROOT), ros_domain_id=62, localhost_only=True,
            algorithm='XFeat + LighterGlue stereo only; normalized transport, no guard/EKF or LiDAR/IMU' ,
            rate=1.0, frames=174, input_policy='identical indexed timestamps and visual pixels; mono+resize moved before DDS transport' ,
            other_live_P3_processes_present=True, source_snapshot=str(snapshot),
            note='Only adapter-equivalent normalized stereo images and /clock are published. Shared Orin load limits timing comparisons.'), indent=2))
        start('mapper', ['ros2','launch',str(ROOT/'scripts/visual_only.launch.py'),
                          'profile:='+str(OUT/'profile.yaml'),'output_dir:='+str(OUT),'normalized_transport:=true' ])
        deadline = time.monotonic()+180
        while time.monotonic() < deadline:
            if processes['mapper'].poll() is not None: raise RuntimeError('Mapper exited during startup')
            if 'Learned frontend ready:' in (OUT/'mapper.log').read_text(): break
            monitor(); time.sleep(1)
        else: raise RuntimeError('Mapper readiness timed out')
        start('observer', ['/usr/bin/python3', str(ROOT/'scripts/visual_only_observer.py'), str(OUT)])
        deadline = time.monotonic()+20
        while not (OUT/'observer.ready').exists():
            if time.monotonic()>deadline: raise RuntimeError('Observer not ready')
            time.sleep(.2)
        player = start('player', ['/usr/bin/python3', str(snapshot/'replay_bag.py'),
            '/home/yanfa/Env_X/InterFace/bags/20260824_235238', '--rate','1',
            '--frame-index', str(ROOT/'test_data/20260824_235238/first_10min_index.json'),
             '--frames','174','--images-only','--normalized-stereo-profile',str(OUT/'profile.yaml'),'--report' ,str(OUT/'replay.json'),
            '--timeline',str(OUT/'replay_frames.csv')])
        print('Pure-visual comparison replay started', OUT, flush=True)
        while player.poll() is None:
            if processes['mapper'].poll() is not None: raise RuntimeError('Mapper exited during replay')
            monitor(); time.sleep(2)
        if player.returncode: raise RuntimeError('Replay failed')
        # Allow the last processing item and persistence queue to drain.
        for _ in range(5): monitor(); time.sleep(2)
        (OUT/'observer.stop').write_text('EOF\n')
        processes['observer'].wait(timeout=10)
        stop('mapper')
        (OUT/'benchmark_complete.json').write_text(json.dumps(dict(
            completed=True, peak_process_tree_rss_mib=peak, wall_finished=time.time()), indent=2))
        print('Pure-visual comparison finished', flush=True)
    except BaseException as error:
        (OUT/'benchmark_error.json').write_text(json.dumps(dict(error=str(error)), indent=2))
        raise
    finally:
        (OUT/'observer.stop').write_text('stop\n')
        for name in ['player','observer','mapper']:
            if name in processes:
                try: stop(name)
                except Exception as error: print('Own process retained:', name, error, flush=True)
        save()


if __name__ == '__main__':
    main()
