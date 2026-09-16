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
OUT = ROOT / 'results/p3_roma_long_80m_20260915'


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
        if int(previous.get("domain",-1))==59 and any(alive(v) for v in previous.get("processes",{}).values()):
            raise RuntimeError("An owned comparison is already active in this domain: "+str(path.parent))
    OUT.mkdir(exist_ok=False)
    state = dict(output=str(OUT), domain=59, processes={})
    snapshot = OUT / 'source_snapshot'
    snapshot.mkdir()
    sources = [P3/'workspace/src/t3_semantic_mapping/t3_semantic_mapping',
               P3/'third_party/RoMa/romatch']
    for source in sources:
        shutil.copytree(source, snapshot/source.name,
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    for source in [P3/'third_party/RoMa/roma_vo.py',
                   P3/'workspace/src/t3_semantic_mapping/launch/t3_envx_demo.launch.py']:
        shutil.copy2(source, snapshot/source.name)
    manifest = {str(p.relative_to(snapshot)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in snapshot.rglob('*') if p.is_file()}
    (snapshot/'manifest.json').write_text(json.dumps(manifest, indent=2))
    command = 'source /opt/ros/humble/setup.bash && source ' + shlex.quote(str(
        P3/'workspace/install_native/setup.bash')) + ' && env -0'
    result = subprocess.run(['bash', '-c', command], check=True, stdout=subprocess.PIPE)
    env = dict(e.split('=', 1) for e in result.stdout.decode().split('\0') if '=' in e)
    env.update(ROS_DOMAIN_ID='59', ROS_LOCALHOST_ONLY='1',
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
            source_program=str(P3), ros_domain_id=59, localhost_only=True,
            algorithm='P3 current RoMa stereo VO plus variance elevation fusion',
            rate=1.0, frames=174, input_policy='identical indexed LiDAR-receive-time batches',
            other_live_P3_processes_present=True, source_snapshot=str(snapshot),
            note='No reference odometry/TF or IMU is published. Shared Orin load limits timing comparisons.'), indent=2))
        start('mapper', ['ros2', 'launch', str(snapshot/'t3_envx_demo.launch.py'),
             'output_dir:='+str(OUT), 'publish_input_processed:=true', 'use_sim_time:=true',
             'enable_semantics:=false', 'elevation_fusion_mode:=variance'])
        deadline = time.monotonic()+180
        while time.monotonic() < deadline:
            if processes['mapper'].poll() is not None: raise RuntimeError('Mapper exited during startup')
            if 'Mapper ready. Waiting for synchronized image input.' in (OUT/'mapper.log').read_text(): break
            monitor(); time.sleep(1)
        else: raise RuntimeError('Mapper readiness timed out')
        start('observer', ['/usr/bin/python3', str(ROOT/'scripts/p3_benchmark_observer.py'), str(OUT)])
        deadline = time.monotonic()+20
        while not (OUT/'observer.ready').exists():
            if time.monotonic()>deadline: raise RuntimeError('Observer not ready')
            time.sleep(.2)
        player = start('player', ['/usr/bin/python3', str(ROOT/'scripts/replay_bag.py'),
            '/home/yanfa/Env_X/InterFace/bags/20260824_235238', '--rate','1',
            '--frame-index', str(ROOT/'test_data/20260824_235238/first_10min_index.json'),
            '--frames','174', '--report',str(OUT/'replay.json'),
            '--timeline',str(OUT/'replay_frames.csv')])
        print('P3 comparison replay started', OUT, flush=True)
        while player.poll() is None:
            if processes['mapper'].poll() is not None: raise RuntimeError('Mapper exited during replay')
            monitor(); time.sleep(2)
        if player.returncode: raise RuntimeError('Replay failed')
        # Allow the last processing item and persistence queue to drain.
        for _ in range(15): monitor(); time.sleep(2)
        (OUT/'observer.stop').write_text('EOF\n')
        processes['observer'].wait(timeout=10)
        stop('mapper')
        (OUT/'benchmark_complete.json').write_text(json.dumps(dict(
            completed=True, peak_process_tree_rss_mib=peak, wall_finished=time.time()), indent=2))
        print('P3 comparison finished', flush=True)
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
