#!/usr/bin/env python3
"""Replay the saved live sensor stream through the complete repaired pipeline."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import yaml

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'results/live_simulation_20260916_100400'
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--rate', type=float, default=1.)
parser.add_argument('--output', type=Path, default=ROOT / 'results/registration_convergence_fix_20260916/fusion_replay_1x')
args = parser.parse_args()
if args.rate <= 0:
    parser.error('--rate must be positive')
OUTPUT = args.output.resolve()
if not OUTPUT.is_relative_to(ROOT / 'results') or OUTPUT.exists():
    parser.error('Choose a new directory under this workspace results/')


def finish_clock():
    """Let final estimator and map timers run after rosbag stops publishing time."""
    import rclpy
    from rosgraph_msgs.msg import Clock
    metadata = yaml.safe_load((SOURCE/'live_input_bag/metadata.yaml').read_text())['rosbag2_bagfile_information']
    last_ns = metadata['starting_time']['nanoseconds_since_epoch'] + metadata['duration']['nanoseconds']
    rclpy.init()
    node = rclpy.create_node('repaired_replay_clock_tail')
    publisher = node.create_publisher(Clock, '/clock', 10)
    started_tail = time.monotonic()
    try:
        while time.monotonic()-started_tail < 6:
            tick = Clock()
            stamp = last_ns + int((time.monotonic()-started_tail)*1e9)
            tick.clock.sec, tick.clock.nanosec = divmod(stamp, 10**9)
            publisher.publish(tick)
            rclpy.spin_once(node, timeout_sec=.02)
    finally:
        node.destroy_node()
        rclpy.shutdown()


env = dict(os.environ, ROS_DOMAIN_ID='57', ROS_LOCALHOST_ONLY='1')
os.environ.update(ROS_DOMAIN_ID='57', ROS_LOCALHOST_ONLY='1')
started = False
try:
    subprocess.run([sys.executable, str(ROOT/'scripts/control.py'), 'start',
        '--profile', str(SOURCE/'profile.yaml'), '--use-sim-time', '--domain', '57',
        '--verify', '--verify-seconds', str(int(360/args.rate+50)), '--rviz', '--output', str(OUTPUT)],
        cwd=ROOT, env=env, check=True, timeout=100)
    started = True
    shutil.copy2(ROOT/'build_manifest.json', OUTPUT/'build_manifest.json')
    with (OUTPUT/'player.log').open('w') as log:
        subprocess.run(['ros2', 'bag', 'play', str(SOURCE/'live_input_bag'), '--clock', '50',
            '--rate', str(args.rate), '--read-ahead-queue-size', '6', '--disable-keyboard-controls',
            '--delay', '2', '--wait-for-all-acked', '3000'],
            cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=int(360/args.rate+60))
    # The acquisition stamps and their source-side receipt lag remain unchanged.
    # Extend the clock, without adding sensor samples, so the final callbacks drain.
    finish_clock()
    with (OUTPUT/'save_map_result.txt').open('w') as log:
        subprocess.run(['bash', str(ROOT/'save_map.sh')], cwd=ROOT, env=env,
                       stdout=log, stderr=subprocess.STDOUT, check=True, timeout=40)
    (OUTPUT/'verifier.stop').touch()
    deadline = time.monotonic()+12
    while not (OUTPUT/'verification.json').exists() and time.monotonic()<deadline:
        time.sleep(.2)
    subprocess.run([sys.executable, str(ROOT/'scripts/capture_rviz.py'),
                    '--output', str(OUTPUT), '--name', 'rviz_repaired.png'],
                   cwd=ROOT, env=env, check=True, timeout=15)
    (OUTPUT/'replay_complete.json').write_text(json.dumps(dict(
        kind='offline_replay_of_actual_live_input', source_run=str(SOURCE),
        input_bag=str(SOURCE/'live_input_bag'), rate=args.rate, clock_tail_sec=6, headers_modified=False,
        reference_fed_to_estimator=False, vehicle_motion_commanded=False), indent=2)+'\n')
finally:
    if started:
        state = json.loads((ROOT/'run_state.json').read_text())
        if state['output'] == str(OUTPUT):
            subprocess.run([sys.executable, str(ROOT/'scripts/control.py'), 'stop'],
                           cwd=ROOT, env=env, check=True, timeout=95)
if (OUTPUT/'replay_complete.json').exists():
    subprocess.run(['bash', str(ROOT/'export_map.sh'), str(OUTPUT)], cwd=ROOT, env=env,
                   check=True, timeout=60)
    print('REPAIRED FUSION REPLAY COMPLETE', OUTPUT, flush=True)
