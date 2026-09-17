#!/usr/bin/env python3
"""Reuse the current car map or start it; only clean up recorded orphan UI jobs."""
import argparse
import fcntl
import json
import os
import subprocess
from pathlib import Path

import yaml
from control import alive, stop

ROOT = Path(__file__).resolve().parents[1]


def inspect():
    state_file = ROOT / 'run_state.json'
    state = json.loads(state_file.read_text()) if state_file.exists() else {'processes': {}}
    running = {name: record for name, record in state['processes'].items() if alive(record)}
    if 'pipeline' in running:
        profile = yaml.safe_load((Path(state['output']) / 'profile.yaml').read_text())
        if (str(state.get('domain')) != '59' or str(state.get('localhost_only')) != '0'
                or profile.get('mapping_source') != 'stereo'
                or not profile.get('hardware', {}).get('enabled')):
            raise RuntimeError('104 当前运行与双目实车/域 59 配置不同；请先人工检查，未停止当前程序。')
    return state, running


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', help='Only inspect; do not start or stop anything')
    args = parser.parse_args()
    # Use the same controller lock while inspecting and cleaning leftover children.
    # Release it before calling start_hardware.sh, whose controller takes this lock itself.
    with (ROOT / 'run_state.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('104 正在执行其他启停操作，请稍后重试。')
        state, running = inspect()
        if args.check:
            print(json.dumps({'pipeline_running': 'pipeline' in running,
                              'running_processes': list(running), 'output': state.get('output'),
                              'next_action': 'reuse' if 'pipeline' in running else 'start'}, ensure_ascii=False))
            return
        if 'pipeline' in running:
            print('104 建图已在运行，复用当前地图：' + str(state['output']), flush=True)
            return
        sensor_env = dict(os.environ, ROS_DOMAIN_ID='19', ROS_LOCALHOST_ONLY='0',
                          CYCLONEDDS_URI='file:///home/yanfa/program/cyclonedds.xml')
        inputs = subprocess.run(['/usr/bin/python3', str(ROOT / 'scripts/check_hardware.py'), '--seconds', '3'],
                                cwd=ROOT, env=sensor_env, capture_output=True, text=True)
        if inputs.returncode:
            print(inputs.stdout, inputs.stderr, flush=True)
            raise RuntimeError('传感器采集未就绪。请先在 237 执行 ./start_104_capture.sh，再启动建图。')
        if running:
            print('清理这套 104 工程已停止建图后留下的窗口/辅助进程。', flush=True)
            for name in ('live_relay', 'monitor', 'capture', 'player', 'verifier', 'rviz', 'visuals'):
                stop(state, name)
    env = dict(os.environ, P3_HARDWARE_DOMAIN='59')
    subprocess.run(['bash', str(ROOT / 'start_hardware.sh')], cwd=ROOT, env=env, check=True)
    print('104 已创建新地图并启动。P4 使用前请对齐新地图原点。', flush=True)


if __name__ == '__main__':
    main()
