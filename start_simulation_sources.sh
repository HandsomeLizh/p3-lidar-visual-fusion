#!/usr/bin/env bash
# Start or reuse UE capture, the car bridge, and the human-operated control GUI.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec /usr/bin/python3 - "$ROOT" "$@" <<'PY'
import argparse
import fcntl
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

DOMAIN = '10'
HOST = '192.168.10.22'
SERVICES = {
    'sensor_capture_node': ('capture', 'start_capture.sh', ['--host', HOST], '6665'),
    'lunar_car_node': ('车辆控制', 'start_car.sh', [HOST, '6668', 'false'], '6668'),
    'lunar_car_gui': ('车辆控制窗口', 'start_gui.sh', [], None),
}


def snapshot():
    found = {name: [] for name in SERVICES}
    for directory in Path('/proc').iterdir():
        if not directory.name.isdigit():
            continue
        try:
            argv = (directory/'cmdline').read_bytes().decode(errors='replace').strip('\0').split('\0')
            name = Path(argv[0]).name
            if name not in SERVICES and len(argv) > 1 and Path(argv[1]).name == 'lunar_car_gui':
                name = 'lunar_car_gui'
            if name not in SERVICES:
                continue
            env = dict(s.split('=', 1) for s in (directory/'environ').read_bytes().decode(errors='replace').split('\0') if '=' in s)
            params = dict(s.split(':=', 1) for s in argv if ':=' in s)
            found[name].append(dict(pid=int(directory.name), domain=env.get('ROS_DOMAIN_ID', '0'),
                host=params.get('tcp_host'), port=params.get('tcp_port'),
                localhost=env.get('ROS_LOCALHOST_ONLY', '0')))
        except (OSError, ValueError, IndexError):
            continue
    return found


def check_existing(found):
    errors = []
    for name, (label, _, _, port) in SERVICES.items():
        entries = found[name]
        if len(entries) > 1:
            errors.append(f'{label} 有多个进程：{[p["pid"] for p in entries]}')
        for p in entries:
            if name == 'lunar_car_gui':
                if (p['domain'], p['localhost']) != (DOMAIN, '0'):
                    errors.append(f'{label} PID {p["pid"]}：当前域 {p["domain"]}、localhost={p["localhost"]}；需要域 {DOMAIN}、localhost=0')
            elif (p['domain'], p['host'], p['port'], p['localhost']) != (DOMAIN, HOST, port, '0'):
                errors.append(f'{label} PID {p["pid"]}：当前域 {p["domain"]}、地址 {p["host"]}:{p["port"]}、localhost={p["localhost"]}；需要域 {DOMAIN}、{HOST}:{port}、localhost=0')
    if errors:
        raise RuntimeError('\n'.join(errors) + '\n请在原启动终端按 Ctrl+C 关闭配置冲突的旧节点，再运行此脚本。未停止任何旧进程。')


def stop_children(children):
    for label, process in children:
        if process.poll() is None:
            print(f'[停止] 本脚本启动的 {label}', flush=True)
            try:
                os.killpg(process.pid, signal.SIGINT)
            except ProcessLookupError:
                pass
    for _, process in children:
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()


def main(root, check_only=False):
    # Keep this lock for the session; concurrent invocations cannot start duplicates.
    lock = None
    if not check_only:
        lock = (root/'simulation_sources.lock').open('a')
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('本启动脚本已在另一个终端运行；可用 --check 查看节点。')
    found = snapshot()
    check_existing(found)  # Check all roles before starting any missing process.
    for name, (label, _, _, _) in SERVICES.items():
        print(f'[复用] {label} PID {found[name][0]["pid"]}，ROS 域 {DOMAIN}' if found[name]
              else f'[未运行] {label}', flush=True)
    if check_only:
        return 0 if all(found.values()) else 1
    if all(found.values()):
        print('采集、车辆控制和控制窗口已运行，可直接启动实时建图。', flush=True)
        return 0
    runtime = Path(os.environ.get('P3_RUNTIME_ROOT', str(root.parent/'roma_t3_algorithm_bundle_20260825/envx_runtime')))
    for path in [runtime/'install/setup.bash', *(runtime/s[1] for s in SERVICES.values())]:
        if not path.is_file():
            raise RuntimeError(f'缺少现有 P3 启动文件：{path}')
    env = dict(os.environ, ROS_DOMAIN_ID=DOMAIN, ROS_LOCALHOST_ONLY='0', RMW_IMPLEMENTATION='rmw_cyclonedds_cpp')
    env.pop('CYCLONEDDS_URI', None)  # Match the existing source services' network setup.
    if not env.get('DISPLAY'):
        env['DISPLAY'] = ':0'
    authority = Path(f'/run/user/{os.getuid()}/gdm/Xauthority')
    if not env.get('XAUTHORITY') and authority.is_file():
        env['XAUTHORITY'] = str(authority)
    env.setdefault('XDG_RUNTIME_DIR', f'/run/user/{os.getuid()}')
    logs = root/'logs'/('simulation_sources_'+time.strftime('%Y%m%d_%H%M%S')+'_'+str(os.getpid()))
    logs.mkdir(parents=True)
    children = []
    try:
        for name, (label, script, arguments, _) in SERVICES.items():
            if found[name]:
                continue
            with (logs/(name+'.log')).open('w') as log:
                process = subprocess.Popen(['bash', str(runtime/script), *arguments], cwd=runtime,
                    env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            children.append((label, process))
            print(f'[启动] {label}，日志：{logs/(name+".log")}', flush=True)
        deadline = time.monotonic()+30
        while True:
            for label, process in children:
                if process.poll() is not None:
                    raise RuntimeError(f'{label} 启动退出，请查看日志：{logs}')
            current = snapshot()
            check_existing(current)
            if all(current.values()):
                break
            if time.monotonic() >= deadline:
                raise RuntimeError(f'等待节点启动超时，请查看日志：{logs}')
            time.sleep(.3)
        print('采集、车辆控制和控制窗口已启动。窗口显示在远程桌面；在另一个终端启动实时建图。', flush=True)
        print('由人操作控制窗口；脚本不会选择驾驶模式或设定非零速度。', flush=True)
        print('保持本终端打开；Ctrl+C 只停止本脚本新启动的节点，已复用节点继续运行。', flush=True)
        while True:
            for label, process in children:
                if process.poll() is not None:
                    raise RuntimeError(f'{label} 已退出，请查看日志：{logs}')
            time.sleep(1)
    finally:
        stop_children(children)
        if lock:
            lock.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='start_simulation_sources.sh', description='启动或复用 UE capture、车辆控制和人工控制窗口。')
    parser.add_argument('--check', action='store_true', help='只检查已有节点，不启动或停止进程')
    args = parser.parse_args(sys.argv[2:])
    def interrupted(*_):
        raise KeyboardInterrupt
    signal.signal(signal.SIGINT, interrupted)
    signal.signal(signal.SIGTERM, interrupted)
    try:
        sys.exit(main(Path(sys.argv[1]), args.check))
    except KeyboardInterrupt:
        print('已结束；原来就在运行的节点未受影响。', flush=True)
        sys.exit(130)
    except (RuntimeError, OSError) as error:
        print(f'[错误] {error}', file=sys.stderr, flush=True)
        sys.exit(2)
PY
