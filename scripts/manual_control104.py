#!/usr/bin/env python3
"""Open the existing manual GUI and reuse the real chassis driver when present."""
import argparse
import fcntl
import json
import os
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHASSIS = Path('/home/yanfa/program/Mars_Car2')
DRIVER = CHASSIS / 'install/mars_car2/lib/mars_car2/chassis_node'
GUI = Path('/home/yanfa/program/UI/src/mars_car_gui.py')


def running(name):
    records = []
    for proc in Path('/proc').glob('[0-9]*'):
        try:
            args = [p.decode(errors='replace') for p in proc.joinpath('cmdline').read_bytes().split(b'\0') if p]
            if not any(Path(arg).name == name for arg in args):
                continue
            if proc.joinpath('stat').read_text().rsplit(')', 1)[1].split()[0] == 'Z':
                continue
        except OSError:
            continue
        try:
            env = dict(item.split(b'=', 1) for item in proc.joinpath('environ').read_bytes().split(b'\0') if b'=' in item)
        except FileNotFoundError:
            continue
        except PermissionError as error:
            raise RuntimeError('发现已有底盘/控制窗口，但无法核验其环境，未启动新实例。') from error
        records.append({'pid': int(proc.name), 'args': args,
                        'domain': env.get(b'ROS_DOMAIN_ID', b'0').decode(),
                        'display': env.get(b'DISPLAY', b'').decode()})
    return records


def driver_status():
    drivers = running('chassis_node')
    if len(drivers) > 1 or any(d['domain'] != '19' or str(DRIVER) not in d['args'] for d in drivers):
        raise RuntimeError('已有底盘进程的路径/ROS 域与 104 配置不一致，请检查；未停止或重复启动底盘。')
    return drivers


def ensure_driver():
    drivers = driver_status()
    if drivers:
        print('复用已有 104 底盘驱动，PID=' + str(drivers[0]['pid']), flush=True)
        return
    print('按原 Mars_Car2 流程启动底盘：上电、上位机控制与编码器基准初始化。', flush=True)
    subprocess.run(['bash', str(CHASSIS / 'start_mars_car.sh')], cwd=CHASSIS,
                   stdin=subprocess.DEVNULL, start_new_session=True, check=True)
    deadline = time.monotonic() + 12
    while time.monotonic() < deadline:
        if driver_status():
            return
        time.sleep(.2)
    raise RuntimeError('未发现底盘节点，请检查 /home/yanfa/program/Mars_Car2/.mars_car.log')


def check_display():
    if not os.environ.get('DISPLAY'):
        raise RuntimeError('没有图形显示。请从 237 桌面使用 start_104_control.sh，或在 104 桌面运行。')
    import tkinter
    root = tkinter.Tk()
    try:
        root.withdraw()
        root.update()
    finally:
        root.destroy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', help='Read-only process/dependency check')
    args = parser.parse_args()
    for path in (DRIVER, CHASSIS / 'start_mars_car.sh', GUI):
        if not path.is_file():
            raise RuntimeError('缺少原底盘程序：' + str(path))
    drivers = driver_status()
    panels = running('mars_car_gui.py')
    if args.check:
        print(json.dumps({'domain': 19, 'drivers': drivers, 'panels': panels, 'gui': str(GUI),
                          'commands_published': 0}, ensure_ascii=False))
        return
    lock_file = ROOT / 'results/manual_control104.lock'
    lock_file.parent.mkdir(parents=True, exist_ok=True)
    with lock_file.open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('本入口的手动控制窗口已在运行，请切回已有窗口。')
        if panels:
            raise RuntimeError('已有手动控制面板：' + json.dumps(panels, ensure_ascii=False) + '；未打开第二套控制面板。')
        check_display()  # Verify X11 before starting anything that enables the motors.
        ensure_driver()
        print('打开原手动控制面板。操作结束先点“停止”，确认车辆停稳，再关闭窗口。', flush=True)
        subprocess.run(['/usr/bin/python3', str(GUI)], cwd=GUI.parents[1], check=True)


if __name__ == '__main__':
    main()
