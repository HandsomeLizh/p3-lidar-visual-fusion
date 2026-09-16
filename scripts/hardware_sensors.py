#!/usr/bin/env python3
"""Start only missing real sensor drivers; stop only recorded owned processes."""
import argparse,fcntl,json,os,signal,subprocess,time
from pathlib import Path
import rclpy
from rclpy.node import Node
from control import alive,identity

ROOT=Path(__file__).resolve().parents[1];STATE=ROOT/'hardware_sensor_state.json'


def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['start','stop','status']);args=parser.parse_args()
    lock=(ROOT/'hardware_sensors.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    state=json.loads(STATE.read_text()) if STATE.exists() else {'processes':{}}
    def save():
        tmp=STATE.with_suffix('.tmp');tmp.write_text(json.dumps(state,indent=2));tmp.replace(STATE)
    if args.action=='status':
        print(json.dumps({k:dict(v,running=alive(v)) for k,v in state['processes'].items()},indent=2));return
    if args.action=='stop':
        for name,record in reversed(list(state['processes'].items())):
            if not alive(record):continue
            os.killpg(record['pid'],signal.SIGINT);deadline=time.monotonic()+25
            while alive(record) and time.monotonic()<deadline:time.sleep(.1)
            if alive(record):
                os.killpg(record['pid'],signal.SIGTERM);deadline=time.monotonic()+5
                while alive(record) and time.monotonic()<deadline:time.sleep(.1)
            if alive(record):raise RuntimeError('Owned sensor did not stop: '+name)
            print('Stopped owned sensor process:',name)
        save();return
    # Old, already-stopped records must not make a reused external driver look
    # like a failed newly started process.
    state['processes']={k:r for k,r in state['processes'].items() if alive(r)};save()
    rclpy.init();node=Node('p3_real_sensor_preflight');deadline=time.monotonic()+2
    while time.monotonic()<deadline:rclpy.spin_once(node,timeout_sec=.1)
    topics={key:node.count_publishers(topic) for key,topic in {
        'lidar':'/Car/T5/OS1/points','imu':'/Car/T5/OS1/imu',
        'left':'/Car/T5/Cam_Left/image_mono/mapping','right':'/Car/T5/Cam_Right/image_mono/mapping',
        'legacy_left':'/Car/T5/Cam_Left/image_raw/color','legacy_right':'/Car/T5/Cam_Right/image_raw/color',
        'trigger':'/sensors_trigger'}.items()}
    node.destroy_node();rclpy.shutdown()
    if bool(topics['lidar'])!=bool(topics['imu']):raise RuntimeError('Ouster is only partially available; enable both PCL and IMU in the existing driver')
    if bool(topics['left'])!=bool(topics['right']):raise RuntimeError('Only one camera publisher exists; inspect that driver before starting another')
    if not topics['left'] and (topics['legacy_left'] or topics['legacy_right']):
        raise RuntimeError('An existing camera driver lacks compact mapping images. Stop that driver before starting the private mapping camera, or configure raw-image bridge input explicitly.')
    out=ROOT/'results'/time.strftime('hardware_sensors_%Y%m%d_%H%M%S');out.mkdir(parents=True,exist_ok=True)
    def spawn(name,command):
        if alive(state['processes'].get(name)):
            print('Owned process already running:',name);return
        path=out/(name+'.log');env=dict(os.environ);env['SAVE_ENABLE']='false'
        with path.open('ab',buffering=0) as log:
            proc=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,stdin=subprocess.DEVNULL)
        state['processes'][name]=dict(pid=proc.pid,identity=identity(proc.pid),command=command,log=str(path));save()
        print('Started',name,'PID',proc.pid,'log:',path,flush=True)
    if not topics['lidar']:spawn('ouster',['bash','/home/yanfa/program/Lidar2/start_ouster.sh'])
    else:print('Reusing existing Ouster LiDAR + IMU driver')
    if not topics['left']:spawn('stereo',['bash',str(ROOT/'scripts/start_hardware_camera.sh')])
    else:print('Reusing existing compact stereo driver')
    if not topics['trigger']:spawn('trigger',['python3','/home/yanfa/program/trigger_publisher.py','/sensors_trigger','5'])
    else:print('Reusing existing camera trigger')
    time.sleep(2.)
    failed=[name for name,record in state['processes'].items() if not alive(record)]
    if failed:raise RuntimeError('Sensor process exited; inspect logs: '+','.join(failed))
    print('Sensor processes started/reused. Verify actual messages with scripts/check_hardware.py.')


if __name__=='__main__':main()
