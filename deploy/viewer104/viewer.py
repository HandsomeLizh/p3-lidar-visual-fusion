"""Own only this display's processes; never launch mapping or vehicle control."""
import argparse,fcntl,json,os,signal,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
STATE=ROOT/'state.json'
DISPLAY_PREFIX='/viewer104_237'


def display_commands(runtime):
 """Source topics stay on 104; only this window's derived outputs are private."""
 return {
  'monitor':['/usr/bin/python3',str(ROOT/'scripts/p3_visual_monitor.py'),'--ros-args',
   '-r','__node:=t3_104_237_monitor','-r','__ns:='+DISPLAY_PREFIX,
   '-r','/T3/demo/rviz_cloud:='+DISPLAY_PREFIX+'/rviz_cloud',
   '-r','/T3/demo/markers:='+DISPLAY_PREFIX+'/markers',
   '-r','/T3/mapping/global_grid_map:=/Car/T3/mapping/global_grid_map',
   '-r','/T3/mapping/lidar_status:=/fusion/map_status',
   '-r','/Car/T5/Cam_Left/image_raw/color:=/fusion/left',
   '-r','/Car/T5/Cam_Right/image_raw/color:=/fusion/right',
   '-r','/Car/T3/metrics/frame_timing:=/fusion/learned_status','-p','use_sim_time:=false'],
  'window':[str(ROOT/'build/t3_visual_window'),str(ROOT/'config/p3_visual_window.rviz'),str(runtime),
   '--ros-args','-r','__node:=t3_104_237_rviz','-r','__ns:='+DISPLAY_PREFIX,'-p','use_sim_time:=false']}

def identity(pid):
 try:
  fields=Path('/proc/'+str(pid)+'/stat').read_text().rsplit(')',1)[1].split()
  return None if fields[0]=='Z' else fields[19]
 except OSError:return None

def alive(record):return bool(record and record.get('identity') and identity(record['pid'])==record['identity'])

def write(data):
 tmp=STATE.with_suffix('.tmp');tmp.write_text(json.dumps(data,indent=2));tmp.replace(STATE)

def main():
 parser=argparse.ArgumentParser();parser.add_argument('action',choices=['start','stop','status']);args=parser.parse_args()
 previous=json.loads(STATE.read_text()) if STATE.exists() else {}
 if args.action=='status':
  print(json.dumps(dict(previous,running=alive(previous),children={k:dict(v,running=alive(v)) for k,v in previous.get('children',{}).items()}),indent=2));return
 if args.action=='stop':
  if alive(previous):os.kill(previous['pid'],signal.SIGINT)
  else:print('Viewer already stopped')
  return
 lock=(ROOT/'viewer.lock').open('a')
 try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 except BlockingIOError:raise SystemExit('104 viewer is already running; use ./status.sh')
 assert os.environ['ROS_DOMAIN_ID']=='59' and os.environ['ROS_LOCALHOST_ONLY']=='0'
 executable=ROOT/'build/t3_visual_window'
 if not executable.is_file():raise SystemExit('Run ./build.sh first')
 out=ROOT/'results'/time.strftime('%Y%m%d_%H%M%S');out.mkdir(parents=True,exist_ok=False)
 runtime=out/'runtime';runtime.mkdir()
 env=dict(os.environ,T3_VISUAL_RUNTIME=str(runtime))
 state=dict(pid=os.getpid(),identity=identity(os.getpid()),domain=59,source='192.168.100.104',
  display_prefix=DISPLAY_PREFIX,output=str(out),children={})
 stopped=False;processes=[]
 def stop(*_):
  nonlocal stopped
  stopped=True
 signal.signal(signal.SIGINT,stop);signal.signal(signal.SIGTERM,stop)
 def spawn(name,command):
  log=(out/(name+'.log')).open('wb')
  process=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
  processes.append((process,log))
  state['children'][name]=dict(pid=process.pid,identity=identity(process.pid),command=command)
  write(state)
 try:
  for name,command in display_commands(runtime).items():spawn(name,command)
  print('104 viewer on domain 59; display topics '+DISPLAY_PREFIX+'/*. Logs: '+str(out),flush=True)
  while not stopped and all(p.poll() is None for p,_ in processes):time.sleep(.2)
 finally:
  for process,log in reversed(processes):
   if process.poll() is None:
    os.killpg(process.pid,signal.SIGINT)
    try:process.wait(timeout=8)
    except subprocess.TimeoutExpired:
     os.killpg(process.pid,signal.SIGTERM);process.wait(timeout=5)
   log.close()
  state['stopped_at']=time.time();write(state)

if __name__=='__main__':main()
