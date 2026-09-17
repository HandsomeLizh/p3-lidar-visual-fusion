"""Explicit short UE motion test with a separate elevation map (domain 87).

The running visual frontend supplies qualified raw visual poses. Reference
feedback is used only for stopping and scoring, never for mapping/localization.
No motion unless --execute is supplied. The idle P4 command relay is suspended
only during the bounded test, then resumed after parking. P4 code is not edited.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import numpy as np
import rclpy
from rclpy.context import Context
from rclpy.executors import SingleThreadedExecutor
from rclpy.qos import QoSProfile,DurabilityPolicy,ReliabilityPolicy,qos_profile_sensor_data
from rclpy.signals import SignalHandlerOptions
from geometry_msgs.msg import PoseStamped,Twist
from nav_msgs.msg import Odometry,Path as RosPath
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import String,Bool
from grid_map_msgs.msg import GridMap
import yaml
from t3_lidar_visual_fusion.ros_utils import cloud_arrays,stamp_sec

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--turns',action='store_true',help='Two mirrored out-and-back arcs, 1.5 m per leg')
    parser.add_argument('--relay-pid',type=int,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();out=args.output.resolve()
    if not out.is_relative_to(ROOT/'results'):raise ValueError('Output must be in this fusion project')
    out.mkdir(parents=True,exist_ok=False)
    cfg=yaml.safe_load((ROOT/'config/simulation_live.yaml').read_text())
    cfg.update(stereo_mapping_topic='/fusion/stereo_points',map_window=32.,map_pending_scans=8,
               semantic_topic='',tof_sources=[],mapping_pose_settle_sec=0.)
    cfg['dynamic_map']['enabled']=False
    profile=out/'profile.yaml';profile.write_text(yaml.safe_dump(cfg,sort_keys=False))
    package=ROOT/'install/t3_lidar_visual_fusion/lib/python3.10/site-packages/t3_lidar_visual_fusion'
    (out/'source_manifest.json').write_text(json.dumps({name:hashlib.sha256((package/name).read_bytes()).hexdigest()
        for name in ['probabilistic_elevation.py','height_bias.py','disk_map.py','terrain_mapper.py']},indent=2))
    contexts=[];executors=[];nodes=[]
    for domain in [57,87,10]:
        os.environ['ROS_LOCALHOST_ONLY']='1' if domain==87 else '0'
        c=Context();rclpy.init(context=c,domain_id=domain,signal_handler_options=SignalHandlerOptions.NO)
        n=rclpy.create_node('height_motion_check_'+str(domain),context=c)
        e=SingleThreadedExecutor(context=c);e.add_node(n)
        contexts.append(c);nodes.append(n);executors.append(e)
    source,target,car=nodes
    last={};seen={};counts={};epoch=[None];moving=[False];conflict=[None]
    def received(key,value):last[key]=value;seen[key]=time.monotonic();counts[key]=counts.get(key,0)+1
    pose_pub=target.create_publisher(Odometry,'/T3/semantic/current_pose',20)
    lidar_pub=target.create_publisher(PointCloud2,'/fusion/lidar',2)
    stereo_pub=target.create_publisher(PointCloud2,'/fusion/stereo_points',2)
    def visual(m):
        covariance=np.asarray(m.pose.covariance).reshape(6,6)
        valid=(np.isfinite(covariance).all() and np.linalg.eigvalsh(covariance).min()>=0 and
               max(np.diag(covariance)[:3])<=cfg['qualified_position_variance'] and
               max(np.diag(covariance)[3:])<=cfg['qualified_rotation_variance'])
        if epoch[0] is None:epoch[0]=m.header.frame_id
        if m.header.frame_id!=epoch[0]:conflict[0]='Visual origin reset';return
        if not valid:return
        received('visual',m);m=copy.deepcopy(m);m.header.frame_id='odom';pose_pub.publish(m)
    def cloud(key,m,pub):received(key,m);pub.publish(m)
    source.create_subscription(Odometry,'/fusion/learned_raw',visual,qos_profile_sensor_data)
    reliable_cloud=QoSProfile(depth=2,reliability=ReliabilityPolicy.RELIABLE)
    source.create_subscription(PointCloud2,'/fusion/lidar',lambda m:cloud('lidar',m,lidar_pub),reliable_cloud)
    source.create_subscription(PointCloud2,'/fusion/stereo_points',lambda m:cloud('stereo',m,stereo_pub),reliable_cloud)
    source.create_subscription(String,'/P4/control/relay_status',lambda m:received('relay',json.loads(m.data)),10)
    def p4_command(m):
        if moving[0] and (abs(m.linear.x)>.001 or abs(m.angular.z)>.001):conflict[0]='P4 requested motion concurrently'
    source.create_subscription(Twist,'/P4/debug/cmd_vel',p4_command,10)
    source.create_subscription(RosPath,'/Car/T4/planning/local_path',
        lambda m:conflict.__setitem__(0,'New P4 path during manual test') if moving[0] and m.poses else None,10)
    car.create_subscription(PoseStamped,'/car/pose',lambda m:received('truth',m),qos_profile_sensor_data)
    car.create_subscription(String,'/car/set_mode',
        lambda m:conflict.__setitem__(0,'External park command') if moving[0] and m.data=='park' else None,10)
    retained=QoSProfile(depth=1,reliability=ReliabilityPolicy.RELIABLE,durability=DurabilityPolicy.TRANSIENT_LOCAL)
    target.create_subscription(GridMap,'/Car/T3/mapping/grid_map',lambda m:received('map',m),retained)
    target.create_subscription(String,'/fusion/map_status',lambda m:received('map_status',json.loads(m.data)),10)
    cmd=car.create_publisher(Twist,'/car/cmd_vel',10)
    mode=car.create_publisher(String,'/car/set_mode',10)
    lease=car.create_publisher(Bool,'/car/external_control_lease',10)
    child=None;paused=False;identity=None;commands=False
    result=dict(execute_requested=args.execute,command_sent=False,phases=[],mapping_domain=87,
                visual_source_domain=57,reference_used_for_mapping=False)
    trace=(out/'supervision.jsonl').open('w',buffering=1)
    def spin(seconds):
        end=time.monotonic()+seconds
        while time.monotonic()<end:
            for e in executors:e.spin_once(timeout_sec=.002)
    def xyz(m):
        p=m.pose.position if isinstance(m,PoseStamped) else m.pose.pose.position
        return np.array([p.x,p.y,p.z])
    def ready():
        if conflict[0]:raise RuntimeError(conflict[0])
        for key,age in [('truth',.6),('visual',4.),('lidar',4.),('map',7.)]:
            if key not in seen or time.monotonic()-seen[key]>age:raise RuntimeError('Stale input: '+key)
        if time.time()-stamp_sec(last['visual'])>cfg['visual_max_age_sec']:
            raise RuntimeError('Visual measurement timestamp too old')
        if child.poll() is not None:raise RuntimeError('Test mapper stopped')
    def snapshot(name):
        m=last['map'];arrays={}
        for key,d in zip(m.layers,m.data):
            ny,nx=(v.size for v in d.layout.dim)
            arrays[key]=np.asarray(d.data).reshape(ny,nx)[::-1,::-1].copy()
        origin=np.array([m.info.pose.position.x-m.info.length_x/2,m.info.pose.position.y-m.info.length_y/2])
        arrays.update(origin=origin,resolution=np.array(m.info.resolution),pose=xyz(last['visual']))
        np.savez_compressed(out/(name+'.npz'),**arrays)
        return arrays
    def hold(seconds):
        end=time.monotonic()+seconds
        while time.monotonic()<end:
            if paused:lease.publish(Bool(data=True));cmd.publish(Twist())
            spin(.1)
    def park():
        moving[0]=False
        for _ in range(5):mode.publish(String(data='park'));cmd.publish(Twist());spin(.1)
    def heading(m):
        q=m.pose.orientation
        return np.arctan2(2*(q.w*q.z+q.x*q.y),1-2*(q.y*q.y+q.z*q.z))
    def move(velocity,label,angular=0.,distance_limit=1.):
        ready();start=xyz(last['truth']);visual_start=xyz(last['visual'])
        initial_heading=heading(last['truth'])
        began=time.monotonic();moving[0]=True
        phase=dict(name=label,distance_m=0.,seconds=0.,fresh_pose_waits=0)
        result['phases'].append(phase)
        mode.publish(String(data='auto'))
        while True:
            try:ready()
            except RuntimeError as exc:
                if 'timestamp too old' not in str(exc) and not str(exc).startswith('Stale input:'):raise
                park();phase['fresh_pose_waits']+=1;waiting=time.monotonic()
                while True:
                    if time.monotonic()-waiting>10.:raise RuntimeError('Fresh input did not recover while parked')
                    hold(.1)
                    try:
                        ready()
                        if time.time()-stamp_sec(last['visual'])>cfg['visual_max_age_sec']-1.:continue
                        break
                    except RuntimeError as error:
                        if 'timestamp too old' not in str(error) and not str(error).startswith('Stale input:'):raise
                moving[0]=True;mode.publish(String(data='auto'))
            distance=float(np.linalg.norm(xyz(last['truth'])-start))
            phase.update(distance_m=distance,seconds=time.monotonic()-began)
            angle=heading(last['truth'])-initial_heading
            phase['heading_change_deg']=float(np.rad2deg(np.arctan2(np.sin(angle),np.cos(angle))))
            if abs(np.linalg.norm(xyz(last['visual'])-visual_start)-distance)>.5:
                raise RuntimeError('Visual travel differs from reference by over 0.5 m')
            trace.write(json.dumps(dict(phase=label,wall=time.time(),distance_m=distance,
                visual=xyz(last['visual']).tolist(),truth=xyz(last['truth']).tolist(),
                map_stamp=stamp_sec(last['map']),height_bias=last.get('map_status',{}).get('height_bias')))+'\n')
            if distance>=distance_limit:break
            if time.monotonic()-began>65.:raise RuntimeError('Short motion leg timed out')
            lease.publish(Bool(data=True));m=Twist();m.linear.x=velocity;m.angular.z=angular;cmd.publish(m);spin(.1)
        park()
    def compare(before,after):
        res=float(before['resolution']);a=before['elevation'];b=after['elevation']
        rr,cc=np.indices(a.shape);delta=np.rint((before['origin']-after['origin'])/res).astype(int)
        br,bc=rr+delta[1],cc+delta[0]
        valid=(br>=0)&(br<b.shape[0])&(bc>=0)&(bc<b.shape[1])&np.isfinite(a)
        x=before['origin'][0]+(cc+.5)*res;y=before['origin'][1]+(rr+.5)*res
        valid&=(np.hypot(x-before['pose'][0],y-before['pose'][1])<8.)&(before['observation_count']>=3)
        rr,cc,br,bc=rr[valid],cc[valid],br[valid],bc[valid]
        observed=np.isfinite(b[br,bc])&(after['observation_count'][br,bc]>before['observation_count'][rr,cc])
        differences=b[br[observed],bc[observed]]-a[rr[observed],cc[observed]]
        if not len(differences):raise RuntimeError('No reobserved height cells to score')
        return dict(reobserved_cells=len(differences),median_m=float(np.median(differences)),
            abs_p95_m=float(np.percentile(abs(differences),95)),abs_max_m=float(abs(differences).max()),
            changed_over_3cm=int((abs(differences)>.03).sum()))
    def interrupted(*unused):raise KeyboardInterrupt('Test interrupted')
    signal.signal(signal.SIGINT,interrupted);signal.signal(signal.SIGTERM,interrupted)
    try:
        env=dict(os.environ,ROS_DOMAIN_ID='87',ROS_LOCALHOST_ONLY='1')
        with (out/'mapper.log').open('wb') as log:
            child=subprocess.Popen([str(ROOT/'install/t3_lidar_visual_fusion/lib/t3_lidar_visual_fusion/terrain_mapper'),
                '--ros-args','-p','profile_path:='+str(profile),'-p','output_dir:='+str(out/'map')],
                env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        spin(16.);ready();baseline=snapshot('before')
        status=last['relay'];result['relay_before']=status
        if status.get('state')!='PARKED' or status.get('path') is not None or time.monotonic()-seen['relay']>2.:
            raise RuntimeError('P4 is not idle; no movement')
        body=cloud_arrays(last['lidar']);mount=np.asarray(cfg['base_from_lidar'])
        body=body@mount[:3,:3].T+mount[:3,3]
        clear=[]
        for direction in [1,-1]:
            p=body[(body[:,0]*direction>.8)&(body[:,0]*direction<(2.8 if args.turns else 2.3))&
                   (abs(body[:,1])<(1.1 if args.turns else .65))]
            if len(p)>100 and np.percentile(p[:,2],99)-np.percentile(p[:,2],10)<.08:clear.append(direction)
        if not clear:raise RuntimeError('No observed flat short corridor; no movement')
        result['direction']=clear[0]
        print('READY '+json.dumps(dict(direction=clear[0],layers=list(last['map'].layers),cells=int(np.isfinite(baseline['elevation']).sum()))),flush=True)
        if not args.execute:
            result['outcome']='observation_only';return
        proc=Path(f'/proc/{args.relay_pid}')
        command=(proc/'cmdline').read_bytes().replace(b'\0',b' ').decode()
        if '/home/yanfa/P4/debug/p3_joint/persistent_command_relay.py' not in command:raise RuntimeError('Wrong relay process')
        identity=(proc/'stat').read_text().split(') ',1)[1].split()[19]
        os.kill(args.relay_pid,signal.SIGSTOP);paused=True
        hold(.5);commands=True;result['command_sent']=True
        if args.turns:
            for name,angular in [('left',.025),('right',-.025)]:
                move(.1*clear[0],name+'_arc',angular,1.5);hold(5.);ready()
                result[name+'_arc_height_change']=compare(baseline,snapshot(name+'_arc'))
                move(-.1*clear[0],name+'_return',-angular,1.5);hold(7.);ready()
                result[name+'_return_height_change']=compare(baseline,snapshot(name+'_return'))
            final=snapshot('returned')
        else:
            move(.1*clear[0],'outward');hold(7.);ready();middle=snapshot('outward')
            result['outward_height_change']=compare(baseline,middle)
            move(-.1*clear[0],'return');hold(9.);ready();final=snapshot('returned')
        result['returned_height_change']=compare(baseline,final)
        result['map_status']=last.get('map_status');result['outcome']='completed'
    except BaseException as exc:
        result.update(outcome='stopped',error=type(exc).__name__+': '+str(exc))
        print(result['error'],flush=True)
    finally:
        try:
            if commands:
                park();start=xyz(last['truth']);hold(2.)
                result['park_displacement_m']=float(np.linalg.norm(xyz(last['truth'])-start))
        finally:
            if paused:
                proc=Path(f'/proc/{args.relay_pid}')
                if proc.exists() and (proc/'stat').read_text().split(') ',1)[1].split()[19]==identity:
                    os.kill(args.relay_pid,signal.SIGCONT);result['relay_resumed']=True
                spin(.5);lease.publish(Bool(data=False));spin(.2)
        if child is not None and child.poll() is None:
            child.send_signal(signal.SIGINT)
            try:child.wait(timeout=12.)
            except subprocess.TimeoutExpired:
                child.terminate()
                try:child.wait(timeout=5.)
                except subprocess.TimeoutExpired:
                    child.kill();child.wait(timeout=3.);result['test_mapper_forced_exit']=True
        result['counts']=counts
        (out/'result.json').write_text(json.dumps(result,indent=2))
        print(json.dumps(result),flush=True);trace.close()
        for e,n,c in zip(executors,nodes,contexts):e.shutdown();n.destroy_node();c.try_shutdown()


if __name__=='__main__':main()
