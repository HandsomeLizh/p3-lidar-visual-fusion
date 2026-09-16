#!/usr/bin/env python3
"""Explicit, bounded UE/P4 test. Without --execute this only observes.

The normal launch scripts never invoke this helper. Reference poses are logged
for assessment and stopping only; this script never publishes localization.
"""
import argparse
import json
import math
import os
from pathlib import Path
import time
import numpy as np
import rclpy
from rclpy.context import Context
from rclpy.executors import SingleThreadedExecutor
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from rclpy.signals import SignalHandlerOptions
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Odometry, Path as RosPath
from grid_map_msgs.msg import GridMap
from std_msgs.msg import String


def stamp(msg):
    s=msg.header.stamp
    return s.sec+s.nanosec*1e-9


def pose(msg):
    p=msg.pose.pose.position;q=msg.pose.pose.orientation
    return [p.x,p.y,p.z,q.x,q.y,q.z,q.w]


def yaw(p):
    x,y,z,w=p[3:]
    return math.atan2(2*(w*z+x*y),1-2*(y*y+z*z))


def main():
    a=argparse.ArgumentParser(description=__doc__)
    a.add_argument('--output',type=Path,required=True)
    a.add_argument('--execute',action='store_true')
    a.add_argument('--distance',type=float,default=3.4)
    a.add_argument('--angle',type=float,default=0.,help='Relative target direction, radians')
    a.add_argument('--timeout',type=float,default=65.)
    x=a.parse_args()
    if not 2.5<=x.distance<=5 or abs(x.angle)>.5 or not 10<=x.timeout<=100:
        a.error('Allowed: distance 2.5..5 m, angle +/-0.5 rad, timeout 10..100 s')
    x.output.mkdir(parents=True,exist_ok=False)
    runtime=Path(__file__).resolve().parents[1]/'run_state.json'
    if runtime.exists():(x.output/'runtime_at_start.json').write_text(runtime.read_text())
    log=(x.output/'observations.jsonl').open('w',buffering=1)
    c=Context();legacy=Context()
    rclpy.init(context=c,domain_id=57,signal_handler_options=SignalHandlerOptions.NO)
    rclpy.init(context=legacy,domain_id=10,signal_handler_options=SignalHandlerOptions.NO)
    node=rclpy.create_node('p3_live_motion_check_'+str(os.getpid()),context=c)
    stopper=rclpy.create_node('p3_live_motion_stop_'+str(os.getpid()),context=legacy)
    executor=SingleThreadedExecutor(context=c);executor.add_node(node)
    mode=stopper.create_publisher(String,'/car/set_mode',10)
    goal_pub=node.create_publisher(PoseStamped,'/Car/T4/rviz_goal',10)
    sensor=QoSProfile(depth=100,reliability=ReliabilityPolicy.BEST_EFFORT)
    retained=QoSProfile(depth=1,reliability=ReliabilityPolicy.RELIABLE,durability=DurabilityPolicy.TRANSIENT_LOCAL)
    last={}; counts={}; state={'command_sent':False,'outcome':'observe_only'}
    maps={}; first_motion=None; last_truth=None; travel=0.; last_heartbeat=0.

    def write(kind,value):
        record=dict(kind=kind,wall=time.time(),monotonic=time.monotonic(),**value)
        log.write(json.dumps(record,allow_nan=False)+'\n')
        last[kind]=record;counts[kind]=counts.get(kind,0)+1

    def on_pose(kind,m):
        nonlocal first_motion,last_truth,travel
        v=pose(m)
        if not np.isfinite(v).all():return
        write(kind,dict(stamp=stamp(m),frame=m.header.frame_id,pose=v,
                       covariance=[m.pose.covariance[i*7] for i in range(6)]))
        if kind=='truth':
            if last_truth is not None and state['command_sent']:
                delta=float(np.linalg.norm(np.asarray(v[:3])-last_truth))
                travel+=delta
                if delta>.002 and first_motion is None:first_motion=time.time()
            last_truth=np.asarray(v[:3])

    def on_json(kind,m):
        try:write(kind,dict(value=json.loads(m.data)))
        except (ValueError,TypeError):write(kind,dict(text=m.data[:2000]))

    def on_grid(kind,m):
        maps[kind]=m
        write(kind,dict(stamp=stamp(m),layers=list(m.layers),length=[m.info.length_x,m.info.length_y]))

    for kind,topic in [('truth','/P4/input/odometry'),('fused','/T3/semantic/current_pose'),
                       ('lidar','/fusion/lio_raw'),('visual','/fusion/learned_raw')]:
        node.create_subscription(Odometry,topic,lambda m,k=kind:on_pose(k,m),sensor)
    for kind,topic in [('fusion','/fusion/status'),('feedback','/P4/input/feedback_status'),
                       ('relay','/P4/control/relay_status')]:
        node.create_subscription(String,topic,lambda m,k=kind:on_json(k,m),10)
    for kind,topic in [('local_grid','/Car/T3/mapping/grid_map'),('global_grid','/T3/mapping/global_grid_map')]:
        node.create_subscription(GridMap,topic,lambda m,k=kind:on_grid(k,m),retained)
    for kind,topic in [('global_path','/Car/T4/planning/global_route'),('local_path','/Car/T4/planning/local_path')]:
        node.create_subscription(RosPath,topic,lambda m,k=kind:write(k,dict(stamp=stamp(m),points=len(m.poses))),retained)
    node.create_subscription(Twist,'/P4/debug/cmd_vel',lambda m:write('command',dict(v=m.linear.x,w=m.angular.z)),10)

    def spin(seconds):
        end=time.monotonic()+seconds
        while time.monotonic()<end:executor.spin_once(timeout_sec=.05)

    def value_at(grid,px,py):
        ox=grid.info.pose.position.x-grid.info.length_x/2
        oy=grid.info.pose.position.y-grid.info.length_y/2
        values={};res=grid.info.resolution
        col,row=math.floor((px-ox)/res),math.floor((py-oy)/res)
        if grid.outer_start_index or grid.inner_start_index:raise RuntimeError('Unsupported circular map')
        for layer,data in zip(grid.layers,grid.data):
            ny,nx=(int(d.size) for d in data.layout.dim)
            if not (0<=col<nx and 0<=row<ny):return {}
            v=float(data.data[(ny-1-row)*nx+(nx-1-col)])
            values[layer]=v if math.isfinite(v) else None
        return values

    try:
        spin(8.)
        required={'truth':.5,'fused':4.,'fusion':3.,'feedback':2.,'relay':2.,'local_grid':6.}
        stale=[k for k,limit in required.items() if k not in last or time.monotonic()-last[k]['monotonic']>limit]
        if stale:raise RuntimeError('Inputs not ready: '+','.join(stale))
        if last['feedback']['value'].get('reason')!='READY':raise RuntimeError('P4 reference not aligned')
        if not last['fusion']['value'].get('localization_valid'):raise RuntimeError('Localization unavailable')
        p=last['truth']['pose'];heading=yaw(p)+x.angle
        gx=p[0]+x.distance*math.cos(heading);gy=p[1]+x.distance*math.sin(heading)
        grid=maps['local_grid'];goal_cell=value_at(grid,gx,gy)
        footprint=[value_at(grid,gx+dx,gy+dy) for dx in [-.65,0,.65] for dy in [-.55,0,.55]]
        state.update(initial_pose=p,goal=[gx,gy,heading],goal_cell=goal_cell,footprint=footprint)
        write('candidate',state.copy())
        print(json.dumps(state),flush=True)
        if x.execute:
            relay=last['relay']['value']
            if relay.get('path') is not None or relay.get('state')=='FORWARDING':
                raise RuntimeError('Existing P4 navigation is active; no command sent')
            if any(v.get('elevation') is None or v.get('occupancy') is None or
                   v.get('obstacle',0) is None or v.get('obstacle',0)>.5 or v['occupancy']>=.65 for v in footprint):
                raise RuntimeError('Goal footprint is unknown or occupied; no command sent')
            if goal_pub.get_subscription_count()==0:raise RuntimeError('No P4 goal subscriber')
            goal=PoseStamped();goal.header.frame_id='map';goal.header.stamp=node.get_clock().now().to_msg()
            goal.pose.position.x=gx;goal.pose.position.y=gy;goal.pose.position.z=goal_cell['elevation']
            goal.pose.orientation.z=math.sin(heading/2);goal.pose.orientation.w=math.cos(heading/2)
            state['command_sent']=True;state['started_wall']=time.time()
            goal_pub.publish(goal);write('goal_sent',dict(goal=state['goal']))
            began=time.monotonic();state['outcome']='timeout'
            while time.monotonic()-began<x.timeout:
                spin(.1);now=time.monotonic()
                if now-last_heartbeat>5:
                    last_heartbeat=now;print(json.dumps(dict(elapsed=now-began,travel_m=travel,mode=last.get('fusion',{}).get('value',{}).get('operating_mode'),relay=last.get('relay',{}).get('value'))),flush=True)
                if (x.output/'stop').exists():raise RuntimeError('Operator stop requested')
                stale=[k for k,limit in required.items() if now-last[k]['monotonic']>limit]
                if stale:raise RuntimeError('Stale during motion: '+','.join(stale))
                if not last['fusion']['value'].get('localization_valid'):raise RuntimeError('Localization lost')
                tp=last['truth']['pose'];fp=last['fused']['pose']
                if np.linalg.norm(np.asarray(tp[:2])-fp[:2])>1.:raise RuntimeError('Planar reference discrepancy exceeds 1 m')
                if travel>6.:raise RuntimeError('Travel bound reached')
                if abs(last.get('command',{}).get('v',0))>.25 or abs(last.get('command',{}).get('w',0))>.08:
                    raise RuntimeError('Command exceeds low-speed test bounds')
                distance=math.hypot(tp[0]-gx,tp[1]-gy)
                angular=abs(math.atan2(math.sin(yaw(tp)-heading),math.cos(yaw(tp)-heading)))
                if distance<.13 and angular<.06:
                    state['outcome']='goal_reached';break
                if now-began>15 and travel<.02:
                    state['outcome']='no_motion_or_planning_rejected';break
    except BaseException as error:
        state['outcome']='stopped';state['error']=str(error)
        print(type(error).__name__+': '+str(error),flush=True)
    finally:
        state['finished_wall']=time.time()
        if state['command_sent']:
            for _ in range(12):mode.publish(String(data='park'));spin(.1)
            before=last.get('truth',{}).get('pose');spin(3.)
            after=last.get('truth',{}).get('pose')
            state['park_displacement_m']=None if before is None or after is None else float(np.linalg.norm(np.asarray(after[:3])-before[:3]))
            state['park_reference_fresh']=time.monotonic()-last.get('truth',{}).get('monotonic',0)<.5
        state.update(counts=counts,travel_m=travel,first_motion_wall=first_motion,final_reference=last.get('truth'),final_fused=last.get('fused'))
        (x.output/'result.json').write_text(json.dumps(state,indent=2)+'\n')
        print(json.dumps({k:v for k,v in state.items() if k not in ('footprint','goal_cell')}),flush=True)
        log.close();executor.shutdown();node.destroy_node();stopper.destroy_node();c.try_shutdown();legacy.try_shutdown()


if __name__=='__main__':main()
