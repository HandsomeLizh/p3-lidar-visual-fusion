#!/usr/bin/env python3
"""Read real ROS sensors, normalize stereo, and share one LiDAR/IMU clock.

No simulation sockets, vehicle command publishers, or ground-truth inputs.
Separate source/target domains keep another running mapper's TF isolated.
"""
import argparse
from collections import Counter
import copy
import json
import os
from pathlib import Path
import signal
import threading
import time

import rclpy
from rclpy.context import Context
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.qos import QoSProfile, ReliabilityPolicy
from rclpy.signals import SignalHandlerOptions
from sensor_msgs.msg import Image, Imu, PointCloud2
from std_msgs.msg import String,Float64MultiArray
import yaml

from stereo_transport import StereoNormalizer
from t3_lidar_visual_fusion.hardware_time import SharedSensorClock


def seconds(msg):return msg.header.stamp.sec+msg.header.stamp.nanosec*1e-9


def set_stamp(msg,stamp):
    ns=round(stamp*1e9);msg.header.stamp.sec=ns//10**9;msg.header.stamp.nanosec=ns%10**9


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--profile',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--stream',choices=['inertial','imu','lidar','stereo'],default='inertial')
    args=parser.parse_args();profile=yaml.safe_load(Path(args.profile).read_text());cfg=profile['hardware']
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    domains=[int(cfg['source_domain']),int(os.environ.get('ROS_DOMAIN_ID','57'))]
    contexts=[Context(),Context()];nodes=[]
    for context,domain,name in zip(contexts,domains,['p3_real_sensor_input','p3_real_sensor_output']):
        rclpy.init(context=context,domain_id=domain,signal_handler_options=SignalHandlerOptions.NO)
        nodes.append(rclpy.create_node(name+'_'+args.stream,context=context))
    source,target=nodes
    clock=SharedSensorClock(**cfg.get('clock',{}));clock_lock=threading.Lock()
    counts=Counter();last={};arrivals={};rejections={}
    keys={'inertial':('imu','lidar'),'imu':('imu',),'lidar':('lidar',),'stereo':('left','right')}[args.stream]
    normalizer=StereoNormalizer(profile) if args.stream=='stereo' else None
    pubs={key:target.create_publisher(typ,topic,QoSProfile(depth=200 if key=='imu' else 2,reliability=ReliabilityPolicy.RELIABLE))
          for key,typ,topic in [('left',Image,'/fusion/left'),('right',Image,'/fusion/right'),
              ('lidar',PointCloud2,profile['lidar_topic']),('imu',Imu,profile['imu_topic'])] if key in keys}
    status_name={'inertial':'hardware_status','lidar':'hardware_status',
                 'imu':'hardware_imu_status','stereo':'hardware_stereo_status'}[args.stream]
    status_pub=target.create_publisher(String,'/fusion/'+status_name,2)
    stopping=threading.Event();fatal=[];groups=[];subs=[]
    clock_reference=cfg.get('clock_reference_topic')
    imu_state={}
    def shared_clock(now):
        nonlocal imu_state
        try:state=json.loads((out/'hardware_imu_status.json').read_text())
        except FileNotFoundError:return False
        if abs(now-float(state.get('report_wall_time',0.)))>2.:
            raise ValueError('Shared IMU clock status is stale')
        with clock_lock:ready=clock.adopt(state['clock'])
        imu_state=state
        return ready
    def observe_clock(msg):
        try:
            if len(msg.data)!=2:raise ValueError('Malformed clock reference')
            with clock_lock:clock.observe_imu(*msg.data)
            counts['clock_references']+=1;arrivals['clock_reference']=time.time()
        except ValueError as error:
            counts['clock_reference_rejected']+=1
            if clock.failure:fatal.append(str(error));stopping.set()
    if args.stream in ('inertial','imu') and clock_reference:
        group=MutuallyExclusiveCallbackGroup();groups.append(group)
        subs.append(source.create_subscription(Float64MultiArray,clock_reference,observe_clock,
            QoSProfile(depth=50,reliability=ReliabilityPolicy.RELIABLE),callback_group=group))
    def receive(key,msg):
        counts[key+'_received']+=1;now=time.time();stamp=seconds(msg);arrivals[key]=now
        if stamp<=last.get(key,-1):
            counts[key+'_old']+=1
            if key!='imu':return
        try:
            if key in ('imu','lidar'):
                if msg.header.frame_id!=profile[key+'_input_frame']:raise ValueError('Unexpected '+key+' frame '+msg.header.frame_id)
                if args.stream=='lidar' and not shared_clock(now):
                    counts['lidar_clock_warmup']+=1;return
                with clock_lock:
                    mapped=clock.observe_imu(stamp,now) if key=='imu' and not clock_reference else clock.convert(stamp,now)
                if args.stream!='lidar' and clock_reference and now-arrivals.get('clock_reference',now)>1.:
                    raise ValueError('Clock reference is stale')
                if mapped is None:counts[key+'_clock_warmup']+=1;return
                if key=='lidar':
                    if len(msg.data)>profile['max_cloud_bytes']:raise ValueError('Cloud byte cap')
                    if stamp-last.get(key,-1)<1./float(cfg.get('max_lidar_hz',5.))-1e-5:
                        counts['lidar_rate_limited']+=1;return
                msg.header=copy.deepcopy(msg.header);set_stamp(msg,mapped)
            else:
                if now-stamp>profile['visual_max_age_sec'] or stamp-now>.05:raise ValueError('Stale/future image')
                if cfg.get('compact_stereo',False):
                    if (msg.width,msg.height)!=tuple(profile['output_image_size']) or msg.encoding!='mono8':
                        raise ValueError('Driver image must match calibrated output size and mono8 encoding')
                    msg.header=copy.deepcopy(msg.header)
                    msg.header.frame_id='camera_left_optical' if key=='left' else 'camera_right_optical'
                else:msg=normalizer.convert(msg,0 if key=='left' else 1)
            pubs[key].publish(msg);last[key]=stamp;counts[key+'_forwarded']+=1
        except (ValueError,TypeError) as error:
            counts[key+'_rejected']+=1
            rejections[key]={'reason':str(error),'raw_stamp':stamp,'wall_time':now,
                'mapped_age_sec':now-(stamp+clock.offset) if key in ('lidar','imu') and clock.offset is not None else None}
            source.get_logger().warning(str(error),throttle_duration_sec=5)
            if clock.failure:fatal.append(clock.failure);stopping.set()
    # IMU cannot wait behind image conversion or a multi-megabyte cloud copy.
    for key,typ in [('imu',Imu),('lidar',PointCloud2),('left',Image),('right',Image)]:
        if key not in keys:continue
        group=MutuallyExclusiveCallbackGroup();groups.append(group)
        topic=cfg[key+'_topic']
        qos=QoSProfile(depth=400 if key=='imu' else 2,reliability=ReliabilityPolicy.BEST_EFFORT)
        subs.append(source.create_subscription(typ,topic,lambda m,k=key:receive(k,m),qos,callback_group=group))
    # In split mode the IMU worker never deserializes a multi-MB PointCloud2.
    # LiDAR adopts its clock estimate from the atomic per-run status file.
    executor=MultiThreadedExecutor(num_threads=2,context=contexts[0]);executor.add_node(source)
    thread=threading.Thread(target=executor.spin,daemon=True);thread.start()
    for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,lambda *_:stopping.set())
    def report():
        with clock_lock:clock_status=clock.status()
        value=dict(source_domain=domains[0],target_domain=domains[1],stream=args.stream,counts=dict(counts),
                   report_wall_time=time.time(),
                   last_rejections=dict(rejections),
                   sensor_idle_seconds={k:time.time()-v for k,v in list(arrivals.items())},vehicle_commands_published=0,
                   raw_source_topics={k:cfg[k+'_topic'] for k in keys})
        if args.stream!='stereo':
            clock_status['reference']='local_driver_dds_publication' if clock_reference else 'python_reception'
            value['clock']=clock_status
        if args.stream=='lidar':
            value['split_inertial_workers']=True
            value['counts'].update(imu_state.get('counts',{}))
            value['raw_source_topics'].update(imu_state.get('raw_source_topics',{}))
            value['imu_status_age_sec']=time.time()-imu_state.get('report_wall_time',0.)
            value['last_rejections'].update(imu_state.get('last_rejections',{}))
            value['sensor_idle_seconds'].update({k:age+value['imu_status_age_sec']
                for k,age in imu_state.get('sensor_idle_seconds',{}).items()})
        payload=json.dumps(value,indent=2);tmp=out/(status_name+'.json.tmp');tmp.write_text(payload);tmp.replace(out/(status_name+'.json'))
        status_pub.publish(String(data=payload))
    print('Real sensor bridge ready: domains',domains,flush=True)
    try:
        while not stopping.wait(1.):report()
    finally:
        report();executor.shutdown();thread.join(timeout=3.)
        for node in nodes:node.destroy_node()
        for context in contexts:context.shutdown()
    if fatal:raise RuntimeError(fatal[-1])


if __name__=='__main__':main()
