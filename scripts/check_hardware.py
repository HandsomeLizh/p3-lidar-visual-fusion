#!/usr/bin/env python3
"""Read-only rover input audit; no driver or motion commands."""
import argparse,collections,json,time
from pathlib import Path
import numpy as np
import yaml,rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image,Imu,PointCloud2

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--seconds',type=float,default=5.);parser.add_argument('--output');args=parser.parse_args()
    cfg=yaml.safe_load((ROOT/'config/hardware104.yaml').read_text())
    rclpy.init(domain_id=int(cfg['hardware']['source_domain']));node=Node('p3_hardware_check')
    records={k:{'samples':collections.deque(maxlen=4000),'frames':set()} for k in ('lidar','imu','left','right')}
    def receive(msg,key):
        r=records[key];r['samples'].append((time.time(),msg.header.stamp.sec+msg.header.stamp.nanosec*1e-9));r['frames'].add(msg.header.frame_id)
        if key in ('left','right'):r['size']=[msg.width,msg.height];r['encoding']=msg.encoding
        if key=='lidar':r['points']=msg.width*msg.height;r['fields']=[f.name for f in msg.fields]
        if key=='imu':r['last_specific_force_mps2']=[msg.linear_acceleration.x,msg.linear_acceleration.y,msg.linear_acceleration.z]
    for key,typ in [('left',Image),('right',Image),('imu',Imu),('lidar',PointCloud2)]:
        node.create_subscription(typ,cfg['hardware'][key+'_topic'],lambda m,k=key:receive(m,k),qos_profile_sensor_data)
    deadline=time.monotonic()+args.seconds
    while time.monotonic()<deadline:rclpy.spin_once(node,timeout_sec=.05)
    passed=True
    for k,r in records.items():
        samples=np.asarray(r.pop('samples'));r['frames']=sorted(r['frames']);r['received']=len(samples)
        r['topic']=cfg['hardware'][k+'_topic'];r['available']=len(samples)>=2;passed &= r['available']
        if len(samples)>1:
            r['rate_hz']=(len(samples)-1)/(samples[-1,0]-samples[0,0]);r['stamp_interval_sec']=float(np.median(np.diff(samples[:,1])))
            r['clock']='device_uptime' if samples[-1,1]<1e9 else 'system_epoch'
    result={'all_inputs_received':bool(passed),'source_domain':cfg['hardware']['source_domain'],'sensors':records,
            'note':'Input connectivity only; calibration and moving accuracy are separate checks'}
    text=json.dumps(result,indent=2);print(text)
    if args.output:Path(args.output).write_text(text)
    node.destroy_node();rclpy.shutdown();return 0 if passed else 1


if __name__=='__main__':raise SystemExit(main())
