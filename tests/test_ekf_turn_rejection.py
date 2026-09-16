#!/usr/bin/env python3
"""Isolate EKF LiDAR innovation gating using recorded, already guarded inputs."""
import copy
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import numpy as np
import yaml
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from rosgraph_msgs.msg import Clock

ROOT=Path(__file__).resolve().parents[1]


def main():
    assert os.environ.get('ROS_DOMAIN_ID')=='58'
    source=ROOT/'results/long_fusion_repaired_clean_final_20260916'
    out=ROOT/'results/ekf_arrival_regression_20260916'
    out.mkdir(exist_ok=False)
    record=json.loads((source/'verification.json').read_text())
    raw={(m['source'],round(m['stamp_sec']*1e9)):m for m in record['raw_poses']}
    assert {m['frame'] for m in record['raw_poses'] if m['source']=='learned_raw'}=={'learned_epoch_1'}
    first=next(m for m in record['raw_poses'] if m['source']=='learned_raw')
    assert np.allclose(first['pose'],[0,0,0,0,0,0,1])
    rows={}
    for m in record['measurements']:
        if m['source'] not in ['lio_guarded','vision_accepted']:continue
        ns=round(m['stamp_sec']*1e9)
        original=raw[('lio_raw' if m['source']=='lio_guarded' else 'learned_raw',ns)]
        # This diagnostic has one identity-aligned visual epoch. Guard covariance
        # floors add diagonal entries and retain the original off-diagonal terms.
        cov=np.array(original['pose_covariance']).reshape(6,6)
        np.fill_diagonal(cov,m['pose_covariance_diagonal'])
        rows.setdefault(ns,{})[m['source']]={'pose':original['pose'],'cov':cov.ravel().tolist()}
    rclpy.init();node=Node('ekf_turn_probe')
    clock=node.create_publisher(Clock,'/clock',10)
    observed={name:{} for name in ['gated','guarded_lidar','lidar_first']}
    pubs={name:{k:node.create_publisher(Odometry,'/turn_probe/'+name+'/'+k,100)
                for k in ['lio_guarded','vision_accepted']} for name in observed}
    def receive(name,m):
        p=m.pose.pose.position;q=m.pose.pose.orientation
        observed[name][m.header.stamp.sec*1000000000+m.header.stamp.nanosec]=[p.x,p.y,p.z,q.x,q.y,q.z,q.w]
    for name in observed:
        node.create_subscription(Odometry,'/turn_probe/'+name,lambda m,n=name:receive(n,m),100)
    procs=[];handles=[]
    try:
        for name in observed:
            cfg=yaml.safe_load((source/'config/ekf.yaml').read_text())['ekf_filter_node']['ros__parameters']
            cfg.update(use_sim_time=True,publish_tf=False,odom0='/turn_probe/'+name+'/lio_guarded',odom1='/turn_probe/'+name+'/vision_accepted',debug=True,debug_out_file=str(out/(name+'_debug.txt')))
            if name!='gated':cfg.pop('odom0_pose_rejection_threshold',None)
            config=out/(name+'.yaml');config.write_text(yaml.safe_dump({name:{'ros__parameters':cfg}}))
            log=(out/(name+'.log')).open('w');handles.append(log)
            procs.append(subprocess.Popen([str(ROOT/'deps/root/opt/ros/humble/lib/robot_localization/ekf_node'),'--ros-args','-r','__node:='+name,'-r','odometry/filtered:=/turn_probe/'+name,'--params-file',str(config)],stdout=log,stderr=subprocess.STDOUT,start_new_session=True))
        deadline=time.monotonic()+12
        while any(pubs[name]['lio_guarded'].get_subscription_count()!=1 for name in observed):
            if time.monotonic()>deadline:raise TimeoutError('EKF discovery')
            rclpy.spin_once(node,timeout_sec=.02)
        for k in range(100):
            c=Clock();c.clock.sec=999;c.clock.nanosec=k*5000000;clock.publish(c)
            until=time.monotonic()+.02
            while time.monotonic()<until:rclpy.spin_once(node,timeout_sec=.002)
        for ns,measurements in sorted(rows.items()):
            for step in range(200):
                c=Clock();c.clock.sec,c.clock.nanosec=divmod(ns+step*10000000,1000000000);clock.publish(c)
                for name in observed:
                    order=(['lio_guarded','vision_accepted'] if name=='lidar_first' else list(measurements))
                    for j,key in enumerate(order):
                        if step!=1+7*j or key not in measurements:continue
                        row=measurements[key];m=Odometry();m.header.stamp.sec,m.header.stamp.nanosec=divmod(ns,1000000000)
                        m.header.frame_id='odom';m.child_frame_id='base_link'
                        p=m.pose.pose.position;q=m.pose.pose.orientation
                        p.x,p.y,p.z,q.x,q.y,q.z,q.w=map(float,row['pose'])
                        m.pose.covariance=row['cov'];pubs[name][key].publish(m)
                until=time.monotonic()+.025
                while time.monotonic()<until:rclpy.spin_once(node,timeout_sec=.002)
                if step>=20 and all(any(abs(t-ns)<3 for t in observed[name]) for name in observed):break
            else:raise TimeoutError("Missing EKF output for "+str(ns))
        (out/"observed.json").write_text(json.dumps(observed))
        times=sorted(rows)
        expected=np.asarray([rows[t]['lio_guarded']['pose'][:3] for t in times])
        result={'qualification':'No truth input. Same guarded constraints, fixed identity visual epoch. Compare recorded receipt order plus joint gating, recorded order without LiDAR joint gating, and LiDAR-first without that gate. Fast diagnostic replay is not timing evidence.','samples':len(times)}
        for name in observed:
            matched=[min(observed[name],key=lambda t:abs(t-ns)) for ns in times]
            assert max(abs(t-ns) for t,ns in zip(matched,times))<3
            positions=np.array([observed[name][t][:3] for t in matched])
            errors=np.linalg.norm(positions-expected,axis=1)
            result[name]={'max_deviation_from_raw_lidar_m':float(errors.max()),'max_frame':int(errors.argmax()),'deviations_m':errors.tolist()}
        result['passed']=(result['gated']['max_deviation_from_raw_lidar_m']>.5
                          and result['guarded_lidar']['max_deviation_from_raw_lidar_m']<.3
                          and result['lidar_first']['max_deviation_from_raw_lidar_m']<.2)
        (out/'result.json').write_text(json.dumps(result,indent=2))
        print(json.dumps(result,indent=2),flush=True)
        assert result['passed'],result
    finally:
        for p in procs:
            if p.poll() is None:os.killpg(p.pid,signal.SIGINT);p.wait(timeout=15)
        for f in handles:f.close()
        node.destroy_node();rclpy.shutdown()


if __name__=='__main__':main()
