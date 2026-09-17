"""Actual EKFs with delayed LiDAR/fast vision; isolated domain, no commands.

Synthetic timing regression, not a real trajectory accuracy benchmark.
"""
import argparse
import copy
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import numpy as np
import rclpy
from nav_msgs.msg import Odometry
from rosgraph_msgs.msg import Clock
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
from generate_config import generate
from t3_lidar_visual_fusion.measurement_order import VisualConstraintQueue
from t3_lidar_visual_fusion.ros_utils import stamp_sec
from t3_lidar_visual_fusion.elevation_fusion import surface_observations


def main():
    p = argparse.ArgumentParser(); p.add_argument('--ekf', required=True); p.add_argument('--output', required=True)
    args = p.parse_args()
    assert os.environ.get('ROS_DOMAIN_ID') == '97'
    out = Path(args.output); out.mkdir(parents=True, exist_ok=False)
    base = yaml.safe_load((ROOT/'config/hardware104.yaml').read_text())
    rclpy.init(); node = rclpy.create_node('delayed_height_fixture')
    clock = node.create_publisher(Clock, '/clock', 10)
    pubs = {}; received = {k:[] for k in ('old', 'fixed')}; procs = []; logs = []
    latest = dict(old=-1., fixed=-1.); latest_input = dict(old=-1., fixed=-1.)
    def receive(mode, msg):
        t = stamp_sec(msg)
        if mode == 'fixed' and t > latest_input[mode]+1e-6:return
        if t < latest[mode]:return
        latest[mode] = t
        received[mode].append(copy.deepcopy(msg))
    def spin(seconds):
        end = time.monotonic()+seconds
        while time.monotonic() < end:rclpy.spin_once(node, timeout_sec=.001)
    def header(msg, t):
        msg.header.stamp.sec, msg.header.stamp.nanosec = divmod(round(t*1e9), 1000000000)
        return msg
    def publish(mode, kind, msg):
        latest_input[mode] = max(latest_input[mode], stamp_sec(msg))
        pubs[mode][kind].publish(msg)
    try:
        for mode in received:
            profile = copy.deepcopy(base)
            profile['ekf_measurement_time_only'] = mode == 'fixed'
            f = out/(mode+'.yaml'); f.write_text(yaml.safe_dump(profile)); generate(f, out/mode)
            cfg = yaml.safe_load((out/mode/'ekf.yaml').read_text())['ekf_filter_node']['ros__parameters']
            cfg.update(use_sim_time=True, publish_tf=False, odom0='/delay/'+mode+'/lidar', odom1='/delay/'+mode+'/visual')
            f = out/(mode+'_ekf.yaml'); f.write_text(yaml.safe_dump({mode:{'ros__parameters':cfg}}))
            pubs[mode] = {k:node.create_publisher(Odometry, '/delay/'+mode+'/'+k, 100) for k in ('lidar','visual')}
            node.create_subscription(Odometry, '/delay/'+mode+'/ekf', lambda m,k=mode:receive(k,m), 100)
            log = (out/(mode+'.log')).open('w'); logs.append(log)
            procs.append(subprocess.Popen([args.ekf, '--ros-args', '-r', '__node:='+mode,
                '--params-file', str(f), '-r', 'odometry/filtered:=/delay/'+mode+'/ekf'],
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True))
        deadline = time.monotonic()+12
        while any(v['lidar'].get_subscription_count()!=1 for v in pubs.values()):
            if time.monotonic()>deadline:raise TimeoutError('EKF discovery')
            spin(.02)
        queue = VisualConstraintQueue(.8); frontier = -1.; last_lidar = -100.
        delayed = []; lidar_stamps = []
        for step in range(241):
            now = 1000.+step*.05
            c = Clock(); c.clock.sec,c.clock.nanosec = divmod(round(now*1e9), 1000000000)
            clock.publish(c)
            if 20 <= step < 200 and step%4 == 0:
                m = header(Odometry(), now); m.header.frame_id='odom'; m.child_frame_id='base_link'
                m.pose.pose.orientation.w=1.; m.pose.pose.position.x=(now-1000.)*.1
                m.pose.covariance=np.diag([.0025]*3+[.0004]*3).ravel().tolist()
                m.twist.twist.linear.x=.1
                m.twist.covariance=np.diag([.05]*6).ravel().tolist()
                delayed.append((now+.2,'visual',copy.deepcopy(m)))
                if step%8 == 0:
                    delayed.append((now+.8,'lidar',m)); lidar_stamps.append(now)
            due=[v for v in delayed if v[0]<=now+1e-7];delayed=[v for v in delayed if v[0]>now+1e-7]
            for _,kind,msg in sorted(due,key=lambda v:v[0]):
                publish('old',kind,msg)
                if kind=='lidar':
                    frontier=stamp_sec(msg);last_lidar=now;publish('fixed',kind,msg)
                else:queue.append(stamp_sec(msg),now,msg)
            for msg in queue.ready(frontier,now,now-last_lidar<1.5):publish('fixed','visual',msg)
            spin(.025)
        spin(.15)
        # Use the same monotonic pose admission as the guard. Compare covariance
        # at the LiDAR scan stamps, rather than averaging repeated EKF messages.
        result={}
        for mode,rows in received.items():
            by_stamp={stamp_sec(m):m for m in rows}; times=sorted(by_stamp)
            values=[]
            for t in lidar_stamps[5:-2]:
                i=int(np.searchsorted(times,t))
                if i<len(times) and abs(times[i]-t)<1e-6:cov=np.array(by_stamp[times[i]].pose.covariance).reshape(6,6)
                elif i and i<len(times) and times[i]-times[i-1]<.65:
                    u=(t-times[i-1])/(times[i]-times[i-1]);cov=sum(w*np.array(by_stamp[s].pose.covariance).reshape(6,6) for w,s in [(1-u,times[i-1]),(u,times[i])])
                else:continue
                points=np.c_[np.linspace(4.,10.,40),np.ones(40),np.zeros(40)]
                obs=surface_observations(points,.2,cov,[0.,0.,0.],base['elevation_fusion'])
                values.append(dict(stamp=t,rotation_variance=float(cov[3,3]),height_accepted=sum(o[3]<=.25**2 for o in obs),cells=len(obs)))
            result[mode]=dict(samples=values,acceptance=sum(v['height_accepted'] for v in values)/max(1,sum(v['cells'] for v in values)))
        result['queue']=queue.status();result['scope']='Synthetic delayed sensor inputs, actual robot_localization EKFs, original height uncertainty limit; no ATE claim.'
        (out/'report.json').write_text(json.dumps(result,indent=2))
        assert result['fixed']['acceptance']>.95,result
        assert result['old']['acceptance']<.5,result
        assert len(result['fixed']['samples'])>=10,result
        print(json.dumps({k:v['acceptance'] for k,v in result.items() if k in ('old','fixed')}))
    finally:
        for proc in procs:
            if proc.poll() is None:os.killpg(proc.pid,signal.SIGINT);proc.wait(timeout=10)
        for log in logs:log.close()
        node.destroy_node();rclpy.try_shutdown()


if __name__=='__main__':main()
