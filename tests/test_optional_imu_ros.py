#!/usr/bin/env python3
"""Exercise the compiled backend through ROS, plus frozen-binary no-IMU replay.

Synthetic stationary IMU verifies availability switching; it is never used as
accuracy evidence and never injected into any recorded-data benchmark.
"""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu, PointCloud2, PointField
from std_msgs.msg import String, Header
from nav_msgs.msg import Odometry
from rosgraph_msgs.msg import Clock

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from indexed_bag import groups


def main():
    if os.environ.get('ROS_DOMAIN_ID')!='58':raise RuntimeError('Use isolated ROS_DOMAIN_ID=58')
    out=ROOT/'results/optional_imu_after_fusion_repair_20260916'
    out.mkdir(exist_ok=True)
    rclpy.init();n=Node('optional_imu_validation')
    pub=n.create_publisher(PointCloud2,'/fusion/lidar',3)
    imu_pub=n.create_publisher(Imu,'/fusion/imu',200)
    clock=n.create_publisher(Clock,'/clock',100)
    metrics,poses={},{}
    for name in ['baseline','auto','off']:
        metrics[name]=[];poses[name]=[]
        n.create_subscription(String,'/'+name+'/metrics',lambda m,k=name:metrics[k].append(json.loads(m.data)),100)
        n.create_subscription(Odometry,'/'+name+'/odom',lambda m,k=name:poses[k].append(m),100)
    procs=[];handles=[]
    def stamp(t):
        h=Header();ns=int(round(t*1e9));h.stamp.sec,h.stamp.nanosec=divmod(ns,1000000000);return h
    def spin(seconds=.1,predicate=None):
        end=time.monotonic()+seconds
        while time.monotonic()<end:
            rclpy.spin_once(n,timeout_sec=.005)
            if predicate and predicate():return
        if predicate:raise RuntimeError('ROS evidence timeout')
    def set_clock(t):
        clock.publish(Clock(clock=stamp(t).stamp));spin(.015)
    def launch(name,binary,mode=None):
        args=[str(binary),'--ros-args','-r','__node:=test_voxel_'+name,
              '-p','use_sim_time:=true','-p','threads:=1',
              '-r','/fusion/lio_raw:=/'+name+'/odom',
              '-r','/fusion/lidar_metrics:=/'+name+'/metrics',
              '-r','/fusion/lidar_quality:=/'+name+'/quality',
              '-r','/fusion/imu_status:=/'+name+'/imu_status']
        if mode:args+=['-p',"imu_mode:='"+mode+"'",'-p','imu_calibration_confirmed:=true']
        f=(out/(name+'.log')).open('w');handles.append(f)
        process=subprocess.Popen(args,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        procs.append(process)
    def stop_all():
        for p in procs:
            if p.poll() is None:
                os.killpg(p.pid,signal.SIGINT);p.wait(timeout=15)
        procs.clear()
    result={}
    try:
        binary=ROOT/'install/t3_voxelmap/lib/t3_voxelmap/voxelmap_node'
        launch('baseline',ROOT/'archive/pre_optional_imu_20260915/voxelmap_node_baseline')
        launch('auto',binary,'auto');launch('off',binary,'off')
        spin(12,lambda:pub.get_subscription_count()==3)
        bag='/home/yanfa/Env_X/InterFace/bags/20260824_235238'
        index=ROOT/'test_data/20260824_235238/first_10min_index.json'
        base=None
        for raw,batch in groups(bag,str(index),12,True):
            if base is None:base=raw
            t=1000.+(raw-base)/1e9;set_clock(t+.001)
            cloud=batch['/Car/T5/OS1/points'];cloud.header=stamp(t);cloud.header.frame_id='lidar'
            before={name:len(metrics[name]) for name in metrics}
            pub.publish(cloud)
            spin(10,lambda:all(len(metrics[k])>before[k] for k in metrics))
        spin(.3)
        def array(name):
            rows=[]
            for m in poses[name]:
                p,q=m.pose.pose.position,m.pose.pose.orientation
                rows.append([p.x,p.y,p.z,q.x,q.y,q.z,q.w,*m.pose.covariance])
            return np.asarray(rows)
        baseline=array('baseline')
        assert len(baseline)==12
        deltas={k:float(np.max(np.abs(array(k)[:,:7]-baseline[:,:7]))) for k in ['auto','off']}
        assert all(v<1e-8 for v in deltas.values()),deltas
        auto_off_delta=float(np.max(np.abs(array('auto')-array('off'))))
        assert auto_off_delta<1e-8,auto_off_delta
        assert all(not m['imu']['mode']=='imu' for m in metrics['auto'])
        result['recorded_no_imu_regression']=dict(frames=12,bag=bag,max_pose_difference_from_pre_imu_binary=deltas,
            auto_off_max_pose_covariance_difference=auto_off_delta,
            qualification='Same raw LiDAR: poses unchanged versus pre-IMU binary; current auto/off pose and covariance identical. Published covariance intentionally changed by directional uncertainty repair. No IMU or truth published.')
        stop_all()
        for name in metrics:metrics[name].clear();poses[name].clear()
        # Independent synthetic availability test, fresh estimator and map.
        launch('auto',binary,'auto');spin(12,lambda:pub.get_subscription_count()==1)
        grid=np.arange(-3.,3.01,.25,dtype=np.float32)
        a,b=np.meshgrid(grid,grid)
        xyz=np.vstack([np.column_stack([a.ravel(),b.ravel(),np.full(a.size,-1.)]),
                       np.column_stack([np.full(a.size,4.),a.ravel(),b.ravel()]),
                       np.column_stack([a.ravel(),np.full(a.size,4.),b.ravel()])]).astype('<f4')
        def scan(t):
            set_clock(t+.005)
            m=PointCloud2();m.header=stamp(t);m.header.frame_id='lidar'
            m.height,m.width=1,len(xyz);m.point_step=12;m.row_step=len(xyz)*12;m.is_dense=True
            m.fields=[PointField(name=k,offset=i*4,datatype=PointField.FLOAT32,count=1) for i,k in enumerate('xyz')]
            m.data=xyz.tobytes();count=len(metrics['auto']);pub.publish(m)
            spin(5,lambda:len(metrics['auto'])>count);return metrics['auto'][-1]['imu']
        def feed(begin,end):
            set_clock(end+.001)
            for ns in range(int(round(begin*100)),int(round(end*100))+1):
                m=Imu();m.header=stamp(ns/100.);m.header.frame_id='imu'
                m.linear_acceleration.z=9.81;m.orientation_covariance[0]=-1.
                imu_pub.publish(m);spin(.003)
            spin(.05)
        modes=[]
        modes.append(scan(2000.));feed(2000.01,2001.2)
        modes.append(scan(2001.2)) # initial interval start unavailable -> CV
        feed(2001.21,2001.3);modes.append(scan(2001.3))
        assert modes[-1]['mode']=='imu',modes
        feed(2001.31,2001.4);modes.append(scan(2001.4))
        modes.append(scan(2001.5));assert modes[-1]['mode']=='constant_velocity'
        # Resume from the last LiDAR timestamp; hysteresis needs 3 covered scans.
        feed(2001.5,2001.6);modes.append(scan(2001.6))
        feed(2001.61,2001.7);modes.append(scan(2001.7))
        feed(2001.71,2001.8);modes.append(scan(2001.8))
        assert modes[-1]['mode']=='imu',modes
        # Malformed inputs are refused even though a publisher is present.
        set_clock(2001.91)
        for bad in ['wrong_frame','missing_acceleration','nan','future','duplicate']:
            m=Imu();m.header=stamp(2001.9);m.header.frame_id='imu';m.linear_acceleration.z=9.81
            if bad=='wrong_frame':m.header.frame_id='unknown'
            if bad=='missing_acceleration':m.linear_acceleration_covariance[0]=-1.
            if bad=='nan':m.angular_velocity.x=float('nan')
            if bad=='future':m.header=stamp(2050.);m.header.frame_id='imu'
            if bad=='duplicate':m.header=stamp(2001.8);m.header.frame_id='imu'
            imu_pub.publish(m);spin(.03)
        modes.append(scan(2001.9));assert modes[-1]['rejected']>=5,modes[-1]
        spin(.2);values=array('auto')
        assert len(values)==len(modes)
        maximum_step=float(np.max(np.linalg.norm(np.diff(values[:,:3],axis=0),axis=1)))
        assert maximum_step<.01,maximum_step
        assert np.isfinite(values).all()
        result['synthetic_switching']=dict(modes=modes,maximum_stationary_position_step_m=maximum_step,
            qualification='Synthetic IMU tests transport/fallback only; not real-sensor accuracy evidence.')
        result['passed']=True
        print(json.dumps(result,indent=2),flush=True)
    finally:
        stop_all()
        for f in handles:f.close()
        (out/'ros_validation.json').write_text(json.dumps(result,indent=2))
        n.destroy_node();rclpy.shutdown()


if __name__=='__main__':main()
