"""Fault injection: raw spinning scans must resume after an IMU gap.

Synthetic stationary room + IMU, isolated ROS domain 77. This is not an ATE test.
"""
from array import array
import json,os,signal,subprocess,time
from pathlib import Path
import numpy as np,rclpy
from sensor_msgs.msg import PointCloud2,PointField,Imu
from std_msgs.msg import Header,String
from rosgraph_msgs.msg import Clock

ROOT=Path(__file__).resolve().parents[1]

def main():
    assert os.environ['ROS_DOMAIN_ID']=='77'
    out=ROOT/'results/hardware104_deployment';out.mkdir(parents=True,exist_ok=True)
    rclpy.init();n=rclpy.create_node('raw_scan_dropout_fixture')
    clouds=n.create_publisher(PointCloud2,'/fusion/lidar',2)
    imus=n.create_publisher(Imu,'/fusion/imu',400)
    clock=n.create_publisher(Clock,'/clock',10);metrics=[];deskewed=[];statuses=[]
    n.create_subscription(String,'/fusion/lidar_metrics',lambda m:metrics.append(json.loads(m.data)),100)
    n.create_subscription(String,'/fusion/imu_status',lambda m:statuses.append(json.loads(m.data)),100)
    n.create_subscription(PointCloud2,'/fusion/lidar_deskewed',deskewed.append,3)
    def spin(seconds=.03):
        end=time.monotonic()+seconds
        while time.monotonic()<end:rclpy.spin_once(n,timeout_sec=.002)
    def stamp(t):
        h=Header();h.stamp.sec,h.stamp.nanosec=divmod(round(t*1e9),10**9);return h.stamp
    def set_clock(t):clock.publish(Clock(clock=stamp(t)));spin()
    def imu(start,end):
        set_clock(end+.01)
        for i,t in enumerate(np.arange(start,end+.0001,.01)):
            m=Imu();m.header=Header(stamp=stamp(t),frame_id='imu');m.linear_acceleration.z=9.81
            m.orientation_covariance[0]=-1.;imus.publish(m)
            if i%10==0:spin(.006)
        spin(.06)
    a,b=np.meshgrid(np.arange(-3.,3.01,.2),np.arange(-3.,3.01,.2))
    xyz=np.vstack([np.c_[a.ravel(),b.ravel(),np.full(a.size,-1.)],
                   np.c_[np.full(a.size,4.),a.ravel(),b.ravel()],
                   np.c_[a.ravel(),np.full(a.size,4.),b.ravel()]])
    points=np.c_[xyz,np.linspace(0.,.1,len(xyz))].astype('<f4')
    def scan(t):
        set_clock(t+.11)
        m=PointCloud2(header=Header(stamp=stamp(t),frame_id='lidar'),height=1,width=len(points),
            fields=[PointField(name=name,offset=4*i,datatype=PointField.FLOAT32,count=1)
                    for i,name in enumerate(('x','y','z','time'))],point_step=16,row_step=len(points)*16,
            is_dense=True,data=array('B',points.tobytes()))
        clouds.publish(m);spin(.35)
    log=(out/'deskew_dropout.log').open('w');proc=None
    try:
        args=[str(ROOT/'install/t3_voxelmap/lib/t3_voxelmap/voxelmap_node'),'--ros-args']
        for value in ('use_sim_time:=true','instantaneous_cloud:=false','deskew_enabled:=true',
            'imu_mode:=auto','imu_calibration_confirmed:=true','voxel_size:=2.0','downsample_size:=0.2','threads:=1'):
            args+=['-p',value]
        proc=subprocess.Popen(args,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        deadline=time.monotonic()+8
        while clouds.get_subscription_count()!=1 and time.monotonic()<deadline:spin(.05)
        assert clouds.get_subscription_count()==1
        imu(999.,1001.12);scan(1001.)
        imu(1001.13,1001.52);scan(1001.4)
        assert len(metrics)>=2 and metrics[-1]['valid_update']
        before=len(metrics);before_cloud=len(deskewed)
        scan(1001.8)
        assert len(metrics)==before and len(deskewed)==before_cloud,'uncompensated scan escaped'
        for t in (1002.2,1002.6,1003.,1003.4,1003.8):
            imu(t-.28,t+.12);scan(t)
        assert len(metrics)>before and metrics[-1]['valid_update'],statuses[-10:]
        assert metrics[-1]['imu']['mode']=='imu'
        assert np.linalg.norm(metrics[-1]['position'])<.01
        assert not any(m.get('visual_seed_used',False) for m in metrics)
        result=dict(passed=True,raw_scan_blocked_during_gap=True,resumed_imu_and_deskew=True,
                    frames_before_gap=before,frames_after_gap=len(metrics)-before,
                    no_visual_seed_override=True,synthetic_fault_injection=True)
        (out/'deskew_dropout.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
    finally:
        if proc and proc.poll() is None:os.kill(proc.pid,signal.SIGINT);proc.wait(timeout=15)
        log.close();n.destroy_node();rclpy.shutdown()

if __name__=='__main__':main()
