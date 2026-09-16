"""Compare real-IMU deskew binaries on identical recorded inputs; never drive.

ROS domain 84, loopback only. Reports consistency and compute time, not ATE.
"""
import argparse,json,os,signal,sqlite3,subprocess,time
from pathlib import Path
import numpy as np
import rclpy,yaml
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import Imu,PointCloud2
from nav_msgs.msg import Odometry
from std_msgs.msg import String
from rosgraph_msgs.msg import Clock


def stamp(msg):return msg.header.stamp.sec+msg.header.stamp.nanosec*1e-9


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('baseline','candidate','bag','profile','output'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--max-scans',type=int,default=75)
    p.add_argument('--candidate-threads',type=int,default=2)
    p.add_argument('--require-equivalent',action='store_true');args=p.parse_args()
    assert os.environ['ROS_DOMAIN_ID']=='84' and os.environ['ROS_LOCALHOST_ONLY']=='1'
    args.output.mkdir(parents=True,exist_ok=False)
    meta=yaml.safe_load((args.bag/'metadata.yaml').read_text())['rosbag2_bagfile_information']
    clouds=[];imus=[]
    for part in meta['relative_file_paths']:
        with sqlite3.connect('file:'+str(args.bag/part)+'?mode=ro',uri=True) as db:
            for topic,data in db.execute("SELECT t.name,m.data FROM messages m JOIN topics t ON t.id=m.topic_id WHERE t.name IN ('/fusion/lidar','/fusion/imu') ORDER BY m.timestamp,m.id"):
                if topic.endswith('imu'):imus.append(deserialize_message(data,Imu))
                else:clouds.append(deserialize_message(data,PointCloud2))
    imus.sort(key=stamp);clouds.sort(key=stamp)
    if not imus or not clouds:raise ValueError('Need normalized raw scans and real IMU in the bag')
    clouds=[c for c in clouds if stamp(c)>=stamp(imus[0])+2. and stamp(c)+.13<stamp(imus[-1])][:args.max_scans]
    if len(clouds)<10:raise ValueError('Not enough scans after IMU warm-up')
    cfg=yaml.safe_load(args.profile.read_text())
    base=dict(cfg['voxelmap'],use_sim_time=True,instantaneous_cloud=False,deskew_enabled=True,
        cloud_motion_compensated=False,imu_mode=cfg['imu_mode'],imu_calibration_confirmed=True,
        max_scan_duration=cfg['max_scan_duration'],visual_recovery_enabled=True,max_iterations=20,threads=2)
    for key in ('base_from_lidar','base_from_imu'):base[key]=np.asarray(cfg[key]).ravel().tolist()
    rclpy.init();node=rclpy.create_node('hardware_recovery_comparison')
    cp=node.create_publisher(PointCloud2,'/fusion/lidar',2)
    ip=node.create_publisher(Imu,'/fusion/imu',400)
    clock=node.create_publisher(Clock,'/clock',10);metrics=[];deskew=[];pose_stamps=[];pose_values=[]
    node.create_subscription(String,'/fusion/lidar_metrics',lambda m:metrics.append(json.loads(m.data)),100)
    node.create_subscription(PointCloud2,'/fusion/lidar_deskewed',lambda m:deskew.append(stamp(m)),3)
    def odometry(message):
        pose_stamps.append(stamp(message));p=message.pose.pose.position;q=message.pose.pose.orientation
        pose_values.append([p.x,p.y,p.z,q.x,q.y,q.z,q.w,*message.pose.covariance])
    node.create_subscription(Odometry,'/fusion/lio_raw',odometry,10)
    def spin(seconds,predicate=None):
        until=time.monotonic()+seconds
        while time.monotonic()<until:
            rclpy.spin_once(node,timeout_sec=.003)
            if predicate and predicate():return
        if predicate:raise RuntimeError('Replay timed out waiting for data')
    def set_clock(t):
        msg=Clock();msg.clock.sec,msg.clock.nanosec=divmod(round(t*1e9),10**9);clock.publish(msg)
    report={}
    try:
        for label,binary in [('baseline',args.baseline),('candidate',args.candidate)]:
            metrics.clear();deskew.clear();pose_stamps.clear();pose_values.clear();case=args.output/label;case.mkdir()
            parameters=case/'parameters.yaml'
            parameters.write_text(yaml.safe_dump({'/**':{'ros__parameters':dict(base,
                threads=args.candidate_threads if label=='candidate' else 2,timing_path=str(case/'metrics.jsonl'))}}))
            with (case/'backend.log').open('w') as log:
                proc=subprocess.Popen([str(binary),'--ros-args','--params-file',str(parameters)],stdout=log,stderr=subprocess.STDOUT)
                try:
                    spin(10.,lambda:cp.get_subscription_count()==1 and ip.get_subscription_count()==1)
                    i=0
                    for n,cloud in enumerate(clouds):
                        t=stamp(cloud);set_clock(t+.13);spin(.02)
                        while i<len(imus) and stamp(imus[i])<=t+.12:
                            ip.publish(imus[i]);i+=1
                            if i%10==0:spin(.008)
                        spin(.05);before=len(metrics);before_deskew=len(deskew);before_pose=len(pose_stamps);cp.publish(cloud)
                        spin(10.,lambda:len(metrics)>before and len(deskew)>before_deskew and len(pose_stamps)>before_pose)
                        assert metrics[-1]['imu']['mode']=='imu',metrics[-1]
                        assert not metrics[-1]['visual_seed_used']
                        # Compare actual ROS stamps; diagnostic JSON has fewer significant digits.
                        assert abs(pose_stamps[-1]-deskew[-1])<1e-7
                        if (n+1)%25==0:print(label,n+1,'/',len(clouds),flush=True)
                    valid=[m for m in metrics if m['valid_update']]
                    np.savez_compressed(case/'odometry.npz',stamp=pose_stamps,pose_covariance=pose_values)
                    positions=np.asarray([m['position'] for m in valid]);times=[m['processing_sec'] for m in metrics]
                    report[label]=dict(frames=len(metrics),valid=len(valid),imu_frames=sum(m['imu']['mode']=='imu' for m in metrics),
                        median_sec=float(np.median(times)),p95_sec=float(np.percentile(times,95)),
                        max_valid_step_m=float(np.linalg.norm(np.diff(positions,axis=0),axis=1).max()) if len(valid)>1 else None,
                        first_last_displacement_m=float(np.linalg.norm(positions[-1]-positions[0])) if len(valid)>1 else None,
                        max_iterations_used=max(m['iterations'] for m in metrics),peak_rss_mib=max(m['peak_rss_mib'] for m in metrics),
                        visual_seed_overrides=sum(m['visual_seed_used'] for m in metrics),metrics=list(metrics))
                    print(json.dumps({label:{k:v for k,v in report[label].items() if k!='metrics'}}),flush=True)
                finally:
                    if proc.poll() is None:
                        proc.send_signal(signal.SIGINT)
                        try:proc.wait(timeout=10)
                        except subprocess.TimeoutExpired:proc.kill();proc.wait()
            spin(.5)
        report['scope']='Same recorded actual rover cloud/IMU, independent replay with 20 iterations and fresh map per version; no motion truth, no vehicle commands, not ATE.'
        report['passed']=(report['candidate']['valid']>=report['baseline']['valid'] and
            report['candidate']['valid']==len(clouds) and report['candidate']['visual_seed_overrides']==0)
        if args.require_equivalent:
            with np.load(args.output/'baseline/odometry.npz') as a,np.load(args.output/'candidate/odometry.npz') as b:
                np.testing.assert_array_equal(a['stamp'],b['stamp'])
                np.testing.assert_allclose(a['pose_covariance'],b['pose_covariance'],atol=1e-10,rtol=1e-10)
                report['maximum_pose_covariance_difference']=float(np.abs(a['pose_covariance']-b['pose_covariance']).max())
            assert [m['iterations'] for m in report['baseline']['metrics']]==[m['iterations'] for m in report['candidate']['metrics']]
        (args.output/'comparison.json').write_text(json.dumps(report,indent=2)+'\n')
    finally:node.destroy_node();rclpy.try_shutdown()
    if not report['passed']:raise SystemExit(1)


if __name__=='__main__':main()
