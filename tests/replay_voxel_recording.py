"""Compare two binaries on identical recorded normalized clouds, without UE control."""
import argparse
import json
import os
from pathlib import Path
import signal
import sqlite3
import subprocess
import time
import numpy as np
import rclpy
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import PointCloud2
from nav_msgs.msg import Odometry
from rosgraph_msgs.msg import Clock
from std_msgs.msg import String
import yaml


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ['baseline','candidate','bag','profile','output']:parser.add_argument('--'+key,type=Path,required=True)
    args=parser.parse_args()
    assert os.environ['ROS_DOMAIN_ID']=='76' and os.environ['ROS_LOCALHOST_ONLY']=='1'
    args.output.mkdir(parents=True,exist_ok=False)
    meta=yaml.safe_load((args.bag/'metadata.yaml').read_text())['rosbag2_bagfile_information']
    clouds=[]
    for part in meta['relative_file_paths']:
        with sqlite3.connect('file:'+str(args.bag/part)+'?mode=ro',uri=True) as db:
            for (data,) in db.execute("SELECT m.data FROM messages m JOIN topics t ON t.id=m.topic_id WHERE t.name='/fusion/lidar' ORDER BY m.timestamp,m.id"):
                clouds.append(deserialize_message(data,PointCloud2))
    if not clouds:raise ValueError('No normalized /fusion/lidar clouds in bag')
    profile=yaml.safe_load(args.profile.read_text())
    base={key:profile[key] for key in ['instantaneous_cloud','cloud_motion_compensated','imu_mode','imu_calibration_confirmed'] if key in profile}
    for key in ['base_from_lidar','base_from_imu']:base[key]=np.asarray(profile[key]).reshape(-1).tolist()
    base.update(profile.get('voxelmap',{}))
    base.update(use_sim_time=True,visual_recovery_enabled=False,threads=2)
    rclpy.init();node=rclpy.create_node('isolated_voxel_recording_comparison')
    cloud_pub=node.create_publisher(PointCloud2,'/fusion/lidar',3)
    clock_pub=node.create_publisher(Clock,'/clock',10)
    metrics=[];poses=[]
    node.create_subscription(String,'/fusion/lidar_metrics',lambda m:metrics.append(json.loads(m.data)),100)
    node.create_subscription(Odometry,'/fusion/lio_raw',poses.append,100)
    def spin(seconds,predicate=None):
        until=time.monotonic()+seconds
        while time.monotonic()<until:
            rclpy.spin_once(node,timeout_sec=.005)
            if predicate and predicate():return
        if predicate:raise RuntimeError('Timed out waiting for isolated replay')
    results={}
    try:
        for label,binary,iterations in [('baseline_20',args.baseline,20),('baseline_30',args.baseline,30),('fixed_30',args.candidate,30)]:
            metrics.clear();poses.clear();case=args.output/label;case.mkdir()
            params=dict(base,max_iterations=iterations,timing_path=str(case/'lidar_metrics.jsonl'))
            path=case/'parameters.yaml';path.write_text(yaml.safe_dump({'/**':{'ros__parameters':params}}))
            with (case/'backend.log').open('w') as log:
                process=subprocess.Popen([str(binary),'--ros-args','--params-file',str(path)],stdout=log,stderr=subprocess.STDOUT)
                try:
                    spin(10.,lambda:cloud_pub.get_subscription_count()==1 and node.count_publishers('/fusion/lio_raw')==1)
                    for cloud in clouds:
                        clock_pub.publish(Clock(clock=cloud.header.stamp));spin(.01)
                        before=len(metrics);before_pose=len(poses);cloud_pub.publish(cloud)
                        spin(10.,lambda:len(metrics)>before and len(poses)>before_pose)
                        assert poses[-1].header.stamp==cloud.header.stamp
                    valid=[m for m in metrics if m['valid_update']]
                    times=[m['processing_sec'] for m in metrics]
                    results[label]=dict(frames=len(metrics),valid=len(valid),median_sec=float(np.median(times)),
                        p95_sec=float(np.percentile(times,95)),max_iterations_used=max(m['iterations'] for m in metrics),
                        max_valid_step_m=float(np.linalg.norm(np.diff([m['position'] for m in valid],axis=0),axis=1).max()) if len(valid)>1 else 0.,
                        peak_rss_mib=max(m['peak_rss_mib'] for m in metrics),metrics=list(metrics))
                    print(json.dumps({label:{k:v for k,v in results[label].items() if k!='metrics'}}),flush=True)
                finally:
                    if process.poll() is None:
                        process.send_signal(signal.SIGINT)
                        try:process.wait(timeout=10)
                        except subprocess.TimeoutExpired:process.kill();process.wait()
            spin(.4)
        results['scope']='Recorded normalized clouds, fresh LiDAR map each case, no visual seed/vehicle/control/ground truth; not ATE.'
        (args.output/'comparison.json').write_text(json.dumps(results,indent=2)+'\n')
    finally:
        node.destroy_node();rclpy.try_shutdown()


if __name__=='__main__':main()
