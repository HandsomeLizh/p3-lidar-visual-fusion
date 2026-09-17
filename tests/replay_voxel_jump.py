"""Replay recorded LiDAR into one isolated estimator; never publish to the live domain."""
import argparse
import copy
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
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--binary', type=Path, required=True)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--assert-convergence', action='store_true')
    p.add_argument('--visual-records', type=Path)
    p.add_argument('--profile', type=Path)
    p.add_argument('--max-frames', type=int, default=0)
    a = p.parse_args()
    if os.environ.get('ROS_DOMAIN_ID') != '84' or os.environ.get('ROS_LOCALHOST_ONLY') != '1':
        raise RuntimeError('This fixture requires isolated ROS domain 84 on localhost')
    a.out.mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load((a.profile or a.run/'profile.yaml').read_text())
    params = dict(cfg['voxelmap'], use_sim_time=True,
                  base_from_lidar=np.asarray(cfg['base_from_lidar']).reshape(-1).tolist(),
                  base_from_imu=np.asarray(cfg['base_from_imu']).reshape(-1).tolist(),
                  imu_mode=str(cfg['imu_mode']), imu_calibration_confirmed=False,
                  instantaneous_cloud=cfg['instantaneous_cloud'],
                  timing_path=str(a.out/'lidar_metrics.jsonl'))
    visual_records=json.loads(a.visual_records.read_text()) if a.visual_records else []
    if visual_records:params.update(independent_visual_topic='/fusion/learned_raw',independent_visual_wait_sec=.45)
    paramfile = a.out/'parameters.yaml'
    paramfile.write_text(yaml.safe_dump({'/**': {'ros__parameters': params}}))
    rclpy.init()
    n = rclpy.create_node('isolated_recorded_jump_replay')
    cloud = n.create_publisher(PointCloud2, '/fusion/lidar', 2)
    visual = n.create_publisher(Odometry, '/fusion/learned_raw', 2)
    clock = n.create_publisher(Clock, '/clock', 2)
    metrics, poses, quality = [], [], []
    n.create_subscription(String, '/fusion/lidar_metrics', lambda m: metrics.append(json.loads(m.data)), 100)
    n.create_subscription(String, '/fusion/lidar_quality', lambda m: quality.append(json.loads(m.data)), 100)
    def pose(m):
        v=m.pose.pose
        poses.append(dict(stamp=m.header.stamp.sec+m.header.stamp.nanosec/1e9,
                          xyz=[v.position.x,v.position.y,v.position.z],
                          quaternion=[v.orientation.x,v.orientation.y,v.orientation.z,v.orientation.w],
                          covariance=m.pose.covariance.tolist()))
    n.create_subscription(Odometry, '/fusion/lio_raw', pose, 100)
    def spin_until(predicate, timeout):
        end=time.monotonic()+timeout
        while not predicate() and time.monotonic()<end:
            rclpy.spin_once(n,timeout_sec=.01)
        if not predicate():raise RuntimeError('Replay estimator timed out')
    process=None
    log=(a.out/'node.log').open('w')
    try:
        process=subprocess.Popen([str(a.binary), '--ros-args', '--params-file', str(paramfile)],
                                 stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        spin_until(lambda:cloud.get_subscription_count()==1,15.)
        dbpath=next((a.run/'live_input_bag').glob('*.db3'))
        db=sqlite3.connect(f'file:{dbpath}?mode=ro',uri=True)
        topic=db.execute('select id from topics where name=?',(cfg['lidar_topic'],)).fetchone()[0]
        t0=time.monotonic()
        for index,(data,) in enumerate(db.execute('select data from messages where topic_id=? order by timestamp',(topic,))):
            if a.max_frames and index>=a.max_frames:break
            m=deserialize_message(data,PointCloud2)
            m.header.frame_id='lidar'
            clock.publish(Clock(clock=copy.deepcopy(m.header.stamp)))
            before=len(metrics)
            cloud.publish(m)
            if visual_records:
                from scipy.spatial.transform import Rotation
                stamp=m.header.stamp.sec+m.header.stamp.nanosec/1e9
                record=min(visual_records,key=lambda v:abs(v['stamp']-stamp))
                if abs(record['stamp']-stamp)<.01 and record.get('pose') is not None:
                    v=Odometry();v.header=copy.deepcopy(m.header);v.header.frame_id=record['epoch'];v.child_frame_id='base_link'
                    matrix=np.asarray(record['pose']);v.pose.pose.position.x,v.pose.pose.position.y,v.pose.pose.position.z=map(float,matrix[:3,3])
                    q=Rotation.from_matrix(matrix[:3,:3]).as_quat()
                    v.pose.pose.orientation.x,v.pose.pose.orientation.y,v.pose.pose.orientation.z,v.pose.pose.orientation.w=map(float,q)
                    v.pose.covariance=record['covariance'];visual.publish(v)
            spin_until(lambda:len(metrics)>before,20.)
            spin_until(lambda:len(quality)>=len(metrics) and len(poses)>=len(metrics),5.)
            if index%15==0:print('replayed',index+1,'valid',metrics[-1]['valid_update'],flush=True)
        db.close()
        report=dict(frames=len(metrics),wall_seconds=time.monotonic()-t0,
                    accepted=sum(r['valid_update'] for r in metrics),
                    nonconverged_accepted=[r['frame'] for r in metrics if r['valid_update'] and not r['solver_converged']],
                    rejected=[{k:r[k] for k in ['frame','stamp_sec','tracking_reason','map_keyframes','position']}
                              for r in metrics if not r['valid_update']],
                    last_frames=[{k:r[k] for k in ['frame','stamp_sec','valid_update','solver_converged','map_keyframes','position','last_translation_correction_m']}
                                 for r in metrics[-12:]])
        xyz=np.asarray([r['xyz'] for r in poses]);step=np.linalg.norm(np.diff(xyz,axis=0),axis=1)
        report['maximum_published_step_m']=float(step.max())
        report['last_12_maximum_published_step_m']=float(step[-12:].max())
        (a.out/'report.json').write_text(json.dumps(report,indent=2))
        (a.out/'poses.json').write_text(json.dumps(poses))
        (a.out/'quality.json').write_text(json.dumps(quality,indent=2))
        if a.assert_convergence:
            assert not report['nonconverged_accepted'],report
            for i,r in enumerate(metrics):
                if not r['valid_update']:
                    assert not quality[i]['reliable'],quality[i]
                    assert max(poses[i]['covariance'][::7])>=1e6,poses[i]
                    if i:assert r['map_keyframes']==metrics[i-1]['map_keyframes'],r
            # This recorded tail is a small in-place turn followed by unchanged
            # scans. The original 0.9 m out-and-back must not survive the repair.
            assert report['last_12_maximum_published_step_m']<.15,report
            assert metrics[-1]['valid_update'],'LiDAR did not recover on subsequent scans'
        print(json.dumps(report,indent=2),flush=True)
    finally:
        if process is not None and process.poll() is None:
            os.killpg(process.pid,signal.SIGINT)
            process.wait(timeout=15)
        log.close();n.destroy_node();rclpy.shutdown()


if __name__=='__main__':main()
