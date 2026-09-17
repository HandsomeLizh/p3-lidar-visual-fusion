"""Replay a stopped live bag through the complete pipeline in localhost domain 90."""
import argparse,copy,json,os,signal,sqlite3,subprocess,time
from pathlib import Path
import numpy as np
import rclpy
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import Image,PointCloud2
from nav_msgs.msg import Odometry
from rosgraph_msgs.msg import Clock
from std_msgs.msg import String
import yaml


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['workspace','run','out']:parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--frames',type=int,default=30)
    parser.add_argument('--assert-recovery',action='store_true')
    parser.add_argument('--profile',type=Path)
    parser.add_argument('--blackout-seconds',type=float,default=0.)
    args=parser.parse_args()
    assert os.environ.get('ROS_DOMAIN_ID')=='90' and os.environ.get('ROS_LOCALHOST_ONLY')=='1'
    args.workspace=args.workspace.resolve();args.run=args.run.resolve();args.out=args.out.resolve()
    # The host login profile may provide a DDS participant limit for its own
    # application. Use this workspace's bounded configuration for all test nodes.
    os.environ['CYCLONEDDS_URI']='file://'+str(args.workspace/'config/cyclonedds.xml')
    os.environ['ROS_LOG_DIR']=str(args.out/'ros_logs')
    state=json.loads((args.workspace/'run_state.json').read_text())
    assert args.run!=Path(state['output']).resolve(),'Use a stopped bag; do not read the active recorder'
    args.out.mkdir(parents=True,exist_ok=False)
    assert 0.<=args.blackout_seconds<=110.
    cfg=yaml.safe_load((args.profile or args.run/'profile.yaml').read_text())
    profile=args.out/'profile.yaml';profile.write_text(yaml.safe_dump(cfg))
    kinds={'/fusion/left':Image,'/fusion/right':Image,cfg['lidar_topic']:PointCloud2}
    samples={}
    database=next((args.run/'live_input_bag').glob('*.db3'))
    with sqlite3.connect('file:'+str(database)+'?mode=ro',uri=True) as db:
        topics=dict(db.execute('select id,name from topics'))
        for topic,data in db.execute('select topic_id,data from messages order by timestamp'):
            name=topics[topic]
            if name not in kinds:continue
            msg=deserialize_message(data,kinds[name]);stamp=msg.header.stamp.sec+msg.header.stamp.nanosec/1e9
            samples.setdefault(stamp,{})[name]=msg
            if sum(len(v)==3 for v in samples.values())>=args.frames:break
    frames=sorted((stamp,messages) for stamp,messages in samples.items() if len(messages)==3)[:args.frames]
    assert len(frames)==args.frames
    rclpy.init();node=rclpy.create_node('isolated_full_startup_replay')
    pubs={name:node.create_publisher(kind,name,3) for name,kind in kinds.items()}
    clock=node.create_publisher(Clock,'/clock',3)
    outputs=[];origins=[];states=[];visuals=[]
    def pose(msg):
        v=msg.pose.pose.position;outputs.append({'stamp':msg.header.stamp.sec+msg.header.stamp.nanosec/1e9,
                                               'xyz':[v.x,v.y,v.z]})
    node.create_subscription(Odometry,'/T3/semantic/current_pose',pose,100)
    node.create_subscription(Odometry,'/fusion/visual_epoch_origin',origins.append,10)
    node.create_subscription(Odometry,'/fusion/learned_raw',
        lambda m:visuals.append(max(m.pose.covariance)<1e5),100)
    node.create_subscription(String,'/fusion/status',lambda m:states.append(json.loads(m.data)),100)
    def tick(stamp):
        msg=Clock();msg.clock.sec,msg.clock.nanosec=divmod(round(stamp*1e9),10**9);clock.publish(msg)
    def spin(duration,stamp,limit=None):
        begin=time.monotonic()
        while time.monotonic()-begin<duration:
            elapsed=time.monotonic()-begin;tick(min(stamp+elapsed,limit) if limit is not None else stamp+elapsed)
            rclpy.spin_once(node,timeout_sec=.02)
    process=None;log=(args.out/'pipeline.log').open('w')
    try:
        process=subprocess.Popen(['ros2','launch',str(args.workspace/'scripts/fusion.launch.py'),
            'profile:='+str(profile),'output_dir:='+str(args.out),'use_sim_time:=true'],
            stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        deadline=time.monotonic()+40
        while time.monotonic()<deadline:
            tick(frames[0][0]);rclpy.spin_once(node,timeout_sec=.05)
            if process.poll() is not None:raise RuntimeError('Pipeline exited during startup')
            if (pubs['/fusion/left'].get_subscription_count()>=2 and
                    pubs['/fusion/right'].get_subscription_count()>=2 and
                    pubs[cfg['lidar_topic']].get_subscription_count()>=1 and
                    (args.out/'learned_metrics.json').exists() and
                    (args.out/'map_statistics.json').exists()):break
        else:raise RuntimeError('Pipeline readiness timed out')
        started=time.monotonic()
        offset=0.;blackout_interval=None
        for i,(original_stamp,messages) in enumerate(frames):
            if args.blackout_seconds and i==5:
                blackout_interval=(frames[i-1][0]+1.,original_stamp+args.blackout_seconds)
                # Only the isolated test domain sees these deliberately black
                # images and a simultaneous LiDAR gap. Sensor time advances;
                # no live capture or vehicle topic is touched.
                for dark_stamp in np.linspace(*blackout_interval,10,endpoint=False):
                    tick(dark_stamp)
                    for name in ['/fusion/left','/fusion/right']:
                        dark=copy.deepcopy(messages[name])
                        dark.header.stamp.sec,dark.header.stamp.nanosec=divmod(round(dark_stamp*1e9),10**9)
                        dark.data=bytes(len(dark.data));pubs[name].publish(dark)
                    spin(1.5,dark_stamp,dark_stamp+1.)
                offset=args.blackout_seconds
            stamp=original_stamp+offset
            tick(stamp)
            for name in ['/fusion/left','/fusion/right',cfg['lidar_topic']]:
                message=copy.deepcopy(messages[name])
                message.header.stamp.sec,message.header.stamp.nanosec=divmod(round(stamp*1e9),10**9)
                pubs[name].publish(message)
            gap=frames[i+1][0]-original_stamp if i+1<len(frames) else 1.6
            spin(max(1.5,gap),stamp,stamp+gap-.01)
            if i%10==0:print('REPLAY',i+1,'FORMAL_OUTPUTS',len(outputs),'ORIGINS',len(origins),flush=True)
            if process.poll() is not None:raise RuntimeError('Pipeline exited while replaying')
        spin(2.,frames[-1][0]+offset+1.6,frames[-1][0]+offset+3.)
        unique={round(v['stamp'],5):v for v in outputs}
        xyz=np.asarray([v['xyz'] for v in unique.values()])
        mapped=json.loads((args.out/'map_statistics.json').read_text())
        status=json.loads((args.out/'fusion_status.json').read_text())
        report={'input_frames':len(frames),'visual_tracked':int(sum(visuals)),'origin_events':len(origins),
                'formal_unique_stamps':len(unique),'mapped_scans':mapped['mapped_scans'],
                'map_revision':mapped['revision'],'last_output_source':status['output_source'],
                'last_localization_valid':status['localization_valid'],
                'last_visual_continuity':status['visual_continuity'],
                'output_discontinuity_reports':sum(v.get('filtered_quality',{}).get('reason')=='output_discontinuity' for v in states),
                'maximum_output_step_m':float(np.linalg.norm(np.diff(xyz,axis=0),axis=1).max()) if len(xyz)>1 else None,
                'wall_seconds':time.monotonic()-started,
                'scope':'Stopped bag replay; no online driving, ground truth, or ATE'}
        if blackout_interval:
            metrics=[json.loads(line) for line in (args.out/'learned_metrics.jsonl').read_text().splitlines()]
            report.update(blackout_sensor_seconds=args.blackout_seconds,
                dark_rejections=sum(v.get('reason')=='underexposed' for v in metrics),
                recovery_confirmations=max((v.get('recovery_confirmations',0) for v in metrics),default=0),
                formal_outputs_during_blackout=sum(blackout_interval[0]<=v['stamp']<blackout_interval[1] for v in outputs),
                formal_outputs_after_blackout=sum(v['stamp']>=blackout_interval[1] for v in outputs))
        (args.out/'report.json').write_text(json.dumps(report,indent=2))
        (args.out/'formal_poses.json').write_text(json.dumps(outputs))
        print(json.dumps(report,indent=2),flush=True)
        if args.assert_recovery:
            allowance=3 if blackout_interval else 0
            assert report['origin_events']>=1 and report['visual_tracked']>=len(frames)-2-allowance,report
            assert report['formal_unique_stamps']>=len(frames)-5-allowance,report
            assert report['mapped_scans']>=len(frames)-6-allowance,report
            assert report['last_localization_valid'] and report['output_discontinuity_reports']==0,report
            assert report['maximum_output_step_m']<.15,report
            if blackout_interval:
                assert report['origin_events']==1 and report['dark_rejections']>=8,report
                assert report['recovery_confirmations']==2 and report['formal_outputs_during_blackout']==0,report
                assert report['formal_outputs_after_blackout']>=len(frames)-9,report
    finally:
        if process is not None and process.poll() is None:
            os.kill(process.pid,signal.SIGINT)
            try:process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid,signal.SIGTERM);process.wait(timeout=10)
        log.close();node.destroy_node();rclpy.shutdown()


if __name__=='__main__':main()
