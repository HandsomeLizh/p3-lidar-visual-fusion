#!/usr/bin/env python3
"""Bounded passive real-sensor integration check in a separate ROS domain.

Never sends vehicle commands. Only processes started by this test are stopped.
An input driver already running in the sensor domain is reused.
"""
import argparse, collections, hashlib, json, os, signal, subprocess, time
from pathlib import Path
import numpy as np
import rclpy
from rclpy.context import Context
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image,Imu,PointCloud2
from nav_msgs.msg import Odometry
from std_msgs.msg import String
from grid_map_msgs.msg import GridMap

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--seconds',type=float,default=65.);parser.add_argument('--output',required=True)
    parser.add_argument('--require-stereo-map',action='store_true')
    parser.add_argument('--profile',type=Path,default=ROOT/'config/hardware104.yaml')
    parser.add_argument('--max-output-gap-sec',type=float,default=0.,help='Optional continuity bound after warmup; zero only reports it')
    parser.add_argument('--warmup-sec',type=float,default=30.)
    args=parser.parse_args();out=Path(args.output).resolve();out.mkdir(parents=True,exist_ok=False)
    domain=int(os.environ['ROS_DOMAIN_ID'])
    if domain in (10,19,57,59):raise ValueError('Choose a separate test domain, such as 73')
    source_context=Context();rclpy.init(context=source_context,domain_id=19)
    source=Node('p3_camera_precheck',context=source_context)
    source_executor=SingleThreadedExecutor(context=source_context);source_executor.add_node(source)
    deadline=time.monotonic()+2
    while time.monotonic()<deadline:source_executor.spin_once(timeout_sec=.1)
    cameras=[source.count_publishers(t)>0 for t in ['/Car/T5/Cam_Left/image_mono/mapping','/Car/T5/Cam_Right/image_mono/mapping']]
    legacy=any(source.count_publishers(t) for t in ['/Car/T5/Cam_Left/image_raw/color','/Car/T5/Cam_Right/image_raw/color'])
    source_executor.shutdown();source.destroy_node();source_context.shutdown()
    if any(cameras) and not all(cameras):raise RuntimeError('Only one camera publisher exists')
    if not any(cameras) and legacy:raise RuntimeError('Existing camera driver lacks compact mapping topics')
    camera_present=all(cameras)
    rclpy.init();node=Node('p3_hardware_live_observer');counts=collections.Counter();last={};poses=[];ages=collections.defaultdict(list);received_at={};first_received={}
    covariances=collections.defaultdict(list);pose_sources=collections.Counter();map_timings=collections.defaultdict(list)
    gap_samples=collections.defaultdict(lambda:collections.deque(maxlen=4000));max_gaps=collections.defaultdict(float)
    gap_violations=collections.Counter();last_stamp={};stamp_regressions=collections.Counter()
    regression_events=collections.defaultdict(list);publishers_seen=collections.defaultdict(dict)
    publisher_counts=collections.defaultdict(int);topic_by_key={}
    map_history=[];started=time.monotonic();observe_from=started+args.warmup_sec
    required_outputs=('deskewed','fused','gridmap','global_gridmap')
    def status(msg,key):
        counts[key]+=1
        try:
            last[key]=json.loads(msg.data)
            if key=='fusion':pose_sources[last[key].get('output_source','unknown')]+=1
            if key=='mapping':
                if len(map_history)<4000:
                    map_history.append(dict(elapsed_sec=time.monotonic()-started,
                        **{k:last[key].get(k) for k in ('map_age_sec','pending','range_pending','stereo_pending','vmrss_mib','vmhwm_mib','dropped_scans','mapped_scans')}))
                for field in ('last_map_update_sec','last_local_publish_sec','last_global_publish_sec','last_checkpoint_sec'):
                    value=last[key].get(field)
                    if isinstance(value,(int,float)) and len(map_timings[field])<4000:map_timings[field].append(value)
                for prefix,field in (('stage_','map_stage_sec'),('queue_','queue_wait_sec_by_source')):
                    for stage,value in last[key].get(field,{}).items():
                        if len(map_timings[prefix+stage])<4000:map_timings[prefix+stage].append(value)
        except ValueError:last[key]={'invalid_json':True}
    def publisher_snapshot(key):
        endpoints=node.get_publishers_info_by_topic(topic_by_key[key])
        publisher_counts[key]=max(publisher_counts[key],len(endpoints))
        current={bytes(e.endpoint_gid).hex():dict(node=e.node_name,namespace=e.node_namespace) for e in endpoints}
        publishers_seen[key].update(current)
        return current
    def receive(msg,key):
        counts[key]+=1
        now=time.monotonic();previous=received_at.get(key)
        if now>=observe_from:
            gap=now-max(previous if previous is not None else observe_from,observe_from);gap_samples[key].append(gap)
            max_gaps[key]=max(max_gaps[key],gap)
            if args.max_output_gap_sec>0 and gap>args.max_output_gap_sec:gap_violations[key]+=1
        received_at[key]=now
        first_received.setdefault(key,received_at[key])
        t=msg.header.stamp.sec+msg.header.stamp.nanosec*1e-9
        if t<last_stamp.get(key,-float('inf'))-1e-6:
            stamp_regressions[key]+=1
            if len(regression_events[key])<32:
                regression_events[key].append(dict(elapsed_sec=now-started,previous=last_stamp[key],
                    current=t,backward_sec=last_stamp[key]-t,publishers=publisher_snapshot(key)))
        last_stamp[key]=t
        if len(ages[key])<4000:ages[key].append(time.time()-t)
        if key=='fused':
            p=msg.pose.pose.position;poses.append([t,p.x,p.y,p.z])
        if isinstance(msg,Odometry) and len(covariances[key])<4000:
            covariance=np.asarray(msg.pose.covariance).reshape(6,6)
            if np.isfinite(covariance).all():
                covariances[key].append([float(np.linalg.eigvalsh(covariance[:3,:3])[-1]),
                                        float(np.linalg.eigvalsh(covariance[3:,3:])[-1])])
    for key,typ,topic in [('imu',Imu,'/fusion/imu'),('cloud',PointCloud2,'/fusion/lidar'),
            ('deskewed',PointCloud2,'/fusion/lidar_deskewed'),('left',Image,'/fusion/left'),('right',Image,'/fusion/right'),
            ('fused',Odometry,'/T3/semantic/current_pose'),('visual_pose',Odometry,'/fusion/learned_raw'),
            ('lidar_pose',Odometry,'/fusion/lio_raw'),('ekf_pose',Odometry,'/fusion/ekf')]:
        topic_by_key[key]=topic
        def callback(k):
            def observed(m):receive(m,k)
            return observed
        node.create_subscription(typ,topic,callback(key),qos_profile_sensor_data)
    node.create_subscription(GridMap,'/Car/T3/mapping/grid_map',callback('gridmap'),qos_profile_sensor_data)
    node.create_subscription(GridMap,'/Car/T3/mapping/global_grid_map',callback('global_gridmap'),qos_profile_sensor_data)
    node.create_subscription(PointCloud2,'/fusion/stereo_points',callback('stereo_points'),qos_profile_sensor_data)
    topic_by_key.update(gridmap='/Car/T3/mapping/grid_map',global_gridmap='/Car/T3/mapping/global_grid_map',
                        stereo_points='/fusion/stereo_points')
    for key,topic in [('imu_status','/fusion/imu_status'),('fusion','/fusion/status'),('hardware','/fusion/hardware_status'),('mapping','/fusion/map_status'),('learned','/fusion/learned_status')]:
        node.create_subscription(String,topic,lambda m,k=key:status(m,k),10)
    # A numerical domain is shared across the LAN, not just this computer.
    # Refuse a known occupied test domain before starting another mapper.
    deadline=time.monotonic()+2.
    while time.monotonic()<deadline:rclpy.spin_once(node,timeout_sec=.1)
    occupied={topic:node.count_publishers(topic) for topic in (
        '/T3/semantic/current_pose','/fusion/lidar_deskewed','/Car/T3/mapping/grid_map',
        '/Car/T3/mapping/global_grid_map','/fusion/ekf') if node.count_publishers(topic)}
    if occupied:
        node.destroy_node();rclpy.shutdown()
        raise RuntimeError('Test domain already contains pipeline publishers: '+str(occupied))
    processes=[]
    def launch(name,command):
        log=(out/(name+'.log')).open('w')
        proc=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        processes.append((name,proc,log));return proc
    try:
        if not camera_present:launch('camera',['bash',str(ROOT/'scripts/start_hardware_camera.sh')])
        pipeline=launch('pipeline',['ros2','launch',str(ROOT/'scripts/fusion.launch.py'),
            'profile:='+str(args.profile.resolve()),'output_dir:='+str(out)])
        started=time.monotonic();observe_from=started+args.warmup_sec
        deadline=started+args.seconds;next_progress=started+15.
        while time.monotonic()<deadline:
            rclpy.spin_once(node,timeout_sec=.05)
            if pipeline.poll() is not None:break
            now=time.monotonic()
            if now>=next_progress:
                for key in required_outputs:publisher_snapshot(key)
                progress=dict(elapsed_sec=now-started,counts=dict(counts),
                    idle_sec={k:now-v for k,v in received_at.items()},max_completed_gap_sec=dict(max_gaps),
                    mapping=last.get('mapping',{}),pipeline_running=pipeline.poll() is None,
                    stamp_regressions=dict(stamp_regressions),publisher_graph=dict(publishers_seen))
                temporary=out/'continuity_progress.json.tmp';temporary.write_text(json.dumps(progress,indent=2));temporary.replace(out/'continuity_progress.json')
                next_progress=now+15.
        lidar=[]
        if (out/'lidar_metrics.jsonl').exists():
            for line in (out/'lidar_metrics.jsonl').read_text().splitlines():
                try:lidar.append(json.loads(line))
                except ValueError:pass
        for key in required_outputs:publisher_snapshot(key)
        idle={k:time.monotonic()-t for k,t in received_at.items()}
        imu_seed_override=any(r.get('imu',{}).get('mode')=='imu' and r.get('visual_seed_used',False) for r in lidar)
        lidar_passed=(counts['deskewed']>=10 and counts['fused']>=10 and counts['gridmap']>=3 and
                idle.get('deskewed',1e9)<3. and idle.get('fused',1e9)<3. and not imu_seed_override and
                any(r.get('imu',{}).get('mode')=='imu' for r in lidar) and pipeline.poll() is None)
        visual_passed=(counts['visual_pose']>=10 and idle.get('visual_pose',1e9)<3. and
                       last.get('fusion',{}).get('visual_accepted',0)>=10)
        stereo_passed=(last.get('mapping',{}).get('stereo_scans',0)>=3 and last.get('mapping',{}).get('stereo_cells',0)>0)
        passed=lidar_passed and visual_passed and (not args.require_stereo_map or stereo_passed)
        maximum_gaps={k:max(max_gaps[k],time.monotonic()-max(received_at.get(k,observe_from),observe_from)) for k in required_outputs}
        continuity_passed=(args.seconds>args.warmup_sec and pipeline.poll() is None and
            all(k in received_at and gap_samples[k] and maximum_gaps[k]<=args.max_output_gap_sec
                and stamp_regressions[k]==0 and len(publishers_seen[k])==1 and publisher_counts[k]==1
                for k in required_outputs)) if args.max_output_gap_sec>0 else None
        if continuity_passed is not None:passed=passed and continuity_passed
        report=dict(passed=passed,lidar_imu_mapping_passed=lidar_passed,
            visual_pipeline_passed=visual_passed,stereo_mapping_passed=stereo_passed,domain=domain,counts=dict(counts),last=last,
            profile=str(args.profile.resolve()),profile_sha256=hashlib.sha256(args.profile.read_bytes()).hexdigest(),
            vehicle_commands_published=0,replay_used=False,camera_reused=camera_present,
            message_idle_sec=idle,imu_seed_overridden_by_visual=imu_seed_override,
            observed_rate_hz={k:(counts[k]-1)/(t-first_received[k]) for k,t in received_at.items() if counts[k]>1 and t>first_received[k]},
            message_age_sec={k:{'median':float(np.median(v)),'p95':float(np.quantile(v,.95))} for k,v in ages.items() if v},
            covariance_max_eigenvalue={k:dict(samples=len(v),position_m2_median=float(np.median(np.asarray(v)[:,0])),
                rotation_rad2_median=float(np.median(np.asarray(v)[:,1])),
                position_m2_p95=float(np.quantile(np.asarray(v)[:,0],.95)),
                rotation_rad2_p95=float(np.quantile(np.asarray(v)[:,1],.95))) for k,v in covariances.items() if v},
            output_source_status_counts=dict(pose_sources),
            map_timing_status_samples={k:dict(samples=len(v),median=float(np.median(v)),p95=float(np.quantile(v,.95))) for k,v in map_timings.items() if v},
            continuity=dict(passed=continuity_passed,warmup_sec=args.warmup_sec,
                max_allowed_gap_sec=args.max_output_gap_sec,observed_sec=time.monotonic()-started,
                maximum_gap_sec=maximum_gaps,stamp_regressions=dict(stamp_regressions),gap_violations=dict(gap_violations),
                stamp_regression_events=dict(regression_events),
                publisher_graph=dict(publishers_seen),maximum_graph_publisher_count=dict(publisher_counts),
                first_receipt_sec={k:v-started for k,v in first_received.items()},
                receive_gap_sec={k:dict(samples=len(v),median=float(np.median(v)),p95=float(np.quantile(v,.95)),maximum=max_gaps[k]) for k,v in gap_samples.items() if v},
                map_status_history=map_history),
            lidar_frames=len(lidar),lidar_valid=sum(r['valid_update'] for r in lidar),
            lidar_imu_scans=sum(r.get('imu',{}).get('mode')=='imu' for r in lidar),
            lidar_processing_median_sec=float(np.median([r['processing_sec'] for r in lidar])) if lidar else None,
            pose_displacement_m=float(np.linalg.norm(np.asarray(poses[-1][1:])-np.asarray(poses[0][1:]))) if poses else None,
            note='Passive interface/static sample check, not moving trajectory ATE or a no-jump guarantee')
        (out/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2),flush=True)
    finally:
        for name,proc,log in reversed(processes):
            if proc.poll() is None:
                os.kill(proc.pid,signal.SIGINT)
                try:proc.wait(timeout=25)
                except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGTERM);proc.wait(timeout=10)
            log.close()
        node.destroy_node();rclpy.shutdown()
    return 0 if passed else 1


if __name__=='__main__':raise SystemExit(main())
