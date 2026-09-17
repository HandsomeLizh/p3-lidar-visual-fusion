"""Recorded images/clouds and estimator outputs through the real EKF/guard/map.

This is an isolated estimator-integration replay, not a vehicle or latency test.
The learned poses and LiDAR outputs must come from separate recorded replays.
"""
import argparse,copy,importlib.util,json,os,signal,sqlite3,subprocess,time
from pathlib import Path
import numpy as np
import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import Image,PointCloud2
from nav_msgs.msg import Odometry
from rosgraph_msgs.msg import Clock
from std_msgs.msg import String
from scipy.spatial.transform import Rotation
import yaml
from t3_lidar_visual_fusion.adaptive_guard import AdaptiveGuard
from t3_lidar_visual_fusion.terrain_mapper import TerrainMapper

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['workspace','run','estimates','visual','profile','out']:
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--rate',type=float,default=1.)
    a=p.parse_args()
    assert os.environ.get('ROS_DOMAIN_ID')=='87' and os.environ.get('ROS_LOCALHOST_ONLY')=='1'
    a.out.mkdir(parents=True,exist_ok=False)
    cfg=yaml.safe_load(a.profile.read_text())
    profile=a.out/'profile.yaml';profile.write_text(yaml.safe_dump(cfg))
    spec=importlib.util.spec_from_file_location('fixture_generator',a.workspace/'scripts/generate_config.py')
    gen=importlib.util.module_from_spec(spec);spec.loader.exec_module(gen);gen.generate(profile,a.out/'config')
    records=json.loads((a.estimates/'poses.json').read_text())
    quality=json.loads((a.estimates/'quality.json').read_text())
    visual=json.loads(a.visual.read_text())
    dbpath=next((a.run/'live_input_bag').glob('*.db3'))
    db=sqlite3.connect(f'file:{dbpath}?mode=ro',uri=True)
    ids={name:i for i,name in db.execute('select id,name from topics')}
    image_index={}
    for topic_id,data in db.execute('select topic_id,data from messages where topic_id in (?,?) order by timestamp',
                                   (ids['/fusion/left'],ids['/fusion/right'])):
        m=deserialize_message(data,Image);ns=m.header.stamp.sec*10**9+m.header.stamp.nanosec
        image_index[(ns,topic_id)]=m
    clouds=db.execute('select data from messages where topic_id=? order by timestamp',(ids[cfg['lidar_topic']],))
    rclpy.init(args=['--ros-args','-p','profile_path:='+str(profile),'-p','output_dir:='+str(a.out),'-p','use_sim_time:=true'])
    guard=AdaptiveGuard();mapper=TerrainMapper();n=rclpy.create_node('recorded_fusion_fixture')
    ex=SingleThreadedExecutor()
    for node in [guard,mapper,n]:ex.add_node(node)
    pubs={name:n.create_publisher(kind,name,10) for name,kind in [('/fusion/lio_raw',Odometry),
        ('/fusion/learned_raw',Odometry),('/fusion/lidar_quality',String),('/fusion/lidar',PointCloud2),
        ('/fusion/left',Image),('/fusion/right',Image),('/clock',Clock)]}
    output=[];states=[]
    close_events=[];original_close=guard.close_vision
    def close(reason):
        close_events.append(dict(stamp=guard.get_clock().now().nanoseconds/1e9,reason=reason))
        return original_close(reason)
    guard.close_vision=close
    def receive(m):
        s=m.header.stamp.sec+m.header.stamp.nanosec/1e9;v=m.pose.pose.position
        output.append(dict(stamp=s,xyz=[v.x,v.y,v.z]))
    n.create_subscription(Odometry,'/T3/semantic/current_pose',receive,100)
    def spin(seconds):
        until=time.monotonic()+seconds
        while time.monotonic()<until:ex.spin_once(timeout_sec=.002)
    def pose(header,xyz,q,cov,frame='odom'):
        m=Odometry();m.header=copy.deepcopy(header);m.header.frame_id=frame;m.child_frame_id='base_link'
        m.pose.pose.position.x,m.pose.pose.position.y,m.pose.pose.position.z=map(float,xyz)
        m.pose.pose.orientation.x,m.pose.pose.orientation.y,m.pose.pose.orientation.z,m.pose.pose.orientation.w=map(float,q)
        m.pose.covariance=list(map(float,cov));return m
    def clock(stamp):
        m=Clock();m.clock.sec,m.clock.nanosec=divmod(round(stamp*1e9),10**9);pubs['/clock'].publish(m)
    log=(a.out/'ekf.log').open('w');process=None
    try:
        process=subprocess.Popen(['ros2','run','robot_localization','ekf_node','--ros-args',
            '--params-file',str(a.out/'config/ekf.yaml'),'-p','use_sim_time:=true',
            '-r','__node:=ekf_filter_node','-r','odometry/filtered:=/fusion/ekf'],
            stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        spin(1.5)
        wall_start=time.monotonic();first_stamp=records[0]['stamp']
        for i,(r,q) in enumerate(zip(records,quality)):
            cloud=deserialize_message(next(clouds)[0],PointCloud2);cloud.header.frame_id='lidar'
            t=cloud.header.stamp.sec+cloud.header.stamp.nanosec/1e9
            ns=cloud.header.stamp.sec*10**9+cloud.header.stamp.nanosec
            # The guard deliberately measures source arrival time in wall time.
            # Preserve recording cadence instead of letting CPU contention in
            # an accelerated fixture manufacture frontend timeouts.
            spin(max(0.,wall_start+(t-first_stamp)/a.rate-time.monotonic()))
            clock(t);spin(.025)
            for topic in ['/fusion/left','/fusion/right']:
                message=image_index.get((ns,ids[topic]))
                if message is not None:pubs[topic].publish(message)
            deadline=time.monotonic()+1.
            while not all(hist and abs(hist[-1][0]-t)<.01 for hist in guard.quality) and time.monotonic()<deadline:
                ex.spin_once(timeout_sec=.002)
            pubs['/fusion/lidar'].publish(cloud);spin(.015)
            pubs['/fusion/lidar_quality'].publish(String(data=json.dumps(q)))
            pubs['/fusion/lio_raw'].publish(pose(cloud.header,r['xyz'],r['quaternion'],r['covariance']))
            v=min(visual,key=lambda x:abs(x['stamp']-t))
            if abs(v['stamp']-t)<.01 and v.get('pose') is not None:
                m=np.asarray(v['pose']);pubs['/fusion/learned_raw'].publish(pose(cloud.header,m[:3,3],
                    Rotation.from_matrix(m[:3,:3]).as_quat(),v['covariance'],v['epoch']))
            spin(.06);clock(t+.1);spin(.08)
            states.append(dict(frame=i+1,stamp=t,qualified=guard.output_qualified,source=guard.output_source,
                               reason=guard.filter_quality.get('reason'),visual=guard.visual_usable,lidar=guard.lidar_usable,
                               visual_reason=guard.gate.reason))
            if i%50==0:print('fusion replay',i+1,states[-1],flush=True)
        # Drain the real mapper's bounded pose-settling queue without extra data.
        spin(2.2);clock(t+.3);spin(.3)
        unique={round(o['stamp'],5):o for o in output};poses=list(unique.values())
        tail=np.asarray([o['xyz'] for o in poses if o['stamp']>=records[-12]['stamp']])
        tail_step=float(np.linalg.norm(np.diff(tail,axis=0),axis=1).max()) if len(tail)>1 else float('inf')
        result=dict(frames=len(states),formal_stamps=len(poses),final=states[-1],tail_max_step_m=tail_step,
                    visual_usable_frames=sum(s['visual'] for s in states),
                    lidar_unavailable_frames=sum(not s['lidar'] for s in states),
                    visual_outputs_during_lidar_failure=sum(s['qualified'] and s['source']=='visual' and not s['lidar'] for s in states),
                    discontinuity_frames=[s['frame'] for s in states if s['reason']=='output_discontinuity'],
                    scope='Recorded estimator outputs, actual image/cloud inputs, real EKF/guard/mapper; isolated domain 87')
        (a.out/'report.json').write_text(json.dumps(result,indent=2));(a.out/'states.json').write_text(json.dumps(states))
        (a.out/'visual_close_events.json').write_text(json.dumps(close_events,indent=2))
        (a.out/'formal_poses.json').write_text(json.dumps(poses))
        print(json.dumps(result,indent=2),flush=True)
        assert states[-1]['qualified'],result
        assert not result['discontinuity_frames'],result
        assert len(tail)>=6 and tail_step<.15,result
        assert poses[-1]['stamp']>=records[-3]['stamp'],result
    finally:
        if process is not None and process.poll() is None:
            os.killpg(process.pid,signal.SIGINT);process.wait(timeout=15)
        log.close();db.close();ex.shutdown()
        for item in ['dense_writer','grid','cloud_grid','delivery_cloud_grid','tum']:
            resource=getattr(mapper,item,None)
            if resource is not None and hasattr(resource,'close'):resource.close()
        for node in [guard,mapper,n]:node.destroy_node()
        rclpy.shutdown()

if __name__=='__main__':main()
