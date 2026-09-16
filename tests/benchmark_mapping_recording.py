"""Offline recorded-cloud map benchmark in an isolated ROS domain; no motion."""
import argparse,cProfile,hashlib,json,os,pstats,sqlite3,time
from pathlib import Path
import numpy as np,rclpy,yaml
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import PointCloud2
from nav_msgs.msg import Odometry
from t3_lidar_visual_fusion.terrain_mapper import TerrainMapper

def main():
    p=argparse.ArgumentParser();p.add_argument('--bag',type=Path,required=True)
    p.add_argument('--profile',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--scans',type=int,default=6);a=p.parse_args()
    assert os.environ['ROS_DOMAIN_ID']=='86' and os.environ['ROS_LOCALHOST_ONLY']=='1'
    a.output.mkdir(parents=True,exist_ok=False)
    cfg=yaml.safe_load(a.profile.read_text());cfg['mapping_pose_settle_sec']=0.
    cfg['map_publish_period']=cfg['global_publish_period']=1000.
    profile=a.output/'profile.yaml';profile.write_text(yaml.safe_dump(cfg))
    messages=[]
    for part in sorted(a.bag.glob('*.db3')):
        with sqlite3.connect('file:'+str(part)+'?mode=ro',uri=True) as db:
            for row in db.execute("SELECT m.data FROM messages m JOIN topics t ON m.topic_id=t.id WHERE t.name='/fusion/lidar' ORDER BY m.timestamp LIMIT ?",(a.scans,)):
                messages.append(deserialize_message(row[0],PointCloud2))
                if len(messages)>=a.scans:break
        if len(messages)>=a.scans:break
    assert messages
    rclpy.init(args=['--ros-args','-p','profile_path:='+str(profile),'-p','output_dir:='+str(a.output/'map')])
    mapper=TerrainMapper();timings=[];profiler=cProfile.Profile()
    try:
        for i,msg in enumerate(messages):
            pose=Odometry();pose.header.stamp=msg.header.stamp;pose.header.frame_id='odom'
            pose.child_frame_id='base_link';pose.pose.pose.orientation.w=1.
            pose.pose.covariance=(np.eye(6)*.0001).ravel().tolist();mapper.odom(pose)
            mapper.enqueue(msg,'lidar',cfg['base_from_lidar'])
            if i==len(messages)-1:profiler.enable()
            start=time.perf_counter();mapper.process();timings.append(time.perf_counter()-start)
            if i==len(messages)-1:profiler.disable()
        start=time.perf_counter();mapper.publish();local_time=time.perf_counter()-start
        start=time.perf_counter();mapper.publish_global();global_time=time.perf_counter()-start
        arrays={}
        for key,tile in mapper.grid.tiles.cache.items():
            for field in ('elevation_count','elevation_mean','elevation_M2','elevation_min','elevation_max','stereo_owned'):
                arrays[str(key)+':'+field]=getattr(tile,field)
        np.savez_compressed(a.output/'layers.npz',**arrays)
        digest=hashlib.sha256()
        for row in mapper.cloud.connection.execute('SELECT ix,iy,iz,x,y,z FROM voxels ORDER BY ix,iy,iz'):
            digest.update(repr(row).encode())
        report={'timings_sec':timings,'median_unprofiled_sec':float(np.median(timings[:-1])),
            'local_publish_sec':local_time,'global_publish_sec':global_time,
            'cloud_count':mapper.cloud.count,'cloud_digest':digest.hexdigest(),'stats':mapper.stats}
        (a.output/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
        with (a.output/'profile.txt').open('w') as stream:pstats.Stats(profiler,stream=stream).sort_stats('cumtime').print_stats(35)
    finally:
        mapper.dense_writer.close(flush_revision=mapper.last_dense_revision if mapper.last_dense_revision>=0 else None)
        mapper.grid.close();mapper.delivery.close();mapper.cloud.close();mapper.tum.close();mapper.destroy_node();rclpy.shutdown()
if __name__=='__main__':main()
