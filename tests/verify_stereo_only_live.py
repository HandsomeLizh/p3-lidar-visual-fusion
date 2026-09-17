"""Passive live source/interface check. No publishers, goals or vehicle commands."""
from collections import Counter
import json
from pathlib import Path
import time
import numpy as np
import rclpy
from rclpy.qos import QoSProfile, DurabilityPolicy, qos_profile_sensor_data
from grid_map_msgs.msg import GridMap
from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import String

ROOT=Path(__file__).resolve().parents[1]


def main():
    state=json.loads((ROOT/'run_state.json').read_text());live=Path(state['output'])
    before=json.loads((live/'map_statistics.json').read_text())
    rclpy.init(domain_id=57);node=rclpy.create_node('stereo_only_passive_verifier')
    counts=Counter();stamps={};grids={};poses=[];health=[];sources=[];cloud_sizes=[]
    retained=QoSProfile(depth=1,durability=DurabilityPolicy.TRANSIENT_LOCAL)
    def stamp(message):return message.header.stamp.sec+message.header.stamp.nanosec*1e-9
    def observe(name,message):
        counts[name]+=1;stamps.setdefault(name,[]).append(stamp(message))
    def grid(name,message):
        observe(name,message)
        if 'elevation' not in message.layers:return
        layer=message.data[message.layers.index('elevation')]
        z=np.asarray(layer.data);valid=z[np.isfinite(z)]
        grids[name]=dict(frame=message.header.frame_id,layers=list(message.layers),
            length_m=[message.info.length_x,message.info.length_y],resolution_m=message.info.resolution,
            known_cells=len(valid),height_p05_p50_p95=np.percentile(valid,[5,50,95]).tolist() if len(valid) else [])
    def pose(message):
        observe('pose',message);p=message.pose.pose.position;poses.append([p.x,p.y,p.z])
    def cloud(message):
        observe('map_cloud',message);cloud_sizes.append(message.width*message.height)
    subscriptions=[
        node.create_subscription(Odometry,'/T3/semantic/current_pose',pose,qos_profile_sensor_data),
        node.create_subscription(PointCloud2,'/T3/mapping/lidar_map',cloud,retained),
        node.create_subscription(PointCloud2,'/fusion/stereo_points',lambda m:observe('stereo',m),2),
        node.create_subscription(PointCloud2,'/fusion/lidar',lambda m:observe('lidar_input',m),2),
        node.create_subscription(Odometry,'/fusion/lio_raw',lambda m:observe('lidar_pose',m),10),
        node.create_subscription(Odometry,'/fusion/learned_raw',lambda m:observe('visual_pose',m),10),
        node.create_subscription(String,'/fusion/status',lambda m:health.append(json.loads(m.data)),10),
    ]
    for name,topic in [('local','/Car/T3/mapping/grid_map'),('global','/Car/T3/mapping/global_grid_map')]:
        subscriptions.append(node.create_subscription(GridMap,topic,lambda m,k=name:grid(k,m),retained))
    started=time.monotonic()
    try:
        while time.monotonic()-started<35.:rclpy.spin_once(node,timeout_sec=.02)
        mapper_topics={topic:[i.node_name for i in node.get_subscriptions_info_by_topic(topic)
            if i.node_name=='fusion_terrain_mapper'] for topic in ['/fusion/lidar','/fusion/stereo_points']}
        after=json.loads((live/'map_statistics.json').read_text())
        checks=dict(only_stereo=after.get('mapping_sources')==['stereo'] and after['lidar_scans']==0,
            source_subscription_exclusive=not mapper_topics['/fusion/lidar'] and bool(mapper_topics['/fusion/stereo_points']),
            mapping_advances=after['stereo_scans']>before['stereo_scans'] and after['revision']>before['revision'],
            localization_and_clouds=all(counts[k]>=3 for k in ['pose','stereo','lidar_input','lidar_pose','visual_pose','map_cloud']),
            height_only_maps=all(grids.get(k,{}).get('layers')==['elevation','elevation_variance','observation_count'] and
                grids[k]['known_cells']>0 for k in ['local','global']),
            local_32m=grids.get('local',{}).get('length_m')==[32.,32.])
        report=dict(passed=all(checks.values()),checks=checks,scope='Passive stationary/live smoke test, not moving accuracy',
            output=str(live),elapsed_sec=time.monotonic()-started,vehicle_commands_published=0,
            counts=dict(counts),mapper_subscriptions=mapper_topics,grids=grids,
            stereo_scans_before=before['stereo_scans'],stereo_scans_after=after['stereo_scans'],
            revision_before=before['revision'],revision_after=after['revision'],
            mapping_sources=after['mapping_sources'],lidar_scans=after['lidar_scans'],
            map_cloud_points_min_max=[min(cloud_sizes),max(cloud_sizes)] if cloud_sizes else [],
            pose_span_xyz_m=np.ptp(poses,axis=0).tolist() if poses else [],
            health_samples=len(health),localization_valid_fraction=float(np.mean([h.get('localization_valid',False) for h in health])) if health else None,
            operating_modes=dict(Counter(h.get('operating_mode') for h in health)),
            last_health=health[-1] if health else None,last_mapping=after,
            intervals={k:dict(median_sec=float(np.median(np.diff(v))),max_sec=float(np.max(np.diff(v))))
                for k,v in stamps.items() if len(v)>1})
        (live/'stereo_only_verification.json').write_text(json.dumps(report,indent=2))
        print(json.dumps(report,indent=2),flush=True)
        assert report['passed'],checks
    finally:node.destroy_node();rclpy.try_shutdown()


if __name__=='__main__':main()
