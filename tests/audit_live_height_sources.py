"""Read-only paired sensor audit for a few suspect height cells; no commands."""
import json
from pathlib import Path
import time
from collections import deque
import numpy as np
import rclpy
from rclpy.qos import QoSProfile, qos_profile_sensor_data
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import PointCloud2
from nav_msgs.msg import Odometry
import yaml
from t3_lidar_visual_fusion.ros_utils import cloud_arrays, stamp_sec
from t3_lidar_visual_fusion.probabilistic_elevation import scan_observations, height_variances

ROOT=Path(__file__).resolve().parents[1]
out=ROOT/'results/elevation_fusion_validation_20260917/p4_map_audit_v2'
live=Path(json.loads((ROOT/'visual_only_live_state.json').read_text())['output'])
cfg=yaml.safe_load((live/'profile.yaml').read_text())
cells=np.floor(np.array([[-3.1,.3],[-2.7,.1],[-1.7,1.1],[-1.5,-.3],[3.3,-1.1],[1.1,-3.9]])/.2).astype(int)
rclpy.init(domain_id=57);node=rclpy.create_node('p3_height_source_audit')
poses=deque(maxlen=30);pending=deque(maxlen=16);rows=[]
node.create_subscription(Odometry,'/Car/T3/localization/odometry',poses.append,qos_profile_sensor_data)
for source,topic in [('lidar','/fusion/lidar'),('stereo','/fusion/stereo_points')]:
    node.create_subscription(PointCloud2,topic,lambda msg,s=source:pending.append((s,msg,time.monotonic())),QoSProfile(depth=2))
deadline=time.monotonic()+18.
try:
    while time.monotonic()<deadline:
        rclpy.spin_once(node,timeout_sec=.01)
        for _ in range(len(pending)):
            source,msg,received=pending.popleft();stamp=stamp_sec(msg)
            pose=min(poses,key=lambda m:abs(stamp_sec(m)-stamp)) if poses else None
            if pose is None or abs(stamp_sec(pose)-stamp)>.02:
                if time.monotonic()-received<5.:pending.append((source,msg,received))
                continue
            p=pose.pose.pose.position;q=pose.pose.pose.orientation
            transform=np.eye(4);transform[:3,:3]=Rotation.from_quat([q.x,q.y,q.z,q.w]).as_matrix()
            transform[:3,3]=[p.x,p.y,p.z]
            mount=np.array(cfg['base_from_lidar']) if source=='lidar' else np.eye(4)
            raw=cloud_arrays(msg);ranges=np.linalg.norm(raw,axis=1)
            raw=raw[np.isfinite(raw).all(axis=1)&(ranges>=cfg['min_range'])&(ranges<=cfg['max_range'])]
            base=raw@mount[:3,:3].T+mount[:3,3];points=base@transform[:3,:3].T+transform[:3,3]
            sensor,common=height_variances(base,transform,pose.pose.covariance,source,cfg,cfg['elevation_fusion'])
            obs=scan_observations(points,sensor,common,.2,cfg['elevation_fusion']['max_cells_per_scan'])
            bias=json.loads((live/'map_statistics.json').read_text())['height_bias']['offset_m']
            bins=np.floor(points[:,:2]/.2).astype(int)
            for cell in cells:
                z=points[np.all(bins==cell,axis=1),2]-bias
                selected=np.flatnonzero(np.all(obs['cells']==cell,axis=1))
                if len(z):rows.append(dict(source=source,stamp=stamp,cell=cell.tolist(),point_count=len(z),
                    z_p10_median_p90=np.percentile(z,[10,50,90]).tolist(),
                    selected_for_fusion=bool(len(selected)),
                    observation_height=float(obs['height'][selected[0]]-bias) if len(selected) else None))
    (out/'paired_sensor_height_audit.json').write_text(json.dumps(rows,indent=2))
    print(json.dumps(rows))
finally:
    node.destroy_node();rclpy.shutdown()
