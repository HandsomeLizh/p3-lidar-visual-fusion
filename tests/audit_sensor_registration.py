"""Read-only, same-stamp LiDAR/stereo comparison in base_link before mapping."""
from collections import deque
import json
import os
from pathlib import Path
import time
import numpy as np
import rclpy
from rclpy.qos import QoSProfile, qos_profile_sensor_data
from scipy.spatial import cKDTree
from sensor_msgs.msg import PointCloud2
from nav_msgs.msg import Odometry
import yaml
from t3_lidar_visual_fusion.ros_utils import cloud_arrays, stamp_sec

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/elevation_fusion_validation_20260917'/time.strftime('sensor_alignment_%H%M%S')
OUT.mkdir(exist_ok=False)
state=json.loads((ROOT/'visual_only_live_state.json').read_text())
cfg=yaml.safe_load((Path(state['output'])/'profile.yaml').read_text())
mount=np.asarray(cfg['base_from_lidar'])
os.environ['ROS_LOCALHOST_ONLY']='0'
rclpy.init(domain_id=57);node=rclpy.create_node('p3_sensor_alignment_audit')
clouds={'lidar':deque(maxlen=10),'stereo':deque(maxlen=10)}
feedback=[];frames=[];used=set();matches=[]
for name in clouds:
    node.create_subscription(PointCloud2,'/fusion/'+('lidar' if name=='lidar' else 'stereo_points'),
                             lambda msg,k=name:clouds[k].append(msg),QoSProfile(depth=2))
node.create_subscription(Odometry,'/P4/input/odometry',lambda m:feedback.append(
    [m.pose.pose.position.x,m.pose.pose.position.y,m.pose.pose.position.z]),qos_profile_sensor_data)
deadline=time.monotonic()+24.
try:
    while time.monotonic()<deadline and len(frames)<10:
        rclpy.spin_once(node,timeout_sec=.01)
        for stereo in list(clouds['stereo']):
            stamp=stamp_sec(stereo)
            if stamp in used or not clouds['lidar']:continue
            lidar=min(clouds['lidar'],key=lambda m:abs(stamp_sec(m)-stamp))
            if abs(stamp_sec(lidar)-stamp)>.001:continue
            if lidar.header.frame_id!='lidar' or stereo.header.frame_id!='base_link':
                raise ValueError('Unexpected sensor frames: '+str((lidar.header.frame_id,stereo.header.frame_id)))
            used.add(stamp)
            laser=cloud_arrays(lidar);laser=laser[np.isfinite(laser).all(axis=1)]
            laser=laser@mount[:3,:3].T+mount[:3,3]
            vision=cloud_arrays(stereo);vision=vision[np.isfinite(vision).all(axis=1)]
            laser=laser[(laser[:,0]>.5)&(laser[:,0]<10)&(abs(laser[:,1])<5)]
            tree=cKDTree(laser[:,:2]);dist,indices=tree.query(vision[:,:2],k=20)
            rows=[]
            for point,dd,ii in zip(vision,dist,indices):
                valid=(dd<.18)&(ii<len(laser))
                if valid.sum()<8:continue
                neighbors=laser[ii[valid]];center=neighbors.mean(axis=0)
                _,singular,vt=np.linalg.svd(neighbors-center,full_matrices=False)
                normal=vt[-1];normal*=1 if normal[2]>0 else -1
                # Compare interior, planar ground patches, excluding discontinuities.
                if normal[2]<.95 or singular[1]/np.sqrt(len(neighbors))<.018 or singular[2]/np.sqrt(len(neighbors))>.012:continue
                rows.append([*point,float((point-center)@normal),len(neighbors)])
            rows=np.asarray(rows,dtype=float).reshape(-1,5)
            frames.append(dict(stamp=stamp,stamp_skew_sec=abs(stamp_sec(lidar)-stamp),
                lidar_points=len(laser),stereo_points=len(vision),planar_matches=len(rows)))
            if len(rows):matches.append(rows)
            if len(frames)==1:np.savez_compressed(OUT/'first_pair.npz',lidar_base=laser,stereo_base=vision,planar_matches=rows)
    if not matches:raise RuntimeError('No observed common planar patches')
    all_rows=np.concatenate(matches);xyz=all_rows[:,:3];residual=all_rows[:,3]
    # Only diagnose the observable height/tilt components. Do not change extrinsics.
    design=np.column_stack([np.ones(len(xyz)),xyz[:,:2]])
    keep=np.abs(residual)<.3
    fit=np.zeros(3)
    for _ in range(6):
        fit=np.linalg.lstsq(design[keep],residual[keep],rcond=None)[0]
        errors=residual-design@fit
        median=np.median(errors[keep]);mad=max(.005,1.4826*np.median(abs(errors[keep]-median)))
        keep=(abs(errors-median)<2.5*mad)&(abs(residual)<.3)
    errors=residual-design@fit
    def stats(values):return dict(count=len(values),median_m=float(np.median(values)),
        abs_p95_m=float(np.percentile(abs(values),95)),rms_m=float(np.sqrt(np.mean(values**2))))
    motion_span=float(np.linalg.norm(np.ptp(np.asarray(feedback),axis=0))) if feedback else None
    report=dict(scope='Same published timestamp, base_link before odometry/map/height bias; not a full extrinsic calibration. Check reference motion before interpreting as stationary.',
        frames=frames,frame_count=len(frames),raw_point_to_lidar_plane=stats(residual),
        robust_subset=stats(residual[keep]),after_diagnostic_height_tilt_fit=stats(errors[keep]),
        fit=dict(height_intercept_m=float(fit[0]),x_slope=float(fit[1]),y_slope=float(fit[2])),
        forward_range_m=[float(xyz[:,0].min()),float(xyz[:,0].max())],
        lateral_range_m=[float(xyz[:,1].min()),float(xyz[:,1].max())],
        reference_motion_span_m=motion_span,
        stationary_translation_observed=(motion_span<=.01) if motion_span is not None else None,
        stationary_rotation_checked=False,
        calibration_confirmed=cfg.get('calibration_confirmed',False),extrinsics_modified=False)
    report['forward_bins']=[dict(range_m=[low,high],**stats(residual[(xyz[:,0]>=low)&(xyz[:,0]<high)]))
        for low,high in [(1,2),(2,3),(3,4),(4,5),(5,8)] if ((xyz[:,0]>=low)&(xyz[:,0]<high)).any()]
    np.savez_compressed(OUT/'planar_residuals.npz',observations=all_rows,robust_mask=keep)
    (OUT/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
finally:
    node.destroy_node();rclpy.shutdown()
