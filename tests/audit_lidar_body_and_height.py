"""Read-only base-frame body candidates and scan/map height disagreement."""
from collections import deque
import json, os, time
from pathlib import Path
import numpy as np
import rclpy
from rclpy.qos import QoSProfile
from sensor_msgs.msg import PointCloud2
from nav_msgs.msg import Odometry
from grid_map_msgs.srv import GetGridMap
import yaml
from t3_lidar_visual_fusion.ros_utils import cloud_arrays, stamp_sec, transform_from_pose
from t3_lidar_visual_fusion.probabilistic_elevation import height_variances, scan_observations

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results'/time.strftime('lidar_body_height_audit_%Y%m%d_%H%M%S')
OUT.mkdir()
state=json.loads((ROOT/'visual_only_live_state.json').read_text());run=Path(state['output'])
cfg=yaml.safe_load((run/'profile.yaml').read_text());mount=np.asarray(cfg['base_from_lidar'])
os.environ['ROS_LOCALHOST_ONLY']='0';rclpy.init(domain_id=57)
node=rclpy.create_node('read_only_lidar_body_height_audit')
clouds=deque(maxlen=5);poses=deque(maxlen=30)
node.create_subscription(PointCloud2,'/fusion/lidar',clouds.append,QoSProfile(depth=3))
node.create_subscription(Odometry,'/T3/semantic/current_pose',poses.append,QoSProfile(depth=20))
client=node.create_client(GetGridMap,'/Car/T3/mapping/get_grid_map')
try:
    end=time.monotonic()+20;pair=None
    while time.monotonic()<end:
        rclpy.spin_once(node,timeout_sec=.05)
        if clouds and poses:
            cloud=clouds[-1];pose=min(poses,key=lambda p:abs(stamp_sec(p)-stamp_sec(cloud)))
            if abs(stamp_sec(pose)-stamp_sec(cloud))<.001:pair=(cloud,pose);break
    if pair is None:raise RuntimeError('No exact-stamp LiDAR/visual pose pair')
    cloud,pose=pair;transform=transform_from_pose(pose.pose.pose)
    xyz=cloud_arrays(cloud);ranges=np.linalg.norm(xyz,axis=1)
    xyz=xyz[np.isfinite(xyz).all(axis=1)&(ranges>=cfg['min_range'])&(ranges<=cfg['max_range'])]
    base=xyz@mount[:3,:3].T+mount[:3,3]
    world=base@transform[:3,:3].T+transform[:3,3]
    inside=(abs(base[:,0])<=.591)&(abs(base[:,1])<=.409)
    nearby=(abs(base[:,0])<=3)&(abs(base[:,1])<=3)
    def stats(a):
        return dict(count=len(a),quantiles=np.percentile(a,[0,5,25,50,75,95,100]).tolist()) if len(a) else dict(count=0)
    report=dict(run=str(run),stamp=stamp_sec(cloud),sensor_frame=cloud.header.frame_id,
        base_footprint_source='/home/yanfa/P4/config/wheel.yaml; planning envelope only, base_footprint to base_link offset not verified',
        reference_xy_limits_m=[[-.591,.591],[-.409,.409]],body_candidate_height=stats(base[inside,2]),
        nearby_height=stats(base[nearby,2]),current_map_pose=transform.tolist(),
        no_filter_or_vehicle_commands_applied=True)
    report['footprint_height_bands']=[dict(z_m=[low,high],count=int((inside&(base[:,2]>=low)&(base[:,2]<high)).sum()))
        for low,high in [(-1.,-.3),(-.3,-.1),(-.1,.1),(.1,.3),(.3,.6),(.6,1.),(1.,1.5)]]
    np.savez_compressed(OUT/'sensor_geometry.npz',sensor=xyz,base=base,world=world,pose=transform)
    request=GetGridMap.Request();request.frame_id='map';request.position_x=float(transform[0,3]);request.position_y=float(transform[1,3]);request.length_x=24.;request.length_y=24.;request.layers=['elevation','elevation_variance']
    if not client.wait_for_service(timeout_sec=3.):raise RuntimeError('Map query service unavailable')
    future=client.call_async(request);rclpy.spin_until_future_complete(node,future,timeout_sec=15.)
    if not future.done():raise RuntimeError('Map query timed out')
    msg=future.result().map
    if 'elevation' not in msg.layers:raise RuntimeError('Map query returned no height')
    layers={}
    for name,data in zip(msg.layers,msg.data):
        ny,nx=(int(d.size) for d in data.layout.dim)
        layers[name]=np.asarray(data.data).reshape(ny,nx)[::-1,::-1]
    covariance=np.asarray(pose.pose.covariance).reshape(6,6)
    sv,pv=height_variances(base,transform,covariance,'lidar',cfg,cfg.get('elevation_fusion',{}))
    obs=scan_observations(world,sv,pv,cfg['map_resolution'],6000,priority_center=transform[:2,3],priority_radius=8.)
    xy=(obs['cells']+.5)*cfg['map_resolution'];origin=np.array([msg.info.pose.position.x-msg.info.length_x/2,msg.info.pose.position.y-msg.info.length_y/2])
    ij=np.floor((xy-origin)/msg.info.resolution).astype(int)
    within=(ij[:,0]>=0)&(ij[:,1]>=0)&(ij[:,0]<nx)&(ij[:,1]<ny)&(obs['relief']<=.06)
    ids=np.flatnonzero(within);old=layers['elevation'][ij[ids,1],ij[ids,0]]
    valid=np.isfinite(old)&(layers['elevation_variance'][ij[ids,1],ij[ids,0]]<=.25)
    ids,old=ids[valid],old[valid];residual=obs['height'][ids]-old
    current=json.loads((run/'map_statistics.json').read_text());offset=current.get('height_bias',{}).get('offset_m',0.)
    report.update(last_mapper_status=current,raw_height_difference=stats(residual),after_current_offset=stats(residual-offset),query_map_stamp=stamp_sec(msg))
    if len(ids)>20:
        local=xy[ids]-transform[:2,3];design=np.column_stack([np.ones(len(ids)),local]);keep=np.ones(len(ids),bool)
        for _ in range(5):
            fit=np.linalg.lstsq(design[keep],residual[keep],rcond=None)[0];errors=residual-design@fit
            center=np.median(errors[keep]);mad=max(.01,1.4826*np.median(abs(errors[keep]-center)));keep=abs(errors-center)<2.5*mad
        report['diagnostic_plane_only']=dict(intercept_m=float(fit[0]),x_slope=float(fit[1]),y_slope=float(fit[2]),inliers=int(keep.sum()),residual=stats(errors[keep]))
        np.savez_compressed(OUT/'height_comparison.npz',xy=xy[ids],new_height=obs['height'][ids],old_height=old,residual=residual,mask=keep)
    (OUT/'report.json').write_text(json.dumps(report,indent=2));print('OUTPUT',OUT);print(json.dumps(report))
finally:
    node.destroy_node();rclpy.shutdown()
