#!/usr/bin/env python3
"""Run the real mapper on parked camera/LiDAR records; no moving accuracy claim."""
import argparse,json,os,time
from pathlib import Path
import cv2,numpy as np,rclpy,yaml
from nav_msgs.msg import Odometry
from std_msgs.msg import Header,String
from t3_lidar_visual_fusion.learned_matching import LearnedMatcher
from t3_lidar_visual_fusion.learned_tracker import LearnedStereoTracker
from t3_lidar_visual_fusion.stereo_mapping import sparse_map_points
from t3_lidar_visual_fusion.terrain_mapper import TerrainMapper
from t3_lidar_visual_fusion.ros_utils import xyz_cloud


def stats(values):
    a=np.asarray(values,dtype=float)
    return dict(median=float(np.median(a)),p95=float(np.percentile(a,95))) if len(a) else None


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root',type=Path,required=True);ap.add_argument('--data',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
    assert os.environ.get('ROS_DOMAIN_ID')=='98' and os.environ.get('ROS_LOCALHOST_ONLY')=='1'
    args.output.mkdir(parents=True,exist_ok=False)
    cfg=yaml.safe_load((args.data/'profile.yaml').read_text())
    cfg['learned_visual'].update(device='cpu',cpu_threads=2);cv2.setNumThreads(1)
    tracker=LearnedStereoTracker(cfg,LearnedMatcher(args.root,cfg['learned_visual']))
    frames=[]
    for frame in json.loads((args.data/'capture.json').read_text())['frames']:
        i=frame['index'];stamp=frame['image_stamp'];cv2.setRNGSeed(0)
        result=tracker.process(stamp,cv2.imread(str(args.data/f'{i:02d}_left.png'),0),cv2.imread(str(args.data/f'{i:02d}_right.png'),0))
        xyz,var=(np.empty((0,3)),np.empty(0)) if result.anchor else sparse_map_points(tracker.geometry,result.map_points,cfg['stereo_mapping'])
        frames.append((stamp,np.load(args.data/f'{i:02d}_lidar.npz')['xyz'],xyz,var))
        print(json.dumps(dict(frame=i,stereo_points=len(xyz))),flush=True)
    del tracker
    reports={};plots={}
    for mode in ('stereo','lidar','hybrid'):
        out=args.output/mode;out.mkdir()
        cfg=yaml.safe_load((args.data/'profile.yaml').read_text())
        cfg.update(map_window=64.,mapping_source='stereo' if mode=='stereo' else 'range',
                   instantaneous_cloud=True,deskew={'enabled':False},mapping_lidar_topic='/fusion/lidar',
                   mapping_pose_settle_sec=0.,map_publish_period=1000.,global_publish_period=1000.)
        cfg['stereo_mapping'].update(enabled=mode!='lidar',preserve_fill_xyz=mode=='hybrid')
        cfg['mapping_camera_view']=dict(enabled=mode!='stereo',full_density_range_m=10.,range_feather_m=5.,image_feather_fraction=.15)
        p=out/'profile.yaml';p.write_text(yaml.safe_dump(cfg))
        rclpy.init(args=['--ros-args','-p','profile_path:='+str(p),'-p','output_dir:='+str(out)])
        mapper=TerrainMapper();times={'lidar':[],'stereo':[]}
        def header(t,frame):
            h=Header(frame_id=frame);h.stamp.sec,h.stamp.nanosec=divmod(round(t*1e9),10**9);return h
        try:
            for stamp,lidar,stereo,var in frames:
                pose=Odometry();pose.header=header(stamp,'odom');pose.child_frame_id='base_link';pose.pose.pose.orientation.w=1.
                pose.pose.covariance=(np.eye(6)*1e-5).ravel().tolist();mapper.odom(pose)
                if mode!='stereo':
                    cloud=xyz_cloud(lidar,header(stamp,'sensor_frame'))
                    begin=time.perf_counter();mapper.enqueue(cloud,'lidar',cfg['base_from_lidar']);mapper.process()
                    times['lidar'].append(time.perf_counter()-begin)
                if mode!='lidar' and len(stereo):
                    mapper.fusion_status(String(data=json.dumps(dict(vision_enabled=True,localization_valid=True))))
                    cloud=xyz_cloud(stereo,header(stamp,'camera_left_optical'),var,'position_variance')
                    begin=time.perf_counter();mapper.enqueue_stereo(cloud);mapper.process()
                    times['stereo'].append(time.perf_counter()-begin)
            mapper.publish();mapper.publish_global();mapper.save();mapper.report_status()
            points=np.array(mapper.cloud.connection.execute('SELECT x,y,z FROM voxels').fetchall()).reshape(-1,3)
            fill=mapper.stereo_fill.preview() if mapper.stereo_fill is not None else np.empty((0,3))
            grid=mapper.grid.extract_window(center_x=0.,center_y=0.,length_x=64.,length_y=64.)
            rows,cols=np.where(np.isfinite(grid.elevation))
            terrain=np.c_[grid.geometry.origin_x+(cols+.5)*.2,grid.geometry.origin_y+(rows+.5)*.2,grid.elevation[rows,cols]]
            np.savez_compressed(out/'geometry.npz',cloud=points,stereo_fill=fill,terrain=terrain)
            reports[mode]=dict(grid_cells=len(terrain),saved_cloud_points=len(points),stereo_fill_points=len(fill),
                process_sec={k:stats(v) for k,v in times.items()},local_publish_sec=mapper.stats.get('last_local_publish_sec'),
                stats=mapper.stats)
            plots[mode]=(points,fill,terrain)
            print(json.dumps(dict(mode=mode,grid_cells=len(terrain),saved_cloud_points=len(points),stereo_fill_points=len(fill))),flush=True)
        finally:
            mapper.dense_writer.close();mapper.grid.close();mapper.delivery.close();mapper.cloud.close();mapper.tum.close()
            if mapper.stereo_fill is not None:mapper.stereo_fill.close()
            mapper.destroy_node();rclpy.try_shutdown()
    lidar_cells={tuple(k) for k in np.floor(plots['lidar'][2][:,:2]/.2).astype(int)}
    hybrid_cells={tuple(k) for k in np.floor(plots['hybrid'][2][:,:2]/.2).astype(int)}
    fill_cells={tuple(k) for k in np.floor(plots['hybrid'][1][:,:2]/.2).astype(int)}
    assert fill_cells and not (fill_cells&lidar_cells)
    assert lidar_cells.issubset(hybrid_cells)
    report=dict(scope='16 parked captures, host-receipt pairing; fixed identity pose with test covariance. Not moving accuracy, latency, or hardware sync validation.',
                frames=len(frames),added_grid_cells_over_lidar=len(hybrid_cells-lidar_cells),
                duplicate_lidar_fill_cells=len(fill_cells&lidar_cells),modes=reports)
    (args.output/'report.json').write_text(json.dumps(report,indent=2))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,3,figsize=(14,8),layout='constrained')
    for col,mode in enumerate(('stereo','lidar','hybrid')):
        cloud,fill,terrain=plots[mode];combined=np.r_[cloud,fill]
        ax=axes[0,col];ax.scatter(combined[:,0],combined[:,1],c=combined[:,2],s=1,vmin=-.6,vmax=.3,cmap='viridis')
        ax.set(xlim=(-1,16),ylim=(-8,8),title=f'{mode}: {len(combined)} saved 3D points',xlabel='Base X [m]',ylabel='Base Y [m]');ax.set_aspect('equal')
        ax=axes[1,col];ax.scatter(terrain[:,0],terrain[:,1],s=12,c='#777777',marker='s')
        if len(fill):ax.scatter(fill[:,0],fill[:,1],s=14,c='#007acc',marker='s',label='Stereo fill');ax.legend()
        ax.set(xlim=(0,5),ylim=(-2.5,2.5),title=f'{len(terrain)} observed 0.2 m cells',xlabel='Base X [m]',ylabel='Base Y [m]');ax.set_aspect('equal')
    fig.suptitle('104 parked real data | calibrated transforms, no fitted alignment | blue = stereo-only terrain fill')
    fig.savefig(args.output/'comparison.png',dpi=150);plt.close(fig)
    print(json.dumps({k:v for k,v in report.items() if k!='modes'},indent=2),flush=True)


if __name__=='__main__':main()
