#!/usr/bin/env python3
"""Side-by-side mapping from the current 104 poses; no drivers or control output."""
import argparse,json,os,time
from datetime import datetime
from pathlib import Path
import rclpy,yaml
from t3_lidar_visual_fusion.terrain_mapper import TerrainMapper

ROOT=Path(__file__).resolve().parents[1]
PREFIX='/fusion/hybrid_preview'
OUTPUTS=[
    '/T3/mapping/elevation_map','/T3/mapping/grid_map','/Car/T3/mapping/grid_map',
    '/T3/mapping/global_grid_map','/Car/T3/mapping/global_grid_map',
    '/T3/mapping/global_map_revision','/Car/T3/mapping/global_map_revision',
    '/T3/mapping/lidar_map','/T3/mapping/stereo_map','/T3/mapping/elevation_cloud',
    '/fusion/map_status','/T3/mapping/get_grid_map','/Car/T3/mapping/get_grid_map',
    '/T3/mapping/save','/T3/mapping/save_grid_map','/Car/T3/mapping/save_grid_map']


def profile(base):
    cfg=yaml.safe_load(Path(base).read_text())
    patch=yaml.safe_load((ROOT/'config/hardware104_hybrid_preview.yaml').read_text())
    for key,value in patch.items():
        if isinstance(value,dict):cfg.setdefault(key,{}).update(value)
        else:cfg[key]=value
    if cfg.get('map_output')!='elevation_only' or not cfg.get('hardware',{}).get('enabled'):
        raise ValueError('Preview requires the real vehicle height-only profile')
    return cfg


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seconds',type=float,default=0.,help='0: run until Ctrl-C')
    args=parser.parse_args()
    if os.environ.get('ROS_DOMAIN_ID')!='59' or os.environ.get('ROS_LOCALHOST_ONLY')!='0':
        raise RuntimeError('Use preview_hybrid_map.sh: 104 domain 59, network enabled')
    # Do not start another estimator: reuse the current live origin and pose.
    from control import alive
    state=json.loads((ROOT/'run_state.json').read_text())
    if str(state.get('domain'))!='59' or not alive(state['processes'].get('pipeline',{})):
        raise RuntimeError('Start the normal 104 mapping pipeline first')
    cfg=profile(Path(state['output'])/'profile.yaml')
    out=ROOT/'results'/datetime.now().strftime('hybrid_preview_%Y%m%d_%H%M%S')
    out.mkdir();path=out/'profile.yaml';path.write_text(yaml.safe_dump(cfg,sort_keys=False))
    ros_args=['--ros-args','-r','__node:=hardware104_hybrid_preview',
              '-p','profile_path:='+str(path),'-p','output_dir:='+str(out)]
    for topic in OUTPUTS:ros_args+=['-r',topic+':='+PREFIX+topic]
    rclpy.init(args=ros_args);node=TerrainMapper()
    print('Preview output: '+str(out),flush=True)
    print('RViz point clouds: '+PREFIX+'/T3/mapping/lidar_map and '+PREFIX+'/T3/mapping/stereo_map',flush=True)
    print('RViz fixed frame: odom. Planner topics and the active map origin are unchanged.',flush=True)
    start=time.monotonic()
    try:
        while rclpy.ok() and (args.seconds<=0 or time.monotonic()-start<args.seconds):
            rclpy.spin_once(node,timeout_sec=.05)
    except KeyboardInterrupt:pass
    finally:
        try:node.save();node.report_status()
        finally:
            node.dense_writer.close();node.grid.close();node.delivery.close();node.cloud.close()
            if node.stereo_fill is not None:node.stereo_fill.close()
            node.tum.close();node.destroy_node();rclpy.try_shutdown()
    print('Preview stopped; normal mapping remains running.',flush=True)


if __name__=='__main__':main()
