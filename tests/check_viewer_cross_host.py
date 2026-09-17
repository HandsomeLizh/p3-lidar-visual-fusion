"""Replay a saved map across hosts in test domain 97; never drive or localize."""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from grid_map_msgs.msg import GridMap
from std_msgs.msg import Header

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'visual'),str(ROOT/'scripts')]


def map_messages(path):
    from t3_lidar_visual_fusion.legacy.grid_map_message import make_grid_map_message
    from t3_lidar_visual_fusion.legacy.semantic_grid import GridGeometry
    with np.load(path,allow_pickle=False) as data:
        geometry=GridGeometry(float(data['resolution']),int(data['width']),int(data['height']),
                              float(data['origin_x']),float(data['origin_y']))
        layers={name:np.asarray(data[name],np.float32) for name in
                ('elevation','elevation_variance','height_range','roughness','observation_count')}
    global_map=make_grid_map_message(header=Header(frame_id='map'),geometry=geometry,layers=layers)
    size=round(64/geometry.resolution)
    window=GridGeometry(geometry.resolution,size,size,-32.,-32.)
    local_layers={name:np.full((size,size),np.nan,np.float32) for name in layers}
    dx=round((geometry.origin_x-window.origin_x)/window.resolution)
    dy=round((geometry.origin_y-window.origin_y)/window.resolution)
    x0,x1=max(0,dx),min(size,dx+geometry.width);y0,y1=max(0,dy),min(size,dy+geometry.height)
    if x1>x0 and y1>y0:
        for name,values in layers.items():local_layers[name][y0:y1,x0:x1]=values[y0-dy:y1-dy,x0-dx:x1-dx]
    local_map=make_grid_map_message(header=Header(frame_id='odom'),geometry=window,layers=local_layers)
    return {'global':global_map,'local':local_map}


def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['source','sink'])
    p.add_argument('--map',type=Path);p.add_argument('--report',type=Path,required=True)
    p.add_argument('--seconds',type=float,default=70.);args=p.parse_args()
    assert os.environ['ROS_DOMAIN_ID']=='97' and os.environ['ROS_LOCALHOST_ONLY']=='0'
    assert 30<=args.seconds<=120
    from viewer_wire import PREFIX
    rclpy.init(args=['--ros-args','-p','compressed_display_prefix:='+PREFIX,'-p','retain_stale_grid_sec:=30.0'])
    executor=SingleThreadedExecutor();nodes=[];samples=[];seen=[]
    start=time.monotonic();next_sample=start;next_publish=start
    args.report.parent.mkdir(parents=True,exist_ok=True)
    if args.mode=='source':
        from viewer_transport import ViewerTransport
        maps=map_messages(args.map);node=ViewerTransport()
        driver=Node('saved_map_test_source');nodes=[node,driver]
        qos=QoSProfile(depth=1,reliability=ReliabilityPolicy.RELIABLE,durability=DurabilityPolicy.TRANSIENT_LOCAL)
        pubs={k:driver.create_publisher(GridMap,'/Car/T3/mapping/'+('global_grid_map' if k=='global' else 'grid_map'),qos) for k in maps}
    else:
        from visual_monitor import Monitor
        node=Monitor();nodes=[node]
    for n in nodes:executor.add_node(n)
    try:
        while time.monotonic()-start<args.seconds:
            now=time.monotonic();elapsed=now-start
            if args.mode=='source' and now>=next_publish:
                next_publish=now+1.
                if not 18<=elapsed<31:  # Controlled source pause, not a live network change.
                    for k,m in maps.items():m.header.stamp=driver.get_clock().now().to_msg();pubs[k].publish(m)
            executor.spin_once(timeout_sec=.02)
            if now>=next_sample:
                next_sample=now+1.
                rss=int(next(line for line in Path('/proc/self/status').read_text().splitlines() if line.startswith('VmRSS:')).split()[1])/1024
                if args.mode=='source':row=dict(elapsed=elapsed,rss_mib=rss,transport=dict(node.stats))
                else:
                    row=dict(elapsed=elapsed,rss_mib=rss,source=node.grid_source,known_cells=node.grid[2] if node.grid else 0,
                             fresh=node.grid_is_fresh(),counts=dict(node.grid_message_count),stamps=dict(node.grid_stamps))
                    if node.grid is not None and node.grid_is_fresh() and not seen:
                        node.grid[0].save(args.report.with_suffix('.png'));seen.append(elapsed)
                samples.append(row)
        if args.mode=='source':
            passed=all(node.stats.get(k,{}).get('messages',0)>=20 for k in ('local','global'))
        else:
            stale=[i for i,row in enumerate(samples) if row['source']=='stale' and row['known_cells']>0]
            recovered=bool(stale and any(row['fresh'] for row in samples[stale[0]+1:]))
            passed=(sum(row['fresh'] for row in samples)>=15 and recovered and
                    all(node.grid_message_count[k]>=20 for k in ('local','global')))
        report=dict(passed=passed,mode=args.mode,domain=97,production_mapping_started=False,
                    vehicle_commands_published=0,controlled_source_pause_sec=[18,31],samples=samples)
        args.report.write_text(json.dumps(report,indent=2))
        print(json.dumps(dict(passed=passed,mode=args.mode,report=str(args.report),
            samples=len(samples),last=samples[-1],first_grid_sec=seen[0] if seen else None)))
        if not passed:raise SystemExit(1)
    finally:
        executor.shutdown()
        for n in nodes:n.destroy_node()
        rclpy.shutdown()


if __name__=='__main__':main()
