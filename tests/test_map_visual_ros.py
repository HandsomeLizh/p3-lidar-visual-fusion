"""Real ROS grid/path/goal transport and optional isolated Tk interaction."""
import json
from collections import deque
import math
import os
from pathlib import Path
import sys
import tempfile
import threading
import time
from types import SimpleNamespace
import numpy as np
import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.qos import QoSProfile,DurabilityPolicy,ReliabilityPolicy
from nav_msgs.msg import Path as RosPath
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Header,String
from grid_map_msgs.msg import GridMap
import yaml
from t3_lidar_visual_fusion.terrain_mapper import TerrainMapper

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import p3_visual_monitor as frontend
ui=frontend.visual_monitor


def main():
    assert os.environ['ROS_DOMAIN_ID']=='69'
    out=ROOT/'results/map_visual_20260916';out.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=ROOT/'build',prefix='map_visual_') as temporary:
        tmp=Path(temporary);cfg=yaml.safe_load((ROOT/'config/simulation_live.yaml').read_text())
        cfg.update(map_window=32.,tile_cells=32,semantic_topic='',map_publish_period=100.,global_publish_period=100.)
        profile=tmp/'profile.yaml';profile.write_text(yaml.safe_dump(cfg))
        rclpy.init(args=['--ros-args','-p','profile_path:='+str(profile),'-p','output_dir:='+str(tmp/'map')])
        mapper=TerrainMapper();monitor=ui.Monitor();driver=rclpy.create_node('isolated_planning_fixture')
        executor=SingleThreadedExecutor()
        for n in [mapper,monitor,driver]:executor.add_node(n)
        retained=QoSProfile(depth=1,reliability=ReliabilityPolicy.RELIABLE,durability=DurabilityPolicy.TRANSIENT_LOCAL)
        routes={label:driver.create_publisher(RosPath,topic,retained) for label,topic in [
            ('global','/Car/T4/planning/global_route'),('local','/Car/T4/planning/local_path')]}
        trace=driver.create_publisher(RosPath,'/T3/semantic/trajectory',retained)
        goals=[];driver.create_subscription(PoseStamped,'/Car/T4/rviz_goal',goals.append,10)
        grids=deque(maxlen=2);driver.create_subscription(GridMap,'/T3/mapping/global_grid_map',grids.append,retained)
        window=None;stop=threading.Event()
        def drain(seconds):
            end=time.monotonic()+seconds
            while time.monotonic()<end:
                executor.spin_once(timeout_sec=.003)
                if window:window.root.update()
        def path(points):
            m=RosPath();m.header=Header(frame_id='map');m.header.stamp=driver.get_clock().now().to_msg()
            for x,y in points:
                p=PoseStamped();p.header=m.header;p.pose.position.x=float(x);p.pose.position.y=float(y);p.pose.orientation.w=1.;m.poses.append(p)
            return m
        try:
            xx,yy=np.meshgrid(np.arange(-16,16,.2)+.01,np.arange(-16,16,.2)+.01)
            ground=np.column_stack([xx.ravel(),yy.ravel(),np.zeros(xx.size)])
            obstacle=np.array([[2.01,.01,.8],[2.21,.01,.8],[2.01,.21,.8],[2.21,.21,.8]])
            mapper.grid.update_elevation_only(points_map=np.r_[ground,obstacle]);mapper.cloud.append(np.r_[ground,obstacle])
            mapper.last_header=Header(frame_id='map');mapper.last_header.stamp=driver.get_clock().now().to_msg();mapper.last_pose=np.eye(4)
            mapper.dirty=True
            drain(.5);started=time.monotonic();mapper.publish();mapper.publish_global();publication=time.monotonic()-started
            trace.publish(path([(-4.,0.),(-2.,0.),(0.,0.)]));routes['global'].publish(path([(0,0),(0,3),(3,3)]));routes['local'].publish(path([(0,0),(0,1),(0,2)]))
            drain(.8)
            assert grids and {'elevation','occupancy','obstacle','traversability'}.issubset(grids[-1].layers)
            assert monitor.grid and monitor.display_points>0
            assert np.any(np.all(np.array(monitor.grid[0])==0,axis=2)), 'Global obstacles were not rendered black'
            assert not monitor.request_goal(2.05,.05,0.), 'Obstacle goal accepted'
            assert not monitor.request_goal(100.,100.,0.), 'Unknown goal accepted'
            assert len(monitor.planning_paths['global'])==3 and len(monitor.planning_paths['local'])==3
            if os.environ.get('DISPLAY'):
                window=ui.Window(monitor,stop);window.root.geometry('785x985+20+20');drain(.4);window.redraw()
                assert window.follow_vehicle
                center,scale=window.current_view
                np.testing.assert_allclose(center,[0.,0.])
                np.testing.assert_allclose(window.pixel_to_map(window.canvas.winfo_width()/2,window.canvas.winfo_height()/2),center)
                window.toggle_goal();center,scale=window.current_view
                px=window.canvas.winfo_width()/2+(3-center[0])*scale;py=window.canvas.winfo_height()/2-(3-center[1])*scale
                window.press(SimpleNamespace(x=px,y=py));window.motion(SimpleNamespace(x=px,y=py-40));window.release(SimpleNamespace(x=px,y=py-40))
                assert not window.goal_mode
            else:assert monitor.request_goal(3.,3.,math.pi/2)
            drain(.6)
            assert len(goals)==1 and goals[0].header.frame_id=='map'
            np.testing.assert_allclose([goals[0].pose.position.x,goals[0].pose.position.y],[3.,3.],atol=1e-6)
            assert abs(goals[0].pose.orientation.z-math.sqrt(.5))<1e-6
            drain(.3);assert len(goals)==1,'Goal was repeated'
            monitor.on_fusion_health(String(data=json.dumps(dict(localization_valid=False))))
            assert not monitor.request_goal(3.,3.,0.),'Goal accepted without qualified localization'
            if window:
                window.refresh();window.root.update()
                assert '正在恢复' in window.status.cget('text')
            monitor.on_fusion_health(String(data=json.dumps(dict(localization_valid=True,output_source='visual'))))
            assert monitor.request_goal(3.,3.,0.)
            monitor.on_fusion_health(String(data=json.dumps(dict(localization_valid=False))))
            monitor.send_requested_goal();drain(.1)
            assert len(goals)==1,'Queued goal sent after localization became invalid'
            monitor.on_fusion_health(String(data=json.dumps(dict(localization_valid=True,output_source='visual'))))
            monitor.fusion_health_at-=4.
            assert not monitor.request_goal(3.,3.,0.),'Goal accepted with stale localization health'
            monitor.fusion_health=None
            intervals=[];previous=time.monotonic();rss=[]
            for i in range(600 if window else 60):
                routes['local'].publish(path([(0,0),(0,1.+.005*i),(0,2.+.005*i)]))
                if i%10==0:mapper.publish_global()
                drain(.03);now=time.monotonic();intervals.append(now-previous);previous=now
                rss.append(int(next(l for l in Path('/proc/self/status').read_text().splitlines() if l.startswith('VmRSS:')).split()[1])/1024)
            if window:
                window.follow_view();window.redraw();drain(.2)
                from PIL import ImageGrab
                ImageGrab.grab(xdisplay=os.environ['DISPLAY']).save(out/'interface.png')
                window.reset_view();window.redraw();drain(.1)
                window.follow_view();window.redraw();drain(.1)
            routes['local'].publish(path([]));drain(.2)
            assert len(monitor.planning_paths['local'])==0
            # Explicit global invalidation clears obsolete image and goal checks.
            for publisher in mapper.global_pubs:publisher.publish(GridMap())
            drain(.3);assert monitor.grid is None
            assert not monitor.request_goal(3,3,0)
            result=dict(passed=True,gui_tested=window is not None,operator_goals=len(goals),
                localization_health_display_and_goal_checks=True,
                global_layers=list(grids[0].layers),display_points=monitor.display_points,
                initial_map_publication_sec=publication,ui_loop_p95_sec=float(np.percentile(intervals,95)),
                ui_loop_max_sec=max(intervals),rss_first_mib=rss[0],rss_last_mib=rss[-1],rss_peak_mib=max(rss),
                scope='Isolated ROS 69, synthetic geometry and planning routes; no live vehicle commands')
            (out/'interface_verification.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
        finally:
            if window:window.root.destroy()
            executor.shutdown();mapper.dense_writer.close();mapper.grid.close();mapper.delivery.close();mapper.cloud.close();mapper.tum.close()
            for n in [mapper,monitor,driver]:n.destroy_node()
            rclpy.try_shutdown()

if __name__=='__main__':main()
