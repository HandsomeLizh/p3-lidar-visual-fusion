"""Real ROS display fallback without driving, publishing goals or allocating a large map."""
import copy,json,os,sys,tempfile,time
from pathlib import Path
import numpy as np,rclpy,yaml
from rclpy.executors import SingleThreadedExecutor
from std_msgs.msg import Header
from grid_map_msgs.msg import GridMap
from t3_lidar_visual_fusion.terrain_mapper import TerrainMapper
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from p3_visual_monitor import visual_monitor

def main():
 assert os.environ['ROS_DOMAIN_ID']=='95' and os.environ['ROS_LOCALHOST_ONLY']=='1'
 with tempfile.TemporaryDirectory(dir=ROOT/'build') as directory:
  out=Path(directory);cfg=yaml.safe_load((ROOT/'config/hardware104.yaml').read_text())
  cfg.update(elevation_fusion={'enabled':False},map_publish_period=1000.,global_publish_period=1000.,tile_cells=16,map_window=32.)
  profile=out/'profile.yaml';profile.write_text(yaml.safe_dump(cfg))
  rclpy.init(args=['--ros-args','-p','profile_path:='+str(profile),'-p','output_dir:='+str(out/'map')])
  mapper=TerrainMapper();monitor=visual_monitor.Monitor();executor=SingleThreadedExecutor()
  executor.add_node(mapper);executor.add_node(monitor)
  def drain():
   until=time.monotonic()+.6
   while time.monotonic()<until:executor.spin_once(timeout_sec=.01)
  try:
   mapper.grid.update_elevation_only(points_map=np.array([[.1,.1,.25],[5.1,.1,.4],[30.1,.1,.7]]))
   mapper.last_header=Header(frame_id='map');mapper.last_header.stamp.sec=100
   mapper.last_pose=np.eye(4);mapper.dirty=True;drain();mapper.publish();mapper.publish_global();drain()
   assert monitor.grid_source=='global' and monitor.grid is not None
   mapper.pressure=True;mapper.publish_global();drain()
   assert not mapper.global_available and monitor.grid_source=='local' and monitor.grid is not None
   assert monitor.grid[1][3]*monitor.grid[1][2]==32.
   monitor.localization_unavailable=lambda:False
   assert monitor.request_goal(.1,.1,0.)
   monitor.goal_requested=None
   assert not monitor.request_goal(30.1,.1,0.),'Fallback must not retain global-only selectable cells'
   # Empty global repeated; malformed local cannot erase a healthy global.
   mapper.publish_global();drain();assert monitor.grid_source=='local'
   mapper.pressure=False;mapper.publish_global();drain();assert monitor.grid_source=='global'
   monitor.on_local_grid(GridMap());assert monitor.grid_source=='global'
   # Recovery after area-limit invalidation follows the same fallback route.
   mapper.dirty=True;mapper.publish();drain()
   monitor.global_grid_at-=9.;monitor.check_grid_freshness()
   assert monitor.grid_source=='local'
   monitor.local_grid_at-=6.;monitor.check_grid_freshness()
   assert monitor.grid is None and monitor.grid_source=='none'
   assert not monitor.request_goal(.1,.1,0.)
   mapper.publish_global();drain();assert monitor.grid_source=='global'
   bad=copy.deepcopy(mapper.cached_global);bad.data[0].data.pop()
   monitor.on_grid(bad);assert monitor.grid_source=='none'
   report=dict(passed=True,real_ros_transport=True,actual_mapper_pressure_invalidation=True,
    local_32m_fallback=True,global_recovery=True,stale_maps_cleared=True,
    malformed_messages_rejected=True,global_only_cells_not_selectable_in_fallback=True,
    vehicle_commands_published=0,goal_messages_published=0)
   target=ROOT/'results/hardware104_display_fallback_20260917/report.json';target.parent.mkdir(parents=True,exist_ok=True)
   target.write_text(json.dumps(report,indent=2));print(json.dumps(report))
  finally:
   mapper.dense_writer.close();mapper.grid.close();mapper.delivery.close();mapper.cloud.close();mapper.tum.close()
   executor.shutdown();mapper.destroy_node();monitor.destroy_node();rclpy.shutdown()

if __name__=='__main__':main()
