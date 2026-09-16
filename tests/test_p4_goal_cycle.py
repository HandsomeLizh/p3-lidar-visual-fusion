"""Private ROS 70: actual P4 planner/controller, with a software-only vehicle.

No UE connections. Controller output is remapped to /test/p4_cmd, consumed only
by this test's kinematic plant. Goal originates in the production P3 interface.
"""
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import numpy as np
import rclpy
from rclpy.executors import SingleThreadedExecutor
from geometry_msgs.msg import TransformStamped,Twist
from nav_msgs.msg import Odometry,Path as RosPath
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Header
from tf2_msgs.msg import TFMessage
import yaml
from t3_lidar_visual_fusion.terrain_mapper import TerrainMapper
from t3_lidar_visual_fusion.ros_utils import xyz_cloud

ROOT=Path(__file__).resolve().parents[1]
P4=Path('/home/yanfa/P4')
sys.path.insert(0,str(ROOT/'scripts'))
import p3_visual_monitor as frontend


def main():
    assert os.environ['ROS_DOMAIN_ID']=='70'
    out=ROOT/'results/map_visual_20260916/p4_cycle';out.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(dir=ROOT/'build',prefix='p4_cycle_') as directory:
        tmp=Path(directory)
        cfg=yaml.safe_load((ROOT/'config/simulation_live.yaml').read_text())
        cfg.update(map_window=32.,tile_cells=32,semantic_topic='',map_publish_period=.5,global_publish_period=2.,
            mapping_pose_settle_sec=0.,base_from_lidar=np.eye(4).tolist(),min_range=.1,max_range=20.)
        profile=tmp/'profile.yaml';profile.write_text(yaml.safe_dump(cfg))
        rclpy.init(args=['--ros-args','-p','profile_path:='+str(profile),'-p','output_dir:='+str(tmp/'map')])
        mapper=TerrainMapper();monitor=frontend.visual_monitor.Monitor();driver=rclpy.create_node('software_vehicle_fixture')
        ex=SingleThreadedExecutor()
        for node in [mapper,monitor,driver]:ex.add_node(node)
        pose_pub=driver.create_publisher(Odometry,'/P4/input/odometry',10)
        fused=driver.create_publisher(Odometry,'/T3/semantic/current_pose',10)
        cloud=driver.create_publisher(PointCloud2,'/fusion/lidar',2)
        tf=driver.create_publisher(TFMessage,'/P4/input/map_to_odom',10)
        command=np.zeros(2);received=[]
        def on_command(m):
            command[:]=[m.linear.x,m.angular.z];received.append((time.monotonic(),float(command[0]),float(command[1])))
        driver.create_subscription(Twist,'/test/p4_cmd',on_command,10)
        params=yaml.safe_load((P4/'debug/p3_joint/navigation.yaml').read_text())
        common=params['common'];nav=dict(params['navigation'])
        nav.update(platform_type=common['platform_type'],platform_config=common['platform_config'],
            coarse_resolution_m=common['coarse_resolution_m'],local_map_topic=common['local_map_topic'],
            odometry_topic=common['odometry_topic'],tf_topic=common['tf_topic'],action_name=common['navigation_action'],
            map_frame='map',odom_frame='odom',base_frame='base_link',use_sim_time=False,
            local_map_qos_reliability='reliable',local_map_qos_durability='transient_local')
        configs=[nav,dict(expected_frame='map',goal_topic='/Car/T4/rviz_goal',action_name=common['navigation_action']),
            dict(input_mode='incremental_reference',platform_config=common['platform_config'],
                odometry_topic=common['odometry_topic'],tf_topic=common['tf_topic'],command_topic='/test/p4_cmd',
                goal_position_tolerance_m=.08,goal_yaw_tolerance_rad=.05,max_linear_mps=.2)]
        prefix=P4/'artifacts/humble/install'
        binaries=[prefix/'lunar_incremental_navigation_ros/lib/lunar_incremental_navigation_ros/lunar_incremental_navigation_node',
            prefix/'lunar_incremental_navigation_ros/lib/lunar_incremental_navigation_ros/lunar_incremental_rviz_goal_bridge',
            prefix/'lunar_pure_wheeled_controller/lib/lunar_pure_wheeled_controller/lunar_pure_wheeled_controller_node.py']
        processes=[];logs=[];state=np.zeros(3);previous=time.monotonic();last_state=0.;last_scan=0.
        max_paths={'global':0,'local':0};route_first=None
        xx,yy=np.meshgrid(np.arange(-8,8,.2)+.01,np.arange(-8,8,.2)+.01)
        ground=np.column_stack([xx.ravel(),yy.ravel(),np.zeros(xx.size)])
        mapper.grid.update_elevation_only(points_map=ground);mapper.cloud.append(ground)
        sensor=np.array([[x,y,-1.] for x in np.arange(-3,4,.2) for y in np.arange(-3,3,.2)])
        def advance(seconds):
            nonlocal previous,last_state,last_scan,route_first
            end=time.monotonic()+seconds
            while time.monotonic()<end:
                now=time.monotonic();dt=min(now-previous,.1);previous=now
                v,w=command;state[0]+=v*math.cos(state[2])*dt;state[1]+=v*math.sin(state[2])*dt;state[2]+=w*dt
                if now-last_state>=.04:
                    last_state=now;m=Odometry();m.header.stamp=driver.get_clock().now().to_msg();m.header.frame_id='odom';m.child_frame_id='base_link'
                    m.pose.pose.position.x=float(state[0]);m.pose.pose.position.y=float(state[1]);m.pose.pose.position.z=1.
                    m.pose.pose.orientation.z=math.sin(state[2]/2);m.pose.pose.orientation.w=math.cos(state[2]/2)
                    m.twist.twist.linear.x=float(v);m.twist.twist.angular.z=float(w)
                    pose_pub.publish(m);fused.publish(m)
                    transform=TransformStamped();transform.header.stamp=m.header.stamp;transform.header.frame_id='map';transform.child_frame_id='odom';transform.transform.rotation.w=1.
                    tf.publish(TFMessage(transforms=[transform]))
                    if now-last_scan>.8:
                        last_scan=now;cloud.publish(xyz_cloud(sensor,Header(stamp=m.header.stamp,frame_id='lidar')))
                ex.spin_once(timeout_sec=.002)
                for key in max_paths:
                    size=len(monitor.planning_paths.get(key,[]));max_paths[key]=max(max_paths[key],size)
                    if size and route_first is None:route_first=time.monotonic()
        try:
            for i,(binary,parameters) in enumerate(zip(binaries,configs)):
                f=tmp/f'p4_{i}.yaml';f.write_text(yaml.safe_dump({'/**':{'ros__parameters':parameters}}))
                log=(out/f'process_{i}.log').open('w');logs.append(log)
                processes.append(subprocess.Popen([str(binary),'--ros-args','--params-file',str(f)],
                    cwd=ROOT,env=os.environ.copy(),stdout=log,stderr=subprocess.STDOUT,start_new_session=True))
            advance(3.)
            mapper.last_header=Header(frame_id='map');mapper.last_header.stamp=driver.get_clock().now().to_msg()
            mapper.dirty=True;mapper.publish();mapper.publish_global();advance(.5)
            assert monitor.grid is not None and monitor.goal_pub.get_subscription_count()>0
            assert monitor.request_goal(2.,0.,0.)
            requested=time.monotonic()
            while time.monotonic()-requested<35. and np.linalg.norm(state[:2]-[2.,0.])>.12:
                advance(.1)
                if any(p.poll() is not None for p in processes):raise RuntimeError('P4 test process exited')
            result=dict(max_path_points=max_paths,final_pose=state.tolist(),
                goal_error_m=float(np.linalg.norm(state[:2]-[2.,0.])),
                first_path_delay_sec=None if route_first is None else route_first-requested,
                seconds=time.monotonic()-requested,controller_commands=len(received),
                nonzero_commands=sum(abs(v)+abs(w)>1e-5 for _,v,w in received),
                mapped_scans=mapper.stats['mapped_scans'],map_revision=mapper.grid.update_id,
                scope='Actual P4 binaries in private ROS 70; software-only vehicle on /test/p4_cmd; no UE TCP')
            result['passed']=all(max_paths.values()) and result['goal_error_m']<.15 and result['mapped_scans']>5 and result['nonzero_commands']>0
            (out/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
            assert result['passed'],result
        finally:
            for p in processes:
                if p.poll() is None:os.killpg(p.pid,signal.SIGINT)
            for p in processes:
                try:p.wait(timeout=5)
                except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGTERM);p.wait(timeout=5)
            for log in logs:log.close()
            ex.shutdown();mapper.dense_writer.close();mapper.grid.close();mapper.delivery.close();mapper.cloud.close();mapper.tum.close()
            for node in [mapper,monitor,driver]:node.destroy_node()
            rclpy.try_shutdown()

if __name__=='__main__':main()
