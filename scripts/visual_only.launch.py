from pathlib import Path
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument,OpaqueFunction,RegisterEventHandler,EmitEvent
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
ROOT=Path(__file__).resolve().parents[1]
def nodes(context):
 profile=LaunchConfiguration("profile").perform(context)
 output=LaunchConfiguration("output_dir").perform(context)
 compact=LaunchConfiguration("normalized_transport").perform(context).lower()=="true"
 common={"use_sim_time":True,"profile_path":profile}
 nodes=[
  Node(package="t3_lidar_visual_fusion",executable="sensor_adapter",parameters=[common],output="screen"),
  Node(package="t3_lidar_visual_fusion",executable="learned_odometry",
   parameters=[dict(common,workspace_root=str(ROOT),output_dir=output)],output="screen")]
 if compact:nodes=nodes[1:]
 return nodes+[RegisterEventHandler(OnProcessExit(target_action=n,on_exit=[EmitEvent(event=Shutdown(reason="Visual benchmark component exited"))])) for n in nodes]
def generate_launch_description():
 return LaunchDescription([DeclareLaunchArgument("profile"),DeclareLaunchArgument("output_dir"),DeclareLaunchArgument("normalized_transport",default_value="false"),OpaqueFunction(function=nodes)])
