from pathlib import Path
import sys
import json
import numpy as np
from scipy.spatial.transform import Rotation
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument,OpaqueFunction,RegisterEventHandler,EmitEvent,LogInfo
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
from generate_config import generate


def nodes(context):
    value=lambda n:LaunchConfiguration(n).perform(context)
    profile=Path(value("profile")).resolve()
    out=Path(value("output_dir")).resolve()
    p=generate(profile,out/"config")
    sim=value("use_sim_time").lower()=="true"
    params={"use_sim_time":sim,"profile_path":str(profile)}
    internal_tf=[("/tf","/fusion/internal_tf"),("/tf_static","/fusion/internal_tf_static")]
    n=[
      Node(package="t3_lidar_visual_fusion",executable="sensor_adapter",parameters=[dict(params,normalized_camera_input=p.get("normalized_camera_input",False))],output="screen"),
      Node(package="t3_voxelmap",executable="voxelmap_node",
           parameters=[{"use_sim_time":sim,"base_from_lidar":np.asarray(p["base_from_lidar"]).reshape(-1).tolist(),
               "instantaneous_cloud":p["instantaneous_cloud"],
               "cloud_motion_compensated":p.get("cloud_motion_compensated",False),
               "base_from_imu":np.asarray(p["base_from_imu"]).reshape(-1).tolist(),
                "imu_mode":ParameterValue(p["imu_mode"],value_type=str),"imu_calibration_confirmed":p["imu_calibration_confirmed"],
                # Same-epoch raw poses are already gated by the learned
                # frontend's keyframe recovery. Keep the recovery horizon in
                # sync; the LiDAR bridge additionally bounds motion and drift.
                "independent_visual_max_gap_sec":float(min(
                    p.get("learned_visual",{}).get("max_recovery_gap_sec",6.),
                    p.get("visual_continuity",{}).get("max_gap",6.)))
                    if p.get("visual_source")=="learned" else 6.,
               "timing_path":str(out/"lidar_metrics.jsonl")},
               p.get("voxelmap",{})],output="screen"),
      Node(package="vins",executable="vins_node",name="vins_estimator",arguments=[str(out/"config/vins.yaml")],
           parameters=[{"use_sim_time":sim}],
           remappings=[("odometry","/fusion/vins_raw"),("point_cloud","/fusion/vins_features"),*internal_tf],
           output="screen"),
      Node(package="robot_localization",executable="ekf_node",name="ekf_filter_node",
           parameters=[str(out/"config/ekf.yaml"),{"use_sim_time":sim}],
           remappings=[("odometry/filtered","/fusion/ekf")]+(internal_tf if p.get("adaptive_source_selection",False) else []),output="screen"),
      Node(package="t3_lidar_visual_fusion",executable=("adaptive_guard" if p.get("adaptive_source_selection",False) else "odometry_guard"),parameters=[dict(params,output_dir=str(out))],output="screen"),
      Node(package="t3_lidar_visual_fusion",executable="terrain_mapper",
           parameters=[params,{"output_dir":str(out)}],output="screen")]
    optional_visual=[]
    optional_motion=[]
    if value("external_estimates").lower()=="true":n=n[3:]
    elif p.get("visual_source","vins") in ("roma_external","none"):n.pop(2)
    elif p.get("visual_source")=="learned":
        n[2]=Node(package="t3_lidar_visual_fusion",executable="learned_odometry",
            parameters=[dict(params,workspace_root=str(ROOT),output_dir=str(out))],output="screen")
    if value("external_estimates").lower()!="true" and p.get("visual_source","vins") in ("vins","learned"):
        optional_visual=[n[2]]
    def visual_exited(event,context):
        stereo_only=p.get('mapping_sources')==['stereo']
        (out/"visual_unavailable.json").write_text(json.dumps(dict(
            reason="visual_process_exited",returncode=getattr(event,"returncode",None),
            action="continue_lidar_localization_stereo_mapping_paused" if stereo_only else "continue_lidar_only")))
        return [LogInfo(msg="Visual frontend exited; LiDAR localization continues; stereo map updates paused"
            if stereo_only else "Visual frontend exited; continuing LiDAR localization and mapping")]
    if p.get("telemetry_motion",{}).get("enabled",False):
        telemetry=Node(package="t3_lidar_visual_fusion",executable="telemetry_motion",
            parameters=[dict(params,output_dir=str(out))],output="screen")
        n.append(telemetry);optional_motion.append(telemetry)
    transforms=[("map","odom",np.eye(4)),("base_link","lidar",np.asarray(p["base_from_lidar"])),
        ("base_link","imu",np.asarray(p["base_from_imu"])),
        ("base_link","camera_left_optical",np.asarray(p["base_from_camera_left"])),
        ("base_link","camera_right_optical",np.asarray(p["base_from_camera_right"]))]
    for parent,child,t in transforms:
        q=Rotation.from_matrix(t[:3,:3]).as_quat()
        args=["--frame-id",parent,"--child-frame-id",child]
        for key,val in zip(["--x","--y","--z","--qx","--qy","--qz","--qw"],[*t[:3,3],*q]):
            args += [key,str(float(val))]
        n.append(Node(package="tf2_ros",executable="static_transform_publisher",
                      name="fusion_static_"+child,arguments=args,output="screen"))
    core_handlers=[RegisterEventHandler(OnProcessExit(target_action=node,
        on_exit=[EmitEvent(event=Shutdown(reason="Critical LiDAR/fusion/map component exited"))]))
         for node in n if node not in optional_visual+optional_motion]
    visual_handlers=[RegisterEventHandler(OnProcessExit(target_action=node,on_exit=visual_exited))
        for node in optional_visual]
    motion_handlers=[RegisterEventHandler(OnProcessExit(target_action=node,
        on_exit=[LogInfo(msg="Telemetry relay exited; continuing LiDAR/visual fusion without fresh velocity assistance")]))
        for node in optional_motion]
    return n+core_handlers+visual_handlers+motion_handlers


def generate_launch_description():
    return LaunchDescription([
      DeclareLaunchArgument("profile",default_value=str(ROOT/"config/simulation.yaml")),
      DeclareLaunchArgument("output_dir",default_value=str(ROOT/"results/live")),
      DeclareLaunchArgument("external_estimates",default_value="false"),
      DeclareLaunchArgument("use_sim_time",default_value="false"),OpaqueFunction(function=nodes)])
