#!/usr/bin/env python3
"""Generate VoxelMap, optional visual, and EKF settings from one calibration."""
from pathlib import Path
import argparse,json
import numpy as np
import yaml

ROOT=Path(__file__).resolve().parents[1]


def generate(profile_path, output):
    p=yaml.safe_load(Path(profile_path).read_text())
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    p.setdefault("lidar_backend","voxelmap")
    if p["lidar_backend"]!="voxelmap":
        raise ValueError("This pipeline now supports VoxelMap only")
    mode=p.get("imu_mode","auto" if p.get("use_imu",False) else "off")
    if isinstance(mode,bool):mode="auto" if mode else "off"  # YAML 1.1 parses bare off as False.
    if mode not in ("auto","off"):
        raise ValueError("imu_mode must be auto or off")
    p["imu_mode"]=mode
    p["imu_calibration_confirmed"]=bool(p.get("imu_calibration_confirmed",p.get("calibration_confirmed",False)))
    if p.get("use_imu",False) and p.get("visual_source","vins")=="vins" and not p["imu_calibration_confirmed"]:
        raise ValueError("VINS IMU requires measured calibration; VoxelMap auto can continue without IMU")
    deskew=p.get("deskew",{}).get("enabled",False)
    if deskew and (p["instantaneous_cloud"] or p.get("cloud_motion_compensated",False) or
                   mode!="auto" or not p["imu_calibration_confirmed"]):
        raise ValueError("Raw scan deskew requires calibrated real IMU and spinning input")
    if deskew:p["mapping_lidar_topic"]="/fusion/lidar_deskewed"
    if not p["instantaneous_cloud"] and not p.get("cloud_motion_compensated",False) and not deskew:
        raise ValueError("VoxelMap needs instantaneous or upstream motion-compensated clouds; raw spinning scans are not yet deskewed here")
    for name in ("base_from_imu","base_from_lidar"):
        t=np.asarray(p[name],dtype=float)
        if (t.shape!=(4,4) or not np.isfinite(t).all() or
            not np.allclose(t[3],[0,0,0,1],atol=1e-8) or
            not np.allclose(t[:3,:3]@t[:3,:3].T,np.eye(3),atol=1e-4) or
            not np.isclose(np.linalg.det(t[:3,:3]),1.,atol=1e-4)):
            raise ValueError(name+" must be a proper rigid transform")
    Tbi=np.asarray(p["base_from_imu"])
    width,height=p["output_image_size"];iw,ih=p["input_image_size"]
    k=np.asarray(p["camera_k"]).reshape(3,3).copy();k[0]*=width/iw;k[1]*=height/ih
    for side in [0,1]:
        cal={"model_type":"PINHOLE","camera_name":"camera"+str(side),"image_width":width,"image_height":height,
             "distortion_parameters":{"k1":float(p.get("distortion",[0,0,0,0])[0]),
                                      "k2":float(p.get("distortion",[0,0,0,0])[1]),
                                      "p1":float(p.get("distortion",[0,0,0,0])[2]),
                                      "p2":float(p.get("distortion",[0,0,0,0])[3])},
             "projection_parameters":{"fx":float(k[0,0]),"fy":float(k[1,1]),"cx":float(k[0,2]),"cy":float(k[1,2])}}
        (output/f"cam{side}.yaml").write_text("%YAML:1.0\n---\n"+yaml.safe_dump(cal,sort_keys=False))
    v={"imu":int(p.get("use_imu",False)),"num_of_cam":2,"imu_topic":"/fusion/imu",
       "image0_topic":"/fusion/left","image1_topic":"/fusion/right",
       "output_path":str(output)+"/","cam0_calib":"cam0.yaml","cam1_calib":"cam1.yaml",
       "image_width":width,"image_height":height,"estimate_extrinsic":0,"multiple_thread":0,"use_gpu":0,"use_gpu_acc_flow":0,"use_gpu_ceres":0,
       "max_cnt":220,"min_dist":15,"freq":10,"F_threshold":1.0,"show_track":0,"flow_back":1,
       "max_solver_time":.06,"max_num_iterations":8,"keyframe_parallax":8.0,
       "acc_n":.1,"gyr_n":.01,"acc_w":.001,"gyr_w":.0001,"g_norm":9.81,
       "estimate_td":0,"td":0.0,"rolling_shutter":0,"rolling_shutter_tr":0.0}
    txt="%YAML:1.0\n---\n"+yaml.safe_dump(v,sort_keys=False)
    for side,key in [(0,"base_from_camera_left"),(1,"base_from_camera_right")]:
        tic=np.linalg.inv(Tbi)@np.asarray(p[key])
        txt+=f"body_T_cam{side}: !!opencv-matrix\n   rows: 4\n   cols: 4\n   dt: d\n   data: "+json.dumps(tic.reshape(-1).tolist())+"\n"
    (output/"vins.yaml").write_text(txt)
    cfg_pose=[True]*6+[False]*9
    pose_axes=("x","y","z","roll","pitch","yaw")
    visual_axes=p.get("visual_pose_axes",list(pose_axes))
    if (not isinstance(visual_axes,list) or not visual_axes
        or any(axis not in pose_axes for axis in visual_axes)
        or len(set(visual_axes))!=len(visual_axes)):
        raise ValueError("visual_pose_axes must be unique pose axes: x, y, z, roll, pitch, yaw")
    # A surface-vehicle profile can let LiDAR observe full 3-D terrain while
    # stereo contributes x/y/yaw. Do not let drifting visual height reject all
    # useful horizontal observations through one joint innovation test.
    visual_pose=[axis in visual_axes for axis in pose_axes]+[False]*9
    ekf={"frequency":30.0,"sensor_timeout":.3,"two_d_mode":False,"publish_tf":True,
         "map_frame":"map","odom_frame":"odom","base_link_frame":"base_link","world_frame":"odom",
         "print_diagnostics":True,"reset_on_time_jump":True,"smooth_lagged_data":True,"history_length":p.get("ekf_history_seconds",10.),
         "predict_to_current_time":False,"odom0":"/lio/odom","odom0_config":cfg_pose,
         "odom0_differential":False,"odom0_relative":False,"odom0_queue_size":100,
         "odom1":"/fusion/vision_odom_guarded","odom1_config":visual_pose,
         "odom1_differential":True,"odom1_relative":False,"odom1_queue_size":100,
         "odom1_pose_rejection_threshold":5.0,"odom1_twist_rejection_threshold":5.0}
    if p.get("adaptive_source_selection",False):
        # Only qualified output may own odom->base_link. Otherwise EKF TF can
        # move RViz even while the guard has stopped odometry/map publication.
        ekf["publish_tf"]=False
        # Match the declared healthy sample interval. With sparse bag input,
        # premature prediction advances EKF time past the arriving cloud pose.
        # Publish measurement-time states; do not fabricate high-rate evidence.
        ekf["sensor_timeout"]=float(p.get("ekf_sensor_timeout",max(.3,p["vision_gate"]["max_gap"])))
        if p.get('ekf_measurement_time_only', False):
            # Source health is handled by the guard. Keep idle wall-time
            # prediction out of the delayed measurement history as well.
            ekf['sensor_timeout']=max(ekf['sensor_timeout'],float(ekf['history_length']))
        ekf["permit_corrected_publication"]=True
        # Each frontend keeps one fixed transform per declared epoch. A new
        # epoch requires a reliable same-time reference; outages keep the transform.
        visual_mode=p.get("visual_constraint_mode","absolute")
        if visual_mode not in ("absolute","differential","body_twist"):
            raise ValueError("visual_constraint_mode must be absolute, differential or body_twist")
        ekf["odom1_differential"]=visual_mode=="differential"
        if visual_mode=="body_twist":
            ekf["odom1_config"]=[False]*6+[True]*6+[False]*3
        # LiDAR already passed registration, covariance and motion checks.
        # Do not let a visual-first update make EKF reject healthy LiDAR.
        # Retain the visual innovation threshold; directional covariance
        # controls how much each LiDAR direction contributes.
        initial=[1.]*6+[.1]*6+[1e-12]*3
        process=[.001]*6+[.01]*6+[1e-12]*3
        # Neither frontend measures acceleration. Keep that inactive state from
        # causing long, unconstrained acceleration extrapolation between scans.
        ekf["initial_estimate_covariance"]=np.diag(initial).reshape(-1).tolist()
        ekf["process_noise_covariance"]=np.diag(process).reshape(-1).tolist()
    telemetry=p.get("telemetry_motion",{})
    if telemetry.get("enabled",False) and telemetry.get("fuse_velocity",False):
        if not telemetry.get("calibration_confirmed",False):
            raise ValueError("Telemetry integration requires confirmed units and body-frame convention")
        if (telemetry.get("velocity_frame","body")=="world" and
                not telemetry.get("world_frame_alignment_confirmed",False)):
            raise ValueError("World velocity integration requires confirmed velocity/attitude frame alignment")
        # Independent velocity feedback supports the existing EKF propagation.
        # Do not consume the UE absolute pose or disguise velocity as IMU data.
        ekf.update(twist0="/fusion/telemetry_twist",
            twist0_config=[False]*6+[True]*6+[False]*3,
            twist0_queue_size=10,twist0_rejection_threshold=5.0)
    (output/"ekf.yaml").write_text(yaml.safe_dump({"ekf_filter_node":{"ros__parameters":ekf}},sort_keys=False))
    return p

if __name__=="__main__":
    a=argparse.ArgumentParser();a.add_argument("profile");a.add_argument("output")
    x=a.parse_args();generate(x.profile,x.output)
