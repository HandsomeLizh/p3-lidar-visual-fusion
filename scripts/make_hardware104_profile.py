#!/usr/bin/env python3
"""Generate the rover profile from existing measured files; preserve provenance."""
import hashlib
import json
from pathlib import Path
import numpy as np
import yaml

ROOT=Path(__file__).resolve().parents[1]
SOURCE=Path('/home/yanfa/P3/roma_t3_algorithm_bundle_20260825/workspace/src/t3_semantic_mapping/config/t5_rgb10_rgb11.yaml')
META=Path('/home/yanfa/program/Lidar2/src/ouster-ros/ouster-ros/config/192.168.19-metadata.json')


def main():
    original=yaml.safe_load(SOURCE.read_text());meta=json.loads(META.read_text())
    p=yaml.safe_load((ROOT/'config/simulation_live.yaml').read_text())
    # XLS vehicle and mounting reference use FRD; published sensor_frame has
    # z up (verified stationary specific force). Keep this explicit, never
    # silently substitute the old mapping-only matrix that inverted the floor.
    flu_from_frd=np.diag([1.,-1.,-1.,1.]);m=original['mounting']
    bl=flu_from_frd@np.asarray(m['transform_vehicle_lidar'])@flu_from_frd
    si=np.asarray(meta['imu_intrinsics']['imu_to_sensor_transform'],dtype=float).reshape(4,4)
    si[:3,3]*=.001  # Ouster metadata translation is in mm.
    left=flu_from_frd@np.asarray(m['transform_vehicle_left'])
    stereo=np.asarray(original['raw_stereo']['transform_right_to_left'])
    p.update(name='Rover 104 real Ouster OS1-128 + internal IMU + Galaxy stereo',
        use_imu=True,imu_mode='auto',imu_calibration_confirmed=True,
        pose_source_preference='fused',visual_motion_information_scale=1.,
        calibration_confirmed=False,instantaneous_cloud=False,cloud_motion_compensated=False,
        deskew={'enabled':True},mapping_lidar_topic='/fusion/lidar_deskewed',
        lidar_topic='/hardware/lidar',imu_topic='/hardware/imu',
        lidar_input_frame='sensor_frame',imu_input_frame='imu_frame',
        lidar_input_basis='ouster_sensor_frame',
        lidar_basis_conversion='XLS FRD mounting basis -> published z-up sensor_frame: D*T*D; yaw mounting needs moving cross-sensor validation.',
        base_from_lidar=bl.tolist(),base_from_imu=(bl@si).tolist(),
        base_from_camera_left=left.tolist(),base_from_camera_right=(left@stereo).tolist(),
        camera_k=original['raw_stereo']['left']['k'],camera_k_right=original['raw_stereo']['right']['k'],
        distortion=original['raw_stereo']['left']['d'],distortion_right=original['raw_stereo']['right']['d'],
        normalized_camera_input=True,visual_max_age_sec=1.5,visual_wall_timeout_sec=1.5,
        source_wall_timeout=1.5,visual_reference_wait=.3,stereo_max_skew=.01,
        pose_max_gap=.6,pose_tolerance=.12,mapping_pose_settle_sec=.4,
        mapping_wait_timeout=3.,ekf_sensor_timeout=.5,
        ekf_measurement_time_only=True,ekf_visual_sync_wait_sec=.8,
        map_window=64.,map_publish_period=1.,global_publish_period=2.,
        mapping_dds_config='config/cyclonedds_mapping104.xml',
        min_range=.7,max_range=50.,max_scan_duration=.12,
        tof_sources=[],telemetry_motion={'enabled':False},
        self_filter={'enabled':True,'min_xyz_m':[-1.47,-1.,.13],'max_xyz_m':[.53,1.,1.03]},
        mapping_source='range',
        mapping_camera_view={'enabled':True,'full_density_range_m':10.,'range_feather_m':5.,'image_feather_fraction':.15},
        stereo_mapping={'enabled':False,'preserve_fill_xyz':True,'topic':'/fusion/stereo_points','max_hz':2.,
            'max_points':512,'min_depth_m':.5,'max_depth_m':8.,'pixel_sigma':1.,
            'max_point_std_m':.15,'max_height_std_m':.25,'voxel_size_m':.1,
            'confirmation_frames':2,'confirmation_timeout_sec':5.,'pending_cells':4096,
            'variance_floor_m2':.0025,'max_position_variance':.04,'max_rotation_variance':.01,
            'preview_points':20000},
        hardware={'enabled':True,'source_domain':19,'max_lidar_hz':5.,'split_inertial':True,
            'clock_reference_topic':'/P3/hardware/clock_reference',
            'lidar_topic':'/Car/T5/OS1/points','imu_topic':'/Car/T5/OS1/imu',
            'left_topic':'/Car/T5/Cam_Left/image_mono/mapping',
            'right_topic':'/Car/T5/Cam_Right/image_mono/mapping','compact_stereo':True,
            'clock':{'mode':'device_uptime','warmup_seconds':1.,'min_samples':80,
                     'max_jitter':.03,'max_drift':.10,'max_age':.75}})
    p['vision_gate']['max_gap']=1.5  # Accept a 1 Hz trigger with small scheduling jitter.
    p['learned_visual'].update(max_tracking_gap_sec=2.,max_recovery_gap_sec=30.,max_processing_hz=5.)
    p['visual_continuity'].update(enabled=True,max_gap=30.)
    p['voxelmap'].update(max_iterations=20,local_map_radius=50.,max_root_voxels=8000,
        imu_init_seconds=1.5,imu_init_min_samples=100,
        imu_max_gap_sec=.04,imu_end_tolerance_sec=.015)
    p['dynamic_map']['enabled']=False  # Re-enable after real-sensor pose/noise validation.
    records={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in (SOURCE,META)}
    p['calibration_provenance']={'sha256':records,'imu_relative_transform':'Ouster factory imu_to_sensor_transform; sensor_frame cloud',
        'stereo_source':'P3 measured RGB10/RGB11 K,D and right-to-left transform',
        'camera_driver_info':'placeholder; not used as calibration',
        'mounting_unchanged_since_20260817':True,
        'mounting_confirmation':'User confirmed installation unchanged on 2026-09-16',
        'body_mounting_status':'2026-08-17 mounting unchanged; FRD-to-FLU published-frame conversion still requires projection/yaw validation',
        'cross_sensor_time_status':'software offset from local driver DDS publication timestamps; not hardware/PTP synchronization'}
    (ROOT/'config/hardware104.yaml').write_text(yaml.safe_dump(p,sort_keys=False,allow_unicode=True))
    path=ROOT/'results/hardware104_deployment';path.mkdir(parents=True,exist_ok=True)
    (path/'source_calibration.yaml').write_text(SOURCE.read_text())
    (path/'ouster_metadata.json').write_text(META.read_text())
    print('Generated config/hardware104.yaml; source hashes and remaining mounting checks recorded')


if __name__=='__main__':main()
