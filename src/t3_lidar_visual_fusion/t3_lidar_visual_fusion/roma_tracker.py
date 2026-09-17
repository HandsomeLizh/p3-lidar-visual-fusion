"""Pose-only adapter for the existing P3 RoMa frontend, with bounded history.

The original model, stereo triangulation and temporal PnP remain in roma_vo.
No original project files are modified. Backend/loop corrections are disabled:
this mapper cannot replay old observations after a historical pose correction.
"""
from contextlib import contextmanager
import hashlib
import importlib
from pathlib import Path
import sys
from urllib.parse import urlparse

import cv2
import numpy as np

from .core import inverse, motion, rigid
from .learned_tracker import TrackerResult
from .stationary import StationaryDetector
from .stereo_geometry import StereoGeometry, TrackingFailure, coverage


def camera_pose_to_body(camera_pose, base_from_rect):
    """T_B0_Bt = T_B_C * T_C0_Ct * T_C_B, including the lever arm."""
    mount = rigid(base_from_rect)
    return rigid(mount @ rigid(camera_pose) @ inverse(mount))


def bound_history(slam, keyframe_limit=12, pose_limit=128):
    if slam.enable_backend:
        raise ValueError('Cannot trim a frontend with an active global backend')
    removed = slam.keyframes[:-keyframe_limit]
    slam.keyframes[:] = slam.keyframes[-keyframe_limit:]
    keep_ids = {k.id for k in slam.keyframes}
    for k in removed:
        slam.relocalizer.remove(k.id)
    active_points = {p.id for k in slam.keyframes for p in k.mappoints if p is not None}
    for key in list(slam.global_mps):
        if key not in active_points:
            del slam.global_mps[key]
        else:
            point = slam.global_mps[key]
            for observation in list(point.observations):
                if observation not in keep_ids:
                    point.remove_observation(observation)
    keep_indices = set(range(max(0, len(slam.poses) - pose_limit), len(slam.poses)))
    keep_indices.update(i for k, i in slam._kf_id_to_pose_idx.items() if k in keep_ids)
    indices = sorted(keep_indices)
    remap = {old: new for new, old in enumerate(indices)}
    for name in ('poses', '_pose_reference_kf_ids', '_pose_relative_to_kf'):
        values = getattr(slam, name)
        setattr(slam, name, [values[i] for i in indices])
    slam._kf_id_to_pose_idx = {k: remap[i] for k, i in slam._kf_id_to_pose_idx.items() if k in keep_ids}
    slam.trim_keyframe_images(2)


@contextmanager
def offline_weights(torch, directory):
    """Use existing checkpoint files; memory-map CPU storage during startup."""
    original = torch.hub.load_state_dict_from_url
    def local(url, *args, file_name=None, **kwargs):
        path = Path(directory) / (file_name or Path(urlparse(url).path).name)
        if not path.is_file():
            raise FileNotFoundError('Existing RoMa checkpoint missing: ' + str(path))
        return torch.load(path, map_location='cpu', weights_only=True, mmap=True)
    torch.hub.load_state_dict_from_url = local
    try:
        yield
    finally:
        torch.hub.load_state_dict_from_url = original


class RomaStereoTracker:
    def __init__(self, profile):
        self.cfg = profile['learned_visual']
        self.geometry = StereoGeometry(profile)
        self.epoch = 1
        self.last_pose = None
        self.stationary = StationaryDetector(require_cloud=False,
            max_gap=float(self.cfg.get('roma_stationary_max_gap_sec',6.)))
        self.correction = np.eye(4)
        self.poisoned = False
        self.recovery_gap = float(self.cfg.get('max_recovery_gap_sec', 30.))
        self.keyframe_limit = int(self.cfg.get('roma_keyframe_limit', 12))
        if not 2 <= self.keyframe_limit <= 32:
            raise ValueError('RoMa keyframe limit must be 2..32')
        available = next(int(l.split()[1]) / 1024 for l in Path('/proc/meminfo').read_text().splitlines()
                         if l.startswith('MemAvailable:'))
        minimum=float(self.cfg.get('roma_min_available_mib',8192))
        if available < minimum:
            raise RuntimeError(f'RoMa needs at least {minimum:.0f} MiB available at startup; currently {available:.0f} MiB')
        source = Path(self.cfg['roma_source_root']).resolve()
        source_file = source / 'roma_vo.py'
        digest = hashlib.sha256(source_file.read_bytes()).hexdigest()
        if digest != self.cfg['roma_source_sha256']:
            raise RuntimeError('Original RoMa source changed; revalidate the adapter before starting')
        sys.path.insert(0, str(source))
        import torch
        self.torch = torch
        if not torch.cuda.is_available():
            raise RuntimeError('RoMa online trial requires CUDA')
        torch.set_num_threads(2)
        torch.set_float32_matmul_precision('highest')
        module = importlib.import_module('roma_vo')
        k = self.geometry.k
        calibration = dict(camera=dict(fx=k[0, 0], fy=k[1, 1], cx=k[0, 2], cy=k[1, 2]),
                           baseline=abs(self.geometry.p1[0, 3] / k[0, 0]))
        with offline_weights(torch, self.cfg['roma_checkpoint_dir']):
            self.slam = module.StereoSLAMv2(
                calib=calibration, device='cuda', coarse_res=280, upsample_res=280,
                stereo_matches=5000, temporal_matches=2000, use_custom_corr=False,
                stereo_upsample_refinement=True, temporal_upsample_refinement=False,
                symmetric_matching=False, enable_feature_reuse=True, batch_coarse_pairs=True,
                enable_klt_temporal_tracking=False, min_depth=.4, max_depth=60.,
                epipolar_threshold=1.5, keyframe_interval=5,
                enable_backend=False, enable_loop_closing=False, imu_helper=None)
        self.slam.keyframe_image_cache_limit = 2

    def reject(self, stamp):
        self.stationary.invalidate('visual_observation_rejected')
        # Never disguise lost tracking or a large jump as a fresh origin.
        if self.last_pose is not None and stamp - self.last_pose[0] > self.recovery_gap:
            self.poisoned = True
        return self.last_pose is not None and not self.poisoned

    def process(self, stamp, left, right):
        if self.poisoned:
            raise TrackingFailure('roma_tracking_restart_required')
        if self.last_pose is not None and stamp <= self.last_pose[0]:
            raise TrackingFailure('nonmonotonic_visual_frame')
        if left.shape != (self.geometry.size[1], self.geometry.size[0], 3) or right.shape != left.shape:
            raise TrackingFailure('uncalibrated_roma_image_dimensions')
        left = cv2.remap(left, *self.geometry.map0, cv2.INTER_LINEAR)
        right = cv2.remap(right, *self.geometry.map1, cv2.INTER_LINEAR)
        for side, frame in enumerate((left, right)):
            self.stationary.image(stamp, cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), side)
        with self.torch.inference_mode():
            raw_camera = self.slam.process_frame(left, right)
        status = self.slam.get_tracking_status()
        bound_history(self.slam, self.keyframe_limit)
        metrics = dict(reason=status.reason, pnp_inliers=status.inlier_count,
                       pnp_ratio=status.inlier_count / max(1, status.correspondence_count),
                       tracking_valid=status.success and not status.initialized,
                       reference_reset=False, keyframe_replaced=status.keyframe_created,
                       stereo_points=len(self.slam._kf_pts3d) if self.slam._kf_pts3d is not None else 0)
        if not status.success:
            raise TrackingFailure(status.reason, metrics)
        snapshot=self.slam.get_feature_match_snapshot()
        spread=0.
        if snapshot is not None:
            accepted=np.asarray(snapshot['temporal_inlier_mask'],dtype=bool)
            spread=coverage(np.asarray(snapshot['temporal_current_points'])[accepted],self.geometry.size)
        metrics['pnp_coverage']=spread
        # RoMa dense sampling + nearest stereo-depth association has a different
        # candidate population from sparse descriptor matches. Use its original
        # admission ratio (0.25), and additionally require 50 inliers + coverage.
        if not status.initialized and (status.inlier_count < self.cfg.get('roma_min_pnp_inliers',50)
                                       or metrics['pnp_ratio'] < self.cfg.get('roma_min_pnp_ratio',.25)
                                       or spread < self.cfg.get('roma_min_pnp_coverage',.15)):
            raise TrackingFailure('roma_pnp_quality_rejected',metrics)
        raw_body = camera_pose_to_body(raw_camera, self.geometry.base_from_rect)
        body = self.correction @ raw_body
        if self.last_pose is not None:
            previous_stamp, previous = self.last_pose
            dt = stamp - previous_stamp
            delta, distance, angle = motion(previous, body)
            if dt > self.recovery_gap or distance / dt > self.cfg.get('max_speed_mps', 4.) or angle / dt > 2.:
                self.poisoned = True
                raise TrackingFailure('roma_motion_jump_restart_required', metrics)
            velocity = np.r_[delta[:3, 3], cv2.Rodrigues(delta[:3, :3])[0].reshape(3)] / dt
            if self.stationary.check(stamp, velocity):
                body = previous.copy()
                # Cancel internal stationary creep too, so it cannot reappear on departure.
                self.correction = body @ inverse(raw_body)
                self.stationary.zero_updates += 1
        self.last_pose = (stamp, body.copy())
        metrics['stationary'] = self.stationary.status()
        return TrackerResult(body, self.epoch, bool(status.initialized), metrics)

    def resource_snapshot(self):
        torch = self.torch
        return dict(backend='roma', source='original_P3_StereoSLAMv2',
                    keyframes=len(self.slam.keyframes), keyframe_limit=self.keyframe_limit,
                    map_points=len(self.slam.global_mps), retained_poses=len(self.slam.poses),
                    cuda_allocated_bytes=torch.cuda.memory_allocated(),
                    cuda_reserved_bytes=torch.cuda.memory_reserved(),
                    cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated())
