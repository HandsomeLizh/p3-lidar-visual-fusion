"""Bounded stereo keyframes; failed observations never replace good geometry."""
from dataclasses import dataclass
from collections import deque
import time
import cv2
import numpy as np
from .core import inverse, motion
from .stereo_geometry import StereoGeometry, TrackingFailure
from .stationary import StationaryDetector


@dataclass
class TrackerResult:
    base_pose: np.ndarray
    epoch: int
    anchor: bool
    metrics: dict
    stereo_points: np.ndarray = None


class LearnedStereoTracker:
    def __init__(self,profile,backend):
        self.cfg=profile["learned_visual"]
        self.geometry=StereoGeometry(profile)
        self.backend=backend
        self.stationary=(StationaryDetector(require_cloud=False)
            if self.cfg.get('image_stationary_enabled',False) else None)
        self.keyframe=None
        self.backup=None
        self.last_pose=None
        self.epoch=0
        self.failed_frames=0
        self.current_stereo=None
        self.recovery_gap=float(self.cfg.get("max_recovery_gap_sec",30.))
        if not np.isfinite(self.recovery_gap) or not 0<self.recovery_gap<=120.:
            raise ValueError("max_recovery_gap_sec must be in (0,120]")
        capacity=int(self.cfg.get('recovery_keyframes',0))
        if not 0<=capacity<=4:raise ValueError('recovery_keyframes must be in [0,4]')
        self.recovery_keyframes=deque(maxlen=capacity)
        self.recovery_candidate=None

    def reset(self):
        self.keyframe=None
        self.backup=None
        self.last_pose=None
        self.failed_frames=0
        self.current_stereo=None
        self.recovery_keyframes.clear();self.recovery_candidate=None

    def reject(self,stamp):
        """Keep only recent references; no invented pose across a rejected frame."""
        self.failed_frames+=1
        if self.stationary is not None:self.stationary.invalidate('visual_observation_rejected')
        if self.last_pose is not None and stamp-self.last_pose[0]>self.recovery_gap:
            self.reset()
        if self.backup is not None and stamp-self.backup[0]>self.recovery_gap:self.backup=None
        if self.recovery_candidate is not None and stamp-self.recovery_candidate[1]>6.:
            self.recovery_candidate=None
        return self.keyframe is not None

    def _seed(self,stamp,features,points,metrics,reason):
        if self.stationary is not None:self.stationary.invalidate('new_visual_reference')
        self.epoch+=1
        world_from_rect=self.geometry.base_from_rect.copy()
        self.keyframe=(stamp,features,points,world_from_rect)
        self.backup=None
        self.last_pose=(stamp,np.eye(4))
        self.failed_frames=0
        self.recovery_keyframes.clear();self.recovery_candidate=None
        metrics.update(reason=reason,tracking_valid=False,reference_reset=True)
        return TrackerResult(np.eye(4),self.epoch,True,metrics)

    def process(self,stamp,left,right):
        # This bounded observation belongs to this image pair, independently of
        # whether temporal PnP succeeds. Mapping still needs a qualified body pose.
        self.current_stereo=None
        metrics={}
        if not np.isfinite(stamp) or (self.last_pose is not None and stamp<=self.last_pose[0]):
            raise TrackingFailure("nonmonotonic_visual_frame")
        start=time.perf_counter()
        left,right=self.geometry.rectify(left,right)
        metrics["rectify_sec"]=time.perf_counter()-start
        if self.stationary is not None:
            start=time.perf_counter()
            self.stationary.image(stamp,left,0)
            self.stationary.image(stamp,right,1)
            metrics['stationary_sec']=time.perf_counter()-start
        start=time.perf_counter()
        f0=self.backend.extract(left);f1=self.backend.extract(right)
        metrics["extract_sec"]=time.perf_counter()-start
        metrics.update(left_features=len(f0["pixels"]),right_features=len(f1["pixels"]))
        start=time.perf_counter()
        stereo=self.backend.match(f0,f1)
        metrics["stereo_match_sec"]=time.perf_counter()-start
        start=time.perf_counter()
        points,quality=self.geometry.triangulate(f0["pixels"],f1["pixels"],stereo)
        metrics.update(quality)
        metrics["triangulate_sec"]=time.perf_counter()-start
        if quality["stereo_points"]<self.cfg.get("min_stereo_points",35):
            raise TrackingFailure("insufficient_stereo_points",metrics)
        if quality["stereo_coverage"]<self.cfg.get("min_stereo_coverage",.12):
            raise TrackingFailure("stereo_poor_spatial_coverage",metrics)
        self.current_stereo=(stamp,points)
        if self.keyframe is None:
            return self._seed(stamp,f0,points,metrics,"initialized")
        # A reference matched successfully moments ago remains usable even if
        # it was captured during a long stationary interval.
        if stamp-self.last_pose[0]>self.recovery_gap:
            return self._seed(stamp,f0,points,metrics,"recovery_reference_expired")
        references=[self.keyframe]
        if self.backup is not None and stamp-self.backup[0]<=self.recovery_gap:references.append(self.backup)
        ordinary_count=len(references)
        used_stamps={frame[0] for frame in references}
        references.extend(frame for frame in reversed(self.recovery_keyframes) if frame[0] not in used_stamps)
        failure=None
        for index,(old_stamp,old_features,old_points,world_from_old) in enumerate(references):
            start=time.perf_counter()
            temporal=self.backend.match(old_features,f0)
            metrics["temporal_match_sec"]=metrics.get("temporal_match_sec",0.)+time.perf_counter()-start
            metrics["temporal_matches"]=len(temporal)
            start=time.perf_counter()
            try:
                result=self.geometry.estimate(old_points,points,old_features["pixels"],f0["pixels"],temporal)
                world_from_current=world_from_old@inverse(result.current_from_reference)
                base_pose=world_from_current@inverse(self.geometry.base_from_rect)
                previous_stamp,previous_pose=self.last_pose
                dt=stamp-previous_stamp
                _,distance,angle=motion(previous_pose,base_pose)
                if dt<=0 or distance/dt>self.cfg.get("max_speed_mps",4.) or angle/dt>self.cfg.get("max_angular_speed_rps",2.):
                    raise TrackingFailure("visual_motion_jump",result.metrics)
                archived=index>=ordinary_count
                recovering=(archived or (self.recovery_keyframes.maxlen and
                    dt>self.cfg.get('max_tracking_gap_sec',3.)))
                if recovering:
                    # Older views and long blind intervals need stronger
                    # evidence than an adjacent-frame match on repeated rocks.
                    evidence=result.metrics
                    if (evidence.get('pnp_inliers',0)<max(60,self.cfg.get('min_pnp_inliers',25)) or
                            evidence.get('pnp_ratio',0)<.7 or evidence.get('pnp_coverage',0)<.25 or
                            evidence.get('stereo_motion_ratio',0)<.8):
                        raise TrackingFailure('recovery_keyframe_geometry_weak',evidence)
                    pending=self.recovery_candidate
                    confirmed=False
                    if pending is not None and pending[0]==old_stamp and 0.<stamp-pending[1]<=6.:
                        _,shift,turn=motion(pending[2],base_pose)
                        confirmed=(shift<=min(.5,self.cfg.get('max_speed_mps',4.)*(stamp-pending[1]))
                                   and turn<=.25)
                    if not confirmed:
                        self.recovery_candidate=(old_stamp,stamp,base_pose.copy())
                        waiting=dict(evidence,recovery_keyframe_stamp_sec=old_stamp,recovery_confirmations=1)
                        raise TrackingFailure('recovery_keyframe_confirming',waiting)
                    metrics.update(recovery_keyframe_stamp_sec=old_stamp,recovery_confirmations=2)
                metrics.update(result.metrics)
                metrics.update(reference_stamp_sec=old_stamp,reference_age_sec=stamp-old_stamp,
                    reference_candidates_tried=index+1,recovered_reference=bool(self.failed_frames or index or
                    dt>self.cfg.get("max_tracking_gap_sec",3.)),recovered_archived_keyframe=archived)
                break
            except (TrackingFailure,ValueError) as exc:
                if str(exc)=='recovery_keyframe_confirming':raise
                failure=exc;metrics.update(getattr(exc,'metrics',{}))
            finally:
                metrics["geometry_sec"]=metrics.get("geometry_sec",0.)+time.perf_counter()-start
        else:
            self.recovery_candidate=None
            metrics.update(reference_retained=True,reference_candidates_tried=len(references))
            raise TrackingFailure(str(failure)[:120],metrics)
        if self.stationary is not None:
            delta,_,_=motion(previous_pose,base_pose)
            velocity=np.r_[delta[:3,3],cv2.Rodrigues(delta[:3,:3])[0].reshape(3)]/dt
            if self.stationary.check(stamp,velocity):
                base_pose=previous_pose.copy()
                world_from_current=base_pose@self.geometry.base_from_rect
                self.stationary.zero_updates+=1
            metrics['stationary']=self.stationary.status()
        _,distance,angle=motion(world_from_old,world_from_current)
        age_due=(stamp-old_stamp>=self.cfg.get("keyframe_max_age_sec",1.) and
                 (distance>=self.cfg.get("keyframe_min_translation_m",.05)-1e-6 or
                  angle>=self.cfg.get("keyframe_min_rotation_rad",.01)))
        # Keep a common reference for small motion; poses still update every
        # frame, including genuine slow movement. Time alone must not integrate
        # fresh triangulation noise on each stationary observation.
        replace=(archived or age_due
                 or distance>=self.cfg.get("keyframe_translation_m",.25)
                 or angle>=self.cfg.get("keyframe_rotation_rad",.15)
                 or result.metrics["pnp_inliers"]<self.cfg.get("keyframe_min_tracks",60))
        if replace:
            self.backup=(old_stamp,old_features,old_points,world_from_old)
            self.keyframe=(stamp,f0,points,world_from_current.copy())
        self.last_pose=(stamp,base_pose.copy())
        self.failed_frames=0
        self.recovery_candidate=None
        if self.recovery_keyframes.maxlen:
            remember=not self.recovery_keyframes
            if not remember:
                _,shift,turn=motion(self.recovery_keyframes[-1][3],world_from_current)
                remember=shift>=2. or turn>=.5
            if remember:self.recovery_keyframes.append((stamp,f0,points,world_from_current.copy()))
        metrics['retained_recovery_keyframes']=len(self.recovery_keyframes)
        metrics.update(reason="tracked",tracking_valid=True,reference_reset=False,keyframe_replaced=replace)
        return TrackerResult(base_pose,self.epoch,False,metrics,points)
