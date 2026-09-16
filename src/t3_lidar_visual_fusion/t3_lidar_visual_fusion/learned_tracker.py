"""Two bounded stereo references; failed observations never replace good geometry."""
from dataclasses import dataclass
import time
import numpy as np
from .core import inverse, motion
from .stereo_geometry import StereoGeometry, TrackingFailure


@dataclass
class TrackerResult:
    base_pose: np.ndarray
    epoch: int
    anchor: bool
    metrics: dict


class LearnedStereoTracker:
    def __init__(self,profile,backend):
        self.cfg=profile["learned_visual"]
        self.geometry=StereoGeometry(profile)
        self.backend=backend
        self.keyframe=None
        self.backup=None
        self.last_pose=None
        self.epoch=0
        self.failed_frames=0
        self.recovery_gap=float(self.cfg.get("max_recovery_gap_sec",30.))
        if not np.isfinite(self.recovery_gap) or not 0<self.recovery_gap<=120.:
            raise ValueError("max_recovery_gap_sec must be in (0,120]")

    def reset(self):
        self.keyframe=None
        self.backup=None
        self.last_pose=None
        self.failed_frames=0

    def reject(self,stamp):
        """Keep only recent references; no invented pose across a rejected frame."""
        self.failed_frames+=1
        if self.keyframe is not None and stamp-self.keyframe[0]>self.recovery_gap:
            self.reset()
        if self.backup is not None and stamp-self.backup[0]>self.recovery_gap:self.backup=None
        return self.keyframe is not None

    def _seed(self,stamp,features,points,metrics,reason):
        self.epoch+=1
        world_from_rect=self.geometry.base_from_rect.copy()
        self.keyframe=(stamp,features,points,world_from_rect)
        self.backup=None
        self.last_pose=(stamp,np.eye(4))
        self.failed_frames=0
        metrics.update(reason=reason,tracking_valid=False,reference_reset=True)
        return TrackerResult(np.eye(4),self.epoch,True,metrics)

    def process(self,stamp,left,right):
        metrics={}
        if not np.isfinite(stamp) or (self.last_pose is not None and stamp<=self.last_pose[0]):
            raise TrackingFailure("nonmonotonic_visual_frame")
        start=time.perf_counter()
        left,right=self.geometry.rectify(left,right)
        metrics["rectify_sec"]=time.perf_counter()-start
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
        if self.keyframe is None:
            return self._seed(stamp,f0,points,metrics,"initialized")
        if stamp-self.keyframe[0]>self.recovery_gap:
            return self._seed(stamp,f0,points,metrics,"recovery_reference_expired")
        references=[self.keyframe]
        if self.backup is not None and stamp-self.backup[0]<=self.recovery_gap:references.append(self.backup)
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
                metrics.update(result.metrics)
                metrics.update(reference_stamp_sec=old_stamp,reference_age_sec=stamp-old_stamp,
                    reference_candidates_tried=index+1,recovered_reference=bool(self.failed_frames or index or
                    stamp-old_stamp>self.cfg.get("max_tracking_gap_sec",3.)))
                break
            except (TrackingFailure,ValueError) as exc:
                failure=exc;metrics.update(getattr(exc,'metrics',{}))
            finally:
                metrics["geometry_sec"]=metrics.get("geometry_sec",0.)+time.perf_counter()-start
        else:
            metrics.update(reference_retained=True,reference_candidates_tried=len(references))
            raise TrackingFailure(str(failure)[:120],metrics)
        _,distance,angle=motion(world_from_old,world_from_current)
        replace=(stamp-old_stamp>=self.cfg.get("keyframe_max_age_sec",1.)
                 or distance>=self.cfg.get("keyframe_translation_m",.25)
                 or angle>=self.cfg.get("keyframe_rotation_rad",.15)
                 or result.metrics["pnp_inliers"]<self.cfg.get("keyframe_min_tracks",60))
        if replace:
            self.backup=(old_stamp,old_features,old_points,world_from_old)
            self.keyframe=(stamp,f0,points,world_from_current.copy())
        self.last_pose=(stamp,base_pose.copy())
        self.failed_frames=0
        metrics.update(reason="tracked",tracking_valid=True,reference_reset=False,keyframe_replaced=replace)
        return TrackerResult(base_pose,self.epoch,False,metrics)
