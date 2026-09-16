"""Check visual motion against independent, timestamp-aligned LiDAR motion.

Agreement is a rejection aid, not proof that either estimator is correct.
Both poses must describe base_link; their world origins may differ.
"""
from collections import deque
from dataclasses import dataclass
import numpy as np
from scipy.spatial.transform import Rotation
from .core import inverse, motion, rigid
from .continuous_frame import pose_covariance


def increment_covariance(previous, current, previous_cov, current_cov):
    """Conservative relative-pose uncertainty from two ROS pose covariances.

    Translation is in the previous body frame; angle error is in the current
    body frame. Cross-time correlation is unavailable, so endpoints are treated
    independently. This is a rejection margin, not an accuracy guarantee.
    """
    r0, r1 = previous[:3, :3].T, current[:3, :3].T
    x, y, z = current[:3, 3]-previous[:3, 3]
    skew = np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])
    j0, j1 = np.zeros((6, 6)), np.zeros((6, 6))
    j0[:3, :3], j1[:3, :3] = -r0, r0
    j0[:3, 3:] = r0@skew
    j0[3:, 3:], j1[3:, 3:] = -r1, r1
    return j0@previous_cov@j0.T+j1@current_cov@j1.T


@dataclass(frozen=True)
class ConsistencyResult:
    ready: bool
    reason: str
    score: float = 0.
    translation_error: float = 0.
    rotation_error: float = 0.
    interval: float = 0.


class MotionConsistency:
    def __init__(self, translation_floor=.15, translation_ratio=.20,
                 rotation_floor=.14, rotation_ratio=.20,
                 min_span=.5, history_seconds=4., max_gap=6., capacity=240):
        self.translation_floor=float(translation_floor)
        self.translation_ratio=float(translation_ratio)
        self.rotation_floor=float(rotation_floor)
        self.rotation_ratio=float(rotation_ratio)
        self.min_span=float(min_span)
        self.history_seconds=float(history_seconds)
        self.max_gap=float(max_gap)
        if min(self.translation_floor,self.rotation_floor,self.min_span,self.history_seconds,self.max_gap)<=0:
            raise ValueError("Motion consistency limits must be positive")
        if min(self.translation_ratio,self.rotation_ratio)<0 or capacity<2:
            raise ValueError("Invalid motion consistency ratio/capacity")
        self.samples=deque(maxlen=int(capacity))

    def reset(self):
        self.samples.clear()

    def check(self,stamp,visual,reference,reference_covariance=None):
        visual,reference=rigid(visual),rigid(reference)
        covariance=None if reference_covariance is None else pose_covariance(reference_covariance)
        if self.samples and (stamp<=self.samples[-1][0] or stamp-self.samples[-1][0]>self.max_gap):
            self.reset()
        while len(self.samples)>1 and stamp-self.samples[1][0]>self.history_seconds:
            self.samples.popleft()
        if not self.samples:
            self.samples.append((stamp,visual.copy(),reference.copy(),covariance))
            return ConsistencyResult(False,"motion_warmup")
        # Compare the recent increment and a longer interval, catching steady
        # scale/direction errors that a per-frame displacement threshold misses.
        anchors=[self.samples[-1]]
        if self.samples[0] is not self.samples[-1]:
            anchors.append(self.samples[0])
        worst=0.;details=(0.,0.,0.)
        for old_stamp,old_visual,old_reference,old_covariance in anchors:
            visual_delta=inverse(old_visual)@visual
            reference_delta=inverse(old_reference)@reference
            _,distance,angle=motion(old_reference,reference)
            displacement_error=visual_delta[:3,3]-reference_delta[:3,3]
            error=float(np.linalg.norm(displacement_error))
            _,_,rotation=motion(reference_delta,visual_delta)
            trans_limit=self.translation_floor+self.translation_ratio*distance
            rot_limit=self.rotation_floor+self.rotation_ratio*angle
            ratio=max(error/trans_limit,rotation/rot_limit)
            if covariance is not None and old_covariance is not None:
                # A weak LiDAR direction cannot be used as exact truth to reject
                # an independently qualified visual observation. Preserve the
                # existing model-error limits and add a directional 3-sigma
                # reference margin. Well-constrained directions remain checked.
                uncertainty=increment_covariance(old_reference,reference,old_covariance,covariance)
                translation_budget=np.eye(3)*trans_limit**2+9.*uncertainty[:3,:3]
                rotation_budget=np.eye(3)*rot_limit**2+9.*uncertainty[3:,3:]
                rotation_error=Rotation.from_matrix(reference_delta[:3,:3].T@visual_delta[:3,:3]).as_rotvec()
                ratio=max(float(np.sqrt(max(0.,displacement_error@np.linalg.solve(translation_budget,displacement_error)))),
                          float(np.sqrt(max(0.,rotation_error@np.linalg.solve(rotation_budget,rotation_error)))))
            if ratio>=worst:
                worst=ratio;details=(error,rotation,stamp-old_stamp)
            if ratio>1.:
                self.reset()
                return ConsistencyResult(False,"visual_lidar_disagreement",0.,error,rotation,stamp-old_stamp)
        span=stamp-self.samples[0][0]
        self.samples.append((stamp,visual.copy(),reference.copy(),covariance))
        if span<self.min_span:
            return ConsistencyResult(False,"motion_warmup",0.,*details)
        return ConsistencyResult(True,"motion_consistent",max(.25,1.-.5*worst),*details)
