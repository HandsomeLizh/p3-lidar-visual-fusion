"""Check visual motion against independent, timestamp-aligned LiDAR motion.

Agreement is a rejection aid, not proof that either estimator is correct.
Both poses must describe base_link; their world origins may differ.
"""
from collections import deque
from dataclasses import dataclass
import numpy as np
from .core import inverse, motion, rigid


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

    def check(self,stamp,visual,reference):
        visual,reference=rigid(visual),rigid(reference)
        if self.samples and (stamp<=self.samples[-1][0] or stamp-self.samples[-1][0]>self.max_gap):
            self.reset()
        while len(self.samples)>1 and stamp-self.samples[1][0]>self.history_seconds:
            self.samples.popleft()
        if not self.samples:
            self.samples.append((stamp,visual.copy(),reference.copy()))
            return ConsistencyResult(False,"motion_warmup")
        # Compare the recent increment and a longer interval, catching steady
        # scale/direction errors that a per-frame displacement threshold misses.
        anchors=[self.samples[-1]]
        if self.samples[0] is not self.samples[-1]:
            anchors.append(self.samples[0])
        worst=0.;details=(0.,0.,0.)
        for old_stamp,old_visual,old_reference in anchors:
            visual_delta=inverse(old_visual)@visual
            reference_delta=inverse(old_reference)@reference
            _,distance,angle=motion(old_reference,reference)
            error=float(np.linalg.norm(visual_delta[:3,3]-reference_delta[:3,3]))
            _,_,rotation=motion(reference_delta,visual_delta)
            trans_limit=self.translation_floor+self.translation_ratio*distance
            rot_limit=self.rotation_floor+self.rotation_ratio*angle
            ratio=max(error/trans_limit,rotation/rot_limit)
            if ratio>=worst:
                worst=ratio;details=(error,rotation,stamp-old_stamp)
            if ratio>1.:
                self.reset()
                return ConsistencyResult(False,"visual_lidar_disagreement",0.,error,rotation,stamp-old_stamp)
        span=stamp-self.samples[0][0]
        self.samples.append((stamp,visual.copy(),reference.copy()))
        if span<self.min_span:
            return ConsistencyResult(False,"motion_warmup",0.,*details)
        return ConsistencyResult(True,"motion_consistent",max(.25,1.-.5*worst),*details)
