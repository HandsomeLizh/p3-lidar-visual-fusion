"""Bound output discontinuities, including repeated corrections of one stamp."""
import numpy as np
from .core import motion


class OutputContinuity:
    def __init__(self, max_speed=4., max_angular_speed=2., max_step=6., max_angle=1.,
                 correction_translation=.5, correction_rotation=.15):
        values=[max_speed,max_angular_speed,max_step,max_angle,correction_translation,correction_rotation]
        if any(not np.isfinite(v) or v<=0 for v in values):
            raise ValueError("Output continuity limits must be finite and positive")
        self.max_speed,self.max_angular_speed=max_speed,max_angular_speed
        self.max_step,self.max_angle=max_step,max_angle
        self.correction_translation,self.correction_rotation=correction_translation,correction_rotation
        self.latest=None
        self.stamp_origin=None

    def check(self, stamp, transform):
        if self.latest is None:return "qualified"
        old_stamp,old=self.latest
        dt=stamp-old_stamp
        if dt<0:return "out_of_order"
        if dt==0:
            # Compare to the first accepted result of this timestamp so a
            # sequence of small corrections cannot bypass the total bound.
            _,distance,angle=motion(self.stamp_origin,transform)
            if distance>self.correction_translation or angle>self.correction_rotation:
                return "same_stamp_discontinuity"
        else:
            _,distance,angle=motion(old,transform)
            if (distance>min(self.max_step,self.max_speed*dt+self.correction_translation) or
                angle>min(self.max_angle,self.max_angular_speed*dt+self.correction_rotation)):
                return "output_discontinuity"
        return "qualified"

    def accept(self, stamp, transform):
        if self.latest is None or stamp>self.latest[0]:self.stamp_origin=transform.copy()
        self.latest=(stamp,transform.copy())
