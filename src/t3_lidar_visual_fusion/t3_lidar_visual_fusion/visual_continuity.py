"""Continuous visual pose in the established odom frame, with bounded history.

The visual frontend owns its trajectory. A co-timed qualified reference supplies
only the frame transform. A new visual epoch or unobserved gap requires a new
reference; it must never be attached to the last pose at a different time.
"""
from collections import deque
import numpy as np
from .core import inverse,rigid,motion
from .continuous_frame import pose_covariance
from .body_motion import _perturb


class VisualContinuity:
    def __init__(self,max_gap=6.,stamp_tolerance=.08,history_samples=128,
                 translation_drift_ratio=.01,rotation_drift_per_m=.001,
                 position_random_walk=.00002,rotation_random_walk=.000001):
        self.max_gap=float(max_gap);self.tolerance=float(stamp_tolerance)
        self.distance_noise=float(translation_drift_ratio)
        self.angle_noise=float(rotation_drift_per_m)
        self.position_walk=float(position_random_walk);self.rotation_walk=float(rotation_random_walk)
        bounds=[self.max_gap,self.tolerance,self.distance_noise,self.angle_noise,self.position_walk,self.rotation_walk]
        if not np.isfinite(bounds).all() or min(bounds)<=0 or history_samples<2:
            raise ValueError('Invalid visual continuity limits')
        self.raw=deque(maxlen=int(history_samples));self.reference=None
        self.distance=0.;self.reason='waiting_for_reference';self.epoch=None
        self.origin=None;self.segment_start=None

    def remember_origin(self,stamp,epoch,transform):
        """Remember an explicit frontend gauge definition, never a motion sample.

        The frontend publishes this only when it creates an identity epoch after
        validating stereo geometry. Ordinary rejected/held poses do not qualify.
        A later tracked frame and a co-timed qualified output are both required.
        """
        transform=rigid(transform)
        if (not np.isfinite(stamp) or not epoch or epoch in ('map','odom') or
                not np.allclose(transform,np.eye(4),rtol=0,atol=1e-8)):
            raise ValueError('Invalid visual epoch origin')
        if self.origin is not None and stamp<=self.origin[0]:return False
        self.origin=(float(stamp),epoch,transform.copy())
        return True

    def observe(self,stamp,epoch,transform,covariance):
        transform=rigid(transform);covariance=pose_covariance(covariance)
        if not np.isfinite(stamp):raise ValueError('Nonfinite visual timestamp')
        if self.raw and stamp<=self.raw[-1][0]:raise ValueError('Nonmonotonic visual timestamp')
        if epoch!=self.epoch or (self.raw and stamp-self.raw[-1][0]>self.max_gap):
            self.raw.clear();self.reference=None;self.distance=0.
            self.segment_start=float(stamp)
            self.reason='new_visual_segment_requires_reference'
        if self.raw:self.distance+=motion(self.raw[-1][1],transform)[1]
        self.epoch=epoch
        self.raw.append((float(stamp),transform.copy(),covariance.copy(),self.distance))

    def anchor(self,stamp,transform,covariance):
        """May be called after delayed visual arrival; acquisition times match."""
        if not self.raw:return False
        sample=min(self.raw,key=lambda v:abs(v[0]-stamp))
        if abs(sample[0]-stamp)>self.tolerance:
            origin=self.origin
            first=self.raw[0]
            if not (origin is not None and origin[1]==self.epoch and
                    abs(origin[0]-stamp)<=self.tolerance and
                    first[0]==self.segment_start and
                    0.<first[0]-origin[0]<=self.max_gap):return False
            # The identity origin defines coordinates; it is not a measured
            # zero-motion pose. Reference and tracked-pose covariance remain in
            # estimate(), including the entire observed displacement from origin.
            sample=(origin[0],origin[2],np.zeros((6,6)),
                    -motion(origin[2],first[1])[1])
        if self.reference and sample[0]<=self.reference[0][0]:return False
        transform=rigid(transform);covariance=pose_covariance(covariance)
        previous=self.reference
        before=self.estimate() if previous is not None else None
        self.reference=(sample,transform.copy(),covariance.copy())
        if before is not None:
            after=self.estimate()
            # A recent but weak fused reference must not make an independently
            # continuous visual fallback less certain than its existing frame.
            # Compare both propagated candidates at the same current timestamp,
            # including the existing distance/time drift allowance.
            for block in [slice(0,3),slice(3,6)]:
                old=float(np.linalg.eigvalsh(before[2][block,block])[-1])
                new=float(np.linalg.eigvalsh(after[2][block,block])[-1])
                if new>old+1e-9:
                    self.reference=previous;self.reason='reference_uncertainty_worse';return False
        self.reason='anchored';return True

    def estimate(self):
        if not self.raw or self.reference is None:return None
        anchor,reference,reference_cov=self.reference
        current=self.raw[-1]
        if current[0]<anchor[0]:return None
        def compose(a,b,c):return a@inverse(b)@c
        result=compose(reference,anchor[1],current[1])
        # Fixed-axis ROS pose errors, propagated through the frame join. The
        # factor of three bounds unknown cross-correlation between its inputs;
        # repeated frames do not turn the frame anchor into new observations.
        covariance=np.zeros((6,6));epsilon=1e-5
        values=[reference,anchor[1],current[1]]
        for side,source_cov in enumerate([reference_cov,anchor[2],current[2]]):
            jacobian=np.empty((6,6))
            for axis in range(6):
                plus=list(values);minus=list(values)
                plus[side]=_perturb(values[side],axis,epsilon)
                minus[side]=_perturb(values[side],axis,-epsilon)
                a=compose(*plus);b=compose(*minus)
                jacobian[:3,axis]=(a[:3,3]-b[:3,3])/(2*epsilon)
                rotation=a[:3,:3]@b[:3,:3].T
                jacobian[3:,axis]=np.array([rotation[2,1]-rotation[1,2],rotation[0,2]-rotation[2,0],
                                           rotation[1,0]-rotation[0,1]])/(4*epsilon)
            covariance+=3.*jacobian@source_cov@jacobian.T
        distance=max(0.,current[3]-anchor[3]);elapsed=max(0.,current[0]-anchor[0])
        covariance[:3,:3]+=np.eye(3)*((self.distance_noise*distance)**2+self.position_walk*elapsed)
        covariance[3:,3:]+=np.eye(3)*((self.angle_noise*distance)**2+self.rotation_walk*elapsed)
        covariance=pose_covariance(.5*(covariance+covariance.T))
        self.reason='continuous_visual_pose'
        return current[0],result,covariance

    def status(self):
        return dict(reason=self.reason,epoch=self.epoch,anchored=self.reference is not None,
                    history_samples=len(self.raw),path_length_m=self.distance,
                    epoch_origin_stamp_sec=(self.origin[0] if self.origin is not None
                        and self.origin[1]==self.epoch else None))
