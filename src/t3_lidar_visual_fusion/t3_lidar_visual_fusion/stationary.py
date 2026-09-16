"""Conservative image + range-change stationarity evidence, without IMU/truth."""
from collections import deque
import time
import cv2
import numpy as np


class StationaryDetector:
    def __init__(self, confirm_frames=3, max_gap=3., stamp_tolerance=.05,
                 flow_limit_px=.25, range_limit_m=.03, speed_limit_mps=.003,
                 angular_limit_rps=.002):
        self.confirm_frames=int(confirm_frames)
        self.max_gap=float(max_gap);self.tolerance=float(stamp_tolerance)
        self.flow_limit=float(flow_limit_px);self.range_limit=float(range_limit_m)
        self.speed_limit=float(speed_limit_mps);self.angular_limit=float(angular_limit_rps)
        values=[self.max_gap,self.tolerance,self.flow_limit,self.range_limit,self.speed_limit,self.angular_limit]
        if self.confirm_frames<3 or not np.isfinite(values).all() or min(values)<=0:
            raise ValueError('Invalid stationarity thresholds')
        self.images=[None,None];self.image_last=[None,None]
        self.image_evidence=[deque(maxlen=12),deque(maxlen=12)]
        self.cloud_anchor=None;self.cloud_last=None;self.cloud_evidence=deque(maxlen=12)
        self.consecutive=0;self.last_check=None;self.last_wall=0.;self.zero_updates=0
        self.result={'state':'unknown','reason':'waiting_for_evidence'}
        self.telemetry_evidence=deque(maxlen=256)
        self.telemetry_wall=0.

    def velocity_feedback(self,stamp,velocity):
        velocity=np.asarray(velocity)
        if (velocity.shape!=(6,) or not np.isfinite(velocity).all() or not np.isfinite(stamp)
                or (self.telemetry_evidence and stamp<=self.telemetry_evidence[-1][0])):
            raise ValueError('Invalid or out-of-order velocity feedback')
        speed=float(np.linalg.norm(velocity[:3]));angular=float(np.linalg.norm(velocity[3:]))
        moving=speed>self.speed_limit or angular>self.angular_limit
        self.telemetry_evidence.append((stamp,moving))
        self.telemetry_wall=time.monotonic()
        if moving:
            self.invalidate('telemetry_motion_detected')
            self.result['state']='moving'

    def invalidate(self,reason):
        self.consecutive=0
        self.result={'state':'unknown','reason':reason}

    def image(self,stamp,array,side,valid=True):
        old_time=self.image_last[side];self.image_last[side]=stamp
        if not valid or array.ndim!=2 or array.dtype!=np.uint8:
            self.images[side]=None;self.image_evidence[side].append((stamp,None))
            self.invalidate('unusable_image');return
        scale=min(1.,320./array.shape[1])
        small=cv2.resize(array,(round(array.shape[1]*scale),round(array.shape[0]*scale)),interpolation=cv2.INTER_AREA)
        anchor=self.images[side]
        if old_time is None or not 0<stamp-old_time<=self.max_gap or anchor is None or anchor.shape!=small.shape:
            self.images[side]=small.copy();self.image_evidence[side].append((stamp,None));return
        corners=cv2.goodFeaturesToTrack(anchor,maxCorners=160,qualityLevel=.02,minDistance=8)
        evidence=None
        if corners is not None and len(corners)>=40:
            tracked,ok,_=cv2.calcOpticalFlowPyrLK(anchor,small,corners,None,winSize=(15,15),maxLevel=2)
            if tracked is not None:
                back,back_ok,_=cv2.calcOpticalFlowPyrLK(small,anchor,tracked,None,winSize=(15,15),maxLevel=2)
                if back is not None:
                    fb=np.linalg.norm(back[:,0]-corners[:,0],axis=1)
                    keep=ok[:,0].astype(bool)&back_ok[:,0].astype(bool)&np.isfinite(tracked[:,0]).all(axis=1)&(fb<.35)
                    points=corners[keep,0]
                    if len(points)>=40 and keep.mean()>=.7:
                        cells=np.clip((points/np.array([small.shape[1],small.shape[0]])*4).astype(int),0,3)
                        coverage=len(np.unique(cells[:,0]+4*cells[:,1]))/16.
                        if coverage>=.25:
                            flow=float(np.percentile(np.linalg.norm(tracked[keep,0]-points,axis=1),90))
                            evidence={'flow_p90_px':flow,'tracked':len(points),'coverage':coverage}
        self.image_evidence[side].append((stamp,evidence))
        # Retain a fixed image anchor during quiet intervals so subpixel creep
        # accumulates instead of being erased by re-anchoring each frame.
        if evidence is None or evidence['flow_p90_px']>self.flow_limit:
            self.images[side]=small.copy()
            self.invalidate('image_changed_or_unobservable')

    def cloud(self,stamp,xyz):
        xyz=np.asarray(xyz);ranges=np.linalg.norm(xyz,axis=1)
        keep=np.isfinite(xyz).all(axis=1)&(ranges>.5)&(ranges<40.)
        xyz=xyz[keep];ranges=ranges[keep]
        az=np.mod(np.arctan2(xyz[:,1],xyz[:,0]),2*np.pi)
        el=np.arctan2(xyz[:,2],np.hypot(xyz[:,0],xyz[:,1]))
        bins=np.clip((az/(2*np.pi)*64).astype(int),0,63)+64*np.clip(((el+np.pi/2)/np.pi*32).astype(int),0,31)
        signature=np.full(2048,np.inf);np.minimum.at(signature,bins,ranges)
        old=self.cloud_anchor;previous=self.cloud_last;self.cloud_last=stamp
        evidence=None
        if old is not None and previous is not None and 0<stamp-previous<=self.max_gap:
            overlap=np.isfinite(old)&np.isfinite(signature)
            union=np.isfinite(old)|np.isfinite(signature)
            if overlap.sum()>=60 and overlap.sum()/max(1,union.sum())>=.8:
                evidence={'range_change_p75_m':float(np.percentile(np.abs(signature[overlap]-old[overlap]),75)),
                          'overlap_bins':int(overlap.sum())}
        self.cloud_evidence.append((stamp,evidence))
        if evidence is None or evidence['range_change_p75_m']>self.range_limit:
            self.cloud_anchor=signature;self.invalidate('lidar_changed_or_unobservable')

    def _at(self,history,stamp):
        if not history:return None
        t,value=min(history,key=lambda item:abs(item[0]-stamp))
        return value if abs(t-stamp)<=self.tolerance else None

    def check(self,stamp,velocity):
        if self.last_check is not None and not 0<stamp-self.last_check<=self.max_gap:
            self.invalidate('nonconsecutive_visual_motion')
        self.last_check=stamp;self.last_wall=time.monotonic()
        telemetry='unavailable'
        if self.telemetry_evidence and time.monotonic()-self.telemetry_wall<=.5:
            t,moving=min(self.telemetry_evidence,key=lambda item:abs(item[0]-stamp))
            aligned=abs(t-stamp)<=.15
            # Current movement vetoes delayed visual claims of stationarity.
            # A quiet telemetry value alone never establishes a zero constraint.
            if self.telemetry_evidence[-1][1] or (aligned and moving):
                self.invalidate('telemetry_motion_detected');self.result['state']='moving'
                return False
            telemetry='quiet' if aligned else 'not_aligned'
        images=[self._at(h,stamp) for h in self.image_evidence]
        cloud=self._at(self.cloud_evidence,stamp)
        if any(v is None for v in images) or cloud is None:
            self.invalidate('missing_current_image_or_lidar_evidence');return False
        speed=float(np.linalg.norm(velocity[:3]));angular=float(np.linalg.norm(velocity[3:]))
        quiet=(np.isfinite(velocity).all() and speed<=self.speed_limit and angular<=self.angular_limit
               and max(v['flow_p90_px'] for v in images)<=self.flow_limit
               and cloud['range_change_p75_m']<=self.range_limit)
        self.consecutive=self.consecutive+1 if quiet else 0
        stopped=self.consecutive>=self.confirm_frames
        self.result={'state':'stationary' if stopped else ('confirming' if quiet else 'moving'),
                     'reason':'image_lidar_visual_motion_consensus','stamp_sec':stamp,
                     'consecutive':self.consecutive,'image_flow_p90_px':[v['flow_p90_px'] for v in images],
                     **cloud,'visual_speed_mps':speed,'visual_angular_speed_rps':angular,
                     'telemetry_check':telemetry}
        return stopped

    def status(self):
        result=dict(self.result,zero_motion_updates=self.zero_updates)
        if self.last_wall and time.monotonic()-self.last_wall>self.max_gap:
            result.update(state='unknown',reason='evidence_expired')
        return result
