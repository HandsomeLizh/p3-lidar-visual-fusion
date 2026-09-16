"""Bounded sparse stereo evidence; no second network or dense disparity pass."""
from collections import OrderedDict
import numpy as np
from .core import inverse


def sparse_map_points(geometry, points, cfg):
    """Original left optical coordinates and an isotropic covariance bound.

    Disparity error produces depth error proportional to z squared. The trace
    bound also includes lateral pixel error, and remains valid after rotation.
    Only independently checked temporal/stereo inliers enter this function.
    """
    points=np.asarray(points,dtype=float).reshape(-1,3)
    points=points[np.isfinite(points).all(axis=1)]
    z=points[:,2]
    points=points[(z>=cfg.get('min_depth_m',.5))&(z<=cfg.get('max_depth_m',8.))]
    z=points[:,2];f=abs(float(geometry.p1[0,3]))
    sigma=float(cfg.get('pixel_sigma',1.))
    depth_var=(sigma*z*z/f)**2
    variance=(depth_var*np.sum((points/z[:,None])**2,axis=1)+
              (sigma*z)**2*(1/geometry.k[0,0]**2+1/geometry.k[1,1]**2)+.01**2)
    valid=np.isfinite(variance)&(variance<=cfg.get('max_point_std_m',.15)**2)
    points,variance=points[valid],variance[valid]
    left_from_rect=inverse(geometry.base_from_left)@geometry.base_from_rect
    points=points@left_from_rect[:3,:3].T+left_from_rect[:3,3]
    # One best measurement per display voxel; all temporal processing is reused.
    order=np.argsort(variance,kind='stable');points,variance=points[order],variance[order]
    keys=np.floor(points/cfg.get('voxel_size_m',.1)).astype(np.int64)
    _,index=np.unique(keys,axis=0,return_index=True)
    index=np.sort(index)[:int(cfg.get('max_points',512))]
    return points[index],variance[index]


class StereoConfirmation:
    """Confirm each cell in distinct frames, with finite pending memory/age."""
    def __init__(self,resolution,cfg):
        self.resolution=resolution;self.cfg=cfg;self.pending=OrderedDict()

    def clear(self):self.pending.clear()

    def observe(self,points,variances,stamp):
        points=np.asarray(points).reshape(-1,3);variances=np.asarray(variances)
        cap=int(self.cfg.get('pending_cells',4096));age=float(self.cfg.get('confirmation_timeout_sec',5.))
        for key in list(self.pending):
            if stamp-self.pending[key][0]>age:del self.pending[key]
        keys=np.floor(points[:,:2]/self.resolution).astype(np.int64)
        # Correlated points in one frame never count as repeated observations.
        selected={}
        for i,key in enumerate(map(tuple,keys)):
            if key not in selected or variances[i]<variances[selected[key]]:selected[key]=i
        output=[]
        for key,i in selected.items():
            z=float(points[i,2]);v=float(variances[i]);old=self.pending.pop(key,None)
            if old is None or stamp<=old[0] or abs(z-old[1])>max(.1,3*np.sqrt(v+old[2])):
                state=(stamp,z,v,1)
            else:
                weight=1/max(v,1e-6);prior=1/max(old[2],self.cfg.get('variance_floor_m2',.0025))
                mean=(prior*old[1]+weight*z)/(prior+weight)
                # Repeated frames share calibration and pose errors. Do not
                # claim a variance below the best individual observation.
                variance=max(self.cfg.get('variance_floor_m2',.0025),min(v,old[2]),1/(prior+weight))
                state=(stamp,mean,variance,old[3]+1)
            if state[3]>=int(self.cfg.get('confirmation_frames',2)):
                xy=(np.asarray(key)+.5)*self.resolution
                output.append([*xy,state[1],state[2],state[3]])
            else:self.pending[key]=state
        while len(self.pending)>cap:self.pending.popitem(last=False)
        return np.asarray(output,dtype=float).reshape(-1,5)
