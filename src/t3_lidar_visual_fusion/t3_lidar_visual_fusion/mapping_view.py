"""Optional camera-view LiDAR crop, adapted from the 216 map experiment.

Only selects existing observations. Edge weights are display/coverage density,
not measurement confidence. The localization input is never cropped here.
"""
import numpy as np
from .stereo_geometry import StereoGeometry,project


def spatial_density_mask(world_points,weights,cell_size):
    points=np.asarray(world_points,dtype=float).reshape(-1,3)
    weights=np.asarray(weights,dtype=float).reshape(-1)
    if (len(points)!=len(weights) or not np.isfinite(cell_size) or cell_size<=0
            or not np.isfinite(weights).all() or np.any((weights<0)|(weights>1))):
        raise ValueError('Invalid spatial density sampling configuration')
    valid=np.isfinite(points).all(axis=1);keep=valid&(weights>=1.)
    ids=np.flatnonzero(valid&(weights>0.)&(weights<1.))
    if not len(ids):return keep
    cells=np.floor(points[ids,:2]/cell_size).astype(np.int64).astype(np.uint64)
    hashed=cells[:,0]*np.uint64(0x9E3779B97F4A7C15)
    hashed ^= (cells[:,1]+np.uint64(0xD1B54A32D192ED03))*np.uint64(0xBF58476D1CE4E5B9)
    hashed ^= hashed>>np.uint64(30);hashed *= np.uint64(0xBF58476D1CE4E5B9)
    hashed ^= hashed>>np.uint64(27);hashed *= np.uint64(0x94D049BB133111EB)
    hashed ^= hashed>>np.uint64(31)
    sample=(hashed>>np.uint64(11)).astype(np.float64)*(1./2**53)
    keep[ids]=sample<weights[ids]
    return keep


class MappingCameraView:
    def __init__(self,profile):
        self.geometry=StereoGeometry(profile)
        c=profile['mapping_camera_view']
        self.full=float(c.get('full_density_range_m',10.))
        self.fade=float(c.get('range_feather_m',5.))
        self.edge=float(c.get('image_feather_fraction',.15))
        self.cell=float(profile['map_resolution'])
        if not np.isfinite([self.full,self.fade,self.edge,self.cell]).all() or min(self.full,self.cell)<=0 or min(self.fade,self.edge)<0:
            raise ValueError('Invalid mapping camera-view limits')

    def weights(self,base):
        g=self.geometry
        rect=(base-g.base_from_rect[:3,3])@g.base_from_rect[:3,:3]
        a,za=project(g.p0,rect);b,zb=project(g.p1,rect)
        def falloff(x):
            t=np.clip(x,0.,1.);return (1.-t)**2*(1.+2.*t)
        weights=(np.isfinite(rect).all(axis=1)&(za>.4)&(zb>.4)).astype(float)
        radius=np.linalg.norm(base[:,:2],axis=1)
        weights *= falloff((radius-self.full)/self.fade) if self.fade else radius<=self.full
        outside=np.zeros((len(base),2));size=np.array(g.size)
        for uv in (a,b):
            weights *= np.isfinite(uv).all(axis=1)
            if self.edge:
                excess=np.maximum(np.maximum(-uv,uv-size),0.)
                outside=np.maximum(outside,excess/(size*self.edge))
            else:weights *= ((uv>=0)&(uv<size)).all(axis=1)
        if self.edge:weights *= falloff(np.linalg.norm(outside,axis=1))
        return np.nan_to_num(weights,nan=0.,posinf=0.,neginf=0.)

    def select(self,world,pose):
        # Evaluate after deskew, in the base frame of the co-timed map pose.
        base=(world-pose[:3,3])@pose[:3,:3]
        weights=self.weights(base)
        return spatial_density_mask(world,weights,self.cell)
