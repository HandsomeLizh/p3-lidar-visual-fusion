"""Select observed ground/low obstacles for a robot-height terrain projection.

The full 3-D cloud is retained separately. No unseen cell is marked free.
"""
import numpy as np


class GroundClearance:
    def __init__(self, enabled=False, clearance_m=.85, fit_radius_m=3.,
                 terrain_radius_m=8., plane_tolerance_m=.08, max_slope_deg=35.,
                 reference_max_age_sec=2.,reference_max_distance_m=3.):
        self.enabled=enabled
        values=[clearance_m,fit_radius_m,terrain_radius_m,plane_tolerance_m,max_slope_deg,
                reference_max_age_sec,reference_max_distance_m]
        if any(not np.isfinite(x) or x<=0 for x in values) or terrain_radius_m<fit_radius_m:
            raise ValueError('Invalid ground clearance configuration')
        self.clearance,self.fit_radius,self.terrain_radius=clearance_m,fit_radius_m,terrain_radius_m
        self.tolerance,self.max_slope=plane_tolerance_m,np.tan(np.deg2rad(max_slope_deg))
        self.stats={};self.plane=None
        self.reference=None;self.reference_max_age=reference_max_age_sec
        self.reference_max_distance=reference_max_distance_m

    def select(self, points, sensor_origin,stamp=None):
        points=np.asarray(points,dtype=float).reshape(-1,3)
        if not self.enabled:return np.arange(len(points))
        self.plane=None
        local=points-np.asarray(sensor_origin)
        radius=np.linalg.norm(local[:,:2],axis=1)
        near=(radius>=.6)&(radius<=self.fit_radius)&(local[:,2]<-.12)&(local[:,2]>-1.5)
        sample=local[near]
        if len(sample)>1200:sample=sample[np.linspace(0,len(sample)-1,1200,dtype=int)]
        self.stats=dict(input_points=len(points),ground_candidates=len(sample),selected_points=0,
                        reason='ground_unobserved')
        if len(sample)<50:return np.empty(0,dtype=int)
        design=np.c_[sample[:,:2],np.ones(len(sample))]
        # RANSAC is small, bounded, deterministic, and constrained against
        # selecting a vertical wall as the supporting ground surface.
        best=np.zeros(len(sample),dtype=bool);best_plane=None
        rng=np.random.default_rng(0)
        for indices in rng.integers(0,len(sample),(40,3)):
            try:plane=np.linalg.solve(design[indices],sample[indices,2])
            except np.linalg.LinAlgError:continue
            if np.linalg.norm(plane[:2])>self.max_slope:continue
            support=np.abs(design@plane-sample[:,2])<=self.tolerance
            if support.sum()>best.sum():best,best_plane=support,plane
        if best.sum()<max(40,.3*len(sample)):return np.empty(0,dtype=int)
        plane=np.linalg.lstsq(design[best],sample[best,2],rcond=None)[0]
        if np.linalg.norm(plane[:2])>self.max_slope:return np.empty(0,dtype=int)
        self.plane=plane
        if stamp is not None and np.isfinite(stamp):
            origin=np.asarray(sensor_origin).copy()
            world_plane=plane.copy();world_plane[2]+=origin[2]-origin[:2]@plane[:2]
            self.reference=(float(stamp),origin,world_plane)
        height=local[:,2]-(local[:,:2]@plane[:2]+plane[2])
        selected=np.flatnonzero((radius<=self.terrain_radius)&(height<=self.clearance))
        self.stats.update(selected_points=len(selected),ground_inliers=int(best.sum()),
                          ground_plane=plane.tolist(),reason='observed_ground',
                          high_points_excluded=int(np.count_nonzero(height>self.clearance)))
        return selected

    def reference_available(self,sensor_origin,stamp):
        if self.reference is None:return False
        observed,origin,_=self.reference
        return bool(np.isfinite(stamp) and abs(stamp-observed)<=self.reference_max_age
            and np.linalg.norm(np.asarray(sensor_origin)[:2]-origin[:2])<=self.reference_max_distance)

    def select_from_reference(self,points,sensor_origin,stamp):
        """Use only a fresh nearby observed ground for sparse stereo points."""
        points=np.asarray(points,dtype=float).reshape(-1,3)
        if not self.enabled:return np.arange(len(points))
        if not self.reference_available(sensor_origin,stamp):return np.empty(0,dtype=int)
        observed,origin,plane=self.reference
        height=points[:,2]-(points[:,:2]@plane[:2]+plane[2])
        radius=np.linalg.norm(points[:,:2]-origin[:2],axis=1)
        return np.flatnonzero((height<=self.clearance)&(radius<=self.terrain_radius))

    def filter_columns(self, points, sensor_origin):
        """Reuse this scan's plane when rebuilding nearby observed columns.

        Do not refit on a single column, or reintroduce its overhead points.
        The caller bounds the rebuilt columns to the current terrain radius.
        """
        if not self.enabled:return points
        if self.plane is None:raise ValueError('Column rebuild requires observed ground')
        local=points-np.asarray(sensor_origin)
        height=local[:,2]-(local[:,:2]@self.plane[:2]+self.plane[2])
        return points[height<=self.clearance]
