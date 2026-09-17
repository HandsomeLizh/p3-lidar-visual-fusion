"""Separate a configured body-region cloud from mapping and registration input.

The box is a configurable selection region, not an inferred or calibrated mesh.
Only current sensor measurements are displayed; no body points are synthesized.
"""
import numpy as np


class VehicleReturns:
    def __init__(self, profile):
        cfg=profile.get('vehicle_returns', {})
        self.enabled=bool(cfg.get('enabled', False))
        self.maximum_points=int(cfg.get('max_display_points', 12000))
        if not 1<=self.maximum_points<=50000:
            raise ValueError('Vehicle display point capacity must be in [1,50000]')
        self.lower=np.asarray(cfg.get('minimum_xyz_m', [0.,0.,0.]),dtype=float)
        self.upper=np.asarray(cfg.get('maximum_xyz_m', [0.,0.,0.]),dtype=float)
        if self.enabled and (self.lower.shape!=(3,) or self.upper.shape!=(3,) or
                not np.isfinite([self.lower,self.upper]).all() or np.any(self.lower>=self.upper)):
            raise ValueError('Vehicle return bounds must define a finite base_link box')

    def mask(self, base_points):
        points=np.asarray(base_points,dtype=float).reshape(-1,3)
        if not self.enabled:return np.zeros(len(points),dtype=bool)
        return (np.isfinite(points).all(axis=1)&(points>=self.lower).all(axis=1)
                &(points<=self.upper).all(axis=1))

    def display(self, base_points):
        points=np.asarray(base_points,dtype=float).reshape(-1,3)
        if len(points)>self.maximum_points:
            points=points[np.linspace(0,len(points)-1,self.maximum_points,dtype=int)]
        return points
