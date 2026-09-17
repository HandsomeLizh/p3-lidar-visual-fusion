"""Exclude a configured vehicle volume in base_link before map integration."""
import numpy as np


class SelfFilter:
    def __init__(self, enabled=False, min_xyz_m=None, max_xyz_m=None):
        self.enabled = bool(enabled)
        self.minimum = np.asarray(min_xyz_m, dtype=float)
        self.maximum = np.asarray(max_xyz_m, dtype=float)
        if self.enabled and (self.minimum.shape != (3,) or self.maximum.shape != (3,)
                or not np.isfinite(self.minimum).all() or not np.isfinite(self.maximum).all()
                or np.any(self.minimum >= self.maximum)):
            raise ValueError('Self filter requires finite, ordered base_link XYZ bounds')

    def keep_mask(self, points_base):
        points = np.asarray(points_base)
        if not self.enabled:
            return np.ones(len(points), dtype=bool)
        return ~np.all((points >= self.minimum) & (points <= self.maximum), axis=1)
