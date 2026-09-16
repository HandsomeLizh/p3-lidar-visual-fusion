"""Fixed-size coarse elevation overview; exact map remains in disk tiles."""
from pathlib import Path
import numpy as np
from .legacy.semantic_grid import GridGeometry


class Overview:
    def __init__(self, size=256, resolution=1.):
        self.size, self.resolution = int(size), float(resolution)
        if self.size < 8 or self.resolution <= 0:
            raise ValueError("Invalid overview size/resolution")
        self.origin = np.zeros(2)
        self.ready = False
        self.minimum = np.full((self.size, self.size), np.inf, np.float32)
        self.maximum = np.full_like(self.minimum, -np.inf)
        self.count = np.zeros_like(self.minimum, dtype=np.uint32)

    def _indices(self, xy):
        return np.floor((xy-self.origin)/self.resolution).astype(np.int64)

    def update(self, points):
        points = np.asarray(points)
        points = points[np.isfinite(points).all(axis=1)]
        if not len(points):
            return
        low, high = points[:, :2].min(axis=0), points[:, :2].max(axis=0)
        if not self.ready:
            self.origin = np.floor(low/self.resolution)*self.resolution
            self.ready = True
        ix = self._indices(points[:, :2])
        if (ix < 0).any() or (ix >= self.size).any():
            rows, cols = np.where(self.count > 0)
            oldxy = self.origin + (np.column_stack([cols, rows])+.5)*self.resolution
            oldmin, oldmax, oldn = self.minimum[rows, cols], self.maximum[rows, cols], self.count[rows, cols]
            # Include whole old cell extents, so successive coarsening never crops.
            if len(oldxy):
                low = np.minimum(low, (oldxy-self.resolution/2).min(axis=0))
                high = np.maximum(high, (oldxy+self.resolution/2).max(axis=0))
            while True:
                origin = np.floor(low/self.resolution)*self.resolution
                if np.all(np.floor((high-origin)/self.resolution) < self.size):
                    break
                self.resolution *= 2.
            self.origin = origin
            self.minimum.fill(np.inf); self.maximum.fill(-np.inf); self.count.fill(0)
            if len(oldxy):
                oldix = self._indices(oldxy)
                addr = oldix[:, 1], oldix[:, 0]
                np.minimum.at(self.minimum, addr, oldmin)
                np.maximum.at(self.maximum, addr, oldmax)
                np.add.at(self.count, addr, oldn)
        ix = self._indices(points[:, :2])
        addr = ix[:, 1], ix[:, 0]
        np.minimum.at(self.minimum, addr, points[:, 2])
        np.maximum.at(self.maximum, addr, points[:, 2])
        if self.count.max() > 2**31:
            self.count //= 2
        np.add.at(self.count, addr, 1)

    def layers(self):
        valid = self.count > 0
        height = np.where(valid, self.minimum, np.nan).astype(np.float32)
        spread = np.where(valid, self.maximum-self.minimum, np.nan).astype(np.float32)
        dy, dx = np.gradient(height, self.resolution)
        slope = np.arctan(np.hypot(dx, dy))
        known = valid & np.isfinite(slope) & (self.count > 1)
        cost = np.maximum(slope/np.deg2rad(35), spread/.3)
        occupancy = np.where(known, np.rint(np.clip(cost, 0, 1)*100), -1).astype(np.int8)
        return height, spread, occupancy

    @property
    def geometry(self):
        return GridGeometry(resolution=self.resolution, width=self.size, height=self.size,
                            origin_x=float(self.origin[0]), origin_y=float(self.origin[1]))

    def save(self, path):
        path = Path(path); tmp = path.with_suffix(".npz.tmp")
        with tmp.open("wb") as stream:
            np.savez_compressed(stream, minimum=self.minimum, maximum=self.maximum,
                count=self.count, resolution=self.resolution, origin=self.origin, ready=self.ready)
        tmp.replace(path)
