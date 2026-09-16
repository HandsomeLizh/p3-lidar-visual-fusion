"""Persistent fine voxels plus a deterministic, bounded whole-map preview."""
import numpy as np
from .legacy.voxel_cloud_store import VoxelCloudStore
from .disk_map import configure_sqlite


class BoundedCloudStore(VoxelCloudStore):
    def __init__(self, path, voxel_size, preview_points=150000, preview_voxel=.3):
        super().__init__(path, voxel_size)
        configure_sqlite(self.connection, cache_mib=16)
        self.preview_cap = int(preview_points)
        self.preview_voxel = max(float(voxel_size), float(preview_voxel))
        self.sample = np.empty((0, 3), dtype=np.float32)
        self.keys = np.empty((0, 3), dtype=np.int64)
        self.rank = np.empty(0, dtype=np.uint64)
        if self.preview_cap < 1:
            raise ValueError("preview_points must be positive")
        # Restart/export-view path, streamed; normal new runs start empty.
        cursor = self.connection.execute("SELECT x,y,z FROM voxels")
        while True:
            rows = cursor.fetchmany(8192)
            if not rows:
                break
            self.observe(np.asarray(rows, dtype=np.float32))

    def observe(self, points):
        points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
        points = points[np.isfinite(points).all(axis=1)]
        if not len(points):
            return
        keys = np.floor(points / self.preview_voxel).astype(np.int64)
        keys, ix = np.unique(keys, axis=0, return_index=True)
        points = points[ix]
        # Vectorized modular uint64 hash, deterministic across replay order.
        k = keys.astype(np.uint64)
        rank = k[:, 0]*np.uint64(0x9e3779b185ebca87)
        rank ^= k[:, 1]*np.uint64(0xc2b2ae3d27d4eb4f)
        rank ^= k[:, 2]*np.uint64(0x165667b19e3779f9)
        rank ^= rank >> np.uint64(30)
        rank *= np.uint64(0xbf58476d1ce4e5b9)
        rank ^= rank >> np.uint64(27)
        if len(self.rank) >= self.preview_cap:
            keep = rank <= self.rank.max()
            keys, points, rank = keys[keep], points[keep], rank[keep]
        keys = np.concatenate([self.keys, keys])
        points = np.concatenate([self.sample, points])
        rank = np.concatenate([self.rank, rank])
        _, ix = np.unique(keys, axis=0, return_index=True)
        keys, points, rank = keys[ix], points[ix], rank[ix]
        if len(rank) > self.preview_cap:
            ix = np.argpartition(rank, self.preview_cap-1)[:self.preview_cap]
            keys, points, rank = keys[ix], points[ix], rank[ix]
        self.keys, self.sample, self.rank = keys, points, rank

    def append(self, points):
        added = super().append(points)
        self.observe(points)
        return added

    def preview(self, max_points=None):
        cap = self.preview_cap if max_points is None else max(1, int(max_points))
        if len(self.sample) <= cap:
            return self.sample
        ix = np.argpartition(self.rank, cap-1)[:cap]
        return self.sample[ix]

    def checkpoint(self):
        self.connection.commit()
        self.connection.execute("PRAGMA wal_checkpoint(PASSIVE)")

    def memory_stats(self):
        return dict(preview_points=len(self.sample),
            preview_array_mib=(self.sample.nbytes+self.keys.nbytes+self.rank.nbytes)/2**20)
