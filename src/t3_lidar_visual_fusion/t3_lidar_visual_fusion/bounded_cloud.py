"""Persistent fine voxels plus a deterministic, bounded whole-map preview."""
import numpy as np
from .legacy.voxel_cloud_store import VoxelCloudStore
from .disk_map import configure_sqlite


class BoundedCloudStore(VoxelCloudStore):
    def __init__(self, path, voxel_size, preview_points=150000, preview_voxel=.3, separate_preview=False):
        super().__init__(path, voxel_size)
        configure_sqlite(self.connection, cache_mib=16)
        self.preview_cap = int(preview_points)
        self.preview_voxel = max(float(voxel_size), float(preview_voxel))
        self.sample = np.empty((0, 3), dtype=np.float32)
        self.keys = np.empty((0, 3), dtype=np.int64)
        self.rank = np.empty(0, dtype=np.uint64)
        self.sweep_column=None
        self.sweep_rowid=0
        if self.preview_cap < 1:
            raise ValueError("preview_points must be positive")
        # Persist display eligibility separately from measured geometry. Hidden
        # edge points remain available to elevation, export and column rebuilds.
        existing = self.connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='preview_voxels'").fetchone()
        self.separate_preview = bool(separate_preview or existing)
        if self.separate_preview and not existing:
            with self.connection:
                self.connection.execute('CREATE TABLE preview_voxels('
                    'ix INTEGER,iy INTEGER,iz INTEGER,PRIMARY KEY(ix,iy,iz)) WITHOUT ROWID')
                # Legacy stores already contained only the visible subset.
                self.connection.execute('INSERT INTO preview_voxels SELECT ix,iy,iz FROM voxels')
        # Restart/export-view path, streamed; normal new runs start empty.
        query = ('SELECT v.x,v.y,v.z FROM voxels v JOIN preview_voxels p USING(ix,iy,iz)'
                 if self.separate_preview else 'SELECT x,y,z FROM voxels')
        cursor = self.connection.execute(query)
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

    def append(self, points, preview_mask=None):
        points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        if preview_mask is not None:
            mask = np.asarray(preview_mask)
            if not self.separate_preview or mask.dtype != np.bool_ or mask.shape != (len(points),):
                raise ValueError('A boolean preview mask requires separate_preview and one value per point')
            visible = points[mask]
        else:
            visible = points
        added = super().append(points)
        visible = visible[np.isfinite(visible).all(axis=1)]
        if self.separate_preview:
            keys = np.unique(np.floor(visible/self.voxel_size).astype(np.int64), axis=0)
            with self.connection:
                self.connection.executemany('INSERT OR IGNORE INTO preview_voxels VALUES(?,?,?)',
                    (tuple(map(int, key)) for key in keys))
        self.observe(visible)
        return added

    def preview(self, max_points=None):
        cap = self.preview_cap if max_points is None else max(1, int(max_points))
        if len(self.sample) <= cap:
            return self.sample
        ix = np.argpartition(self.rank, cap-1)[:cap]
        return self.sample[ix]

    def local_candidates(self, center, radius, limit):
        low=np.floor((np.asarray(center)-radius)/self.voxel_size).astype(int)
        high=np.floor((np.asarray(center)+radius)/self.voxel_size).astype(int)
        if self.sweep_column is None or not low[0]<=self.sweep_column<=high[0]:
            self.sweep_column=int(low[0]);self.sweep_rowid=0
        rows=[]
        # Equality on X plus a Y range uses the existing (ix,iy,iz) index.
        # Never scan all historical rowids as the vehicle moves across the map.
        for _ in range(int(high[0]-low[0]+1)):
            remaining=int(limit)-len(rows)
            batch=self.connection.execute(
                'SELECT rowid,ix,iy,iz,x,y,z FROM voxels WHERE ix=? '
                'AND iy BETWEEN ? AND ? AND iz BETWEEN ? AND ? AND rowid>? '
                'ORDER BY rowid LIMIT ?',
                (self.sweep_column,int(low[1]),int(high[1]),int(low[2]),int(high[2]),self.sweep_rowid,remaining)).fetchall()
            rows.extend(batch)
            if len(batch)==remaining:
                self.sweep_rowid=int(batch[-1][0]);break
            self.sweep_column=self.sweep_column+1 if self.sweep_column<high[0] else int(low[0])
            self.sweep_rowid=0
        return np.asarray(rows,dtype=float).reshape(-1,7)

    def remove_rows(self, rows):
        rows=np.asarray(rows).reshape(-1,7)
        with self.connection:
            before=self.connection.total_changes
            self.connection.executemany('DELETE FROM voxels WHERE rowid=? AND ix=? AND iy=? AND iz=?',
                (tuple(map(int,row[:4])) for row in rows))
            self.count-=self.connection.total_changes-before
            if self.separate_preview:
                self.connection.executemany('DELETE FROM preview_voxels WHERE ix=? AND iy=? AND iz=? '
                    'AND NOT EXISTS(SELECT 1 FROM voxels WHERE ix=? AND iy=? AND iz=?)',
                    (tuple(map(int,row[1:4]))*2 for row in rows))
        deleted={tuple(row[1:4].astype(np.int64)) for row in rows}
        keep=np.array([tuple(k) not in deleted for k in np.floor(self.sample/self.voxel_size).astype(np.int64)],dtype=bool)
        self.sample,self.keys,self.rank=self.sample[keep],self.keys[keep],self.rank[keep]

    def column_points(self, cell, resolution):
        """Read one XY cell in chunks, independent of total map extent."""
        low=np.asarray(cell)*resolution;high=low+resolution
        a=np.floor(low/self.voxel_size).astype(int);b=np.floor(high/self.voxel_size).astype(int)
        for ix in range(a[0],b[0]+1):
            cursor=self.connection.execute('SELECT x,y,z FROM voxels WHERE ix=? AND iy BETWEEN ? AND ?',
                (int(ix),int(a[1]),int(b[1])))
            while True:
                rows=cursor.fetchmany(4096)
                if not rows:break
                points=np.asarray(rows,dtype=float)
                keep=((points[:,:2]>=low)&(points[:,:2]<high)).all(axis=1)
                if keep.any():yield points[keep]

    def checkpoint(self):
        self.connection.commit()
        self.connection.execute("PRAGMA wal_checkpoint(PASSIVE)")

    def memory_stats(self):
        return dict(preview_points=len(self.sample),preview_separate=self.separate_preview,
            preview_array_mib=(self.sample.nbytes+self.keys.nbytes+self.rank.nbytes)/2**20)
