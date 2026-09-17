"""Actual stereo observations in confirmed, range-unobserved terrain cells.

One measured XYZ per XY cell; bounded display cache, persistent disk history.
The height grid owns source arbitration. This store never aligns or invents XYZ.
"""
from collections import OrderedDict
import sqlite3
import numpy as np
from .disk_map import configure_sqlite


class StereoFillStore:
    def __init__(self, path, resolution, preview_cap=20000):
        self.resolution=float(resolution);self.cap=int(preview_cap)
        if not np.isfinite(self.resolution) or self.resolution<=0 or self.cap<1:
            raise ValueError('Invalid stereo fill resolution/cap')
        self.db=sqlite3.connect(str(path));configure_sqlite(self.db,cache_mib=4)
        self.db.execute('CREATE TABLE IF NOT EXISTS points ('
                        'ix INTEGER,iy INTEGER,x REAL,y REAL,z REAL,PRIMARY KEY(ix,iy))')
        # Reuse the existing streaming XYZ export without a second exporter.
        self.db.execute('CREATE VIEW IF NOT EXISTS voxels AS SELECT x,y,z FROM points')
        self.preview_cells=OrderedDict()
        rows=self.db.execute('SELECT ix,iy,x,y,z FROM points ORDER BY rowid DESC LIMIT ?', (self.cap,)).fetchall()
        for row in reversed(rows):
            self.preview_cells[tuple(row[:2])]=tuple(row[2:])
        self.count=self.db.execute('SELECT COUNT(*) FROM points').fetchone()[0]

    def update(self, accepted, observations, variances):
        """Retain the best current measured point in each accepted grid cell."""
        accepted=np.asarray(accepted,dtype=float).reshape(-1,3)
        points=np.asarray(observations,dtype=float).reshape(-1,3)
        variances=np.asarray(variances,dtype=float).reshape(-1)
        if len(points)!=len(variances) or not np.isfinite(accepted).all():
            raise ValueError('Invalid confirmed stereo observations')
        cells={tuple(k) for k in np.floor(accepted[:,:2]/self.resolution).astype(np.int64)}
        chosen={}
        for i in np.argsort(variances,kind='stable'):
            if not np.isfinite(points[i]).all() or not np.isfinite(variances[i]) or variances[i]<=0:continue
            key=tuple(np.floor(points[i,:2]/self.resolution).astype(np.int64))
            if key in cells and key not in chosen:chosen[key]=tuple(map(float,points[i]))
        with self.db:
            # Distinguish insert/update without scanning the full map count.
            before=self.db.total_changes
            self.db.executemany('INSERT OR IGNORE INTO points VALUES(?,?,?,?,?)',
                                (tuple(map(int,k))+p for k,p in chosen.items()))
            self.count+=self.db.total_changes-before
            self.db.executemany('UPDATE points SET x=?,y=?,z=? WHERE ix=? AND iy=?',
                                (p+tuple(map(int,k)) for k,p in chosen.items()))
        for key,point in chosen.items():
            self.preview_cells.pop(key,None);self.preview_cells[key]=point
        while len(self.preview_cells)>self.cap:self.preview_cells.popitem(last=False)
        return len(chosen)

    def remove(self,cells):
        keys=[tuple(map(int,k)) for k in np.asarray(cells).reshape(-1,2)]
        with self.db:
            before=self.db.total_changes
            self.db.executemany('DELETE FROM points WHERE ix=? AND iy=?',keys)
            removed=self.db.total_changes-before;self.count-=removed
        for key in keys:self.preview_cells.pop(key,None)
        return removed

    def preview(self):
        return np.asarray(list(self.preview_cells.values()),dtype=np.float32).reshape(-1,3)

    def checkpoint(self):
        self.db.commit();self.db.execute('PRAGMA wal_checkpoint(PASSIVE)')

    def close(self):self.db.close()
