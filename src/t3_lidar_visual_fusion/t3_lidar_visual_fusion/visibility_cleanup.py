"""Bounded, conservative removal of repeatedly observed free LiDAR voxels."""
import numpy as np
from scipy.spatial import cKDTree


class VisibilityCleanup:
    def __init__(self, cfg=None):
        cfg=cfg or {}
        self.confirmations=int(cfg.get('confirmations',3))
        self.max_gap=float(cfg.get('max_gap_sec',8.))
        self.radius=float(cfg.get('ray_radius_m',.04))
        self.margin=float(cfg.get('range_margin_m',.25))
        self.max_range=float(cfg.get('max_range_m',40.))
        self.budget=int(cfg.get('max_candidates',40000))
        self.cell_budget=int(cfg.get('max_changed_cells',256))
        if self.confirmations<2 or min(self.max_gap,self.radius,self.margin,self.max_range,self.budget,self.cell_budget)<=0:
            raise ValueError('Invalid visibility cleanup bounds')
        self.votes={};self.last_stamp=-float('inf')
        self.stats=dict(checked=0,free_evidence=0,removed=0,pending=0)

    def update(self, store, sensor_points, map_from_sensor, stamp, resolution):
        if not np.isfinite(stamp) or stamp<=self.last_stamp:return np.empty((0,3))
        self.last_stamp=stamp
        self.votes={k:v for k,v in self.votes.items() if 0<stamp-v[1]<=self.max_gap}
        xyz=np.asarray(sensor_points,dtype=float).reshape(-1,3)
        distance=np.linalg.norm(xyz,axis=1)
        keep=np.isfinite(xyz).all(axis=1)&(distance>.5)
        xyz,distance=xyz[keep],distance[keep]
        if len(xyz)<4:return np.empty((0,3))
        origin=map_from_sensor[:3,3]
        rows=store.local_candidates(origin,self.max_range,self.budget)
        if not len(rows):return np.empty((0,3))
        ids=rows[:,0].astype(np.int64);points=rows[:,4:7]
        old_sensor=(points-origin)@map_from_sensor[:3,:3]
        ranges=np.linalg.norm(old_sensor,axis=1)
        valid=(ranges>.5)&(ranges<self.max_range)
        tree=cKDTree(xyz/distance[:,None])
        chord,nearest=tree.query(old_sensor/np.maximum(ranges[:,None],1e-9),k=4,workers=1)
        # A narrow cylinder around an actual return ray is evidence. Missing
        # beams are never free space. Nearby shorter returns veto clearance.
        aligned=chord<=self.radius/np.maximum(ranges[:,None],.5)
        observed=np.min(np.where(aligned,distance[nearest],np.inf),axis=1)
        free=valid&np.isfinite(observed)&(observed>ranges+max(self.margin,store.voxel_size*2))
        chosen=[];cells=set()
        # Most unchanged scans have no free evidence. Avoid building a large
        # endpoint set and visiting every stored voxel on that common path.
        if self.votes:
            for identity in ids[~free]:self.votes.pop(int(identity),None)
        if np.any(free):
            current=xyz@map_from_sensor[:3,:3].T+origin
            occupied={tuple(k) for k in np.floor(current/store.voxel_size).astype(np.int64)}
        for row,identity in zip(rows[free],ids[free]):
            key=tuple(row[1:4].astype(np.int64));identity=int(identity)
            # A current endpoint in the same stored voxel wins over another ray.
            if key in occupied:
                self.votes.pop(identity,None);continue
            count=self.votes.get(identity,(0,stamp))[0]+1
            self.votes[identity]=(count,stamp)
            if count>=self.confirmations:
                cell=tuple(np.floor(row[4:6]/resolution).astype(np.int64))
                if cell in cells or len(cells)<self.cell_budget:
                    cells.add(cell);chosen.append(row)
        # Working state is bounded independently of persistent map size.
        if len(self.votes)>2*self.budget:
            oldest=sorted(self.votes,key=lambda k:self.votes[k][1])[:len(self.votes)-2*self.budget]
            for key in oldest:self.votes.pop(key,None)
        removed=np.asarray(chosen,dtype=float).reshape(-1,7)
        if len(removed):
            store.remove_rows(removed)
            for identity in removed[:,0]:self.votes.pop(int(identity),None)
        self.stats=dict(checked=len(rows),free_evidence=int(free.sum()),removed=len(removed),pending=len(self.votes))
        return removed[:,4:7]
