"""Bounded resident elevation tiles; complete history lives in SQLite.

Reuse the frozen project's cell fusion and GridMap window conventions.
Do not use its unbounded tile dictionary or whole-map snapshot clone.
"""
from collections import OrderedDict
from io import BytesIO
from pathlib import Path
import json
import math
import os
import sqlite3
import numpy as np
import yaml
from .legacy.tiled_semantic_map import TiledSemanticMapManager
from .surface_grid import SurfaceGrid

FIELDS = TiledSemanticMapManager._SNAPSHOT_TILE_FIELDS + (("surface_height_range", np.float32),)


def configure_sqlite(db, cache_mib=16):
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("PRAGMA synchronous=FULL")
    db.execute("PRAGMA cache_size=" + str(-int(cache_mib * 1024)))
    db.execute("PRAGMA mmap_size=0")
    db.execute("PRAGMA temp_store=FILE")
    db.execute("PRAGMA wal_autocheckpoint=256")
    db.execute("PRAGMA journal_size_limit=16777216")


class TileCache:
    """LRU values only, with disk indexed keys and atomic dirty eviction."""
    def __init__(self, owner, database, max_tiles, max_bytes):
        self.owner = owner
        self.db = sqlite3.connect(str(database), timeout=5)
        configure_sqlite(self.db)
        self.db.execute("CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY,value TEXT)")
        self.db.execute("CREATE TABLE IF NOT EXISTS tiles(x INTEGER,y INTEGER,payload BLOB,"
                        "revision INTEGER,PRIMARY KEY(x,y))")
        self.db.execute("CREATE INDEX IF NOT EXISTS tile_revision ON tiles(revision)")
        schema = json.dumps(dict(resolution=owner.resolution, tile_cells=owner.tile_cells,
                                 class_names=owner.class_names), sort_keys=True)
        old = self.db.execute("SELECT value FROM metadata WHERE key='schema'").fetchone()
        if old and old[0] != schema:
            raise ValueError("Stored elevation calibration/schema differs from profile")
        self.db.execute("INSERT OR IGNORE INTO metadata VALUES('schema',?)", (schema,))
        self.db.commit()
        self.max_tiles, self.max_bytes = int(max_tiles), int(max_bytes)
        self.cache = OrderedDict()
        self.dirty = set()
        self.new_keys = set()
        self.nbytes = 0
        self.disk_count = self.db.execute("SELECT COUNT(*) FROM tiles").fetchone()[0]
        self.evictions = self.loads = self.writes = 0
        self.tile_bytes = owner.tile_cells**2 * sum(
            np.dtype(dtype).itemsize * (len(owner.class_names) if name == "semantic_votes" else 1)
            for name, dtype in FIELDS)
        if self.max_tiles < 1 or self.max_bytes < self.tile_bytes:
            raise ValueError("Tile cache must fit at least one complete tile")

    def __len__(self):
        return self.disk_count + len(self.new_keys)

    def get(self, key, default=None, writable=False, create=False):
        key = tuple(map(int, key))
        if key in self.cache:
            self.cache.move_to_end(key)
            if writable:
                self.dirty.add(key)
            return self.cache[key]
        row = self.db.execute("SELECT payload,revision FROM tiles WHERE x=? AND y=?", key).fetchone()
        if row is None and not create:
            return default
        # Evict before allocation; one serialized tile is the only extra load buffer.
        self.trim(reserve=self.tile_bytes)
        tile = self.owner._new_tile(key)
        if row is not None:
            with np.load(BytesIO(row[0]), allow_pickle=False) as data:
                for name, dtype in FIELDS:
                    if name == "surface_height_range" and name not in data:
                        # Old runs did not retain per-scan relief. Preserve their
                        # conservative evidence; do not silently relabel obstacles.
                        value = np.where(tile.elevation_count > 0,
                            tile.elevation_max - tile.elevation_min, 0.).astype(dtype)
                    else:
                        value = np.asarray(data[name], dtype=dtype)
                    if value.shape != getattr(tile, name).shape:
                        raise ValueError("Corrupt elevation tile " + str(key))
                    setattr(tile, name, value.copy())
            tile._fusion_revision = row[1]
            self.loads += 1
        else:
            self.new_keys.add(key)
        self.cache[key] = tile
        self.nbytes += self.tile_bytes
        if writable or create:
            self.dirty.add(key)
        return tile

    def persist(self, key):
        if key not in self.dirty:
            return
        tile = self.cache[key]
        payload = BytesIO()
        np.savez_compressed(payload, **{name: getattr(tile, name) for name, _ in FIELDS})
        with self.db:
            self.db.execute("INSERT INTO tiles VALUES(?,?,?,?) ON CONFLICT(x,y) DO UPDATE "
                            "SET payload=excluded.payload,revision=excluded.revision",
                            (*key, payload.getvalue(), int(getattr(tile,"_fusion_revision",self.owner.update_id))))
        if key in self.new_keys:
            self.new_keys.remove(key)
            self.disk_count += 1
        self.dirty.remove(key)
        self.writes += 1

    def trim(self, reserve=0):
        while self.cache and (len(self.cache) + bool(reserve) > self.max_tiles
                              or self.nbytes + reserve > self.max_bytes):
            key = next(iter(self.cache))
            self.persist(key)  # Failed writes keep the dirty tile in RAM.
            del self.cache[key]
            self.nbytes -= self.tile_bytes
            self.evictions += 1

    def flush(self):
        for key in tuple(self.dirty):
            self.persist(key)
        with self.db:
            self.db.execute("INSERT INTO metadata VALUES('map_revision',?) "
                            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                            (str(self.owner.update_id),))
        self.db.execute("PRAGMA wal_checkpoint(PASSIVE)")

    def bounds(self):
        row = self.db.execute("SELECT MIN(x),MAX(x),MIN(y),MAX(y) FROM tiles").fetchone()
        xs, ys = [], []
        if row[0] is not None:
            xs.extend(row[:2]); ys.extend(row[2:])
        for x, y in self.new_keys:
            xs.append(x); ys.append(y)
        return None if not xs else (min(xs), max(xs), min(ys), max(ys))

    def close(self):
        self.flush()
        self.db.close()


class DiskElevationMap(TiledSemanticMapManager):
    def __init__(self, database, *, resolution=.2, tile_cells=128,
                 max_tiles=64, cache_mib=128, max_window_cells=250000):
        super().__init__(resolution=resolution, tile_cells=tile_cells)
        self.database = Path(database)
        self.database.parent.mkdir(parents=True, exist_ok=True)
        self.max_window_cells = int(max_window_cells)
        self.tiles = TileCache(self, self.database, max_tiles, int(cache_mib * 2**20))
        row = self.tiles.db.execute("SELECT value FROM metadata WHERE key='map_revision'").fetchone()
        maximum=self.tiles.db.execute("SELECT COALESCE(MAX(revision),0) FROM tiles").fetchone()[0]
        self.update_id = max(int(row[0]) if row else 0,int(maximum))

    def _new_tile(self, key):
        x, y = self._tile_origin(key)
        return SurfaceGrid(**self._tile_kwargs, origin_x=x, origin_y=y)

    def update_elevation_only(self, *, points_map):
        points = np.asarray(points_map, dtype=np.float64).reshape(-1, 3)
        points = points[np.isfinite(points).all(axis=1)]
        if not len(points):
            return
        self.update_id += 1
        pairs = np.floor(points[:, :2] / self.tile_length).astype(np.int64)
        for pair in np.unique(pairs, axis=0):
            key = tuple(map(int, pair))
            tile = self.tiles.get(key, writable=True, create=True)
            tile.update_elevation_only(points_map=points[np.all(pairs == pair, axis=1)])
            tile._fusion_revision = self.update_id

    def add_semantics(self, points, labels):
        cells = np.floor(np.asarray(points)[:, :2] / self.resolution).astype(np.int64)
        pairs = np.floor_divide(cells, self.tile_cells)
        for pair in np.unique(pairs, axis=0):
            key = tuple(map(int, pair))
            tile = self.tiles.get(key, writable=True)
            if tile is None:
                continue
            use = np.all(pairs == pair, axis=1)
            local = cells[use] - pair * self.tile_cells
            # Saturate by halving before uint32 overflow, preserving vote proportions.
            if tile.semantic_votes.max() > 2**31:
                tile.semantic_votes //= 2
            np.add.at(tile.semantic_votes, (labels[use], local[:, 1], local[:, 0]), 1)
            tile._fusion_revision = self.update_id

    def rebuild_cells(self, cells, cloud):
        """Replace only columns whose old voxels were confirmed free by rays."""
        cells=np.unique(np.asarray(cells,dtype=np.int64).reshape(-1,2),axis=0)
        if not len(cells):return
        self.update_id+=1
        for cell in cells:
            pair=np.floor_divide(cell,self.tile_cells)
            tile=self.tiles.get(tuple(pair),writable=True)
            if tile is None:continue
            col,row=cell-pair*self.tile_cells
            surface_range = tile.surface_height_range[row,col]
            for name,_ in FIELDS:
                values=getattr(tile,name)
                fill=np.inf if name=='elevation_min' else -np.inf if name=='elevation_max' else 0
                if name=='semantic_votes':values[:,row,col]=fill
                else:values[row,col]=fill
            for points in cloud.column_points(cell,self.resolution):
                tile._update_elevation_cell(row,col,points[:,2])
            # Remaining persistent points mix acquisition times. Cleanup can
            # reduce actual relief, but must not turn temporal drift into relief.
            remaining = (tile.elevation_max[row,col] - tile.elevation_min[row,col]
                         if tile.elevation_count[row,col] else 0.)
            tile.surface_height_range[row,col] = min(surface_range, max(0., remaining))
            tile._fusion_revision=self.update_id

    def window_geometry(self, **kwargs):
        values = np.asarray([kwargs[k] for k in ("center_x", "center_y", "length_x", "length_y")])
        if not np.isfinite(values).all() or np.any(values[2:] <= 0) or np.any(np.abs(values[:2]) > 1e7):
            raise ValueError("Map query requires finite position and positive extent")
        width = math.ceil(kwargs["length_x"] / self.resolution)
        height = math.ceil(kwargs["length_y"] / self.resolution)
        if width * height > self.max_window_cells:
            raise ValueError("Map query exceeds max_query_cells; request smaller regions")
        return super().window_geometry(**kwargs)

    def extract_global(self, cell_limit):
        bounds = self.tiles.bounds()
        if bounds is None:
            return None
        x0, x1, y0, y1 = bounds
        nx, ny = (x1-x0+1)*self.tile_cells, (y1-y0+1)*self.tile_cells
        if nx*ny > cell_limit:
            raise ValueError("Full-resolution global message exceeds global_max_cells")
        # Explicit caller cap; no resolution changes or cropped global maps.
        original = self.max_window_cells
        try:
            self.max_window_cells = cell_limit
            return self.extract_window(center_x=(x0+x1+1)*self.tile_length/2,
                center_y=(y0+y1+1)*self.tile_length/2,
                length_x=nx*self.resolution, length_y=ny*self.resolution)
        finally:
            self.max_window_cells = original

    def memory_stats(self):
        return dict(resident_tiles=len(self.tiles.cache), total_tiles=len(self.tiles),
                    tile_array_mib=self.tiles.nbytes/2**20, dirty_tiles=len(self.tiles.dirty),
                    tile_evictions=self.tiles.evictions, tile_loads=self.tiles.loads,
                    tile_writes=self.tiles.writes, max_resident_tiles=self.tiles.max_tiles)

    def checkpoint(self):
        self.tiles.flush()

    def close(self):
        self.tiles.close()

    def save_snapshot(self, directory, *, timestamp_text, frame_id):
        """Offline export compatible with the original map_index.yaml/tiles format.

        One tile blob and a streamed YAML index; no all-history Python key list.
        Invoke after mapping has stopped to avoid pinning a growing WAL snapshot.
        """
        self.checkpoint()
        directory = Path(directory)
        td = directory / "tiles"
        td.mkdir(parents=True, exist_ok=True)
        index = directory / "map_index.yaml"
        temporary = index.with_suffix(".yaml.tmp")
        metadata = dict(format="t3_tiled_semantic_map", format_version=1,
            map_id="t3_lidar_visual_fusion", frame_id=frame_id,
            timestamp=timestamp_text, resolution=self.resolution, tile_cells=self.tile_cells,
            origin_x=0., origin_y=0., class_names=list(self.class_names),
            map_revision=self.update_id, total_accepted_points=0)
        with temporary.open("w") as stream:
            stream.write(yaml.safe_dump(metadata, sort_keys=False))
            stream.write("tiles:\n" if self.tile_count else "tiles: []\n")
            for x, y, blob, revision in self.tiles.db.execute("SELECT x,y,payload,revision FROM tiles ORDER BY x,y"):
                filename = f"tile_{x}_{y}.npz"
                path = td / filename
                tmp = path.with_suffix(".tmp")
                with tmp.open("wb") as target:
                    target.write(blob); target.flush(); os.fsync(target.fileno())
                tmp.replace(path)
                stream.write(yaml.safe_dump([dict(x=x, y=y, file="tiles/"+filename,
                                                  revision=revision)], sort_keys=False))
            stream.flush(); os.fsync(stream.fileno())
        temporary.replace(index)
        return {"index": index, "tiles": td}
