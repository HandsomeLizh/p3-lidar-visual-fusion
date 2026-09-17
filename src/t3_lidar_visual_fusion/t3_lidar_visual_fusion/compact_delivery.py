"""Original project's SQLite wire format, updated with bounded working memory."""
import json
import sqlite3
import time
from .disk_map import configure_sqlite
from .legacy.compact_grid_store import _metadata, _compressed_tile_payload


class CompactDelivery:
    def __init__(self, path, *, height_only=False):
        self.height_only=bool(height_only)
        self.format_name='t3_compact_elevation_map' if self.height_only else 't3_compact_global_grid_map'
        self.db = sqlite3.connect(str(path), timeout=5)
        configure_sqlite(self.db, cache_mib=16)
        self.db.execute("PRAGMA user_version=1")
        self.db.execute("CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY,value TEXT NOT NULL)")
        self.db.execute("CREATE TABLE IF NOT EXISTS tiles(tile_x INTEGER NOT NULL,tile_y INTEGER NOT NULL,"
            "revision INTEGER NOT NULL,elevation_offset_m REAL NOT NULL,payload BLOB NOT NULL,"
            "PRIMARY KEY(tile_x,tile_y))")
        self.db.commit()
        old=self.db.execute("SELECT value FROM metadata WHERE key='format'").fetchone()
        if old and json.loads(old[0])!=self.format_name:
            self.db.close()
            raise ValueError('Map output mode changed; start a new output directory')
        row = self.db.execute("SELECT value FROM metadata WHERE key='map_revision'").fetchone()
        self.revision = int(json.loads(row[0])) if row else -1

    def checkpoint(self, grid):
        grid.checkpoint()
        if self.revision == grid.update_id:
            return
        metadata = _metadata(grid, frame_id="map", timestamp_text=time.strftime("%Y%m%d_%H%M%S"))
        if self.height_only:
            metadata['format']=self.format_name
            metadata.pop('class_names',None)
            for name in ['occupancy','semantic','semantic_confidence']:metadata['encoding'].pop(name,None)
        self.db.execute("BEGIN IMMEDIATE")
        try:
            # A single indexed stream, never a dict of every historical tile.
            for x, y, rev in grid.tiles.db.execute("SELECT x,y,revision FROM tiles WHERE revision>?",
                                                  (self.revision,)):
                tile = grid.tiles.get((x, y))
                offset, payload = _compressed_tile_payload(tile, 3, height_only=self.height_only)
                self.db.execute("INSERT INTO tiles VALUES(?,?,?,?,?) ON CONFLICT(tile_x,tile_y) "
                    "DO UPDATE SET revision=excluded.revision,elevation_offset_m=excluded.elevation_offset_m,"
                    "payload=excluded.payload", (x, y, rev, offset, payload))
            self.db.executemany("INSERT INTO metadata VALUES(?,?) ON CONFLICT(key) DO UPDATE "
                "SET value=excluded.value", ((k, json.dumps(v)) for k, v in metadata.items()))
            self.db.commit()
            self.revision = grid.update_id
        except Exception:
            self.db.rollback()
            raise
        self.db.execute("PRAGMA wal_checkpoint(PASSIVE)")

    def close(self):
        self.db.close()
