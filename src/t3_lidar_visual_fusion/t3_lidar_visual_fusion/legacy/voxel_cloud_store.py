"""Persistent XYZ voxels; height layers in the same XY cell stay distinct."""

from pathlib import Path
import os
import sqlite3

import numpy as np
from ..array_groups import unique_row_indices


class VoxelCloudStore:
    def __init__(self, path: Path, voxel_size: float):
        if not np.isfinite(voxel_size) or voxel_size <= 0:
            raise ValueError("voxel_size must be finite and positive")
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.voxel_size = float(voxel_size)
        self.connection = sqlite3.connect(str(self.path), timeout=30)
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=NORMAL")
        self.connection.execute(
            "CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY, value TEXT)"
        )
        previous = self.connection.execute(
            "SELECT value FROM metadata WHERE key='voxel_size'"
        ).fetchone()
        if previous is not None and float(previous[0]) != self.voxel_size:
            self.connection.close()
            raise ValueError("Existing cloud database has a different voxel size")
        self.connection.execute(
            "INSERT OR IGNORE INTO metadata VALUES('voxel_size', ?)",
            (str(self.voxel_size),),
        )
        self.connection.execute(
            "CREATE TABLE IF NOT EXISTS voxels("
            "ix INTEGER, iy INTEGER, iz INTEGER, x REAL, y REAL, z REAL, "
            "UNIQUE(ix,iy,iz))"
        )
        self.connection.commit()
        self.count = self.connection.execute("SELECT COUNT(*) FROM voxels").fetchone()[0]

    def append(self, points):
        points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        points = points[np.isfinite(points).all(axis=1)]
        scaled = np.floor(points / self.voxel_size)
        if np.any(np.abs(scaled) >= 2**62):
            raise ValueError("Point coordinate exceeds voxel index range")
        keys = scaled.astype(np.int64)
        indices = unique_row_indices(keys)
        selected = points[indices]
        keys = keys[indices]
        before = self.connection.total_changes
        with self.connection:
            self.connection.executemany(
                "INSERT OR IGNORE INTO voxels VALUES(?,?,?,?,?,?)",
                (k + p for k, p in zip(keys.tolist(), selected.tolist())),
            )
        added = self.connection.total_changes - before
        self.count += added
        return added

    def preview(self, max_points=300000):
        # rowid can have gaps following INSERT OR IGNORE; LIMIT guarantees
        # the preview cap independently of those gaps. The database is full.
        stride = max(1, int(np.ceil(self.count / max(1, max_points))))
        rows = self.connection.execute(
            "SELECT x,y,z FROM voxels WHERE rowid % ? = 0 LIMIT ?",
            (stride, int(max_points)),
        ).fetchall()
        return np.asarray(rows, dtype=np.float32).reshape(-1, 3)

    @staticmethod
    def export_pcd(database: Path, destination: Path):
        """Export one consistent WAL snapshot with bounded working memory."""
        destination = Path(destination)
        temporary = destination.with_name(destination.name + ".tmp")
        connection = sqlite3.connect(str(database), timeout=30)
        try:
            connection.execute("BEGIN")
            count = connection.execute("SELECT COUNT(*) FROM voxels").fetchone()[0]
            header = (
                "# .PCD v0.7 - Point Cloud Data file format\nVERSION 0.7\n"
                "FIELDS x y z\nSIZE 4 4 4\nTYPE F F F\nCOUNT 1 1 1\n"
                f"WIDTH {count}\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\n"
                f"POINTS {count}\nDATA binary\n"
            )
            with temporary.open("wb") as stream:
                stream.write(header.encode("ascii"))
                cursor = connection.execute("SELECT x,y,z FROM voxels")
                while True:
                    rows = cursor.fetchmany(8192)
                    if not rows:
                        break
                    stream.write(np.asarray(rows, dtype="<f4").tobytes())
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, destination)
            return count
        finally:
            connection.close()

    def close(self):
        self.connection.close()
