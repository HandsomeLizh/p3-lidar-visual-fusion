"""Compact, single-file delivery format for the finalized global grid map."""

from __future__ import annotations

import io
import json
import os
import sqlite3
import threading
import time
import zlib
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Tuple

import numpy as np

from .tiled_semantic_map import TiledSemanticMapManager


FORMAT_NAME = "t3_compact_global_grid_map"
FORMAT_VERSION = 1
ELEVATION_SCALE_M = 0.01
VARIANCE_SCALE_M2 = 0.0001
HEIGHT_RANGE_SCALE_M = 0.001
ROUGHNESS_SCALE_M = 0.001
INT16_UNKNOWN = np.int16(-32768)
UINT16_UNKNOWN = np.uint16(65535)


def _quantize_elevation(values: np.ndarray) -> Tuple[float, np.ndarray]:
    source = np.asarray(values, dtype=np.float64)
    encoded = np.full(source.shape, INT16_UNKNOWN, dtype=np.int16)
    valid = np.isfinite(source)
    if not np.any(valid):
        return 0.0, encoded
    minimum = float(np.min(source[valid]))
    maximum = float(np.max(source[valid]))
    offset = 0.5 * (minimum + maximum)
    quantized = np.rint((source[valid] - offset) / ELEVATION_SCALE_M)
    if np.any(quantized < -32767) or np.any(quantized > 32767):
        raise ValueError(
            "tile elevation range exceeds the int16 centimetre encoding"
        )
    encoded[valid] = quantized.astype(np.int16)
    return offset, encoded


def _quantize_unsigned(values: np.ndarray, scale: float) -> np.ndarray:
    source = np.asarray(values, dtype=np.float64)
    encoded = np.full(source.shape, UINT16_UNKNOWN, dtype=np.uint16)
    valid = np.isfinite(source)
    if np.any(valid):
        encoded[valid] = np.clip(
            np.rint(np.maximum(source[valid], 0.0) / float(scale)),
            0,
            65534,
        ).astype(np.uint16)
    return encoded


def _tile_payload(tile) -> Tuple[float, bytes]:
    elevation_offset_m, elevation_cm = _quantize_elevation(
        tile.elevation_layer()
    )
    buffer = io.BytesIO()
    np.savez(
        buffer,
        occupancy=tile.occupancy_int8(),
        semantic=tile.semantic_id_layer(),
        semantic_confidence=np.clip(
            np.rint(tile.semantic_confidence_layer() * 255.0),
            0,
            255,
        ).astype(np.uint8),
        elevation=elevation_cm,
        elevation_variance=_quantize_unsigned(
            tile.elevation_variance_layer(), VARIANCE_SCALE_M2
        ),
        height_range=_quantize_unsigned(
            tile.height_range_layer(), HEIGHT_RANGE_SCALE_M
        ),
        roughness=_quantize_unsigned(
            tile.roughness_layer(), ROUGHNESS_SCALE_M
        ),
        observation_count=np.clip(
            tile.elevation_count, 0, 65535
        ).astype(np.uint16),
    )
    return elevation_offset_m, buffer.getvalue()


def _compressed_tile_payload(tile, compression_level: int) -> Tuple[float, bytes]:
    """Encode one immutable tile and compress it outside the SQLite thread.

    NumPy and zlib execute the expensive parts in native code and release the
    GIL. A thread pool therefore uses multiple CPU cores without copying tile
    arrays through multiprocessing IPC.
    """
    offset, raw_payload = _tile_payload(tile)
    return offset, zlib.compress(raw_payload, int(compression_level))


def _iter_compressed_tiles(
    grid: TiledSemanticMapManager,
    keys: Iterable[Tuple[int, int]],
    *,
    compression_level: int,
    executor: ThreadPoolExecutor | None,
    max_in_flight: int,
) -> Iterator[Tuple[Tuple[int, int], float, bytes]]:
    """Yield encoded tiles in key order with bounded temporary memory."""
    ordered_keys = iter(keys)
    if executor is None or int(max_in_flight) <= 1:
        for key in ordered_keys:
            offset, payload = _compressed_tile_payload(
                grid.tiles[key], compression_level
            )
            yield key, offset, payload
        return

    pending = deque()
    for _ in range(int(max_in_flight)):
        try:
            key = next(ordered_keys)
        except StopIteration:
            break
        pending.append(
            (
                key,
                executor.submit(
                    _compressed_tile_payload,
                    grid.tiles[key],
                    compression_level,
                ),
            )
        )

    while pending:
        key, future = pending.popleft()
        offset, payload = future.result()
        yield key, offset, payload
        try:
            next_key = next(ordered_keys)
        except StopIteration:
            continue
        pending.append(
            (
                next_key,
                executor.submit(
                    _compressed_tile_payload,
                    grid.tiles[next_key],
                    compression_level,
                ),
            )
        )


def _metadata(
    grid: TiledSemanticMapManager,
    *,
    frame_id: str,
    timestamp_text: str,
    map_revision: int | None = None,
) -> Dict[str, Any]:
    return {
        "format": FORMAT_NAME,
        "format_version": FORMAT_VERSION,
        "map_id": grid.map_id,
        "frame_id": str(frame_id),
        "timestamp": str(timestamp_text),
        "map_revision": (
            grid.map_revision if map_revision is None else int(map_revision)
        ),
        "resolution": grid.resolution,
        "tile_cells": grid.tile_cells,
        "tile_length": grid.tile_length,
        "origin_x": grid.origin_x,
        "origin_y": grid.origin_y,
        "tile_count": grid.tile_count,
        "known_cells": grid.known_cells,
        "known_area_m2": grid.known_area_m2,
        "class_names": list(grid.class_names),
        "encoding": {
            "occupancy": "int8:-1_unknown,0_to_100",
            "semantic": "uint8",
            "semantic_confidence": "uint8:0_to_255",
            "elevation": "int16:per_tile_offset+value*0.01m,-32768_unknown",
            "elevation_variance": "uint16:value*0.0001m2,65535_unknown",
            "height_range": "uint16:value*0.001m,65535_unknown",
            "roughness": "uint16:value*0.001m,65535_unknown",
            "observation_count": "uint16:saturating",
        },
    }


def save_compact_global_map(
    grid: TiledSemanticMapManager,
    output_path: Path,
    *,
    frame_id: str,
    timestamp_text: str,
    compression_level: int = 6,
    map_revision: int | None = None,
) -> Path:
    """Atomically save finalized map layers as one indexed SQLite file."""

    level = int(compression_level)
    if not 0 <= level <= 9:
        raise ValueError("compression_level must be between 0 and 9")
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(
        f".{output.name}.{os.getpid()}.{time.time_ns()}.tmp"
    )
    connection = None
    try:
        connection = sqlite3.connect(str(temporary))
        connection.execute("PRAGMA journal_mode=OFF")
        connection.execute("PRAGMA synchronous=OFF")
        connection.execute("PRAGMA user_version=1")
        connection.execute(
            "CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
        )
        connection.execute(
            "CREATE TABLE tiles ("
            "tile_x INTEGER NOT NULL, tile_y INTEGER NOT NULL, "
            "revision INTEGER NOT NULL, elevation_offset_m REAL NOT NULL, "
            "payload BLOB NOT NULL, PRIMARY KEY(tile_x, tile_y))"
        )
        connection.executemany(
            "INSERT INTO metadata(key, value) VALUES(?, ?)",
            [
                (key, json.dumps(value, separators=(",", ":")))
                for key, value in _metadata(
                    grid,
                    frame_id=frame_id,
                    timestamp_text=timestamp_text,
                    map_revision=map_revision,
                ).items()
            ],
        )
        for key in sorted(grid.tiles):
            offset, raw_payload = _tile_payload(grid.tiles[key])
            connection.execute(
                "INSERT INTO tiles VALUES(?, ?, ?, ?, ?)",
                (
                    int(key[0]),
                    int(key[1]),
                    int(grid.tile_revisions.get(key, 0)),
                    float(offset),
                    sqlite3.Binary(zlib.compress(raw_payload, level)),
                ),
            )
        connection.commit()
        connection.close()
        connection = None
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, output)
        return output
    finally:
        if connection is not None:
            connection.close()
        if temporary.exists():
            temporary.unlink()


def update_compact_global_map(
    grid: TiledSemanticMapManager,
    output_path: Path,
    *,
    frame_id: str,
    timestamp_text: str,
    map_revision: int,
    compression_level: int = 6,
    executor: ThreadPoolExecutor | None = None,
    max_in_flight: int = 1,
) -> Dict[str, int]:
    """Transactionally update only changed rows in an existing live map."""

    output = Path(output_path)
    if not output.exists():
        save_compact_global_map(
            grid,
            output,
            frame_id=frame_id,
            timestamp_text=timestamp_text,
            compression_level=compression_level,
            map_revision=map_revision,
        )
        return {"updated_tiles": grid.tile_count, "deleted_tiles": 0}

    with sqlite3.connect(str(output), timeout=5.0) as connection:
        connection.execute("PRAGMA busy_timeout=5000")
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=NORMAL")
        stored_revisions = {
            (int(tile_x), int(tile_y)): int(revision)
            for tile_x, tile_y, revision in connection.execute(
                "SELECT tile_x, tile_y, revision FROM tiles"
            )
        }
        current_keys = set(grid.tiles)
        changed_keys = tuple(
            key
            for key in sorted(current_keys)
            if stored_revisions.get(key)
            != int(grid.tile_revisions.get(key, 0))
        )
        deleted_keys = tuple(sorted(set(stored_revisions) - current_keys))
        metadata = _metadata(
            grid,
            frame_id=frame_id,
            timestamp_text=timestamp_text,
            map_revision=map_revision,
        )
        connection.execute("BEGIN IMMEDIATE")
        try:
            connection.executemany(
                "INSERT INTO metadata(key, value) VALUES(?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                [
                    (key, json.dumps(value, separators=(",", ":")))
                    for key, value in metadata.items()
                ],
            )
            for key, offset, compressed_payload in _iter_compressed_tiles(
                grid,
                changed_keys,
                compression_level=int(compression_level),
                executor=executor,
                max_in_flight=max_in_flight,
            ):
                connection.execute(
                    "INSERT INTO tiles(tile_x, tile_y, revision, "
                    "elevation_offset_m, payload) VALUES(?, ?, ?, ?, ?) "
                    "ON CONFLICT(tile_x, tile_y) DO UPDATE SET "
                    "revision=excluded.revision, "
                    "elevation_offset_m=excluded.elevation_offset_m, "
                    "payload=excluded.payload",
                    (
                        int(key[0]),
                        int(key[1]),
                        int(grid.tile_revisions.get(key, 0)),
                        float(offset),
                        sqlite3.Binary(compressed_payload),
                    ),
                )
            connection.executemany(
                "DELETE FROM tiles WHERE tile_x=? AND tile_y=?",
                [(int(key[0]), int(key[1])) for key in deleted_keys],
            )
            connection.commit()
        except Exception:
            connection.rollback()
            raise
    return {
        "updated_tiles": len(changed_keys),
        "deleted_tiles": len(deleted_keys),
    }


class LiveCompactGlobalMapWriter:
    """Coalescing background writer for the current complete global map."""

    def __init__(
        self,
        output_path: Path,
        *,
        initial_grid: TiledSemanticMapManager,
        frame_id: str,
        compression_level: int = 6,
        parallel_workers: int = 1,
    ) -> None:
        self.output_path = Path(output_path)
        self.frame_id = str(frame_id)
        self.compression_level = int(compression_level)
        self.parallel_workers = max(1, int(parallel_workers))
        save_compact_global_map(
            initial_grid,
            self.output_path,
            frame_id=self.frame_id,
            timestamp_text="initialising",
            compression_level=self.compression_level,
            map_revision=0,
        )
        self._condition = threading.Condition()
        self._pending = None
        self._stop_requested = False
        self._persisted_revision = 0
        self._last_error: Exception | None = None
        self._last_statistics: Dict[str, int] = {
            "updated_tiles": 0,
            "deleted_tiles": 0,
        }
        self._executor = (
            ThreadPoolExecutor(
                max_workers=self.parallel_workers,
                thread_name_prefix="t3-map-tile-encode",
            )
            if self.parallel_workers > 1
            else None
        )
        self._thread = threading.Thread(
            target=self._run,
            name="t3-live-global-map-writer",
            daemon=True,
        )
        self._thread.start()

    @property
    def persisted_revision(self) -> int:
        with self._condition:
            return int(self._persisted_revision)

    @property
    def last_error(self) -> Exception | None:
        with self._condition:
            return self._last_error

    @property
    def last_statistics(self) -> Dict[str, int]:
        with self._condition:
            return dict(self._last_statistics)

    def enqueue(
        self,
        grid: TiledSemanticMapManager,
        *,
        map_revision: int,
        timestamp_text: str,
    ) -> None:
        with self._condition:
            if self._stop_requested:
                raise RuntimeError("live compact-map writer is closed")
            self._pending = (
                grid,
                int(map_revision),
                str(timestamp_text),
            )
            self._condition.notify_all()

    def flush(self, map_revision: int, timeout_sec: float = 30.0) -> None:
        target = int(map_revision)
        deadline = time.monotonic() + float(timeout_sec)
        with self._condition:
            while self._persisted_revision < target:
                if self._last_error is not None:
                    raise RuntimeError(
                        f"live compact-map update failed: {self._last_error}"
                    ) from self._last_error
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    raise TimeoutError(
                        f"compact map did not reach revision {target}"
                    )
                self._condition.wait(timeout=remaining)

    def close(self, *, flush_revision: int | None = None) -> None:
        flush_error = None
        if flush_revision is not None:
            try:
                self.flush(flush_revision)
            except Exception as exception:
                flush_error = exception
        with self._condition:
            self._stop_requested = True
            self._condition.notify_all()
        self._thread.join(timeout=30.0)
        if self._thread.is_alive():
            raise TimeoutError("live compact-map writer did not stop")
        if self._executor is not None:
            self._executor.shutdown(wait=True, cancel_futures=False)
            self._executor = None
        if flush_error is not None:
            raise flush_error

    def _run(self) -> None:
        while True:
            with self._condition:
                while self._pending is None and not self._stop_requested:
                    self._condition.wait()
                if self._pending is None and self._stop_requested:
                    return
                pending = self._pending
                self._pending = None
            grid, revision, timestamp_text = pending
            try:
                statistics = update_compact_global_map(
                    grid,
                    self.output_path,
                    frame_id=self.frame_id,
                    timestamp_text=timestamp_text,
                    map_revision=revision,
                    compression_level=self.compression_level,
                    executor=self._executor,
                    max_in_flight=self.parallel_workers,
                )
            except Exception as exception:
                with self._condition:
                    self._last_error = exception
                    self._condition.notify_all()
                continue
            with self._condition:
                self._persisted_revision = max(
                    self._persisted_revision, revision
                )
                self._last_error = None
                self._last_statistics = statistics
                self._condition.notify_all()


def load_compact_metadata(path: Path) -> Dict[str, Any]:
    """Read delivery metadata without loading any map tile payload."""

    with sqlite3.connect(str(Path(path))) as connection:
        rows = connection.execute("SELECT key, value FROM metadata").fetchall()
    return {str(key): json.loads(value) for key, value in rows}


def load_compact_tile(path: Path, tile_x: int, tile_y: int) -> Dict[str, Any]:
    """Decode one tile for offline consumers and format verification."""

    with sqlite3.connect(str(Path(path))) as connection:
        row = connection.execute(
            "SELECT revision, elevation_offset_m, payload FROM tiles "
            "WHERE tile_x=? AND tile_y=?",
            (int(tile_x), int(tile_y)),
        ).fetchone()
    if row is None:
        raise KeyError((int(tile_x), int(tile_y)))
    revision, elevation_offset_m, compressed = row
    with np.load(
        io.BytesIO(zlib.decompress(compressed)), allow_pickle=False
    ) as archive:
        arrays = {name: archive[name].copy() for name in archive.files}
    encoded_elevation = arrays["elevation"]
    elevation = np.full(encoded_elevation.shape, np.nan, dtype=np.float32)
    valid = encoded_elevation != INT16_UNKNOWN
    elevation[valid] = (
        float(elevation_offset_m)
        + encoded_elevation[valid].astype(np.float32) * ELEVATION_SCALE_M
    )
    arrays["elevation"] = elevation
    arrays["revision"] = int(revision)
    arrays["elevation_offset_m"] = float(elevation_offset_m)
    return arrays
