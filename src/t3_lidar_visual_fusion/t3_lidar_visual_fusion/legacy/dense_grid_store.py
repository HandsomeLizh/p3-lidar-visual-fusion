"""Atomic single-file export of one complete dense semantic GridMap."""

from __future__ import annotations

import os
import threading
import time
import uuid
from pathlib import Path

import numpy as np

from .tiled_semantic_map import SemanticMapWindow


FORMAT_NAME = "t3_dense_global_grid_map"
FORMAT_VERSION = 1


def _scalar(value, dtype):
    return np.asarray(value, dtype=dtype)


def save_dense_global_grid_map(
    path: Path,
    window: SemanticMapWindow,
    *,
    frame_id: str,
    map_revision: int,
    timestamp_text: str,
) -> Path:
    """Write a complete dense map atomically as a compressed NumPy archive.

    Arrays use ``[row, column]`` indexing. Row zero starts at ``origin_y`` and
    column zero starts at ``origin_x``; both axes increase with their map-frame
    coordinate. The payload contains no object arrays and is safe to open with
    ``allow_pickle=False``.
    """

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    geometry = window.geometry
    expected_shape = (int(geometry.height), int(geometry.width))
    arrays = {
        "occupancy": np.asarray(window.occupancy, dtype=np.int8),
        "semantic_id": np.asarray(window.semantic, dtype=np.uint8),
        "semantic_confidence": np.asarray(
            window.semantic_confidence, dtype=np.float32
        ),
        "elevation": np.asarray(window.elevation, dtype=np.float32),
        "elevation_variance": np.asarray(
            window.elevation_variance, dtype=np.float32
        ),
        "height_range": np.asarray(window.height_range, dtype=np.float32),
        "roughness": np.asarray(window.roughness, dtype=np.float32),
        "observation_count": np.asarray(
            window.observation_count, dtype=np.uint32
        ),
        "color_bgr": np.asarray(window.semantic_color_bgr(), dtype=np.uint8),
    }
    for name, array in arrays.items():
        required_shape = expected_shape + ((3,) if name == "color_bgr" else ())
        if array.shape != required_shape:
            raise ValueError(
                f"Dense GridMap layer {name!r} has shape {array.shape}, "
                f"expected {required_shape}"
            )

    payload = {
        "format": _scalar(FORMAT_NAME, np.str_),
        "format_version": _scalar(FORMAT_VERSION, np.uint16),
        "frame_id": _scalar(str(frame_id), np.str_),
        "timestamp": _scalar(str(timestamp_text), np.str_),
        "map_revision": _scalar(map_revision, np.uint64),
        "resolution": _scalar(geometry.resolution, np.float64),
        "origin_x": _scalar(geometry.origin_x, np.float64),
        "origin_y": _scalar(geometry.origin_y, np.float64),
        "center_x": _scalar(geometry.center_x, np.float64),
        "center_y": _scalar(geometry.center_y, np.float64),
        "width": _scalar(geometry.width, np.uint32),
        "height": _scalar(geometry.height, np.uint32),
        "length_x": _scalar(geometry.length_x, np.float64),
        "length_y": _scalar(geometry.length_y, np.float64),
        "class_names": np.asarray(window.class_names, dtype=np.str_),
        **arrays,
    }

    temporary = destination.with_name(
        f".{destination.name}.{uuid.uuid4().hex}.tmp.npz"
    )
    try:
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, **payload)
            stream.flush()
            os.fsync(stream.fileno())
        with np.load(temporary, allow_pickle=False) as archive:
            if str(archive["format"].item()) != FORMAT_NAME:
                raise ValueError("Dense GridMap export validation failed")
            if tuple(archive["occupancy"].shape) != expected_shape:
                raise ValueError("Dense GridMap geometry validation failed")
        os.replace(temporary, destination)
        try:
            directory_fd = os.open(str(destination.parent), os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except OSError:
            # The file has already been atomically committed; directory fsync
            # is not available on every supported filesystem.
            pass
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return destination


class LiveDenseGlobalMapWriter:
    """Coalescing background writer for the latest complete dense map."""

    def __init__(self, output_path: Path, *, frame_id: str) -> None:
        self.output_path = Path(output_path)
        self.frame_id = str(frame_id)
        self._condition = threading.Condition()
        self._pending = None
        self._active_revision = -1
        self._persisted_revision = -1
        self._last_error: Exception | None = None
        self._stop_requested = False
        self._thread = threading.Thread(
            target=self._run,
            name="t3-live-dense-grid-map-writer",
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

    def enqueue(
        self,
        window: SemanticMapWindow,
        *,
        map_revision: int,
        timestamp_text: str,
    ) -> bool:
        revision = int(map_revision)
        with self._condition:
            if self._stop_requested:
                raise RuntimeError("live dense-map writer is closed")
            pending_revision = (
                int(self._pending[1]) if self._pending is not None else -1
            )
            if revision <= max(
                self._persisted_revision,
                self._active_revision,
                pending_revision,
            ):
                return False
            self._pending = (window, revision, str(timestamp_text))
            self._condition.notify_all()
            return True

    def flush(self, map_revision: int, timeout_sec: float = 60.0) -> None:
        target = int(map_revision)
        deadline = time.monotonic() + float(timeout_sec)
        with self._condition:
            while self._persisted_revision < target:
                if (
                    self._last_error is not None
                    and self._active_revision < target
                    and (
                        self._pending is None
                        or int(self._pending[1]) < target
                    )
                ):
                    raise RuntimeError(
                        f"live dense-map update failed: {self._last_error}"
                    ) from self._last_error
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    raise TimeoutError(
                        f"dense map did not reach revision {target}"
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
        self._thread.join(timeout=60.0)
        if self._thread.is_alive():
            raise TimeoutError("live dense-map writer did not stop")
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
                self._active_revision = int(pending[1])
            window, revision, timestamp_text = pending
            try:
                save_dense_global_grid_map(
                    self.output_path,
                    window,
                    frame_id=self.frame_id,
                    map_revision=revision,
                    timestamp_text=timestamp_text,
                )
            except Exception as exception:
                with self._condition:
                    self._last_error = exception
                    self._active_revision = -1
                    self._condition.notify_all()
                continue
            with self._condition:
                self._persisted_revision = max(
                    self._persisted_revision, revision
                )
                self._active_revision = -1
                self._last_error = None
                self._condition.notify_all()
