from __future__ import annotations

import zipfile
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import cv2
import numpy as np
import yaml

from .lidar_occupancy import (
    ENDPOINT_FREE,
    ENDPOINT_OCCUPIED,
    LidarGroundSegmentationConfig,
    validate_endpoint_occupancy_states,
)
from .semantic_grid import (
    CLASS_NAMES,
    PALETTE_BGR,
    GridGeometry,
    LayeredSemanticGridMap,
)
from .snapshot_store import commit_atomic_snapshot


TileKey = Tuple[int, int]


def _bresenham_cells(
    start_col: int,
    start_row: int,
    end_col: int,
    end_row: int,
) -> np.ndarray:
    """Vectorized (row, column) cells, bit-identical to ``_bresenham``.

    闭式解已与逐步生成器在穷举小域及 2 万条随机长线上逐格比对一致。
    占用光线每帧要展开数十万个格子，逐格生成器是 Python 级热点。
    """
    x0, y0 = int(start_col), int(start_row)
    x1, y1 = int(end_col), int(end_row)
    a = abs(x1 - x0)
    b = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    steps = np.arange(max(a, b) + 1, dtype=np.int64)
    if a >= b:
        x = x0 + sx * steps
        if a > 0:
            y = y0 + sy * ((2 * steps * b + a) // (2 * a))
        else:
            y = np.full_like(steps, y0)
    else:
        y = y0 + sy * steps
        x = x0 + sx * ((2 * steps * a + b) // (2 * b))
    return np.stack([y, x], axis=1)


@dataclass(frozen=True)
class DirtyRegion:
    """Inclusive-exclusive dirty bounds in one tile's [row, column] array."""

    min_row: int
    min_column: int
    max_row: int
    max_column: int

    @property
    def width(self) -> int:
        return int(self.max_column - self.min_column)

    @property
    def height(self) -> int:
        return int(self.max_row - self.min_row)

    def merged(self, other: "DirtyRegion") -> "DirtyRegion":
        return DirtyRegion(
            min(self.min_row, other.min_row),
            min(self.min_column, other.min_column),
            max(self.max_row, other.max_row),
            max(self.max_column, other.max_column),
        )


@dataclass(frozen=True)
class SemanticMapWindow:
    """Detached rectangular map payload in the global map frame."""

    geometry: GridGeometry
    class_names: Tuple[str, ...]
    occupancy: np.ndarray
    semantic: np.ndarray
    semantic_confidence: np.ndarray
    elevation: np.ndarray
    elevation_variance: np.ndarray
    height_range: np.ndarray
    roughness: np.ndarray
    observation_count: np.ndarray

    def occupancy_int8(self) -> np.ndarray:
        return self.occupancy

    def semantic_id_layer(self) -> np.ndarray:
        return self.semantic

    def semantic_confidence_layer(self) -> np.ndarray:
        return self.semantic_confidence

    def elevation_layer(self) -> np.ndarray:
        return self.elevation

    def elevation_variance_layer(self) -> np.ndarray:
        return self.elevation_variance

    def height_range_layer(self) -> np.ndarray:
        return self.height_range

    def roughness_layer(self) -> np.ndarray:
        return self.roughness

    def occupancy_probability_layer(self) -> np.ndarray:
        result = np.full(self.occupancy.shape, np.nan, dtype=np.float32)
        observed = self.occupancy >= 0
        result[observed] = self.occupancy[observed].astype(np.float32) / 100.0
        return result

    def semantic_color_bgr(self) -> np.ndarray:
        clipped = np.clip(
            self.semantic.astype(np.int32),
            0,
            len(PALETTE_BGR) - 1,
        )
        return PALETTE_BGR[clipped]

    def semantic_color_packed_float(self) -> np.ndarray:
        bgr = self.semantic_color_bgr()
        rgb_uint = (
            (bgr[:, :, 2].astype(np.uint32) << 16)
            | (bgr[:, :, 1].astype(np.uint32) << 8)
            | bgr[:, :, 0].astype(np.uint32)
        )
        return rgb_uint.view(np.float32)

    def elevation_color_bgr(self) -> np.ndarray:
        valid = np.isfinite(self.elevation)
        normalized = np.zeros(self.elevation.shape, dtype=np.uint8)
        if np.any(valid):
            low, high = np.nanpercentile(self.elevation, [2.0, 98.0])
            if not np.isfinite(low) or not np.isfinite(high) or high <= low:
                low = float(np.nanmin(self.elevation))
                high = float(np.nanmax(self.elevation))
            if high > low:
                scaled = (self.elevation - low) / (high - low)
                normalized[valid] = np.clip(
                    scaled[valid] * 255.0,
                    0,
                    255,
                ).astype(np.uint8)
            else:
                normalized[valid] = 127
        color = cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)
        color[~valid] = 0
        return color

    def all_layers(self) -> Dict[str, np.ndarray]:
        return self.layers()

    def layers(
        self, requested_names: Optional[Iterable[str]] = None
    ) -> Dict[str, np.ndarray]:
        """Materialise only requested GridMap layers in stable wire order."""

        builders = {
            "elevation": lambda: self.elevation,
            "elevation_variance": lambda: self.elevation_variance,
            "occupancy": self.occupancy_probability_layer,
            "semantic_id": lambda: self.semantic.astype(np.float32),
            "semantic_confidence": lambda: self.semantic_confidence,
            "observation_count": lambda: self.observation_count.astype(
                np.float32
            ),
            "height_range": lambda: self.height_range,
            "roughness": lambda: self.roughness,
            "color": self.semantic_color_packed_float,
        }
        names = tuple(builders) if requested_names is None else tuple(
            str(name) for name in requested_names
        )
        unknown = tuple(name for name in names if name not in builders)
        if unknown:
            raise ValueError(
                "Unknown GridMap layers: " + ", ".join(sorted(set(unknown)))
            )
        return {name: builders[name]() for name in names}


@dataclass(frozen=True)
class TileUpdateMetadata:
    tile_x: int
    tile_y: int
    tile_revision: int
    cleared: bool


@dataclass(frozen=True)
class TileUpdate:
    tile_x: int
    tile_y: int
    tile_revision: int
    cleared: bool
    dirty_x: int
    dirty_y: int
    window: SemanticMapWindow


class TiledSemanticMapManager:
    """Unbounded sparse 2.5D map made from fixed-size dense tiles.

    The existing ``LayeredSemanticGridMap`` remains the fusion kernel for one
    tile. Incremental transactions shallow-copy the tile dictionary and clone
    only tiles that are actually touched. Coordinates are indexed against the
    global map origin, so Python floor division correctly handles negative map
    coordinates.
    """

    FORMAT_VERSION = 1
    _SNAPSHOT_TILE_FIELDS = (
        ("occupancy_log_odds", np.float32),
        ("occupancy_observed", np.bool_),
        ("semantic_votes", np.uint32),
        ("elevation_count", np.uint32),
        ("elevation_mean", np.float64),
        ("elevation_M2", np.float64),
        ("elevation_min", np.float64),
        ("elevation_max", np.float64),
    )

    def __init__(
        self,
        *,
        resolution: float = 0.10,
        tile_cells: int = 256,
        origin_x: float = 0.0,
        origin_y: float = 0.0,
        class_names: Optional[Iterable[str]] = None,
        free_log_odds: float = -0.40,
        occupied_log_odds: float = 0.85,
        min_log_odds: float = -4.0,
        max_log_odds: float = 4.0,
        elevation_mad_scale: float = 3.5,
        min_elevation_mad: float = 0.03,
        max_rays_per_update: int = 1200,
        free_class_ids: Iterable[int] = (1,),
        occupied_class_ids: Iterable[int] = (3, 4),
        lidar_ground_max_slope_deg: float = 35.0,
        lidar_ground_neighbor_count: int = 12,
        lidar_ground_max_neighbor_distance_m: float = 0.75,
        map_id: str = "t3_semantic_map",
    ) -> None:
        self.resolution = float(resolution)
        self.tile_cells = int(tile_cells)
        if self.resolution <= 0.0:
            raise ValueError("resolution must be positive")
        if self.tile_cells <= 0:
            raise ValueError("tile_cells must be positive")
        self.origin_x = float(origin_x)
        self.origin_y = float(origin_y)
        if not np.isfinite([self.origin_x, self.origin_y]).all():
            raise ValueError("tile lattice origin must be finite")

        self.class_names = tuple(class_names or CLASS_NAMES)
        self.free_class_ids = frozenset(int(value) for value in free_class_ids)
        self.occupied_class_ids = frozenset(
            int(value) for value in occupied_class_ids
        )
        self.map_id = str(map_id)
        self.max_rays_per_update = int(max_rays_per_update)
        self.lidar_ground_config = LidarGroundSegmentationConfig(
            max_slope_deg=lidar_ground_max_slope_deg,
            neighbor_count=lidar_ground_neighbor_count,
            max_neighbor_distance_m=lidar_ground_max_neighbor_distance_m,
        )
        self._tile_kwargs = {
            "resolution": self.resolution,
            "length_x": self.tile_length,
            "length_y": self.tile_length,
            "class_names": list(self.class_names),
            "free_log_odds": float(free_log_odds),
            "occupied_log_odds": float(occupied_log_odds),
            "min_log_odds": float(min_log_odds),
            "max_log_odds": float(max_log_odds),
            "elevation_mad_scale": float(elevation_mad_scale),
            "min_elevation_mad": float(min_elevation_mad),
            # Occupancy is updated globally below so rays may cross tiles.
            "max_rays_per_update": 0,
        }

        self.tiles: Dict[TileKey, LayeredSemanticGridMap] = {}
        self.tile_revisions: Dict[TileKey, int] = {}
        self._surface_counts: Dict[TileKey, int] = {}
        self._observed_counts: Dict[TileKey, int] = {}
        self._accepted_counts: Dict[TileKey, int] = {}
        self._elevation_ranges: Dict[TileKey, Tuple[float, float]] = {}
        self._dirty_regions: Dict[TileKey, DirtyRegion] = {}
        self._cleared_tiles = set()
        self._shared_tiles = set()
        self._shared_non_elevation_tiles = set()
        self._last_updated_tile_keys: Tuple[TileKey, ...] = ()
        self.update_id = 0
        self.total_accepted_points = 0

    @property
    def tile_length(self) -> float:
        return self.tile_cells * self.resolution

    @property
    def tile_count(self) -> int:
        return len(self.tiles)

    @property
    def known_cells(self) -> int:
        return int(sum(self._observed_counts.values()))

    @property
    def surface_cells(self) -> int:
        return int(sum(self._surface_counts.values()))

    @property
    def known_area_m2(self) -> float:
        return self.known_cells * self.resolution * self.resolution

    @property
    def map_revision(self) -> int:
        return int(self.update_id)

    @property
    def last_updated_tile_keys(self) -> Tuple[TileKey, ...]:
        """Tiles touched by the most recent observation update.

        This includes occupancy-ray tiles, not only endpoint tiles.  It is a
        runtime dependency signal for corrected-map replay and is deliberately
        not part of the ROS or snapshot interface.
        """
        return self._last_updated_tile_keys

    def clone_for_update(self) -> "TiledSemanticMapManager":
        """Create a copy-on-write transaction candidate."""
        candidate = object.__new__(TiledSemanticMapManager)
        candidate.resolution = self.resolution
        candidate.tile_cells = self.tile_cells
        candidate.origin_x = self.origin_x
        candidate.origin_y = self.origin_y
        candidate.class_names = self.class_names
        candidate.free_class_ids = self.free_class_ids
        candidate.occupied_class_ids = self.occupied_class_ids
        candidate.map_id = self.map_id
        candidate.max_rays_per_update = self.max_rays_per_update
        candidate.lidar_ground_config = self.lidar_ground_config
        candidate._tile_kwargs = dict(self._tile_kwargs)
        candidate.tiles = dict(self.tiles)
        candidate.tile_revisions = dict(self.tile_revisions)
        candidate._surface_counts = dict(self._surface_counts)
        candidate._observed_counts = dict(self._observed_counts)
        candidate._accepted_counts = dict(self._accepted_counts)
        candidate._elevation_ranges = dict(self._elevation_ranges)
        candidate._dirty_regions = dict(self._dirty_regions)
        candidate._cleared_tiles = set(self._cleared_tiles)
        candidate._shared_tiles = set(self.tiles)
        candidate._shared_non_elevation_tiles = set()
        candidate._last_updated_tile_keys = ()
        candidate.update_id = self.update_id
        candidate.total_accepted_points = self.total_accepted_points
        return candidate

    def _tile_origin(self, key: TileKey) -> Tuple[float, float]:
        return (
            self.origin_x + key[0] * self.tile_length,
            self.origin_y + key[1] * self.tile_length,
        )

    def _new_tile(self, key: TileKey) -> LayeredSemanticGridMap:
        origin_x, origin_y = self._tile_origin(key)
        if getattr(self,'_temporal_elevation_config',{}).get('enabled',False):
            from ..elevation_fusion import TemporalElevationTile
            return TemporalElevationTile(fusion=self._temporal_elevation_config,
                **self._tile_kwargs,origin_x=origin_x,origin_y=origin_y)
        return LayeredSemanticGridMap(
            **self._tile_kwargs,
            origin_x=origin_x,
            origin_y=origin_y,
        )

    def _writable_tile(self, key: TileKey) -> LayeredSemanticGridMap:
        # A tile can become active again after a corrected rebuild previously
        # retired it. Active data always supersedes the retained tombstone.
        self._cleared_tiles.discard(key)
        tile = self.tiles.get(key)
        if tile is None:
            tile = self._new_tile(key)
            self.tiles[key] = tile
            # A corrected rebuild may have published a higher-revision clear
            # tombstone for this key.  If evidence later returns to the same
            # tile, retain that revision history so receivers cannot mistake
            # the reactivated tile for stale pre-clear data.
            self.tile_revisions.setdefault(key, 0)
            self._surface_counts[key] = 0
            self._observed_counts[key] = 0
            self._accepted_counts[key] = 0
            self._elevation_ranges[key] = (float("nan"), float("nan"))
            return tile
        if key in self._shared_tiles:
            tile = deepcopy(tile)
            self.tiles[key] = tile
            self._shared_tiles.remove(key)
        elif key in self._shared_non_elevation_tiles:
            # A preceding elevation-only update copied only its five height
            # arrays.  Promote it before any semantic/occupancy write.
            tile = deepcopy(tile)
            self.tiles[key] = tile
            self._shared_non_elevation_tiles.remove(key)
        return tile

    def _writable_elevation_tile(
        self, key: TileKey
    ) -> LayeredSemanticGridMap:
        """Return a tile writable only through its elevation arrays."""

        tile = (
            self._writable_tile(key)
            if key not in self.tiles
            else self.tiles[key]
        )
        self._cleared_tiles.discard(key)
        if key in self._shared_tiles:
            tile = tile.clone_elevation_for_update()
            self.tiles[key] = tile
            self._shared_tiles.remove(key)
            self._shared_non_elevation_tiles.add(key)
        return tile

    def _global_cells(
        self,
        x: np.ndarray,
        y: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        columns = np.floor(
            (np.asarray(x, dtype=np.float64) - self.origin_x)
            / self.resolution
        )
        rows = np.floor(
            (np.asarray(y, dtype=np.float64) - self.origin_y)
            / self.resolution
        )
        return rows.astype(np.int64), columns.astype(np.int64)

    def _split_global_cell(
        self,
        row: int,
        column: int,
    ) -> Tuple[TileKey, int, int]:
        tile_x = int(column // self.tile_cells)
        tile_y = int(row // self.tile_cells)
        local_column = int(column - tile_x * self.tile_cells)
        local_row = int(row - tile_y * self.tile_cells)
        return (tile_x, tile_y), local_row, local_column

    def _mark_dirty_bounds(
        self,
        key: TileKey,
        min_row: int,
        min_column: int,
        max_row: int,
        max_column: int,
    ) -> None:
        region = DirtyRegion(
            int(min_row),
            int(min_column),
            int(max_row),
            int(max_column),
        )
        previous = self._dirty_regions.get(key)
        self._dirty_regions[key] = (
            region if previous is None else previous.merged(region)
        )

    def _update_global_occupancy(
        self,
        *,
        global_rows: np.ndarray,
        global_columns: np.ndarray,
        labels: np.ndarray,
        camera_position_map: np.ndarray,
        geometric_endpoint_states: Optional[np.ndarray],
        touched: set,
    ) -> None:
        camera = np.asarray(camera_position_map, dtype=np.float64).reshape(3)
        if not np.isfinite(camera).all():
            return
        camera_rows, camera_columns = self._global_cells(
            np.array([camera[0]]),
            np.array([camera[1]]),
        )
        camera_row = int(camera_rows[0])
        camera_column = int(camera_columns[0])

        point_count = len(global_rows)
        if point_count > self.max_rays_per_update > 0:
            sample_indices = np.linspace(
                0,
                point_count - 1,
                self.max_rays_per_update,
                dtype=np.int64,
            )
        else:
            sample_indices = np.arange(point_count, dtype=np.int64)

        # 所有瓦片共用 _tile_kwargs 中同一组 log-odds 常数，因此自由/占据
        # 增量与原逐格实现一致。格集合仍由同一 _bresenham 逐光线生成；
        # 只是把逐格的瓦片查找、写入和脏区合并改为按瓦片一次性批量累加。
        free_increment = float(self._tile_kwargs["free_log_odds"])
        occupied_increment = float(self._tile_kwargs["occupied_log_odds"])

        update_rows = []
        update_columns = []
        update_increments = []
        for index in sample_indices:
            endpoint_row = int(global_rows[index])
            endpoint_column = int(global_columns[index])
            semantic_id = int(labels[index])
            cells = _bresenham_cells(
                camera_column,
                camera_row,
                endpoint_column,
                endpoint_row,
            )
            if len(cells) == 0:
                continue
            if len(cells) > 1:
                update_rows.append(cells[:-1, 0])
                update_columns.append(cells[:-1, 1])
                update_increments.append(
                    np.full(len(cells) - 1, free_increment)
                )
            if semantic_id in self.occupied_class_ids:
                endpoint_increment = occupied_increment
            elif semantic_id in self.free_class_ids:
                endpoint_increment = free_increment
            elif (
                geometric_endpoint_states is not None
                and geometric_endpoint_states[index] == ENDPOINT_OCCUPIED
            ):
                endpoint_increment = occupied_increment
            elif (
                geometric_endpoint_states is not None
                and geometric_endpoint_states[index] == ENDPOINT_FREE
            ):
                endpoint_increment = free_increment
            else:
                continue
            update_rows.append(cells[-1:, 0])
            update_columns.append(cells[-1:, 1])
            update_increments.append(np.full(1, endpoint_increment))

        if not update_rows:
            return
        rows = np.concatenate(update_rows)
        columns = np.concatenate(update_columns)
        increments = np.concatenate(update_increments)
        tile_x = columns // self.tile_cells
        tile_y = rows // self.tile_cells
        local_rows = rows - tile_y * self.tile_cells
        local_columns = columns - tile_x * self.tile_cells
        # np.unique(..., axis=0) first builds a structured array and sorts the
        # complete two-column payload.  Occupancy updates contain many ray
        # cells but normally touch only a few tiles.  A stable indirect sort
        # on the two integer keys keeps updates within each tile in their
        # original order (important for exact float32 accumulation) and avoids
        # the structured-array conversion plus one full inverse-label array.
        order = np.lexsort((tile_y, tile_x))
        ordered_tile_x = tile_x[order]
        ordered_tile_y = tile_y[order]
        group_starts = np.concatenate(
            (
                np.array([0], dtype=np.int64),
                np.flatnonzero(
                    (ordered_tile_x[1:] != ordered_tile_x[:-1])
                    | (ordered_tile_y[1:] != ordered_tile_y[:-1])
                ).astype(np.int64)
                + 1,
            )
        )
        group_ends = np.concatenate(
            (group_starts[1:], np.array([len(order)], dtype=np.int64))
        )
        for start, end in zip(group_starts, group_ends):
            indices = order[int(start):int(end)]
            key = (
                int(ordered_tile_x[int(start)]),
                int(ordered_tile_y[int(start)]),
            )
            tile_rows = local_rows[indices]
            tile_columns = local_columns[indices]
            tile = self._writable_tile(key)
            np.add.at(
                tile.occupancy_log_odds,
                (tile_rows, tile_columns),
                increments[indices].astype(np.float32),
            )
            tile.occupancy_observed[tile_rows, tile_columns] = True
            self._mark_dirty_bounds(
                key,
                int(tile_rows.min()),
                int(tile_columns.min()),
                int(tile_rows.max()) + 1,
                int(tile_columns.max()) + 1,
            )
            touched.add(key)

    def _refresh_tile_statistics(self, keys: Iterable[TileKey]) -> None:
        for key in keys:
            tile = self.tiles[key]
            surface = tile.elevation_count > 0
            observed = surface | tile.occupancy_observed
            self._surface_counts[key] = int(np.count_nonzero(surface))
            self._observed_counts[key] = int(np.count_nonzero(observed))
            self._accepted_counts[key] = int(
                np.sum(tile.elevation_count, dtype=np.uint64)
            )
            if np.any(surface):
                values = tile.elevation_mean[surface]
                self._elevation_ranges[key] = (
                    float(np.min(values)),
                    float(np.max(values)),
                )
            else:
                self._elevation_ranges[key] = (float("nan"), float("nan"))

    def update(
        self,
        *,
        points_map: np.ndarray,
        semantic_labels: np.ndarray,
        camera_position_map: np.ndarray,
        geometric_endpoint_states: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        self._last_updated_tile_keys = ()
        points = np.asarray(points_map, dtype=np.float64).reshape(-1, 3)
        labels = np.asarray(semantic_labels, dtype=np.int64).reshape(-1)
        if len(points) != len(labels):
            raise ValueError(
                f"points and labels differ: {len(points)} != {len(labels)}"
            )
        endpoint_states = None
        if geometric_endpoint_states is not None:
            endpoint_states = validate_endpoint_occupancy_states(
                geometric_endpoint_states,
                point_count=len(points),
            )

        valid = np.isfinite(points).all(axis=1)
        valid &= (labels >= 0) & (labels < len(self.class_names))
        points = points[valid]
        labels = labels[valid]
        if endpoint_states is not None:
            endpoint_states = endpoint_states[valid]
        if len(points) == 0:
            return {
                "accepted_points": 0,
                "visible_points": 0,
                "known_cells": self.known_cells,
                "known_area_m2": self.known_area_m2,
                "surface_cells": self.surface_cells,
                "updated_tiles": 0,
                "tile_count": self.tile_count,
            }

        global_rows, global_columns = self._global_cells(
            points[:, 0],
            points[:, 1],
        )
        tile_x = np.floor_divide(global_columns, self.tile_cells)
        tile_y = np.floor_divide(global_rows, self.tile_cells)
        touched = set()
        accepted_points = 0

        tile_pairs = np.column_stack((tile_x, tile_y))
        for pair in np.unique(tile_pairs, axis=0):
            key = (int(pair[0]), int(pair[1]))
            mask = (tile_x == pair[0]) & (tile_y == pair[1])
            tile = self._writable_tile(key)
            statistics = tile.update(
                points_map=points[mask],
                semantic_labels=labels[mask],
                camera_position_map=camera_position_map,
                update_occupancy=False,
            )
            accepted_points += int(statistics.get("accepted_points", 0))
            local_rows = global_rows[mask] - key[1] * self.tile_cells
            local_columns = global_columns[mask] - key[0] * self.tile_cells
            self._mark_dirty_bounds(
                key,
                int(np.min(local_rows)),
                int(np.min(local_columns)),
                int(np.max(local_rows)) + 1,
                int(np.max(local_columns)) + 1,
            )
            touched.add(key)

        self._update_global_occupancy(
            global_rows=global_rows,
            global_columns=global_columns,
            labels=labels,
            camera_position_map=camera_position_map,
            geometric_endpoint_states=endpoint_states,
            touched=touched,
        )

        for key in touched:
            tile = self.tiles[key]
            np.clip(
                tile.occupancy_log_odds,
                tile.min_log_odds,
                tile.max_log_odds,
                out=tile.occupancy_log_odds,
            )
            self.tile_revisions[key] = self.tile_revisions.get(key, 0) + 1

        self._refresh_tile_statistics(touched)
        self._last_updated_tile_keys = tuple(sorted(touched))
        self.update_id += 1
        self.total_accepted_points += accepted_points

        finite_ranges = [
            value
            for limits in self._elevation_ranges.values()
            for value in limits
            if np.isfinite(value)
        ]
        return {
            "accepted_points": int(accepted_points),
            "visible_points": int(len(points)),
            "known_cells": self.known_cells,
            "known_area_m2": self.known_area_m2,
            "surface_cells": self.surface_cells,
            "updated_tiles": int(len(touched)),
            "tile_count": self.tile_count,
            "elevation_min": (
                float(min(finite_ranges)) if finite_ranges else float("nan")
            ),
            "elevation_max": (
                float(max(finite_ranges)) if finite_ranges else float("nan")
            ),
        }

    def update_elevation_only(
        self,
        *,
        points_map: np.ndarray,
    ) -> Dict[str, float]:
        """Fuse elevation samples without semantic or occupancy side effects."""

        self._last_updated_tile_keys = ()
        points = np.asarray(points_map, dtype=np.float64).reshape(-1, 3)
        points = points[np.isfinite(points).all(axis=1)]
        if len(points) == 0:
            return {
                "accepted_points": 0,
                "visible_points": 0,
                "known_cells": self.known_cells,
                "known_area_m2": self.known_area_m2,
                "surface_cells": self.surface_cells,
                "updated_tiles": 0,
                "tile_count": self.tile_count,
            }

        global_rows, global_columns = self._global_cells(
            points[:, 0], points[:, 1]
        )
        tile_x = np.floor_divide(global_columns, self.tile_cells)
        tile_y = np.floor_divide(global_rows, self.tile_cells)
        tile_pairs = np.column_stack((tile_x, tile_y))
        touched = set()
        accepted_points = 0

        for pair in np.unique(tile_pairs, axis=0):
            key = (int(pair[0]), int(pair[1]))
            mask = (tile_x == pair[0]) & (tile_y == pair[1])
            tile = self._writable_elevation_tile(key)
            statistics = tile.update_elevation_only(
                points_map=points[mask]
            )
            accepted_points += int(statistics.get("accepted_points", 0))
            local_rows = global_rows[mask] - key[1] * self.tile_cells
            local_columns = global_columns[mask] - key[0] * self.tile_cells
            self._mark_dirty_bounds(
                key,
                int(np.min(local_rows)),
                int(np.min(local_columns)),
                int(np.max(local_rows)) + 1,
                int(np.max(local_columns)) + 1,
            )
            touched.add(key)

        for key in touched:
            self.tile_revisions[key] = self.tile_revisions.get(key, 0) + 1
        self._refresh_tile_statistics(touched)
        self._last_updated_tile_keys = tuple(sorted(touched))
        self.update_id += 1
        self.total_accepted_points += accepted_points

        finite_ranges = [
            value
            for limits in self._elevation_ranges.values()
            for value in limits
            if np.isfinite(value)
        ]
        return {
            "accepted_points": int(accepted_points),
            "visible_points": int(len(points)),
            "known_cells": self.known_cells,
            "known_area_m2": self.known_area_m2,
            "surface_cells": self.surface_cells,
            "updated_tiles": int(len(touched)),
            "tile_count": self.tile_count,
            "elevation_min": (
                float(min(finite_ranges)) if finite_ranges else float("nan")
            ),
            "elevation_max": (
                float(max(finite_ranges)) if finite_ranges else float("nan")
            ),
        }

    def pending_dirty_tile_keys(self) -> Tuple[TileKey, ...]:
        return tuple(sorted(self._dirty_regions))

    def replay_tile_keys(self) -> Tuple[TileKey, ...]:
        """Return authoritative active tiles plus retained clear tombstones."""
        return tuple(sorted(set(self.tiles) | self._cleared_tiles))

    def tile_update_metadata(self, key: TileKey) -> TileUpdateMetadata:
        """Return revision/tombstone metadata without copying tile arrays."""
        key = (int(key[0]), int(key[1]))
        if key not in self.tiles and key not in self._cleared_tiles:
            raise KeyError(f"Unknown tile {key}")
        return TileUpdateMetadata(
            tile_x=key[0],
            tile_y=key[1],
            tile_revision=int(self.tile_revisions.get(key, 0)),
            cleared=key not in self.tiles,
        )

    def mark_tiles_published(self, keys: Iterable[TileKey]) -> None:
        for key in keys:
            normalized = tuple(key)
            self._dirty_regions.pop(normalized, None)
            # Keep clear tombstones after their first publication. A Task-4
            # subscriber that reconnects later must still be able to erase a
            # tile removed by loop/BA correction.

    def mark_tiles_published_at_revisions(
        self,
        revisions: Dict[TileKey, int],
    ) -> None:
        """Clear dirty state only when the streamed revision is still live."""
        for key, revision in revisions.items():
            normalized = (int(key[0]), int(key[1]))
            if int(self.tile_revisions.get(normalized, 0)) == int(revision):
                self._dirty_regions.pop(normalized, None)

    def replace_rebuilt_tiles(
        self,
        *,
        rebuilt: "TiledSemanticMapManager",
        tile_keys: Iterable[TileKey],
    ) -> None:
        """Replace selected tiles from an exact replay candidate.

        ``self`` is expected to be a copy-on-write clone of the live map.
        Unselected tiles stay shared and untouched.  Publication metadata is
        finalized separately by :meth:`mark_rebuild_dirty` so callers can
        still abort the complete transaction before exposing any mutation.
        """
        compatible = (
            self.resolution == rebuilt.resolution
            and self.tile_cells == rebuilt.tile_cells
            and self.origin_x == rebuilt.origin_x
            and self.origin_y == rebuilt.origin_y
            and self.class_names == rebuilt.class_names
            and self.free_class_ids == rebuilt.free_class_ids
            and self.occupied_class_ids == rebuilt.occupied_class_ids
            and self.map_id == rebuilt.map_id
            and self.max_rays_per_update == rebuilt.max_rays_per_update
            and self.lidar_ground_config == rebuilt.lidar_ground_config
            and self._tile_kwargs == rebuilt._tile_kwargs
        )
        if not compatible:
            raise ValueError("Rebuilt tiles use an incompatible map lattice")

        keys = {
            (int(key[0]), int(key[1]))
            for key in tile_keys
        }
        accepted_delta = 0
        for key in keys:
            previous_accepted = int(self._accepted_counts.get(key, 0))
            self._shared_tiles.discard(key)
            self._shared_non_elevation_tiles.discard(key)
            tile = rebuilt.tiles.get(key)
            if tile is None:
                self.tiles.pop(key, None)
                self._surface_counts.pop(key, None)
                self._observed_counts.pop(key, None)
                self._accepted_counts.pop(key, None)
                self._elevation_ranges.pop(key, None)
                accepted_delta -= previous_accepted
                continue
            self.tiles[key] = tile
            self._surface_counts[key] = rebuilt._surface_counts[key]
            self._observed_counts[key] = rebuilt._observed_counts[key]
            self._accepted_counts[key] = rebuilt._accepted_counts[key]
            self._elevation_ranges[key] = rebuilt._elevation_ranges[key]
            self._cleared_tiles.discard(key)
            accepted_delta += (
                int(rebuilt._accepted_counts[key]) - previous_accepted
            )

        self.total_accepted_points += int(accepted_delta)
        self._last_updated_tile_keys = tuple(sorted(keys))

    def mark_rebuild_dirty(
        self,
        previous: "TiledSemanticMapManager",
        tile_keys: Optional[Iterable[TileKey]] = None,
    ) -> None:
        """Publish replacement tiles after a pose-geometry rebuild.

        A loop/BA correction can move evidence out of cells that were already
        consumed downstream. Marking only newly populated cells would leave
        ghosts at the old locations. Full-tile replacement rectangles for the
        union of old and new tile keys explicitly carry unknown values for all
        cells that disappeared, without changing the public message schema.
        """
        previous_keys = set(previous.tiles)
        current_keys = set(self.tiles)
        replacement_keys = (
            previous_keys | current_keys
            if tile_keys is None
            else {
                (int(key[0]), int(key[1]))
                for key in tile_keys
            }
        )
        full = DirtyRegion(0, 0, self.tile_cells, self.tile_cells)
        for key in replacement_keys:
            self._dirty_regions[key] = full
            self.tile_revisions[key] = max(
                int(self.tile_revisions.get(key, 0)),
                int(previous.tile_revisions.get(key, 0)) + 1,
            )
        self._cleared_tiles.difference_update(
            replacement_keys & current_keys
        )
        self._cleared_tiles.update(
            replacement_keys & previous_keys - current_keys
        )

    def _window_from_arrays(
        self,
        *,
        geometry: GridGeometry,
        occupancy: np.ndarray,
        semantic: np.ndarray,
        confidence: np.ndarray,
        elevation: np.ndarray,
        variance: np.ndarray,
        height_range: np.ndarray,
        roughness: np.ndarray,
        observations: np.ndarray,
    ) -> SemanticMapWindow:
        return SemanticMapWindow(
            geometry=geometry,
            class_names=self.class_names,
            occupancy=occupancy,
            semantic=semantic,
            semantic_confidence=confidence,
            elevation=elevation,
            elevation_variance=variance,
            height_range=height_range,
            roughness=roughness,
            observation_count=observations,
        )

    def tile_update(self, key: TileKey, full_tile: bool = False) -> TileUpdate:
        key = (int(key[0]), int(key[1]))
        tile = self.tiles.get(key)
        if tile is None and key not in self._cleared_tiles:
            raise KeyError(f"Unknown tile {key}")
        if full_tile:
            region = DirtyRegion(0, 0, self.tile_cells, self.tile_cells)
        else:
            region = self._dirty_regions.get(key)
            if region is None:
                raise KeyError(f"Tile {key} has no pending dirty region")
        row_slice = slice(region.min_row, region.max_row)
        column_slice = slice(region.min_column, region.max_column)
        origin_x, origin_y = self._tile_origin(key)
        geometry = GridGeometry(
            resolution=self.resolution,
            width=region.width,
            height=region.height,
            origin_x=origin_x + region.min_column * self.resolution,
            origin_y=origin_y + region.min_row * self.resolution,
        )
        if tile is None:
            shape = (region.height, region.width)
            window = self._window_from_arrays(
                geometry=geometry,
                occupancy=np.full(shape, -1, dtype=np.int8),
                semantic=np.zeros(shape, dtype=np.uint8),
                confidence=np.zeros(shape, dtype=np.float32),
                elevation=np.full(shape, np.nan, dtype=np.float32),
                variance=np.full(shape, np.nan, dtype=np.float32),
                height_range=np.full(shape, np.nan, dtype=np.float32),
                roughness=np.full(shape, np.nan, dtype=np.float32),
                observations=np.zeros(shape, dtype=np.uint32),
            )
            revision = int(self.tile_revisions.get(key, 0))
        else:
            window = self._window_from_arrays(
                geometry=geometry,
                occupancy=tile.occupancy_int8()[row_slice, column_slice].copy(),
                semantic=tile.semantic_id_layer()[row_slice, column_slice].copy(),
                confidence=tile.semantic_confidence_layer()[
                    row_slice, column_slice
                ].copy(),
                elevation=tile.elevation_layer()[row_slice, column_slice].copy(),
                variance=tile.elevation_variance_layer()[
                    row_slice, column_slice
                ].copy(),
                height_range=tile.height_range_layer()[
                    row_slice, column_slice
                ].copy(),
                roughness=tile.roughness_layer()[row_slice, column_slice].copy(),
                observations=tile.elevation_count[
                    row_slice, column_slice
                ].copy(),
            )
            revision = int(self.tile_revisions.get(key, 0))
        return TileUpdate(
            tile_x=key[0],
            tile_y=key[1],
            tile_revision=revision,
            cleared=tile is None,
            dirty_x=region.min_column,
            dirty_y=region.min_row,
            window=window,
        )

    def semantic_ids_at(self, points_map: np.ndarray) -> np.ndarray:
        points = np.asarray(points_map, dtype=np.float64).reshape(-1, 3)
        result = np.zeros(len(points), dtype=np.uint8)
        if not len(points):
            return result
        finite = np.isfinite(points[:, :2]).all(axis=1)
        rows, columns = self._global_cells(points[:, 0], points[:, 1])
        for index in np.where(finite)[0]:
            key, local_row, local_column = self._split_global_cell(
                int(rows[index]), int(columns[index])
            )
            tile = self.tiles.get(key)
            if tile is None:
                continue
            votes = tile.semantic_votes[:, local_row, local_column]
            if np.any(votes):
                result[index] = np.uint8(np.argmax(votes))
        return result

    def extract_window(
        self,
        *,
        center_x: float,
        center_y: float,
        length_x: float,
        length_y: float,
    ) -> SemanticMapWindow:
        geometry = self.window_geometry(
            center_x=center_x,
            center_y=center_y,
            length_x=length_x,
            length_y=length_y,
        )
        width = geometry.width
        height = geometry.height
        start_column = int(round(
            (geometry.origin_x - self.origin_x) / self.resolution
        ))
        start_row = int(round(
            (geometry.origin_y - self.origin_y) / self.resolution
        ))
        end_column = start_column + width
        end_row = start_row + height

        shape = (height, width)
        occupancy = np.full(shape, -1, dtype=np.int8)
        semantic = np.zeros(shape, dtype=np.uint8)
        confidence = np.zeros(shape, dtype=np.float32)
        elevation = np.full(shape, np.nan, dtype=np.float32)
        variance = np.full(shape, np.nan, dtype=np.float32)
        height_range = np.full(shape, np.nan, dtype=np.float32)
        roughness = np.full(shape, np.nan, dtype=np.float32)
        observations = np.zeros(shape, dtype=np.uint32)

        min_tile_x = start_column // self.tile_cells
        max_tile_x = (end_column - 1) // self.tile_cells
        min_tile_y = start_row // self.tile_cells
        max_tile_y = (end_row - 1) // self.tile_cells
        for tile_y in range(min_tile_y, max_tile_y + 1):
            for tile_x in range(min_tile_x, max_tile_x + 1):
                key = (tile_x, tile_y)
                tile = self.tiles.get(key)
                if tile is None:
                    continue
                tile_start_column = tile_x * self.tile_cells
                tile_start_row = tile_y * self.tile_cells
                overlap_start_column = max(start_column, tile_start_column)
                overlap_end_column = min(
                    end_column, tile_start_column + self.tile_cells
                )
                overlap_start_row = max(start_row, tile_start_row)
                overlap_end_row = min(end_row, tile_start_row + self.tile_cells)

                source_columns = slice(
                    overlap_start_column - tile_start_column,
                    overlap_end_column - tile_start_column,
                )
                source_rows = slice(
                    overlap_start_row - tile_start_row,
                    overlap_end_row - tile_start_row,
                )
                target_columns = slice(
                    overlap_start_column - start_column,
                    overlap_end_column - start_column,
                )
                target_rows = slice(
                    overlap_start_row - start_row,
                    overlap_end_row - start_row,
                )
                source = (source_rows, source_columns)
                target = (target_rows, target_columns)
                occupancy[target] = tile.occupancy_int8()[source]
                semantic[target] = tile.semantic_id_layer()[source]
                confidence[target] = tile.semantic_confidence_layer()[source]
                elevation[target] = tile.elevation_layer()[source]
                variance[target] = tile.elevation_variance_layer()[source]
                height_range[target] = tile.height_range_layer()[source]
                roughness[target] = tile.roughness_layer()[source]
                observations[target] = tile.elevation_count[source]

        return self._window_from_arrays(
            geometry=geometry,
            occupancy=occupancy,
            semantic=semantic,
            confidence=confidence,
            elevation=elevation,
            variance=variance,
            height_range=height_range,
            roughness=roughness,
            observations=observations,
        )

    def extract_complete_window(self) -> Optional[SemanticMapWindow]:
        """Materialise one dense GridMap spanning every allocated tile.

        This is intentionally separate from the rolling local window.  The
        returned rectangle is tile-aligned, contains every current global-map
        cell, and fills unallocated gaps with the standard unknown values.
        ``None`` means that no map evidence has allocated a tile yet.
        """
        if not self.tiles:
            return None
        tile_x_values = [key[0] for key in self.tiles]
        tile_y_values = [key[1] for key in self.tiles]
        minimum_tile_x = min(tile_x_values)
        maximum_tile_x = max(tile_x_values)
        minimum_tile_y = min(tile_y_values)
        maximum_tile_y = max(tile_y_values)
        tile_count_x = maximum_tile_x - minimum_tile_x + 1
        tile_count_y = maximum_tile_y - minimum_tile_y + 1
        origin_x = self.origin_x + minimum_tile_x * self.tile_length
        origin_y = self.origin_y + minimum_tile_y * self.tile_length
        length_x = tile_count_x * self.tile_length
        length_y = tile_count_y * self.tile_length
        return self.extract_window(
            center_x=origin_x + 0.5 * length_x,
            center_y=origin_y + 0.5 * length_y,
            length_x=length_x,
            length_y=length_y,
        )

    def extract_occupancy_overview(
        self, *, resolution: float
    ) -> Tuple[GridGeometry, np.ndarray]:
        """Build a conservative low-resolution overview without a dense map.

        Only observed cells from allocated tiles are visited.  Each overview
        cell receives the maximum occupancy value of its observed fine cells,
        so any obstacle remains visible after downsampling.
        """

        requested_resolution = float(resolution)
        factor = int(round(requested_resolution / self.resolution))
        if factor <= 0 or not np.isclose(
            requested_resolution,
            factor * self.resolution,
            rtol=0.0,
            atol=1.0e-9,
        ):
            raise ValueError(
                "overview resolution must be a positive integer multiple of "
                "the map resolution"
            )

        keys = tuple(self.tiles)
        if not keys:
            geometry = GridGeometry(
                resolution=factor * self.resolution,
                width=1,
                height=1,
                origin_x=self.origin_x,
                origin_y=self.origin_y,
            )
            return geometry, np.full((1, 1), -1, dtype=np.int8)

        min_fine_column = min(key[0] for key in keys) * self.tile_cells
        max_fine_column = (
            max(key[0] for key in keys) + 1
        ) * self.tile_cells
        min_fine_row = min(key[1] for key in keys) * self.tile_cells
        max_fine_row = (max(key[1] for key in keys) + 1) * self.tile_cells
        min_coarse_column = min_fine_column // factor
        max_coarse_column = -(-max_fine_column // factor)
        min_coarse_row = min_fine_row // factor
        max_coarse_row = -(-max_fine_row // factor)
        width = max_coarse_column - min_coarse_column
        height = max_coarse_row - min_coarse_row
        occupancy = np.full((height, width), -1, dtype=np.int8)
        flat_occupancy = occupancy.reshape(-1)

        for (tile_x, tile_y), tile in self.tiles.items():
            tile_occupancy = tile.occupancy_int8()
            local_rows, local_columns = np.nonzero(tile_occupancy >= 0)
            if not len(local_rows):
                continue
            global_rows = tile_y * self.tile_cells + local_rows
            global_columns = tile_x * self.tile_cells + local_columns
            target_rows = global_rows // factor - min_coarse_row
            target_columns = global_columns // factor - min_coarse_column
            target_flat = target_rows * width + target_columns
            np.maximum.at(
                flat_occupancy,
                target_flat,
                tile_occupancy[local_rows, local_columns],
            )

        coarse_resolution = factor * self.resolution
        geometry = GridGeometry(
            resolution=coarse_resolution,
            width=width,
            height=height,
            origin_x=self.origin_x + min_coarse_column * coarse_resolution,
            origin_y=self.origin_y + min_coarse_row * coarse_resolution,
        )
        return geometry, occupancy

    def window_geometry(
        self,
        *,
        center_x: float,
        center_y: float,
        length_x: float,
        length_y: float,
    ) -> GridGeometry:
        """Return the cell-aligned window geometry without copying layers."""
        width = max(1, int(np.ceil(float(length_x) / self.resolution)))
        height = max(1, int(np.ceil(float(length_y) / self.resolution)))
        start_column = int(np.floor(
            (float(center_x) - 0.5 * width * self.resolution - self.origin_x)
            / self.resolution
        ))
        start_row = int(np.floor(
            (float(center_y) - 0.5 * height * self.resolution - self.origin_y)
            / self.resolution
        ))
        return GridGeometry(
            resolution=self.resolution,
            width=width,
            height=height,
            origin_x=self.origin_x + start_column * self.resolution,
            origin_y=self.origin_y + start_row * self.resolution,
        )

    @staticmethod
    def _tile_filename(key: TileKey) -> str:
        def component(value: int) -> str:
            return f"p{value}" if value >= 0 else f"m{abs(value)}"

        return f"tile_{component(key[0])}_{component(key[1])}.npz"

    def _snapshot_tile_entries(self):
        return [
            {
                "x": key[0],
                "y": key[1],
                "revision": int(self.tile_revisions.get(key, 0)),
                "file": f"tiles/{self._tile_filename(key)}",
                "observed_cells": int(self._observed_counts.get(key, 0)),
                "surface_cells": int(self._surface_counts.get(key, 0)),
            }
            for key in sorted(self.tiles)
        ]

    def _snapshot_metadata(
        self,
        *,
        timestamp_text: str,
        frame_id: str,
    ) -> dict:
        return {
            "format": "t3_tiled_semantic_map",
            "format_version": self.FORMAT_VERSION,
            "map_id": self.map_id,
            "timestamp": timestamp_text,
            "frame_id": frame_id,
            "resolution": self.resolution,
            "tile_cells": self.tile_cells,
            "tile_length": self.tile_length,
            "origin_x": self.origin_x,
            "origin_y": self.origin_y,
            "map_revision": self.map_revision,
            "tile_count": self.tile_count,
            "known_cells": self.known_cells,
            "known_area_m2": self.known_area_m2,
            "surface_cells": self.surface_cells,
            "total_accepted_points": int(self.total_accepted_points),
            "class_names": list(self.class_names),
            "free_class_ids": sorted(self.free_class_ids),
            "occupied_class_ids": sorted(self.occupied_class_ids),
            "cleared_tiles": [
                {
                    "x": key[0],
                    "y": key[1],
                    "revision": int(self.tile_revisions.get(key, 0)),
                }
                for key in sorted(self._cleared_tiles)
            ],
            "fusion": {
                "free_log_odds": self._tile_kwargs["free_log_odds"],
                "occupied_log_odds": self._tile_kwargs["occupied_log_odds"],
                "min_log_odds": self._tile_kwargs["min_log_odds"],
                "max_log_odds": self._tile_kwargs["max_log_odds"],
                "elevation_mad_scale": self._tile_kwargs[
                    "elevation_mad_scale"
                ],
                "min_elevation_mad": self._tile_kwargs["min_elevation_mad"],
                "max_rays_per_update": self.max_rays_per_update,
                "lidar_ground_max_slope_deg": (
                    self.lidar_ground_config.max_slope_deg
                ),
                "lidar_ground_neighbor_count": (
                    self.lidar_ground_config.neighbor_count
                ),
                "lidar_ground_max_neighbor_distance_m": (
                    self.lidar_ground_config.max_neighbor_distance_m
                ),
            },
            "tiles": self._snapshot_tile_entries(),
        }

    def _write_snapshot_directory(
        self,
        output_directory: Path,
        *,
        timestamp_text: str,
        frame_id: str,
        parallel_workers: int = 1,
    ) -> Dict[str, Path]:
        output_directory = Path(output_directory)
        tile_directory = output_directory / "tiles"
        tile_directory.mkdir(parents=True, exist_ok=True)

        def write_tile(key: TileKey) -> None:
            tile = self.tiles[key]
            path = tile_directory / self._tile_filename(key)
            np.savez_compressed(
                path,
                **{
                    name: getattr(tile, name)
                    for name, _ in self._SNAPSHOT_TILE_FIELDS
                },
            )

        keys = iter(sorted(self.tiles))
        worker_count = max(1, int(parallel_workers))
        if worker_count == 1 or self.tile_count < 2:
            for key in keys:
                write_tile(key)
        else:
            # Keep only one task per worker in flight. Each task references a
            # complete tile, so this caps compression temporary memory.
            with ThreadPoolExecutor(
                max_workers=worker_count,
                thread_name_prefix="t3-snapshot-tile",
            ) as executor:
                pending = deque()
                for _ in range(worker_count):
                    try:
                        key = next(keys)
                    except StopIteration:
                        break
                    pending.append(executor.submit(write_tile, key))
                while pending:
                    pending.popleft().result()
                    try:
                        key = next(keys)
                    except StopIteration:
                        continue
                    pending.append(executor.submit(write_tile, key))

        metadata = self._snapshot_metadata(
            timestamp_text=timestamp_text,
            frame_id=frame_id,
        )
        index_path = output_directory / "map_index.yaml"
        index_path.write_text(
            yaml.safe_dump(metadata, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
        return {"index": index_path, "tiles": tile_directory}

    @staticmethod
    def _snapshot_relative_path(value: object) -> Path:
        path = Path(str(value))
        if path.is_absolute() or not path.parts or ".." in path.parts:
            raise ValueError(f"Invalid snapshot payload path: {value!r}")
        return path

    def _validate_snapshot_directory(
        self,
        index_path: Path,
        *,
        timestamp_text: str,
        frame_id: str,
    ) -> None:
        """Verify a staged snapshot without loading a second whole map."""
        index_path = Path(index_path)
        try:
            metadata = yaml.safe_load(index_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, yaml.YAMLError) as exception:
            raise ValueError(
                f"Snapshot index is unreadable: {index_path}"
            ) from exception
        if not isinstance(metadata, dict):
            raise ValueError(f"Snapshot index is not a mapping: {index_path}")

        expected_metadata = self._snapshot_metadata(
            timestamp_text=timestamp_text,
            frame_id=frame_id,
        )
        if metadata.keys() != expected_metadata.keys():
            raise ValueError("Snapshot metadata fields are incomplete or unknown")
        for name, expected in expected_metadata.items():
            if metadata[name] != expected:
                raise ValueError(f"Snapshot metadata mismatch for {name}")

        entries = metadata["tiles"]
        referenced_files = set()
        expected_shape = (self.tile_cells, self.tile_cells)
        for entry in entries:
            key = (entry["x"], entry["y"])
            tile = self.tiles[key]
            relative_path = self._snapshot_relative_path(entry["file"])
            payload_path = index_path.parent / relative_path
            if payload_path in referenced_files:
                raise ValueError(
                    f"Duplicate snapshot tile payload: {relative_path}"
                )
            referenced_files.add(payload_path)
            if not payload_path.is_file():
                raise ValueError(
                    f"Snapshot tile payload is missing: {payload_path}"
                )
            try:
                with np.load(payload_path, allow_pickle=False) as saved:
                    if set(saved.files) != {
                        name for name, _ in self._SNAPSHOT_TILE_FIELDS
                    }:
                        raise ValueError(
                            f"Snapshot tile fields are invalid: {payload_path}"
                        )
                    for name, dtype in self._SNAPSHOT_TILE_FIELDS:
                        value = saved[name]
                        wanted_shape = (
                            (len(self.class_names),) + expected_shape
                            if name == "semantic_votes"
                            else expected_shape
                        )
                        if value.shape != wanted_shape:
                            raise ValueError(
                                f"Invalid {name} shape in {payload_path}: "
                                f"{value.shape} != {wanted_shape}"
                            )
                        if value.dtype != np.dtype(dtype):
                            raise ValueError(
                                f"Invalid {name} dtype in {payload_path}: "
                                f"{value.dtype} != {np.dtype(dtype)}"
                            )
                        try:
                            np.testing.assert_array_equal(
                                value,
                                getattr(tile, name),
                            )
                        except AssertionError as exception:
                            raise ValueError(
                                f"Snapshot tile {key} field {name} differs "
                                "from the live map"
                            ) from exception
            except (
                OSError,
                EOFError,
                ValueError,
                zipfile.BadZipFile,
            ) as exception:
                if isinstance(exception, ValueError) and str(exception).startswith(
                    "Snapshot"
                ):
                    raise
                raise ValueError(
                    f"Snapshot tile payload is unreadable: {payload_path}"
                ) from exception

        tile_directory = index_path.parent / "tiles"
        actual_files = set(tile_directory.glob("*.npz"))
        if actual_files != referenced_files:
            raise ValueError(
                "Snapshot tile directory contains missing or stale payloads"
            )

    def save_snapshot(
        self,
        output_directory: Path,
        *,
        timestamp_text: str,
        frame_id: str,
        parallel_workers: int = 1,
    ) -> Dict[str, Path]:
        """Atomically save a validated immutable map version.

        The public root ``map_index.yaml`` remains the load entry point.  A
        failed or interrupted save leaves the previously committed snapshot
        readable and unchanged.
        """
        return commit_atomic_snapshot(
            Path(output_directory),
            version_label=timestamp_text,
            write_snapshot=lambda staging: self._write_snapshot_directory(
                staging,
                timestamp_text=timestamp_text,
                frame_id=frame_id,
                parallel_workers=parallel_workers,
            ),
            validate_snapshot=lambda index: self._validate_snapshot_directory(
                index,
                timestamp_text=timestamp_text,
                frame_id=frame_id,
            ),
        )

    @classmethod
    def load_snapshot(cls, index_path: Path) -> "TiledSemanticMapManager":
        index_path = Path(index_path)
        metadata = yaml.safe_load(index_path.read_text(encoding="utf-8"))
        if metadata.get("format") != "t3_tiled_semantic_map":
            raise ValueError(f"Unsupported map snapshot: {index_path}")
        if int(metadata.get("format_version", -1)) != cls.FORMAT_VERSION:
            raise ValueError(
                f"Unsupported map format version in {index_path}: "
                f"{metadata.get('format_version')}"
            )
        fusion = metadata.get("fusion", {})
        manager = cls(
            resolution=float(metadata["resolution"]),
            tile_cells=int(metadata["tile_cells"]),
            origin_x=float(metadata.get("origin_x", 0.0)),
            origin_y=float(metadata.get("origin_y", 0.0)),
            class_names=metadata["class_names"],
            free_class_ids=metadata.get("free_class_ids", (1,)),
            occupied_class_ids=metadata.get("occupied_class_ids", (3, 4)),
            map_id=str(metadata.get("map_id", "t3_semantic_map")),
            free_log_odds=float(fusion.get("free_log_odds", -0.40)),
            occupied_log_odds=float(fusion.get("occupied_log_odds", 0.85)),
            min_log_odds=float(fusion.get("min_log_odds", -4.0)),
            max_log_odds=float(fusion.get("max_log_odds", 4.0)),
            elevation_mad_scale=float(
                fusion.get("elevation_mad_scale", 3.5)
            ),
            min_elevation_mad=float(fusion.get("min_elevation_mad", 0.03)),
            max_rays_per_update=int(
                fusion.get("max_rays_per_update", 1200)
            ),
            lidar_ground_max_slope_deg=float(
                fusion.get("lidar_ground_max_slope_deg", 35.0)
            ),
            lidar_ground_neighbor_count=int(
                fusion.get("lidar_ground_neighbor_count", 12)
            ),
            lidar_ground_max_neighbor_distance_m=float(
                fusion.get("lidar_ground_max_neighbor_distance_m", 0.75)
            ),
        )
        manager._temporal_elevation_config=metadata.get('elevation_fusion',{})
        snapshot_fields=cls._SNAPSHOT_TILE_FIELDS
        if manager._temporal_elevation_config.get('enabled',False):
            from ..elevation_fusion import EXTRA_FIELDS
            snapshot_fields+=EXTRA_FIELDS
        expected_shape = (manager.tile_cells, manager.tile_cells)
        for entry in metadata.get("tiles", []):
            key = (int(entry["x"]), int(entry["y"]))
            path = index_path.parent / str(entry["file"])
            with np.load(path, allow_pickle=False) as saved:
                tile = manager._new_tile(key)
                for name, dtype in snapshot_fields:
                    value = np.asarray(saved[name], dtype=dtype).copy()
                    if name == "semantic_votes":
                        wanted = (len(manager.class_names),) + expected_shape
                    else:
                        wanted = expected_shape
                    if value.shape != wanted:
                        raise ValueError(
                            f"Invalid {name} shape in {path}: "
                            f"{value.shape} != {wanted}"
                        )
                    setattr(tile, name, value)
            manager.tiles[key] = tile
            manager.tile_revisions[key] = int(entry.get("revision", 0))
        manager.update_id = int(metadata.get("map_revision", 0))
        manager.total_accepted_points = int(
            metadata.get("total_accepted_points", 0)
        )
        cleared_entries = metadata.get("cleared_tiles", [])
        manager._cleared_tiles = {
            (int(entry["x"]), int(entry["y"]))
            for entry in cleared_entries
        }
        manager._cleared_tiles.difference_update(manager.tiles)
        for entry in cleared_entries:
            key = (int(entry["x"]), int(entry["y"]))
            if key in manager._cleared_tiles:
                # Older format-version-1 snapshots omitted this optional
                # value; zero is the compatible starting revision for them.
                manager.tile_revisions[key] = int(
                    entry.get("revision", 0)
                )
        manager._refresh_tile_statistics(manager.tiles)
        manager._dirty_regions.clear()
        return manager
