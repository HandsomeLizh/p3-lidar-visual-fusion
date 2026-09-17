from __future__ import annotations

from copy import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import cv2
import numpy as np
import yaml

from .lidar_occupancy import (
    ENDPOINT_FREE,
    ENDPOINT_OCCUPIED,
    validate_endpoint_occupancy_states,
)


CLASS_NAMES = [
    "unknown",
    "traversable_ground",
    "shadow",
    "bright_rock",
    "rough_obstacle",
]

# BGR colours used for human-facing image output.
PALETTE_BGR = np.array(
    [
        [0, 0, 0],
        [0, 180, 0],
        [120, 80, 40],
        [0, 220, 255],
        [0, 0, 220],
    ],
    dtype=np.uint8,
)


@dataclass(frozen=True)
class GridGeometry:
    resolution: float
    width: int
    height: int
    origin_x: float
    origin_y: float

    @property
    def length_x(self) -> float:
        return self.width * self.resolution

    @property
    def length_y(self) -> float:
        return self.height * self.resolution

    @property
    def center_x(self) -> float:
        return self.origin_x + 0.5 * self.length_x

    @property
    def center_y(self) -> float:
        return self.origin_y + 0.5 * self.length_y


def _bresenham(
    start_col: int,
    start_row: int,
    end_col: int,
    end_row: int,
):
    """Yield integer cells from start to end, inclusive."""
    x0, y0 = int(start_col), int(start_row)
    x1, y1 = int(end_col), int(end_row)

    dx = abs(x1 - x0)
    sx = 1 if x0 < x1 else -1
    dy = -abs(y1 - y0)
    sy = 1 if y0 < y1 else -1
    error = dx + dy

    while True:
        yield y0, x0
        if x0 == x1 and y0 == y1:
            break
        doubled = 2 * error
        if doubled >= dy:
            error += dy
            x0 += sx
        if doubled <= dx:
            error += dx
            y0 += sy


class LayeredSemanticGridMap:
    """Incremental occupancy, semantic and elevation grid.

    Array convention:
        [row, column] == [Y index, X index]
        origin is the lower-left corner in the ROS map frame.
    """

    def __init__(
        self,
        *,
        resolution: float = 0.10,
        length_x: float = 60.0,
        length_y: float = 60.0,
        origin_x: Optional[float] = None,
        origin_y: Optional[float] = None,
        class_names: Optional[list[str]] = None,
        free_log_odds: float = -0.40,
        occupied_log_odds: float = 0.85,
        min_log_odds: float = -4.0,
        max_log_odds: float = 4.0,
        elevation_mad_scale: float = 3.5,
        min_elevation_mad: float = 0.03,
        max_rays_per_update: int = 1200,
    ) -> None:
        resolution = float(resolution)
        if resolution <= 0.0:
            raise ValueError("resolution must be positive")

        width = max(1, int(np.ceil(float(length_x) / resolution)))
        height = max(1, int(np.ceil(float(length_y) / resolution)))

        if origin_x is None:
            origin_x = -0.5 * width * resolution
        if origin_y is None:
            origin_y = -0.5 * height * resolution

        self.geometry = GridGeometry(
            resolution=resolution,
            width=width,
            height=height,
            origin_x=float(origin_x),
            origin_y=float(origin_y),
        )

        self.class_names = list(class_names or CLASS_NAMES)
        self.class_count = len(self.class_names)

        shape = (height, width)
        self.occupancy_log_odds = np.zeros(shape, dtype=np.float32)
        self.occupancy_observed = np.zeros(shape, dtype=bool)

        self.semantic_votes = np.zeros(
            (self.class_count, height, width),
            dtype=np.uint32,
        )

        self.elevation_count = np.zeros(shape, dtype=np.uint32)
        self.elevation_mean = np.zeros(shape, dtype=np.float64)
        self.elevation_M2 = np.zeros(shape, dtype=np.float64)
        self.elevation_min = np.full(shape, np.inf, dtype=np.float64)
        self.elevation_max = np.full(shape, -np.inf, dtype=np.float64)

        self.free_log_odds = float(free_log_odds)
        self.occupied_log_odds = float(occupied_log_odds)
        self.min_log_odds = float(min_log_odds)
        self.max_log_odds = float(max_log_odds)

        self.elevation_mad_scale = float(elevation_mad_scale)
        self.min_elevation_mad = float(min_elevation_mad)
        self.max_rays_per_update = int(max_rays_per_update)

        self.update_id = 0
        self.total_accepted_points = 0

    def clone_elevation_for_update(self) -> "LayeredSemanticGridMap":
        """Copy only arrays mutated by an elevation-only update.

        Occupancy and semantic layers remain shared and read-only in this
        clone.  A caller that later needs to modify those layers must promote
        the tile to a full writable copy first.
        """

        candidate = copy(self)
        for name in (
            "elevation_count",
            "elevation_mean",
            "elevation_M2",
            "elevation_min",
            "elevation_max",
        ):
            setattr(candidate, name, getattr(self, name).copy())
        return candidate

    def xy_to_indices(
        self,
        x: np.ndarray,
        y: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        columns = np.floor(
            (x - self.geometry.origin_x) / self.geometry.resolution
        ).astype(np.int64)
        rows = np.floor(
            (y - self.geometry.origin_y) / self.geometry.resolution
        ).astype(np.int64)
        valid = (
            (columns >= 0)
            & (columns < self.geometry.width)
            & (rows >= 0)
            & (rows < self.geometry.height)
        )
        return rows, columns, valid

    def xy_to_single_index(self, x: float, y: float) -> Optional[tuple[int, int]]:
        rows, columns, valid = self.xy_to_indices(
            np.array([x]), np.array([y])
        )
        if not bool(valid[0]):
            return None
        return int(rows[0]), int(columns[0])

    def _update_elevation_cell(
        self,
        row: int,
        column: int,
        z_values: np.ndarray,
    ) -> int:
        values = np.asarray(z_values, dtype=np.float64)
        values = values[np.isfinite(values)]
        if values.size == 0:
            return 0

        # One sample has zero deviation.  With two samples both deviations
        # equal the MAD, so both are accepted whenever the configured robust
        # multiplier is at least one.  Avoid two median reductions and a
        # temporary deviation array in those common sparse-cell cases without
        # changing low-scale custom configurations that can intentionally
        # reject a two-point pair.
        if values.size == 1 or (
            values.size == 2
            and self.elevation_mad_scale * 1.4826 >= 1.0
        ):
            accepted = values
        else:
            median = float(np.median(values))
            absolute_deviation = np.abs(values - median)
            mad = float(np.median(absolute_deviation))
            threshold = max(
                self.min_elevation_mad,
                self.elevation_mad_scale * 1.4826 * mad,
            )
            accepted = values[absolute_deviation <= threshold]
            if accepted.size == 0:
                accepted = np.array([median], dtype=np.float64)

        batch_count = int(accepted.size)
        if batch_count == 1:
            batch_mean = batch_min = batch_max = float(accepted[0])
            batch_M2 = 0.
        else:
            batch_mean = float(np.mean(accepted))
            batch_M2 = float(np.sum((accepted - batch_mean) ** 2))
            batch_min = float(np.min(accepted))
            batch_max = float(np.max(accepted))

        old_count = int(self.elevation_count[row, column])
        old_mean = float(self.elevation_mean[row, column])
        old_M2 = float(self.elevation_M2[row, column])

        if old_count == 0:
            new_count = batch_count
            new_mean = batch_mean
            new_M2 = batch_M2
        else:
            new_count = old_count + batch_count
            delta = batch_mean - old_mean
            new_mean = old_mean + delta * batch_count / new_count
            new_M2 = (
                old_M2
                + batch_M2
                + delta * delta * old_count * batch_count / new_count
            )

        self.elevation_count[row, column] = new_count
        self.elevation_mean[row, column] = new_mean
        self.elevation_M2[row, column] = new_M2
        self.elevation_min[row, column] = min(
            self.elevation_min[row, column], batch_min,
        )
        self.elevation_max[row, column] = max(
            self.elevation_max[row, column], batch_max,
        )
        self._observe_elevation_batch(row, column, batch_min, batch_max)
        return batch_count

    def _observe_elevation_batch(self, row, column, minimum, maximum):
        """Optional derived statistics; lifetime height fusion stays unchanged."""

    def update(
        self,
        *,
        points_map: np.ndarray,
        semantic_labels: np.ndarray,
        camera_position_map: np.ndarray,
        update_occupancy: bool = True,
        geometric_endpoint_states: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
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

        finite = np.isfinite(points).all(axis=1)
        label_valid = (labels >= 0) & (labels < self.class_count)
        valid = finite & label_valid
        points = points[valid]
        labels = labels[valid]
        if endpoint_states is not None:
            endpoint_states = endpoint_states[valid]

        if len(points) == 0:
            return {
                "accepted_points": 0,
                "known_cells": int(np.count_nonzero(self.elevation_count)),
            }

        rows, columns, in_map = self.xy_to_indices(
            points[:, 0], points[:, 1]
        )
        points = points[in_map]
        labels = labels[in_map]
        if endpoint_states is not None:
            endpoint_states = endpoint_states[in_map]
        rows = rows[in_map]
        columns = columns[in_map]

        if len(points) == 0:
            return {
                "accepted_points": 0,
                "known_cells": int(np.count_nonzero(self.elevation_count)),
            }

        linear = rows * self.geometry.width + columns
        unique_linear = np.unique(linear)

        accepted_points = 0
        for cell in unique_linear:
            mask = linear == cell
            row = int(cell // self.geometry.width)
            column = int(cell % self.geometry.width)

            cell_labels = labels[mask]
            counts = np.bincount(
                cell_labels,
                minlength=self.class_count,
            ).astype(np.uint32)
            self.semantic_votes[:, row, column] += counts

            accepted_points += self._update_elevation_cell(
                row,
                column,
                points[mask, 2],
            )

        if update_occupancy:
            self._update_occupancy(
                rows=rows,
                columns=columns,
                labels=labels,
                camera_position_map=np.asarray(
                    camera_position_map,
                    dtype=np.float64,
                ).reshape(3),
                geometric_endpoint_states=endpoint_states,
            )

        self.update_id += 1
        self.total_accepted_points += accepted_points

        elevation = self.elevation_layer()
        finite_elevation = elevation[np.isfinite(elevation)]

        return {
            "accepted_points": int(accepted_points),
            "visible_points": int(len(points)),
            "known_cells": int(np.count_nonzero(self.elevation_count)),
            "known_area_m2": float(
                np.count_nonzero(self.elevation_count)
                * self.geometry.resolution
                * self.geometry.resolution
            ),
            "elevation_min": (
                float(np.min(finite_elevation))
                if finite_elevation.size
                else float("nan")
            ),
            "elevation_max": (
                float(np.max(finite_elevation))
                if finite_elevation.size
                else float("nan")
            ),
        }

    def update_elevation_only(
        self,
        *,
        points_map: np.ndarray,
    ) -> Dict[str, float]:
        """Fuse height samples without changing semantics or occupancy."""

        points = np.asarray(points_map, dtype=np.float64).reshape(-1, 3)
        points = points[np.isfinite(points).all(axis=1)]
        if len(points) == 0:
            return {
                "accepted_points": 0,
                "visible_points": 0,
                "known_cells": int(np.count_nonzero(self.elevation_count)),
            }

        rows, columns, in_map = self.xy_to_indices(
            points[:, 0], points[:, 1]
        )
        points = points[in_map]
        rows = rows[in_map]
        columns = columns[in_map]
        if len(points) == 0:
            return {
                "accepted_points": 0,
                "visible_points": 0,
                "known_cells": int(np.count_nonzero(self.elevation_count)),
            }

        linear = rows * self.geometry.width + columns
        accepted_points = 0
        # Group once instead of rescanning all points for every occupied cell.
        # The stable sort retains the original sample order inside each cell,
        # while cells keep the same ascending order produced by np.unique.
        order = np.argsort(linear, kind="stable")
        linear_sorted = linear[order]
        z_sorted = points[order, 2]
        boundaries = np.flatnonzero(
            linear_sorted[1:] != linear_sorted[:-1]
        ) + 1
        starts = np.concatenate(
            (np.array([0], dtype=np.int64), boundaries)
        )
        ends = np.concatenate(
            (boundaries, np.array([len(order)], dtype=np.int64))
        )
        for start, end in zip(starts, ends):
            cell = int(linear_sorted[start])
            row = int(cell // self.geometry.width)
            column = int(cell % self.geometry.width)
            accepted_points += self._update_elevation_cell(
                row, column, z_sorted[start:end]
            )

        self.update_id += 1
        self.total_accepted_points += accepted_points
        elevation = self.elevation_layer()
        finite_elevation = elevation[np.isfinite(elevation)]
        known_cells = int(np.count_nonzero(self.elevation_count))
        return {
            "accepted_points": int(accepted_points),
            "visible_points": int(len(points)),
            "known_cells": known_cells,
            "known_area_m2": float(
                known_cells
                * self.geometry.resolution
                * self.geometry.resolution
            ),
            "elevation_min": (
                float(np.min(finite_elevation))
                if finite_elevation.size
                else float("nan")
            ),
            "elevation_max": (
                float(np.max(finite_elevation))
                if finite_elevation.size
                else float("nan")
            ),
        }

    def _update_occupancy(
        self,
        *,
        rows: np.ndarray,
        columns: np.ndarray,
        labels: np.ndarray,
        camera_position_map: np.ndarray,
        geometric_endpoint_states: Optional[np.ndarray],
    ) -> None:
        camera_cell = self.xy_to_single_index(
            float(camera_position_map[0]),
            float(camera_position_map[1]),
        )
        if camera_cell is None:
            return

        camera_row, camera_column = camera_cell
        point_count = len(rows)

        if point_count > self.max_rays_per_update > 0:
            sample_indices = np.linspace(
                0,
                point_count - 1,
                self.max_rays_per_update,
                dtype=np.int64,
            )
        else:
            sample_indices = np.arange(point_count, dtype=np.int64)

        for index in sample_indices:
            end_row = int(rows[index])
            end_column = int(columns[index])
            semantic_id = int(labels[index])
            cells = list(
                _bresenham(
                    camera_column,
                    camera_row,
                    end_column,
                    end_row,
                )
            )
            if not cells:
                continue

            # Intermediate cells receive free-space evidence.
            for row, column in cells[:-1]:
                if (
                    0 <= row < self.geometry.height
                    and 0 <= column < self.geometry.width
                ):
                    self.occupancy_log_odds[row, column] += self.free_log_odds
                    self.occupancy_observed[row, column] = True

            row, column = cells[-1]
            if not (
                0 <= row < self.geometry.height
                and 0 <= column < self.geometry.width
            ):
                continue

            if semantic_id in {3, 4}:
                increment = self.occupied_log_odds
            elif semantic_id == 1:
                increment = self.free_log_odds
            elif (
                geometric_endpoint_states is not None
                and geometric_endpoint_states[index] == ENDPOINT_OCCUPIED
            ):
                increment = self.occupied_log_odds
            elif (
                geometric_endpoint_states is not None
                and geometric_endpoint_states[index] == ENDPOINT_FREE
            ):
                increment = self.free_log_odds
            else:
                # Shadow/unknown carry elevation and semantic evidence but do
                # not force an occupancy interpretation.
                continue

            self.occupancy_log_odds[row, column] += increment
            self.occupancy_observed[row, column] = True

        np.clip(
            self.occupancy_log_odds,
            self.min_log_odds,
            self.max_log_odds,
            out=self.occupancy_log_odds,
        )

    def elevation_layer(self) -> np.ndarray:
        result = np.full(
            self.elevation_count.shape,
            np.nan,
            dtype=np.float32,
        )
        valid = self.elevation_count > 0
        result[valid] = self.elevation_mean[valid].astype(np.float32)
        return result

    def elevation_variance_layer(self) -> np.ndarray:
        result = np.full(
            self.elevation_count.shape,
            np.nan,
            dtype=np.float32,
        )
        valid_one = self.elevation_count == 1
        valid_many = self.elevation_count > 1
        result[valid_one] = 0.0
        result[valid_many] = (
            self.elevation_M2[valid_many]
            / (self.elevation_count[valid_many] - 1)
        ).astype(np.float32)
        return result

    def height_range_layer(self) -> np.ndarray:
        result = np.full(
            self.elevation_count.shape,
            np.nan,
            dtype=np.float32,
        )
        valid = self.elevation_count > 0
        result[valid] = (
            self.elevation_max[valid] - self.elevation_min[valid]
        ).astype(np.float32)
        return result

    def roughness_layer(self) -> np.ndarray:
        variance = self.elevation_variance_layer()
        result = np.sqrt(np.maximum(variance, 0.0)).astype(np.float32)
        return result

    def semantic_id_layer(self) -> np.ndarray:
        total = np.sum(self.semantic_votes, axis=0)
        semantic = np.argmax(self.semantic_votes, axis=0).astype(np.uint8)
        semantic[total == 0] = 0
        return semantic

    def semantic_confidence_layer(self) -> np.ndarray:
        total = np.sum(self.semantic_votes, axis=0).astype(np.float32)
        maximum = np.max(self.semantic_votes, axis=0).astype(np.float32)
        result = np.zeros(total.shape, dtype=np.float32)
        valid = total > 0.0
        result[valid] = maximum[valid] / total[valid]
        return result

    def occupancy_probability_layer(self) -> np.ndarray:
        probability = 1.0 / (
            1.0 + np.exp(-self.occupancy_log_odds.astype(np.float64))
        )
        result = np.full(probability.shape, np.nan, dtype=np.float32)
        result[self.occupancy_observed] = probability[
            self.occupancy_observed
        ].astype(np.float32)
        return result

    def occupancy_int8(self) -> np.ndarray:
        probability = self.occupancy_probability_layer()
        result = np.full(probability.shape, -1, dtype=np.int8)
        valid = np.isfinite(probability)
        result[valid] = np.clip(
            np.rint(probability[valid] * 100.0),
            0,
            100,
        ).astype(np.int8)
        return result

    def observation_count_layer(self) -> np.ndarray:
        return self.elevation_count.astype(np.float32)

    def semantic_color_bgr(self) -> np.ndarray:
        semantic = self.semantic_id_layer()
        clipped = np.clip(
            semantic.astype(np.int32),
            0,
            len(PALETTE_BGR) - 1,
        )
        return PALETTE_BGR[clipped]

    def semantic_color_packed_float(self) -> np.ndarray:
        """Pack RGB as float32 in the convention used by grid_map color layers."""
        bgr = self.semantic_color_bgr()
        rgb_uint = (
            (bgr[:, :, 2].astype(np.uint32) << 16)
            | (bgr[:, :, 1].astype(np.uint32) << 8)
            | bgr[:, :, 0].astype(np.uint32)
        )
        return rgb_uint.view(np.float32)

    def all_layers(self) -> Dict[str, np.ndarray]:
        return {
            "elevation": self.elevation_layer(),
            "elevation_variance": self.elevation_variance_layer(),
            "occupancy": self.occupancy_probability_layer(),
            "semantic_id": self.semantic_id_layer().astype(np.float32),
            "semantic_confidence": self.semantic_confidence_layer(),
            "observation_count": self.observation_count_layer(),
            "height_range": self.height_range_layer(),
            "roughness": self.roughness_layer(),
            "color": self.semantic_color_packed_float(),
        }

    def elevation_color_bgr(self) -> np.ndarray:
        elevation = self.elevation_layer()
        valid = np.isfinite(elevation)
        normalized = np.zeros(elevation.shape, dtype=np.uint8)

        if np.any(valid):
            low, high = np.nanpercentile(elevation, [2.0, 98.0])
            if not np.isfinite(low) or not np.isfinite(high) or high <= low:
                low = float(np.nanmin(elevation))
                high = float(np.nanmax(elevation))
            if high > low:
                scaled = (elevation - low) / (high - low)
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

    def save_snapshot(
        self,
        output_directory: Path,
        *,
        timestamp_text: str,
        frame_id: str,
    ) -> Dict[str, Path]:
        output_directory = Path(output_directory)
        output_directory.mkdir(parents=True, exist_ok=True)

        layers = self.all_layers()
        occupancy = self.occupancy_int8()
        semantic = self.semantic_id_layer()

        npz_path = output_directory / "grid_map_latest.npz"
        np.savez_compressed(
            npz_path,
            occupancy_int8=occupancy,
            semantic_uint8=semantic,
            **layers,
            resolution=np.float32(self.geometry.resolution),
            origin_x=np.float64(self.geometry.origin_x),
            origin_y=np.float64(self.geometry.origin_y),
            frame_id=np.array(frame_id),
            timestamp=np.array(timestamp_text),
            class_names=np.array(self.class_names),
        )

        occupancy_preview = np.full(
            occupancy.shape,
            205,
            dtype=np.uint8,
        )
        occupancy_preview[occupancy == 0] = 254
        occupied_valid = occupancy > 0
        occupancy_preview[occupied_valid] = np.clip(
            254
            - occupancy[occupied_valid].astype(np.int16) * 2,
            0,
            254,
        ).astype(np.uint8)

        occupancy_path = output_directory / "occupancy_grid.png"
        semantic_path = output_directory / "semantic_grid_color.png"
        elevation_path = output_directory / "elevation_grid_color.png"

        # Images are flipped vertically for conventional image viewing because
        # the grid array origin is lower-left.
        cv2.imwrite(str(occupancy_path), np.flipud(occupancy_preview))
        cv2.imwrite(
            str(semantic_path),
            np.flipud(self.semantic_color_bgr()),
        )
        cv2.imwrite(
            str(elevation_path),
            np.flipud(self.elevation_color_bgr()),
        )

        metadata = {
            "timestamp": timestamp_text,
            "frame_id": frame_id,
            "resolution": float(self.geometry.resolution),
            "width": int(self.geometry.width),
            "height": int(self.geometry.height),
            "origin": [
                float(self.geometry.origin_x),
                float(self.geometry.origin_y),
                0.0,
            ],
            "length_x": float(self.geometry.length_x),
            "length_y": float(self.geometry.length_y),
            "update_id": int(self.update_id),
            "total_accepted_points": int(self.total_accepted_points),
            "layers": list(layers.keys()),
            "class_names": list(self.class_names),
        }
        yaml_path = output_directory / "grid_map_latest.yaml"
        yaml_path.write_text(
            yaml.safe_dump(
                metadata,
                sort_keys=False,
                allow_unicode=True,
            ),
            encoding="utf-8",
        )

        return {
            "npz": npz_path,
            "yaml": yaml_path,
            "occupancy": occupancy_path,
            "semantic": semantic_path,
            "elevation": elevation_path,
        }
