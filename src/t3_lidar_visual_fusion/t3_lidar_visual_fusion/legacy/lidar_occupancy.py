from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree


ENDPOINT_UNKNOWN = np.int8(-1)
ENDPOINT_FREE = np.int8(0)
ENDPOINT_OCCUPIED = np.int8(1)

# A horizontal LiDAR scan ring on a cave wall can fit one mathematical plane
# even though it is only a line on the physical surface.  These two fixed
# quality gates require a genuinely two-dimensional, locally planar patch.
_MIN_PLANARITY = 0.18
_MAX_SURFACE_VARIATION = 0.10
_MIN_SECOND_EIGENVALUE = 1.0e-7


@dataclass(frozen=True)
class LidarGroundSegmentationConfig:
    """Local-surface criteria used only by direct LiDAR map observations."""

    max_slope_deg: float = 35.0
    neighbor_count: int = 12
    max_neighbor_distance_m: float = 0.75

    def __post_init__(self) -> None:
        slope = float(self.max_slope_deg)
        neighbor_value = float(self.neighbor_count)
        distance = float(self.max_neighbor_distance_m)
        if not np.isfinite(slope) or not 0.0 <= slope < 90.0:
            raise ValueError(
                "lidar_ground_max_slope_deg must be finite and in [0, 90)"
            )
        if (
            not np.isfinite(neighbor_value)
            or not neighbor_value.is_integer()
            or neighbor_value < 3.0
        ):
            raise ValueError(
                "lidar_ground_neighbor_count must be an integer of at least 3"
            )
        neighbors = int(neighbor_value)
        if not np.isfinite(distance) or distance <= 0.0:
            raise ValueError(
                "lidar_ground_max_neighbor_distance_m must be finite and "
                "positive"
            )
        object.__setattr__(self, "max_slope_deg", slope)
        object.__setattr__(self, "neighbor_count", neighbors)
        object.__setattr__(self, "max_neighbor_distance_m", distance)


DEFAULT_LIDAR_GROUND_CONFIG = LidarGroundSegmentationConfig()


def validate_endpoint_occupancy_states(
    values: np.ndarray,
    *,
    point_count: int,
) -> np.ndarray:
    """Return validated ``[-1, 0, 1]`` endpoint states as int8."""

    states = np.asarray(values).reshape(-1)
    if len(states) != int(point_count):
        raise ValueError(
            "points and geometric endpoint states differ: "
            f"{int(point_count)} != {len(states)}"
        )
    if not np.all(
        np.isin(states, [ENDPOINT_UNKNOWN, ENDPOINT_FREE, ENDPOINT_OCCUPIED])
    ):
        raise ValueError(
            "geometric endpoint states must use -1 unknown, 0 free, or 1 "
            "occupied"
        )
    return states.astype(np.int8, copy=False)


def classify_lidar_endpoint_occupancy(
    points_map: np.ndarray,
    sensor_origin_map: np.ndarray,
    *,
    config: LidarGroundSegmentationConfig = DEFAULT_LIDAR_GROUND_CONFIG,
) -> np.ndarray:
    """Classify direct-LiDAR endpoints without changing semantic labels.

    A k-nearest-neighbour covariance estimates each return's local surface
    normal.  The sign is chosen to face the sensor: a visible floor therefore
    has an upward normal, while a ceiling has a downward normal and walls have
    near-horizontal normals.  Reliable upward patches within ``max_slope_deg``
    are free endpoints; other reliable surfaces are occupied.  Sparse, linear,
    or rough neighbourhoods remain unknown rather than inventing evidence.

    The returned int8 values are ``-1`` unknown, ``0`` free, and ``1``
    occupied.  Semantic free/occupied classes still take precedence in the map
    fusion layer.
    """

    points = np.asarray(points_map, dtype=np.float64).reshape(-1, 3)
    sensor = np.asarray(sensor_origin_map, dtype=np.float64).reshape(3)
    states = np.full(len(points), ENDPOINT_UNKNOWN, dtype=np.int8)
    if not np.isfinite(sensor).all() or len(points) < config.neighbor_count:
        return states

    finite_indices = np.flatnonzero(np.isfinite(points).all(axis=1))
    if len(finite_indices) < config.neighbor_count:
        return states
    finite_points = points[finite_indices]

    distances, neighbor_indices = cKDTree(finite_points).query(
        finite_points,
        k=config.neighbor_count,
    )
    neighborhoods = finite_points[neighbor_indices]
    centered = neighborhoods - np.mean(neighborhoods, axis=1, keepdims=True)
    covariance = np.einsum(
        "nki,nkj->nij",
        centered,
        centered,
    ) / float(config.neighbor_count)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    normals = eigenvectors[:, :, 0]

    toward_sensor = sensor.reshape(1, 3) - finite_points
    flip = np.einsum("ni,ni->n", normals, toward_sensor) < 0.0
    normals[flip] *= -1.0

    largest = np.maximum(eigenvalues[:, 2], np.finfo(np.float64).eps)
    total = np.maximum(
        np.sum(eigenvalues, axis=1),
        np.finfo(np.float64).eps,
    )
    planarity = (eigenvalues[:, 1] - eigenvalues[:, 0]) / largest
    surface_variation = eigenvalues[:, 0] / total
    reliable = (
        (distances[:, -1] <= config.max_neighbor_distance_m)
        & (eigenvalues[:, 1] >= _MIN_SECOND_EIGENVALUE)
        & (planarity >= _MIN_PLANARITY)
        & (surface_variation <= _MAX_SURFACE_VARIATION)
    )

    upward_threshold = np.cos(np.deg2rad(config.max_slope_deg))
    ground = reliable & (normals[:, 2] >= upward_threshold)
    finite_states = np.full(
        len(finite_points),
        ENDPOINT_UNKNOWN,
        dtype=np.int8,
    )
    finite_states[ground] = ENDPOINT_FREE
    finite_states[reliable & ~ground] = ENDPOINT_OCCUPIED
    states[finite_indices] = finite_states
    return states
