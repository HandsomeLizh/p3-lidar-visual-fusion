from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation


def opencv_world_to_map_rotation(camera_pitch_deg: float) -> np.ndarray:
    """Return the fixed rotation from initial OpenCV world to ROS map.

    OpenCV optical axes:
        X right, Y down, Z forward.

    ROS map axes:
        X forward, Y left, Z up.

    The camera optical axis is pitched downward by camera_pitch_deg relative
    to the horizontal vehicle frame.

    This matrix reproduces the established project conversion, e.g.
    [0.030, -0.100, 0.184] -> approximately [0.209, -0.030, -0.005]
    for a 30 degree downward camera pitch.
    """
    pitch = np.deg2rad(float(camera_pitch_deg))
    sine = np.sin(pitch)
    cosine = np.cos(pitch)
    return np.array(
        [
            [0.0, -sine, cosine],
            [-1.0, 0.0, 0.0],
            [0.0, -cosine, -sine],
        ],
        dtype=np.float64,
    )


def pose_opencv_to_map(
    pose_world_camera: np.ndarray,
    rotation_map_world: np.ndarray,
    translation_offset: np.ndarray | None = None,
) -> np.ndarray:
    pose = np.asarray(pose_world_camera, dtype=np.float64)
    if pose.shape != (4, 4):
        raise ValueError(f"Pose must be 4x4, got {pose.shape}")

    offset = (
        np.zeros(3, dtype=np.float64)
        if translation_offset is None
        else np.asarray(translation_offset, dtype=np.float64).reshape(3)
    )

    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = rotation_map_world @ pose[:3, :3]
    result[:3, 3] = rotation_map_world @ pose[:3, 3] + offset
    return result


def points_opencv_to_map(
    points_world: np.ndarray,
    rotation_map_world: np.ndarray,
    translation_offset: np.ndarray | None = None,
) -> np.ndarray:
    points = np.asarray(points_world, dtype=np.float64).reshape(-1, 3)
    offset = (
        np.zeros(3, dtype=np.float64)
        if translation_offset is None
        else np.asarray(translation_offset, dtype=np.float64).reshape(3)
    )
    return (rotation_map_world @ points.T).T + offset


def matrix_to_quaternion_xyzw(rotation_matrix: np.ndarray) -> np.ndarray:
    # Older SciPy releases do not accept a read-only NumPy view.  Callers may
    # legitimately pass immutable localization estimates, so own the buffer.
    return Rotation.from_matrix(
        np.asarray(rotation_matrix, dtype=np.float64).copy()
    ).as_quat()
