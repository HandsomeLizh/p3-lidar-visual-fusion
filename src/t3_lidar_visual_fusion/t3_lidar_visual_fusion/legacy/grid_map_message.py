from __future__ import annotations

from array import array
from collections import OrderedDict
from typing import Mapping

import numpy as np
from grid_map_msgs.msg import GridMap
from std_msgs.msg import Float32MultiArray, MultiArrayDimension

from .coordinate_utils import matrix_to_quaternion_xyzw
from .semantic_grid import GridGeometry


def _typed_array(values, *, dtype, typecode: str) -> array:
    """Copy NumPy values into a ROS-compatible native typed buffer."""

    flat = np.ascontiguousarray(values, dtype=dtype).reshape(-1)
    typed_data = array(typecode)
    if typed_data.itemsize != flat.dtype.itemsize:
        raise RuntimeError(
            f"array('{typecode}') item size {typed_data.itemsize} does not "
            f"match {flat.dtype} item size {flat.dtype.itemsize}"
        )
    typed_data.frombytes(memoryview(flat).cast("B"))
    return typed_data


def float32_array(values: np.ndarray) -> array:
    """Return a ROS-compatible float32 buffer, preserving IEEE NaNs."""

    return _typed_array(values, dtype=np.float32, typecode="f")


def int8_array(values: np.ndarray) -> array:
    """Return a ROS-compatible signed-int8 buffer."""

    return _typed_array(values, dtype=np.int8, typecode="b")


def uint8_array(values: np.ndarray) -> array:
    """Return a ROS-compatible unsigned-int8 buffer."""

    return _typed_array(values, dtype=np.uint8, typecode="B")


def uint32_array(values: np.ndarray) -> array:
    """Return a ROS-compatible native uint32 buffer."""

    return _typed_array(values, dtype=np.uint32, typecode="I")


def _layer_to_multi_array(layer_yx: np.ndarray) -> Float32MultiArray:
    """Convert [Y, X] NumPy data into the GridMap Eigen-style matrix layout."""
    layer_yx = np.asarray(layer_yx, dtype=np.float32)
    if layer_yx.ndim != 2:
        raise ValueError(f"GridMap layer must be 2-D, got {layer_yx.shape}")

    # grid_map matrices conventionally use the first matrix index for X and
    # the second for Y. Transpose [Y, X] -> [X, Y], then serialize in Eigen
    # column-major order.
    matrix_xy = np.asfortranarray(
        np.flip(layer_yx, axis=(0, 1)).T
    )

    message = Float32MultiArray()
    total_size = int(matrix_xy.size)

    column_dimension = MultiArrayDimension()
    column_dimension.label = "column_index"
    column_dimension.size = int(matrix_xy.shape[1])
    column_dimension.stride = total_size

    row_dimension = MultiArrayDimension()
    row_dimension.label = "row_index"
    row_dimension.size = int(matrix_xy.shape[0])
    row_dimension.stride = int(matrix_xy.shape[0])

    message.layout.dim = [column_dimension, row_dimension]
    message.layout.data_offset = 0
    # Assigning array('f') uses the generated ROS message's typed-buffer path.
    # Unlike its generic Python-list validator, that path correctly preserves
    # IEEE NaN values used for unknown GridMap cells.
    message.data = float32_array(matrix_xy.reshape(-1, order="F"))
    return message


def make_grid_map_message(
    *,
    header,
    geometry: GridGeometry,
    layers: Mapping[str, np.ndarray],
    basic_layers: list[str] | None = None,
    transform_output_map: np.ndarray | None = None,
) -> GridMap:
    if not layers:
        raise ValueError("At least one GridMap layer is required")

    message = GridMap()
    message.header = header
    message.info.resolution = float(geometry.resolution)
    message.info.length_x = float(geometry.length_x)
    message.info.length_y = float(geometry.length_y)
    grid_pose_map = np.eye(4, dtype=np.float64)
    grid_pose_map[0, 3] = float(geometry.center_x)
    grid_pose_map[1, 3] = float(geometry.center_y)
    if transform_output_map is None:
        grid_pose_output = grid_pose_map
    else:
        output_map = np.asarray(transform_output_map, dtype=np.float64)
        if output_map.shape != (4, 4) or not np.isfinite(output_map).all():
            raise ValueError("transform_output_map must be a finite 4x4 matrix")
        grid_pose_output = output_map @ grid_pose_map
    quaternion = matrix_to_quaternion_xyzw(grid_pose_output[:3, :3])
    message.info.pose.position.x = float(grid_pose_output[0, 3])
    message.info.pose.position.y = float(grid_pose_output[1, 3])
    message.info.pose.position.z = float(grid_pose_output[2, 3])
    message.info.pose.orientation.x = float(quaternion[0])
    message.info.pose.orientation.y = float(quaternion[1])
    message.info.pose.orientation.z = float(quaternion[2])
    message.info.pose.orientation.w = float(quaternion[3])

    ordered = OrderedDict(layers)
    message.layers = list(ordered.keys())
    message.basic_layers = list(basic_layers or ["elevation"])
    message.data = [
        _layer_to_multi_array(array)
        for array in ordered.values()
    ]
    message.outer_start_index = 0
    message.inner_start_index = 0
    return message
