#!/usr/bin/env python3
"""Regression of lost outage motion, reset bridging and rotated uncertainty."""
from pathlib import Path
import sys
import numpy as np
from scipy.spatial.transform import Rotation
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src/t3_lidar_visual_fusion'))
from t3_lidar_visual_fusion.continuous_frame import ContinuousFrame, pose_covariance, rotate_covariance

def pose(x):
    value = np.eye(4); value[0,3] = x; return value

gate = ContinuousFrame(recovery_frames=2)
gate.begin_epoch('vision_1'); assert gate.align(pose(0), pose(10))
assert gate.accept(0, pose(0)).transform is None
assert np.isclose(gate.accept(1, pose(1)).transform[0,3], 11.)
before = gate.alignment.copy()
gate.close('temporary_overexposure')
assert gate.accept(8, pose(8)).transform is None
assert np.isclose(gate.accept(9, pose(9)).transform[0,3], 19.)
assert np.array_equal(before, gate.alignment)  # Eight metres must not become zero.
gate.begin_epoch('vision_2')
assert gate.accept(10, pose(-700)).transform is None
assert gate.align(pose(-700), pose(20))
gate.accept(10, pose(-700))
assert np.isclose(gate.accept(11, pose(-699)).transform[0,3], 21.)
# Rotation of a non-axis-aligned weak direction must retain off-diagonal terms.
cov = np.diag([100., .0025, .0025, .04, .01, .005])
rotation = Rotation.from_euler('z', .7).as_matrix()
rotated = rotate_covariance(cov, rotation, np.zeros(6))
assert abs(rotated[0,1]) > 10.
assert np.allclose(np.linalg.eigvalsh(rotated), np.linalg.eigvalsh(cov))
pose_covariance(rotated)
try:
    bad = cov.copy(); bad[0,0] = -1.; pose_covariance(bad)
    raise AssertionError('Negative uncertainty accepted')
except ValueError:
    pass
print('PASS: same-epoch outage motion, explicit reset bridge, rotated full covariance')
