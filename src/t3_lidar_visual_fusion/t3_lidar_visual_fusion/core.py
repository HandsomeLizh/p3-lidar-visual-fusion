"""ROS-independent motion validation. All transforms map child coordinates to parent."""
from collections import deque
from dataclasses import dataclass
import numpy as np
from scipy.spatial.transform import Rotation, Slerp


def rigid(value):
    t = np.asarray(value, dtype=float).reshape(4, 4)
    if (not np.isfinite(t).all() or not np.allclose(t[3], [0, 0, 0, 1])
        or not np.allclose(t[:3, :3].T @ t[:3, :3], np.eye(3), atol=1e-5)
        or not np.isclose(np.linalg.det(t[:3, :3]), 1, atol=1e-5)):
        raise ValueError("Expected finite rigid transform with det(R)=+1")
    return t


def inverse(t):
    out = np.eye(4)
    out[:3, :3] = t[:3, :3].T
    out[:3, 3] = -out[:3, :3] @ t[:3, 3]
    return out


def motion(previous, current):
    delta = inverse(previous) @ current
    return delta, float(np.linalg.norm(delta[:3, 3])), float(Rotation.from_matrix(delta[:3, :3]).magnitude())


class PoseBuffer:
    def __init__(self, capacity=4000):
        self.samples = deque(maxlen=capacity)

    def append(self, stamp, transform):
        if self.samples and stamp <= self.samples[-1][0]:
            return False
        self.samples.append((float(stamp), rigid(transform).copy()))
        return True

    def replace_latest(self, stamp, transform):
        """Accept a corrected estimate at the current timestamp only."""
        if not self.samples or abs(stamp-self.samples[-1][0])>1e-8:
            return False
        self.samples[-1]=(float(stamp),rigid(transform).copy())
        return True

    def at(self, stamp, tolerance=0.06, max_gap=0.5):
        if not self.samples:
            return None
        times = np.array([x[0] for x in self.samples])
        i = int(np.searchsorted(times, stamp))
        if i < len(times) and abs(times[i]-stamp) < 1e-8:
            return self.samples[i][1].copy()
        if i == 0 or i == len(times):
            j = 0 if i == 0 else -1
            return self.samples[j][1].copy() if abs(times[j] - stamp) <= tolerance else None
        ta, a = self.samples[i-1]
        tb, b = self.samples[i]
        if tb-ta > max_gap:
            return None
        u = (stamp-ta)/(tb-ta)
        out = np.eye(4)
        out[:3, 3] = (1-u)*a[:3, 3] + u*b[:3, 3]
        out[:3, :3] = Slerp([ta, tb], Rotation.from_matrix([a[:3, :3], b[:3, :3]]))([stamp]).as_matrix()[0]
        return out


@dataclass
class GateResult:
    transform: object = None
    reason: str = "waiting"
    reseed: bool = False


class VisionGate:
    """Publish only healthy local increments, with a fresh anchor after every outage.

    Never bridge an unobserved interval or a VINS world-frame reset. The first
    re-enabled sample is an unchanged anchor with inflated covariance, so the
    EKF differential history cannot turn a reset into a velocity spike.
    """
    def __init__(self, recovery_frames=5, max_speed=3.0, max_angular_speed=2.5,
                 max_step=1.0, max_angle=0.6, max_gap=0.6):
        self.recovery_frames = recovery_frames
        self.max_speed, self.max_angular_speed = max_speed, max_angular_speed
        self.max_step, self.max_angle, self.max_gap = max_step, max_angle, max_gap
        self.previous = None
        self.count = 0
        self.enabled = False
        self.output = np.eye(4)
        self.reason = "waiting"

    def close(self, reason):
        self.count, self.enabled, self.previous = 0, False, None
        self.reason = reason
        return GateResult(reason=reason)

    def accept(self, stamp, transform, healthy=True, reason="quality"):
        try:
            transform = rigid(transform)
        except (ValueError, TypeError):
            return self.close("invalid_pose")
        if not healthy:
            return self.close(reason)
        if self.previous is None:
            self.previous = (stamp, transform.copy())
            self.count = 1
            self.reason = "recovering"
            return GateResult(reason=self.reason)
        old_stamp, old = self.previous
        dt = stamp - old_stamp
        if dt <= 0:
            return self.close("non_monotonic_stamp")
        delta, distance, angle = motion(old, transform)
        if dt > self.max_gap:
            self.close("odom_timeout")
            self.previous = (stamp, transform.copy())
            self.count = 1
            return GateResult(reason="odom_timeout")
        if (distance > self.max_step or angle > self.max_angle
            or distance/dt > self.max_speed or angle/dt > self.max_angular_speed):
            return self.close("pose_jump")
        self.previous = (stamp, transform.copy())
        self.count += 1
        if not self.enabled:
            if self.count < self.recovery_frames:
                self.reason = "recovering"
                return GateResult(reason=self.reason)
            self.enabled = True
            self.reason = "healthy"
            return GateResult(self.output.copy(), self.reason, True)
        self.output = self.output @ delta
        self.reason = "healthy"
        return GateResult(self.output.copy(), self.reason)
