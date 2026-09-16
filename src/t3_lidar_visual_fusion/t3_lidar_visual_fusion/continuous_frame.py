"""A source-to-odom transform belongs to an epoch, not to a quality gate cycle."""
import numpy as np
from .core import GateResult, inverse, motion, rigid


def pose_covariance(value):
    cov = np.asarray(value, dtype=float).reshape(6, 6)
    if not np.isfinite(cov).all() or not np.allclose(cov, cov.T, atol=1e-7):
        raise ValueError("Nonfinite or asymmetric pose covariance")
    cov = .5 * (cov + cov.T)
    if np.linalg.eigvalsh(cov)[0] < -1e-8 or np.diag(cov).max() <= 0:
        raise ValueError("Invalid pose covariance")
    return cov


def rotate_covariance(cov, rotation, floor):
    """ROS pose errors are displacement and fixed-axis rotation in parent axes."""
    j = np.zeros((6, 6)); j[:3, :3] = rotation; j[3:, 3:] = rotation
    value = j @ cov @ j.T
    # Positive diagonal addition preserves cross-covariances and PSD.
    value += np.diag(np.maximum(0., np.asarray(floor) - np.diag(value)))
    return .5 * (value + value.T)


class ContinuousFrame:
    def __init__(self, recovery_frames=3, max_speed=4., max_angular_speed=2.,
                 max_step=6., max_angle=1., max_gap=6.):
        self.recovery_frames = int(recovery_frames)
        self.max_speed, self.max_angular_speed = max_speed, max_angular_speed
        self.max_step, self.max_angle, self.max_gap = max_step, max_angle, max_gap
        self.epoch = None
        self.alignment = None
        self.previous = None
        self.last_accepted = None
        self.count = 0
        self.enabled = False
        self.reason = "waiting"
        self.output = np.eye(4)

    def close(self, reason):
        self.count = 0; self.enabled = False; self.previous = None
        self.reason = reason
        # The transform and last accepted pose survive a temporary rejection.
        return GateResult(reason=reason)

    def begin_epoch(self, epoch):
        if epoch == self.epoch:
            return False
        self.close("new_epoch_requires_reference")
        self.epoch = epoch; self.alignment = None; self.last_accepted = None
        return True

    def align(self, raw, reference):
        if self.alignment is None and reference is not None:
            self.alignment = rigid(reference) @ inverse(rigid(raw))
            return True
        return False

    def accept(self, stamp, transform, healthy=True, reason="quality"):
        transform = rigid(transform)
        if not healthy:
            return self.close(reason)
        if self.alignment is None:
            return self.close("new_epoch_requires_reference")
        # Check against the last accepted pose even across a temporary outage.
        if self.last_accepted is not None:
            old_stamp, old = self.last_accepted
            dt = stamp - old_stamp
            _, distance, angle = motion(old, transform)
            if dt <= 0 or distance / dt > self.max_speed or angle / dt > self.max_angular_speed:
                return self.close("pose_jump_or_nonmonotonic")
        if self.previous is not None:
            dt = stamp - self.previous[0]
            _, distance, angle = motion(self.previous[1], transform)
            if dt <= 0 or (dt <= self.max_gap and (distance > self.max_step or angle > self.max_angle)):
                return self.close("pose_jump_or_nonmonotonic")
            if dt > self.max_gap:
                self.close("odom_timeout")
        self.previous = (stamp, transform.copy()); self.count += 1
        if not self.enabled and self.count < self.recovery_frames:
            self.reason = "recovering"
            return GateResult(reason=self.reason)
        self.enabled = True; self.reason = "healthy"
        self.output = self.alignment @ transform
        self.last_accepted = (stamp, transform.copy())
        return GateResult(self.output.copy(), self.reason, False)
