"""Bounded stereo pose-to-body-motion adapter; it never joins visual epochs."""
import cv2
import numpy as np
from .core import rigid
from .continuous_frame import pose_covariance


def _skew(v):
    x, y, z = v
    return np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])


def body_velocity(previous, current, dt):
    """SE(3) logarithm / dt: constant body velocity over this observed interval."""
    rotation = previous[:3, :3].T @ current[:3, :3]
    translation = previous[:3, :3].T @ (current[:3, 3]-previous[:3, 3])
    sine_vector = .5*np.array([rotation[2,1]-rotation[1,2], rotation[0,2]-rotation[2,0], rotation[1,0]-rotation[0,1]])
    sine = float(np.linalg.norm(sine_vector))
    cosine = float(np.clip((np.trace(rotation)-1.)*.5, -1., 1.))
    theta = float(np.arctan2(sine, cosine))
    if theta < 1e-4:
        angle = sine_vector*(1.+theta*theta/6.)
    elif sine > 1e-5:
        angle = sine_vector*(theta/sine)
    else:
        angle = cv2.Rodrigues(rotation)[0].reshape(3)
    theta = float(np.linalg.norm(angle)); omega = _skew(angle)
    coefficient = (1./12.+theta*theta/720.) if theta < 1e-4 else (1.-.5*theta/np.tan(.5*theta))/(theta*theta)
    inverse_left_jacobian = np.eye(3)-.5*omega+coefficient*(omega@omega)
    return np.r_[inverse_left_jacobian@translation, angle]/dt


def _perturb(transform, axis, delta):
    value = transform.copy()
    if axis < 3:
        value[axis, 3] += delta
    else:
        rotation = np.zeros(3); rotation[axis-3] = delta
        value[:3, :3] = cv2.Rodrigues(rotation)[0]@value[:3, :3]
    return value


class BodyMotion:
    def __init__(self, max_gap=3., min_gap=.001, variance_floor=None):
        self.max_gap, self.min_gap = float(max_gap), float(min_gap)
        self.floor = np.asarray(variance_floor if variance_floor is not None else [.0004]*3+[.0001]*3)
        if self.floor.shape != (6,) or not np.isfinite(self.floor).all() or (self.floor <= 0).any():
            raise ValueError("Body velocity variance floor must contain six positive finite values")
        self.previous = None
        self.reason = "waiting_for_motion_pair"

    def reset(self):
        self.previous = None
        self.reason = "waiting_for_motion_pair"

    def update(self, stamp, epoch, transform, covariance):
        transform, covariance = rigid(transform), pose_covariance(covariance)
        if not np.isfinite(stamp): raise ValueError("Nonfinite motion timestamp")
        old = self.previous
        self.previous = (float(stamp), epoch, transform.copy(), covariance.copy())
        if old is None or epoch != old[1]:
            self.reason = "new_motion_epoch"; return None
        dt = stamp-old[0]
        if not self.min_gap <= dt <= self.max_gap:
            self.reason = "motion_gap"; return None
        velocity = body_velocity(old[2], transform, dt)
        # ROS pose covariance uses translations and fixed-axis angle errors in
        # the parent frame. Differentiate the relative body motion in those same
        # coordinates. Dividing motion by dt also divides its variance by dt^2.
        jacobians = []
        epsilon = 1e-5
        for side in (0, 1):
            jacobian = np.zeros((6, 6))
            for axis in range(6):
                a, b = old[2], transform
                if side == 0:
                    plus = body_velocity(_perturb(a, axis, epsilon), b, dt)
                    minus = body_velocity(_perturb(a, axis, -epsilon), b, dt)
                else:
                    plus = body_velocity(a, _perturb(b, axis, epsilon), dt)
                    minus = body_velocity(a, _perturb(b, axis, -epsilon), dt)
                jacobian[:, axis] = (plus-minus)/(2.*epsilon)
            jacobians.append(jacobian)
        cov = jacobians[0]@old[3]@jacobians[0].T+jacobians[1]@covariance@jacobians[1].T
        cov = .5*(cov+cov.T)
        cov += np.diag(np.maximum(0., self.floor-np.diag(cov)))
        if not np.isfinite(velocity).all(): raise ValueError("Nonfinite body velocity")
        pose_covariance(cov)
        self.reason = "observed_body_motion"
        return velocity, cov, dt
