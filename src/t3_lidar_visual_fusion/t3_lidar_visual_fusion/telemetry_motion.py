"""Independent, bounded velocity-only relay from the existing vehicle receiver.

No vehicle command publishers, TCP connections, images, or absolute pose inputs.
The UE feedback timestamp is receiver time; this is not a calibrated IMU.
"""
from collections import Counter
import json
from pathlib import Path
import time

import numpy as np
import rclpy
from rclpy.context import Context
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from rclpy.signals import SignalHandlerOptions
from geometry_msgs.msg import TwistWithCovarianceStamped
from nav_msgs.msg import Odometry
from std_msgs.msg import String
import yaml


class VelocityGate:
    """Validate receiver-stamped twists; keep only the latest unread sample."""
    def __init__(self, cfg):
        self.frame = cfg.get('frame_id', 'base_link')
        self.max_age = float(cfg.get('max_age_sec', .3))
        self.future = float(cfg.get('future_tolerance_sec', .05))
        self.max_speed = float(cfg.get('max_speed_mps', 4.))
        self.max_angular = float(cfg.get('max_angular_speed_rps', 2.))
        self.std = np.asarray(cfg.get('standard_deviation', [.05]*3 + [.03]*3), dtype=float)
        self.linear_scale = float(cfg.get('linear_scale', 1.))
        self.angular_scale = float(cfg.get('angular_scale', 1.))
        self.rotation = np.asarray(cfg.get('base_from_feedback_rotation', np.eye(3)), dtype=float)
        values = [self.max_age, self.future, self.max_speed, self.max_angular, self.linear_scale, self.angular_scale]
        if (not np.isfinite(values).all() or min(values) <= 0 or self.std.shape != (6,)
                or not np.isfinite(self.std).all() or np.any(self.std <= 0)):
            raise ValueError('Invalid telemetry velocity limits or uncertainty')
        if (self.rotation.shape != (3,3) or not np.isfinite(self.rotation).all()
                or not np.allclose(self.rotation@self.rotation.T,np.eye(3),atol=1e-6)
                or not np.isclose(np.linalg.det(self.rotation),1.,atol=1e-6)):
            raise ValueError('Feedback-to-base rotation must be a proper calibrated rotation')
        self.counts = Counter()
        self.last_seen = -1.
        self.last_forwarded = None
        self.pending = None
        self.reason = 'waiting_for_velocity'

    def accept(self, stamp, frame, velocity, now):
        self.counts['received'] += 1
        velocity = np.asarray(velocity, dtype=float)
        if velocity.shape == (6,):
            velocity = velocity * np.array([self.linear_scale]*3 + [self.angular_scale]*3)
            velocity = np.r_[self.rotation@velocity[:3],self.rotation@velocity[3:]]
        reason = None
        if velocity.shape != (6,) or not np.isfinite(velocity).all() or not np.isfinite([stamp, now]).all():
            reason = 'nonfinite_or_malformed'
        elif frame != self.frame:
            reason = 'wrong_frame'
        elif stamp <= self.last_seen:
            reason = 'old_or_duplicate'
        elif not -self.future <= now-stamp <= self.max_age:
            reason = 'expired_or_future'
        elif np.linalg.norm(velocity[:3]) > self.max_speed or np.linalg.norm(velocity[3:]) > self.max_angular:
            reason = 'implausible_velocity'
        if reason:
            self.counts[reason] += 1
            self.reason = reason
            return False
        self.last_seen = stamp
        if self.pending is not None:
            self.counts['coalesced'] += 1
        self.pending = (stamp, velocity.copy())
        self.reason = 'qualified'
        return True

    def take(self, now):
        sample, self.pending = self.pending, None
        if sample is None:
            return None
        stamp, velocity = sample
        if not -self.future <= now-stamp <= self.max_age:
            self.counts['expired_before_publish'] += 1
            self.reason = 'expired_before_publish'
            return None
        self.last_forwarded = stamp
        self.counts['forwarded'] += 1
        # Inflate delayed local messages. The upstream sampling delay is unknown.
        inflation = 1. + max(0., now-stamp)/self.max_age
        return stamp, velocity, np.diag((self.std*inflation)**2)


class TelemetryMotion(Node):
    def __init__(self):
        super().__init__('fusion_telemetry_motion')
        self.declare_parameter('profile_path', '')
        self.declare_parameter('output_dir', '')
        self.output = self.get_parameter('output_dir').value
        profile = yaml.safe_load(Path(self.get_parameter('profile_path').value).read_text())
        self.cfg = profile.get('telemetry_motion', {})
        if not self.cfg.get('enabled', False):
            raise ValueError('Telemetry motion must be explicitly enabled in the profile')
        if self.cfg.get('fuse_velocity', False) and not self.cfg.get('calibration_confirmed', False):
            raise ValueError('Velocity integration requires confirmed source units and body frame')
        self.gate = VelocityGate(self.cfg)
        self.output_hz = float(self.cfg.get('max_output_hz', 25.))
        if not np.isfinite(self.output_hz) or not 1 <= self.output_hz <= 100:
            raise ValueError('Telemetry output limit must be between 1 and 100 Hz')
        self.publisher = self.create_publisher(TwistWithCovarianceStamped, '/fusion/telemetry_twist', 10)
        self.status_publisher = self.create_publisher(String, '/fusion/telemetry_status', 3)
        self.create_timer(1./self.output_hz, self.publish_latest)
        self.create_timer(1., self.report)
        self.began = time.monotonic()
        self.get_logger().info('Velocity-only UE telemetry assistance; receiver timestamps; no vehicle commands')

    def receive(self, msg):
        # Deliberately read no fields from msg.pose or its covariance.
        v, w = msg.twist.twist.linear, msg.twist.twist.angular
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec*1e-9
        self.gate.accept(stamp, msg.child_frame_id, [v.x, v.y, v.z, w.x, w.y, w.z],
                         self.get_clock().now().nanoseconds*1e-9)

    def publish_latest(self):
        sample = self.gate.take(self.get_clock().now().nanoseconds*1e-9)
        if sample is None:
            return  # Never restamp or repeat an old measurement as new evidence.
        stamp, velocity, covariance = sample
        msg = TwistWithCovarianceStamped()
        ns = round(stamp*1e9)
        msg.header.stamp.sec, msg.header.stamp.nanosec = divmod(ns, 10**9)
        msg.header.frame_id = 'base_link' if self.cfg.get('calibration_confirmed',False) else 'vehicle_feedback_unverified'
        v, w = msg.twist.twist.linear, msg.twist.twist.angular
        v.x, v.y, v.z, w.x, w.y, w.z = map(float, velocity)
        msg.twist.covariance = covariance.ravel().tolist()
        self.publisher.publish(msg)

    def report(self):
        now = self.get_clock().now().nanoseconds*1e-9
        age = None if self.gate.last_forwarded is None else now-self.gate.last_forwarded
        data = dict(enabled=True, source='ue_vehicle_telemetry', input_topic=self.cfg.get('topic', '/car/odom'),
                    timestamp_basis='receiver_time', absolute_pose_used=False, vehicle_commands_published=0,
                    integrated_in_ekf=bool(self.cfg.get('fuse_velocity', False)),
                    calibration_confirmed=bool(self.cfg.get('calibration_confirmed', False)),
                    linear_scale=self.gate.linear_scale, angular_scale=self.gate.angular_scale,
                    active=age is not None and 0 <= age <= self.gate.max_age,
                    age_sec=age, reason=self.gate.reason, counts=dict(self.gate.counts),
                    elapsed_sec=time.monotonic()-self.began, max_output_hz=self.output_hz,
                    uncertainty='engineering_prior_not_sensor_calibration')
        self.status_publisher.publish(String(data=json.dumps(data)))
        if self.output:
            path = Path(self.output)/'telemetry_status.tmp'
            path.write_text(json.dumps(data, indent=2))
            path.replace(Path(self.output)/'telemetry_status.json')


def main():
    rclpy.init()
    target = TelemetryMotion()
    context = Context()
    rclpy.init(args=[], context=context, domain_id=int(target.cfg.get('source_domain', 10)),
               signal_handler_options=SignalHandlerOptions.NO)
    source = rclpy.create_node('fusion_telemetry_source', context=context)
    source.create_subscription(Odometry, target.cfg.get('topic', '/car/odom'), target.receive,
                               QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE))
    receive_executor = SingleThreadedExecutor(context=context)
    publish_executor = SingleThreadedExecutor()
    receive_executor.add_node(source)
    publish_executor.add_node(target)
    try:
        while rclpy.ok():
            receive_executor.spin_once(timeout_sec=.002)
            publish_executor.spin_once(timeout_sec=.002)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        receive_executor.shutdown()
        publish_executor.shutdown()
        source.destroy_node()
        target.destroy_node()
        context.try_shutdown()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
