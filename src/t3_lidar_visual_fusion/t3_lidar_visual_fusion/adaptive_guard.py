"""Fuse independent source poses without discarding motion at quality recovery.

LiDAR keeps its native odometry gauge. Each visual epoch is aligned once to a
same-time LiDAR pose; temporary quality rejection never changes that transform.
No simulator reference or fused-state feedback is used to align a moving epoch.
"""
import copy
import json
import time
from collections import deque
from pathlib import Path
import numpy as np
import rclpy
from std_msgs.msg import String
from .core import PoseBuffer, motion
from .continuous_frame import ContinuousFrame, pose_covariance, rotate_covariance
from .body_motion import BodyMotion
from .stationary import StationaryDetector
from .odometry_guard import OdometryGuard
from .ros_utils import stamp_sec, transform_from_pose, set_pose, cloud_arrays
from sensor_msgs.msg import PointCloud2
from rclpy.qos import QoSProfile, ReliabilityPolicy
from geometry_msgs.msg import TwistWithCovarianceStamped


class AdaptiveGuard(OdometryGuard):
    def __init__(self):
        self.visual_usable = False; self.lidar_usable = False; self.conflict = False
        self.stationary = None
        super().__init__()
        if self.visual_source not in ("learned", "none"):
            raise ValueError("Adaptive mode requires the independently checked learned frontend")
        self.gate = ContinuousFrame(**self.cfg["vision_gate"])
        self.use_body_motion = self.cfg.get("visual_constraint_mode", "absolute") == "body_twist"
        self.visual_motion = BodyMotion(max_gap=self.cfg["vision_gate"]["max_gap"],
            variance_floor=self.cfg.get("visual_velocity_variance_floor"))
        self.last_visual_motion_interval = None
        self.lidar_arrivals = deque(maxlen=12)
        self.visual_arrivals = deque(maxlen=12)
        lidar_config = dict(self.cfg["vision_gate"])
        lidar_config["recovery_frames"] = self.cfg.get("lidar_recovery_frames", 3)
        self.lidar_gate = ContinuousFrame(**lidar_config)
        self.lidar_gate.begin_epoch("odom")
        self.lidar_gate.alignment = np.eye(4)
        self.lidar_quality = {}; self.lidar_covariance_reliable = False
        self.lidar_covariances = deque(maxlen=self.cfg.get("pose_buffer_samples", 1200))
        self.filtered_poses = PoseBuffer(1200)
        self.last_good_pose = None
        self.output_qualified = False
        self.last_filtered_covariance = None
        self.filter_quality = {"reason": "waiting_for_filter"}
        self.last_lidar_usable_wall = 0.; self.last_visual_usable_wall = 0.
        self.visual_constraints = 0; self.lidar_constraints = 0
        self.visual_anchors = 0; self.lidar_anchors = 0
        self.adaptive_rejections = {"lidar_geometry": 0, "visual_geometry": 0,
                                    "conflict": 0, "unbridged_visual_epoch": 0}
        # Engineering noise floors, not an accuracy calibration. The backend
        # preserves larger directional uncertainty from scan geometry and prior.
        self.lidar_floor = np.asarray(self.cfg.get("lidar_covariance_floor",
                                                    [.0025]*3 + [.0004]*3))
        self.create_subscription(String, "/fusion/lidar_quality", self.registration_quality, 30)
        stationary_cfg=dict(self.cfg.get("stationary",{}))
        if stationary_cfg.pop("enabled",False):
            if not self.use_body_motion:raise ValueError("Stationarity requires visual body-motion observations")
            self.stationary=StationaryDetector(**stationary_cfg)
            self.create_subscription(PointCloud2,"/fusion/lidar",self.stationary_cloud,
                QoSProfile(depth=1,reliability=ReliabilityPolicy.RELIABLE))
        self.telemetry_status={"enabled":bool(self.cfg.get("telemetry_motion",{}).get("enabled",False))}
        self.telemetry_status_wall=0.
        if self.telemetry_status["enabled"]:
            self.create_subscription(String,"/fusion/telemetry_status",self.telemetry_report,3)
            if self.stationary is not None:
                self.create_subscription(TwistWithCovarianceStamped,"/fusion/telemetry_twist",self.telemetry_twist,10)
        self.get_logger().info("Adaptive repair: persistent epoch transforms and full directional covariance")

    def telemetry_report(self,msg):
        try:
            data=json.loads(msg.data)
            if not isinstance(data,dict):return
            self.telemetry_status=data;self.telemetry_status_wall=time.monotonic()
        except (ValueError,TypeError):pass

    def telemetry_twist(self,msg):
        try:
            stamp=stamp_sec(msg)
            age=self.get_clock().now().nanoseconds*1e-9-stamp
            # This detector only uses speed norms, which are rotation-invariant.
            if msg.header.frame_id not in ("base_link","vehicle_feedback_unverified") or not -.05<=age<=.4:return
            v,w=msg.twist.twist.linear,msg.twist.twist.angular
            self.stationary.velocity_feedback(stamp,[v.x,v.y,v.z,w.x,w.y,w.z])
        except ValueError:self.stationary.invalidate("invalid_velocity_feedback")

    def image(self,msg,side):
        super().image(msg,side)
        if self.stationary is None:return
        try:
            healthy=bool(self.quality[side] and self.quality[side][-1][1].valid
                         and stamp_sec(msg)>self.visual_blocked_through)
            self.stationary.image(stamp_sec(msg),self.bridge.imgmsg_to_cv2(msg,desired_encoding="mono8"),side,healthy)
        except Exception:self.stationary.invalidate("invalid_stationarity_image")

    def stationary_cloud(self,msg):
        try:
            if len(msg.data)>self.cfg.get("max_cloud_bytes",16000000):raise ValueError("Cloud too large")
            self.stationary.cloud(stamp_sec(msg),cloud_arrays(msg))
        except (ValueError,RuntimeError):self.stationary.invalidate("invalid_stationarity_cloud")

    def registration_quality(self, msg):
        try:
            self.lidar_quality = json.loads(msg.data)
        except (ValueError, TypeError):
            self.lidar_quality = {"reliable": False, "reason": "invalid_quality_message"}

    def source_timeout(self, arrivals, maximum):
        # Preserve sparse-bag support without waiting the sparse-bag timeout
        # when a high-rate live sensor stops. Arrival intervals use wall time,
        # so paused or accelerated replay clocks cannot extend freshness.
        if len(arrivals) < 4:
            return float(maximum)
        period = float(np.median(np.diff(arrivals)))
        return min(float(maximum), max(self.cfg.get("source_timeout_floor_sec", .6), 3.*period))

    def vision(self, msg):
        previous = self.last_vins_input_wall
        super().vision(msg)
        if self.last_vins_input_wall != previous:
            self.visual_arrivals.append(self.last_vins_input_wall)

    def close_vision(self, reason):
        if self.stationary is not None:self.stationary.invalidate(reason)
        self.visual_usable = False
        if hasattr(self, "visual_motion"):
            self.visual_motion.reset()
        return super().close_vision(reason)

    def lidar_reference_covariance_at(self, stamp):
        if not self.lidar_covariances:
            return None
        t, cov = min(self.lidar_covariances, key=lambda sample: abs(sample[0]-stamp))
        if abs(t-stamp) > self.cfg.get("agreement_stamp_tolerance", .05):
            return None
        if np.linalg.eigvalsh(cov[:3, :3])[-1] > self.cfg.get("lidar_reference_position_variance", 1.):
            return None
        if np.linalg.eigvalsh(cov[3:, 3:])[-1] > self.cfg.get("lidar_reference_rotation_variance", .25):
            return None
        return cov

    def lidar_reference_at(self, stamp):
        if self.lidar_reference_covariance_at(stamp) is None:
            return None
        return self.lio_poses.at(stamp, tolerance=self.cfg.get("agreement_stamp_tolerance", .05),
                                 max_gap=self.cfg.get("agreement_pose_max_gap", .35))

    def lio(self, msg):
        try:
            stamp = stamp_sec(msg)
            if msg.child_frame_id != "base_link" or msg.header.frame_id != "odom":
                raise ValueError("LiDAR input must describe base_link in its stable odom frame")
            transform = transform_from_pose(msg.pose.pose)
            if self.last_lio:
                dt = stamp-self.last_lio[0]
                _, distance, angle = motion(self.last_lio[1], transform)
                if dt <= 0 or distance/max(dt, 1e-6) > self.cfg["lio_max_speed"] or angle/max(dt, 1e-6) > self.cfg["lio_max_angular_speed"]:
                    raise ValueError("LiDAR discontinuity")
            covariance = pose_covariance(msg.pose.covariance)
            self.last_lio = stamp, transform.copy()
            self.last_lio_wall = time.monotonic()
            self.lidar_arrivals.append(self.last_lio_wall)
            self.lio_poses.append(stamp, transform)
            self.lidar_covariances.append((stamp, covariance.copy()))
            # Reject a failed registration, but retain a partial observation.
            # The EKF receives the full matrix, including the weak eigenvectors.
            if np.linalg.eigvalsh(covariance)[0] >= self.cfg.get("lidar_max_pose_variance", 100.):
                self.lidar_usable = False; self.lidar_covariance_reliable = False
                self.lidar_gate.close("lidar_registration_unavailable")
                self.adaptive_rejections["lidar_geometry"] += 1
                self.process_visual(); return
            self.lidar_covariance_reliable = self.lidar_reference_at(stamp) is not None
            # The initial scan defines the gauge; no recovery interval is lost.
            self.lidar_gate.recovery_frames = 1 if self.lidar_constraints == 0 else int(self.cfg.get("lidar_recovery_frames", 3))
            result = self.lidar_gate.accept(stamp, transform)
            if result.transform is None:
                self.lidar_usable = False; self.process_visual(); return
            out = copy.deepcopy(msg); out.header.frame_id = "odom"
            set_pose(out.pose.pose, result.transform)
            out.pose.covariance = rotate_covariance(covariance, np.eye(3), self.lidar_floor).reshape(-1).tolist()
            self.lio_pub.publish(out); self.counters["lio"] += 1
            self.lidar_usable = True; self.last_lidar_usable_wall = time.monotonic()
            self.lidar_constraints += 1
            self.process_visual()
        except ValueError as error:
            self.lidar_covariance_reliable = False; self.lidar_usable = False
            self.lidar_gate.close("invalid_lidar")
            self.get_logger().warn(str(error), throttle_duration_sec=5)

    def process_visual(self):
        if not self.pending_visual:
            return
        queued, msg = self.pending_visual[0]
        if not self.check_visual_timing(msg, queued, "before_fusion"):
            self.pending_visual.popleft(); return
        stamp = stamp_sec(msg)
        # A faster visual callback must not bypass same-frame LiDAR checks.
        # Bound the wait so vision can still continue if LiDAR stops arriving.
        awaiting_lidar = (self.last_lio is None or
                          self.last_lio[0] < stamp-self.cfg.get("agreement_stamp_tolerance", .05))
        lidar_recent = self.lidar_usable and time.monotonic()-self.last_lidar_usable_wall < self.source_timeout(self.lidar_arrivals,self.cfg["source_wall_timeout"])
        wait_budget = min(self.cfg.get("visual_reference_wait", 1.),
                          self.cfg.get("visual_max_age_sec", 1.5)*.5)
        if awaiting_lidar and lidar_recent and time.monotonic()-queued < wait_budget:
            return
        reference = self.lidar_reference_at(stamp)
        # Briefly allow the matching LiDAR callback to arrive for a new epoch.
        new_epoch = self.gate.epoch != msg.header.frame_id or self.gate.alignment is None
        # A matching but weak LiDAR pose cannot bridge a new visual epoch.
        # Wait only if that LiDAR sample has not arrived, within freshness.
        if (not self.use_body_motion and new_epoch and reference is None and self.last_good_pose is not None
            and awaiting_lidar and time.monotonic()-queued < wait_budget):
            return
        self.pending_visual.popleft()
        if self.gate.begin_epoch(msg.header.frame_id):
            self.consistency.reset()
            self.visual_motion.reset()
        try:
            transform = transform_from_pose(msg.pose.pose)
            if msg.child_frame_id != "base_link":
                raise ValueError("Visual estimate must describe base_link")
            covariance = pose_covariance(msg.pose.covariance)
            if reference is None and self.gate.alignment is None and self.last_good_pose is None:
                # Initial co-timed zero origins define a gauge even when the
                # first scan has not yet supplied directional constraints.
                first = self.lio_poses.samples[0] if self.lio_poses.samples else None
                if first is None or (abs(first[0]-stamp)<self.cfg.get("agreement_stamp_tolerance",.05)
                                     and np.allclose(first[1],np.eye(4),atol=1e-6)):
                    reference = np.eye(4)
            if self.use_body_motion and self.gate.alignment is None:
                # A relative body increment is independent of the visual epoch's
                # world origin. This identity is only for internal jump checks;
                # the pose is never used as an absolute EKF observation.
                self.gate.alignment = np.eye(4)
            if not self.use_body_motion and self.gate.align(transform, reference):
                self.visual_anchors += 1
            if self.gate.alignment is None:
                self.close_vision("new_epoch_requires_same_time_reference")
                self.adaptive_rejections["unbridged_visual_epoch"] += 1
                self.counters["vins_rejected"] += 1; return
            if np.diag(covariance).max() >= 1e5:
                self.close_vision("visual_frontend_anchor")
                self.counters["vins_rejected"] += 1; return
            healthy, reason = self.quality_at(stamp)
            if not healthy:
                self.close_vision(reason); self.counters["vins_rejected"] += 1; return
            floor = np.asarray(self.cfg["vision_pose_variance"])
            geometric_score = min(1., float(np.sqrt(np.min(floor/np.maximum(1e-12, np.diag(covariance))))))
            self.quality_weight = min(self.quality_weight, geometric_score)
            if geometric_score < self.cfg.get("independent_visual_min_score", .65):
                self.close_vision("weak_visual_geometry")
                self.adaptive_rejections["visual_geometry"] += 1
                self.counters["vins_rejected"] += 1; return
            if reference is not None:
                reference_covariance = self.lidar_reference_covariance_at(stamp)
                decision = self.consistency.check(stamp, transform, reference, reference_covariance)
                self.motion_status = dict(reason=decision.reason, reference_available=True,
                    reference_uncertainty_considered=reference_covariance is not None,
                    translation_error_m=decision.translation_error, rotation_error_rad=decision.rotation_error,
                    interval_sec=decision.interval, score=decision.score)
                if decision.reason == "visual_lidar_disagreement":
                    # Quarantine the contradictory visual constraint while the
                    # independently qualified LiDAR continues. No global latch.
                    self.close_vision("visual_lidar_disagreement")
                    self.adaptive_rejections["conflict"] += 1
                    self.counters["vins_rejected"] += 1; return
                if decision.ready:
                    self.quality_weight = min(self.quality_weight, decision.score)
            else:
                self.consistency.reset()
                self.motion_status = dict(reason="visual_independent_lidar_partial_or_missing", reference_available=False)
            weighted_covariance = rotate_covariance(covariance, np.eye(3),
                floor/max(.1, self.quality_weight)**2)
            body_motion = (self.visual_motion.update(stamp, msg.header.frame_id,
                transform, weighted_covariance) if self.use_body_motion else None)
            result = self.gate.accept(stamp, transform)
            if result.transform is None:
                if result.reason != "recovering":
                    self.visual_motion.reset()
                self.visual_usable = False; self.counters["vins_rejected"] += 1; return
            out = copy.deepcopy(msg)
            if self.use_body_motion:
                if body_motion is None:
                    self.visual_usable = False; self.counters["vins_rejected"] += 1; return
                velocity, velocity_covariance, interval = body_motion
                if self.stationary is not None and self.stationary.check(stamp,velocity):
                    # Replace the same visual motion observation; do not add a
                    # second correlated measurement or overwrite the output pose.
                    velocity=np.zeros(6)
                    velocity_covariance=np.diag([.005**2]*3+[.003**2]*3)
                    self.stationary.zero_updates+=1
                self.last_visual_motion_interval = interval
                # Preserve the honest source epoch on the inactive pose. Twist
                # is expressed in child_frame_id=base_link, per nav_msgs/Odometry.
                out.pose.covariance = np.diag(np.full(6, 1e6)).reshape(-1).tolist()
                linear, angular = out.twist.twist.linear, out.twist.twist.angular
                linear.x,linear.y,linear.z,angular.x,angular.y,angular.z = map(float, velocity)
                out.twist.covariance = velocity_covariance.reshape(-1).tolist()
            else:
                out.header.frame_id = "odom"
                set_pose(out.pose.pose, result.transform)
                out.pose.covariance = rotate_covariance(covariance, self.gate.alignment[:3, :3],
                    floor/max(.1, self.quality_weight)**2).reshape(-1).tolist()
            self.vision_pub.publish(out); self.vision_legacy_pub.publish(out)
            self.last_vins_wall = time.monotonic(); self.last_visual_usable_wall = self.last_vins_wall
            self.counters["vins_accepted"] += 1; self.visual_constraints += 1
            self.visual_usable = True
        except ValueError:
            self.close_vision("invalid_visual_body_pose")
            self.counters["vins_rejected"] += 1

    def active_sources(self):
        now = time.monotonic()
        return (self.lidar_usable and now-self.last_lidar_usable_wall <= self.source_timeout(self.lidar_arrivals,self.cfg["source_wall_timeout"]),
                self.visual_usable and self.gate.enabled and now-self.last_visual_usable_wall <= self.source_timeout(self.visual_arrivals,self.cfg.get("visual_wall_timeout_sec", 2.5)))

    def filtered(self, msg):
        if not any(self.active_sources()):
            self.output_qualified = False
            self.filter_quality = {"reason": "no_active_source"}
            return
        try:
            transform = transform_from_pose(msg.pose.pose)
            cov = pose_covariance(msg.pose.covariance)
            position_variance = float(np.linalg.eigvalsh(cov[:3, :3])[-1])
            rotation_variance = float(np.linalg.eigvalsh(cov[3:, 3:])[-1])
            self.filter_quality = dict(stamp_sec=stamp_sec(msg),
                position_max_variance=position_variance,
                rotation_max_variance=rotation_variance, reason="qualified")
            if position_variance > self.cfg.get("qualified_position_variance", 4.):
                self.output_qualified = False
                self.filter_quality["reason"] = "position_uncertain"
                return
            if rotation_variance > self.cfg.get("qualified_rotation_variance", .5):
                self.output_qualified = False
                self.filter_quality["reason"] = "rotation_uncertain"
                return
        except ValueError:
            self.output_qualified = False
            self.filter_quality = {"reason": "invalid_filtered_pose_or_covariance"}
            return
        stamp = stamp_sec(msg)
        if stamp < self.last_filter_stamp:
            return
        if stamp == self.last_filter_stamp:
            if (self.last_good_pose is not None and self.last_filtered_covariance is not None
                and np.allclose(transform,self.last_good_pose,rtol=0,atol=1e-10)
                and np.allclose(cov,self.last_filtered_covariance,rtol=0,atol=1e-10)):
                return
            # A late visual measurement can correct the current LiDAR stamp.
            # Publish that correction without inventing a later acquisition time.
            self.pose_pub.publish(msg); self.car_pose_pub.publish(msg); self.filtered_pub.publish(msg)
            self.counters["filtered"] += 1
            if self.path.poses and abs(stamp_sec(self.path.poses[-1])-stamp)<1e-8:
                self.path.poses[-1].pose=copy.deepcopy(msg.pose.pose)
            self.filtered_poses.replace_latest(stamp,transform)
        else:
            before = self.counters["filtered"]
            super().filtered(msg)
            if self.counters["filtered"] == before:
                return
            self.filtered_poses.append(stamp,transform)
        self.last_good_pose = transform.copy()
        self.last_filtered_covariance = cov.copy()
        self.output_qualified = True

    def status(self):
        if self.visual_source == "none":
            self.close_vision("vision_disabled")
        elif time.monotonic()-self.last_vins_input_wall > self.source_timeout(self.visual_arrivals,self.cfg.get("visual_wall_timeout_sec", 2.5)):
            self.close_vision("vision_timeout")
        lidar, visual = self.active_sources()
        mode = "lidar_visual" if lidar and visual else "visual_primary" if visual else "lidar_only" if lidar else "degraded"
        telemetry=dict(self.telemetry_status)
        if telemetry.get("enabled") and time.monotonic()-self.telemetry_status_wall>2.:
            telemetry.update(active=False,reason="telemetry_status_timeout")
        data = dict(self.counters, visual_source=self.visual_source,
            learned_backend=self.cfg.get("learned_visual", {}).get("backend"),
            fusion_strategy="persistent_epoch_full_covariance", operating_mode=mode,
            localization_valid=bool((lidar or visual) and self.output_qualified),
            filtered_quality=self.filter_quality,
            visual_pose_axes=self.cfg.get("visual_pose_axes", ["x","y","z","roll","pitch","yaw"]),
            visual_constraint_mode=self.cfg.get("visual_constraint_mode", "absolute"),
            visual_motion_interval_sec=self.last_visual_motion_interval,
            source_timeouts_sec=dict(lidar=self.source_timeout(self.lidar_arrivals,self.cfg["source_wall_timeout"]),
                visual=self.source_timeout(self.visual_arrivals,self.cfg.get("visual_wall_timeout_sec", 2.5))),
            lidar_usable=lidar, lidar_full_reference=self.lidar_covariance_reliable,
            vision_enabled=visual, visual_epoch=self.last_visual_epoch,
            vision_reason=self.gate.reason, visual_received=self.counters["vins_received"],
            visual_accepted=self.counters["vins_accepted"], visual_rejected=self.counters["vins_rejected"],
            visual_constraints=self.visual_constraints,
            visual_pose_constraints=0 if self.use_body_motion else self.visual_constraints,
            visual_motion_constraints=self.visual_constraints if self.use_body_motion else 0,
            lidar_pose_constraints=self.lidar_constraints,
            visual_anchors=self.visual_anchors, lidar_anchors=self.lidar_anchors,
            quality_basis="overlap_residual_and_directional_covariance",
            stationary=self.stationary.status() if self.stationary is not None else {"state":"disabled"},
            telemetry=telemetry,
            lidar_quality=self.lidar_quality, motion_consistency=self.motion_status,
            adaptive_rejections=self.adaptive_rejections, visual_timing=self.timing_status,
            visual_timing_rejected=self.visual_timing_rejected,
            image_quality=[h[-1][1].dictionary() if h else None for h in self.quality])
        self.status_pub.publish(String(data=json.dumps(data)))
        if self.output_dir:
            path = Path(self.output_dir)/"fusion_status.tmp"
            path.write_text(json.dumps(data)); path.replace(Path(self.output_dir)/"fusion_status.json")
        if self.path.poses:
            self.path.header.frame_id = "map"
            for pose in self.path.poses: pose.header.frame_id = "map"
            self.path_pub.publish(self.path); self.car_path_pub.publish(self.path)


def main():
    rclpy.init(); node = AdaptiveGuard()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok(): rclpy.shutdown()
