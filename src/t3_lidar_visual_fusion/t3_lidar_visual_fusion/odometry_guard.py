"""LIO absolute pose, guarded VINS relative pose, and T3 odometry/path outputs."""
import copy
import math
import json
import time
from pathlib import Path
from collections import deque
import cv2
import numpy as np
import yaml
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data,QoSProfile,DurabilityPolicy,ReliabilityPolicy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, PointCloud
from nav_msgs.msg import Odometry, Path as RosPath
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String
from .core import VisionGate, PoseBuffer, inverse, rigid, motion
from .motion_consistency import MotionConsistency
from .visual_timing import VisualTiming
from .image_quality import assess_image
from .ros_utils import stamp_sec,transform_from_pose,set_pose


class OdometryGuard(Node):
    def __init__(self):
        super().__init__("fusion_odometry_guard")
        self.declare_parameter("profile_path","")
        self.declare_parameter("output_dir","")
        self.output_dir=self.get_parameter("output_dir").value
        self.filtered_ages=deque(maxlen=256)
        self.cfg=yaml.safe_load(Path(self.get_parameter("profile_path").value).read_text())
        self.gate=VisionGate(**self.cfg["vision_gate"])
        self.body_to_base=inverse(rigid(self.cfg["base_from_imu"]))
        self.visual_source=self.cfg.get("visual_source","vins")
        if self.visual_source not in ("vins","roma_external","learned","none"):
            raise ValueError("visual_source must be vins, roma_external, learned or none")
        self.visual_topic=self.cfg.get("visual_odometry_topic",
            "/fusion/roma_raw" if self.visual_source=="roma_external" else ("/fusion/learned_raw" if self.visual_source=="learned" else "/fusion/vins_raw"))
        if self.visual_topic in ("/T3/semantic/current_pose","/Car/T3/localization/odometry",
            "/odometry/filtered","/fusion/ekf","/fusion/vision_odom_guarded","/vins/odom_guarded","/lio/odom"):
            raise ValueError("Visual input must be an unfused private topic, never a fused output")
        self.check_motion=self.cfg.get("require_lidar_agreement",True)
        self.consistency=MotionConsistency(**self.cfg.get("motion_consistency",{}))
        self.motion_status={"reason":"waiting","reference_available":False}
        self.lio_poses=PoseBuffer(self.cfg.get("pose_buffer_samples",1200))
        self.pending_visual=deque(maxlen=self.cfg.get("visual_pending_samples",8))
        self.visual_timing=VisualTiming(self.cfg.get("visual_max_age_sec",1.5),
            self.cfg.get("visual_future_tolerance_sec",.1))
        history=float(self.cfg.get("ekf_history_seconds",10.))
        if not math.isfinite(history) or history<=self.visual_timing.max_age_sec:
            raise ValueError("EKF history must exceed the visual measurement age budget")
        self.timing_status={"reason":"waiting"}
        self.visual_input_ages=deque(maxlen=256)
        self.visual_timing_rejected=0
        self.last_visual_input_stamp=-1.
        self.last_visual_epoch=None
        self.visual_blocked_through=-1.
        self.last_vins_input_wall=0.
        self.bridge=CvBridge()
        self.quality=[deque(maxlen=30),deque(maxlen=30)]
        self.features=deque(maxlen=30)
        self.last_lio=None
        self.last_lio_wall=0.
        self.last_vins_wall=0.
        self.last_filter_stamp=-1.
        self.path=RosPath();self.path.header.frame_id="odom"
        self.lio_pub=self.create_publisher(Odometry,"/lio/odom",30)
        self.vision_pub=self.create_publisher(Odometry,"/fusion/vision_odom_guarded",30)
        self.vision_legacy_pub=self.create_publisher(Odometry,"/vins/odom_guarded",30)
        self.pose_pub=self.create_publisher(Odometry,"/T3/semantic/current_pose",30)
        self.car_pose_pub=self.create_publisher(Odometry,"/Car/T3/localization/odometry",30)
        self.filtered_pub=self.create_publisher(Odometry,"/odometry/filtered",30)
        self.path_pub=self.create_publisher(RosPath,"/T3/semantic/trajectory",QoSProfile(depth=1,durability=DurabilityPolicy.TRANSIENT_LOCAL,reliability=ReliabilityPolicy.RELIABLE))
        self.car_path_pub=self.create_publisher(RosPath,"/Car/T3/debug/trajectory",QoSProfile(depth=1,durability=DurabilityPolicy.TRANSIENT_LOCAL,reliability=ReliabilityPolicy.RELIABLE))
        self.status_pub=self.create_publisher(String,"/fusion/status",10)
        self.counters={"lio":0,"vins_received":0,"vins_accepted":0,"vins_rejected":0,"filtered":0}
        self.create_subscription(Odometry,"/fusion/lio_raw",self.lio,30)
        if self.visual_source!="none":self.create_subscription(Odometry,self.visual_topic,self.vision,30)
        self.create_subscription(Odometry,"/fusion/ekf",self.filtered,30)
        for i,topic in enumerate(["/fusion/left","/fusion/right"] if self.visual_source!="none" else []):
            self.create_subscription(Image,topic,lambda m,i=i:self.image(m,i),QoSProfile(depth=1,reliability=ReliabilityPolicy.RELIABLE))
        self.create_subscription(PointCloud,"/fusion/vins_features",self.feature,10)
        self.create_timer(1.,self.status)
        self.create_timer(.05,self.process_visual)
        self.get_logger().info("Guard ready; LiDAR backend=voxelmap; visual source="+self.visual_source)

    def close_vision(self,reason):
        self.consistency.reset()
        return self.gate.close(reason)

    def block_image_epoch(self,stamp,reason):
        self.visual_blocked_through=max(self.visual_blocked_through,stamp)
        self.counters["vins_rejected"]+=len(self.pending_visual)
        self.pending_visual.clear();self.close_vision(reason)

    def image(self,msg,side):
        try:
            a=self.bridge.imgmsg_to_cv2(msg,desired_encoding="mono8")
            q=assess_image(a,min_brightness=self.cfg["min_brightness"],
                min_features=self.cfg["min_image_features"],
                min_contrast=self.cfg.get("min_image_contrast",18.),
                min_coverage=self.cfg.get("min_feature_coverage",.2),
                min_sharpness=self.cfg.get("min_image_sharpness",8.),
                texture_required=self.visual_source=="vins")
            previous=self.quality[side][-1][1] if self.quality[side] else None
            self.quality[side].append((stamp_sec(msg),q,time.monotonic()))
            if not q.valid:self.block_image_epoch(stamp_sec(msg),q.reason)
            elif previous and abs(previous.mean-q.mean)>self.cfg.get("exposure_change_threshold",65.):
                self.block_image_epoch(stamp_sec(msg),"exposure_transition")
        except Exception:
            self.quality[side].clear();self.block_image_epoch(stamp_sec(msg),"invalid_image")

    def feature(self,msg):
        self.features.append((stamp_sec(msg),len(msg.points)))

    def quality_at(self,stamp):
        self.quality_weight=1.
        for hist in self.quality:
            if not hist:return False,"no_image"
            stamp_q,q,image_wall=min(hist,key=lambda x:abs(x[0]-stamp))
            if abs(stamp_q-stamp)>self.cfg["image_quality_tolerance"]:return False,"image_timeout"
            if not q.valid:return False,q.reason
            if time.monotonic()-image_wall>self.visual_timing.max_age_sec:return False,"visual_image_wall_expired"
            self.quality_weight=min(self.quality_weight,q.score)
        if self.cfg["require_vins_features"]:
            if not self.features:return False,"no_vins_features"
            q=min(self.features,key=lambda x:abs(x[0]-stamp))
            if abs(q[0]-stamp)>self.cfg["feature_tolerance"]:return False,"feature_timeout"
            if q[1]<self.cfg["min_vins_features"]:return False,"low_vins_features"
        return True,"healthy"

    def lio(self,msg):
        try:
            stamp=stamp_sec(msg)
            if msg.child_frame_id!="base_link":raise ValueError("LiDAR input must describe base_link")
            t=transform_from_pose(msg.pose.pose)
            if self.last_lio:
                old_stamp,old=self.last_lio
                dt=stamp-old_stamp
                _,dist,angle=motion(old,t)
                if dt<=0 or dist/max(dt,1e-6)>self.cfg["lio_max_speed"] or angle/max(dt,1e-6)>self.cfg["lio_max_angular_speed"]:
                    raise ValueError("LIO discontinuity")
            out=copy.deepcopy(msg);out.header.frame_id="odom";out.child_frame_id="base_link"
            set_pose(out.pose.pose,t)
            # Conservative body-pose covariance floors; never trust upstream zero covariance.
            diag=np.maximum(np.diag(np.asarray(out.pose.covariance).reshape(6,6)),
                            np.asarray(self.cfg["lio_pose_variance"]))
            if not np.isfinite(diag).all():raise ValueError("LIO covariance is nonfinite")
            out.pose.covariance=np.diag(diag).reshape(-1).tolist()
            self.last_lio=(stamp,t.copy());self.last_lio_wall=time.monotonic()
            self.lio_poses.append(stamp,t)
            self.lio_pub.publish(out);self.counters["lio"]+=1
            self.process_visual()
        except ValueError as e:
            self.get_logger().warn(str(e),throttle_duration_sec=5)

    def check_visual_timing(self,msg,queued,stage):
        result=self.visual_timing.check(stamp_sec(msg),
            self.get_clock().now().nanoseconds/1e9,queued,time.monotonic())
        self.timing_status=dict(result.dictionary(),stage=stage,
            max_age_sec=self.visual_timing.max_age_sec)
        if stage=="received" and result.sensor_age_sec is not None:
            self.visual_input_ages.append(result.sensor_age_sec)
        if not result.valid:
            self.visual_timing_rejected+=1
            self.counters["vins_rejected"]+=1
            self.close_vision(result.reason)
        return result.valid

    def vision(self,msg):
        self.counters["vins_received"]+=1
        queued=time.monotonic()
        if not self.check_visual_timing(msg,queued,"received"):return
        stamp=stamp_sec(msg)
        if stamp<=self.visual_blocked_through:
            self.counters["vins_rejected"]+=1;return
        if stamp<=self.last_visual_input_stamp:
            self.counters["vins_rejected"]+=1
            self.close_vision("visual_input_out_of_order");return
        if self.visual_source=="learned":
            if not msg.header.frame_id.startswith("learned_epoch_"):
                self.counters["vins_rejected"]+=1;self.close_vision("invalid_visual_epoch");return
            if msg.header.frame_id!=self.last_visual_epoch:
                self.counters["vins_rejected"]+=len(self.pending_visual)
                self.pending_visual.clear();self.close_vision("visual_epoch_changed")
                self.last_visual_epoch=msg.header.frame_id
        self.last_visual_input_stamp=stamp
        self.last_vins_input_wall=queued
        if len(self.pending_visual)==self.pending_visual.maxlen:
            self.pending_visual.popleft();self.counters["vins_rejected"]+=1
            self.close_vision("visual_queue_overflow")
        self.pending_visual.append((queued,msg))
        self.process_visual()

    def process_visual(self):
        if not self.pending_visual:return
        queued,msg=self.pending_visual[0]
        if not self.check_visual_timing(msg,queued,"before_fusion"):
            self.pending_visual.popleft();return
        stamp=stamp_sec(msg)
        reference=self.lio_poses.at(stamp,
            tolerance=self.cfg.get("agreement_stamp_tolerance",.05),
            max_gap=self.cfg.get("agreement_pose_max_gap",.35))
        if self.check_motion and reference is None:
            past_gap=bool(self.lio_poses.samples and
                          self.lio_poses.samples[-1][0]>stamp+self.cfg.get("agreement_stamp_tolerance",.05))
            if not past_gap and time.monotonic()-queued<self.cfg.get("visual_reference_wait",2.):return
            self.pending_visual.popleft();self.counters["vins_rejected"]+=1
            self.motion_status={"reason":"no_aligned_lidar_reference","reference_available":False}
            self.close_vision("no_aligned_lidar_reference");return
        self.pending_visual.popleft()
        healthy,reason=self.quality_at(stamp)
        if not healthy:
            self.close_vision(reason);self.counters["vins_rejected"]+=1;return
        try:
            t=transform_from_pose(msg.pose.pose)
            if self.visual_source in ("roma_external","learned"):
                if msg.child_frame_id!="base_link":
                    raise ValueError("External/learned visual input must already describe base_link")
            else:t=t@self.body_to_base
        except ValueError:
            self.close_vision("invalid_visual_body_pose");self.counters["vins_rejected"]+=1;return
        if self.check_motion:
            decision=self.consistency.check(stamp,t,reference)
            self.motion_status=dict(reason=decision.reason,reference_available=True,
                translation_error_m=decision.translation_error,rotation_error_rad=decision.rotation_error,
                interval_sec=decision.interval,score=decision.score)
            if not decision.ready:
                # Warm-up keeps the comparison history, but cannot feed EKF.
                self.gate.close(decision.reason)
                self.counters["vins_rejected"]+=1;return
            self.quality_weight=min(self.quality_weight,decision.score)
        result=self.gate.accept(stamp,t,True,reason)
        if result.transform is None:
            if result.reason not in ("recovering","waiting"):self.consistency.reset()
            self.counters["vins_rejected"]+=1;return
        if result.reseed and reference is not None:
            # Align the new visual epoch's axes with the actual body heading.
            # Large anchor covariance prevents bridging an outage/reset.
            self.gate.output=reference.copy();result.transform=reference.copy()
        out=copy.deepcopy(msg);out.header.frame_id="odom";out.child_frame_id="base_link"
        set_pose(out.pose.pose,result.transform)
        var=np.asarray(self.cfg["vision_pose_variance"])*min(25.,1./max(.1,self.quality_weight)**2)
        if self.visual_source=="learned":
            observed=np.diag(np.asarray(msg.pose.covariance).reshape(6,6))
            if not np.isfinite(observed).all() or (observed<=0).any():
                self.close_vision("invalid_visual_covariance");self.counters["vins_rejected"]+=1;return
            var=np.maximum(var,observed)
        out.pose.covariance=np.diag(np.full(6,1e6) if result.reseed else var).reshape(-1).tolist()
        self.vision_pub.publish(out);self.vision_legacy_pub.publish(out)
        self.last_vins_wall=time.monotonic()
        self.counters["vins_accepted"]+=1

    def filtered(self,msg):
        now=time.monotonic()
        if now-self.last_lio_wall>self.cfg["source_wall_timeout"] and not (
            self.gate.enabled and now-self.last_vins_wall<=self.cfg["source_wall_timeout"]):
            return
        stamp=stamp_sec(msg)
        if stamp<=self.last_filter_stamp:return
        try:transform_from_pose(msg.pose.pose)
        except ValueError:return
        self.last_filter_stamp=stamp
        age=self.get_clock().now().nanoseconds/1e9-stamp
        if math.isfinite(age):self.filtered_ages.append(age)
        self.pose_pub.publish(msg);self.car_pose_pub.publish(msg);self.filtered_pub.publish(msg);self.counters["filtered"]+=1
        if not self.path.poses or stamp-stamp_sec(self.path.poses[-1])>=.2:
            p=PoseStamped();p.header=copy.deepcopy(msg.header);p.pose=copy.deepcopy(msg.pose.pose)
            self.path.header=copy.deepcopy(msg.header);self.path.poses.append(p)
            cap=self.cfg.get("trajectory_max_poses",5000)
            if len(self.path.poses)>cap:self.path.poses=self.path.poses[-cap:]

    def status(self):
        now=time.monotonic()
        if self.visual_source=="none":self.close_vision("vision_disabled")
        elif now-self.last_vins_input_wall>self.cfg.get("visual_wall_timeout_sec",2.5):
            self.close_vision("vision_timeout")
        lio_fresh=now-self.last_lio_wall<=self.cfg["source_wall_timeout"]
        mode=("lidar_visual" if self.gate.enabled else "lidar_only") if lio_fresh else "waiting_for_lidar"
        d=dict(self.counters,visual_source=self.visual_source,visual_input=self.visual_topic,
               learned_backend=self.cfg.get("learned_visual",{}).get("backend"),visual_epoch=self.last_visual_epoch,
               quality_basis="photometry_and_classical_features" if self.visual_source=="vins" else "photometry_and_lidar_motion",
               visual_received=self.counters["vins_received"],visual_accepted=self.counters["vins_accepted"],
               visual_rejected=self.counters["vins_rejected"],motion_consistency=self.motion_status,
               pending_visual=len(self.pending_visual),visual_timing=self.timing_status,
               visual_timing_rejected=self.visual_timing_rejected,
               visual_input_age_samples=len(self.visual_input_ages),
               visual_input_age_p95_sec=(float(np.percentile(self.visual_input_ages,95))
                   if self.visual_input_ages else None),
               vision_enabled=self.gate.enabled,vision_reason=self.gate.reason,
               operating_mode=mode,visual_blocked_through_stamp=self.visual_blocked_through,
               filtered_age_samples=len(self.filtered_ages),
               filtered_output_age_p95_sec=(float(np.percentile(self.filtered_ages,95)) if self.filtered_ages else None),
               lio_fresh=time.monotonic()-self.last_lio_wall<=self.cfg["source_wall_timeout"],
               image_quality=[h[-1][1].dictionary() if h else None for h in self.quality])
        self.status_pub.publish(String(data=json.dumps(d)))
        if self.output_dir:
            try:
                path=Path(self.output_dir)/"fusion_status.tmp"
                path.write_text(json.dumps(d));path.replace(Path(self.output_dir)/"fusion_status.json")
            except OSError as error:self.get_logger().warn("Status snapshot write failed: "+str(error),throttle_duration_sec=10)
        if self.path.poses:
            # map->odom is an explicit identity static transform for this backend.
            self.path.header.frame_id="map"
            for p in self.path.poses:p.header.frame_id="map"
            self.path_pub.publish(self.path);self.car_path_pub.publish(self.path)


def main():
    rclpy.init();node=OdometryGuard()
    try:rclpy.spin(node)
    except KeyboardInterrupt:pass
    finally:
        node.destroy_node()
        if rclpy.ok():rclpy.shutdown()
