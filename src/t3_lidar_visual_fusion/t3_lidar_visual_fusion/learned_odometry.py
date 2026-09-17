"""Asynchronous sparse learned stereo odometry with one pending pair and no TF."""
from collections import deque
import copy,json,logging,threading,time
from logging.handlers import RotatingFileHandler
from pathlib import Path
import cv2
import numpy as np
import yaml
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile,ReliabilityPolicy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image,PointCloud2
from nav_msgs.msg import Odometry
from std_msgs.msg import String
from .image_quality import assess_image
from .learned_matching import LearnedMatcher
from .learned_tracker import LearnedStereoTracker
from .stereo_geometry import TrackingFailure
from .visual_timing import VisualTiming
from .ros_utils import set_pose,stamp_sec,xyz_cloud


class LatestStereoPair:
    """Only the not-yet-started pair may be replaced; a running pair has one owner."""
    def __init__(self):
        self.condition=threading.Condition()
        self.pending=None
        self.stopped=False

    def put(self,pair):
        with self.condition:
            if self.stopped:return False
            replaced=self.pending is not None
            self.pending=pair
            self.condition.notify()
            return replaced

    def take(self,earliest=0.):
        with self.condition:
            while not self.stopped:
                delay=earliest-time.monotonic()
                if self.pending is not None and delay<=0:
                    pair=self.pending;self.pending=None;return pair
                self.condition.wait(timeout=max(.001,min(.2,delay)) if delay>0 else .2)
            return None

    def close(self):
        with self.condition:
            self.stopped=True;self.pending=None;self.condition.notify_all()


class LearnedOdometry(Node):
    def __init__(self):
        super().__init__("fusion_learned_odometry")
        self.declare_parameter("profile_path","")
        self.declare_parameter("workspace_root","")
        self.declare_parameter("output_dir","")
        self.profile=yaml.safe_load(Path(self.get_parameter("profile_path").value).read_text())
        self.cfg=self.profile["learned_visual"]
        if not self.get_parameter("output_dir").value or not self.get_parameter("workspace_root").value:
            raise ValueError("workspace_root and output_dir are required")
        self.output=Path(self.get_parameter("output_dir").value)
        root=Path(self.get_parameter("workspace_root").value)
        self.period=1./float(self.cfg.get("max_processing_hz",5.))
        if self.period<=0 or not np.isfinite(self.period):raise ValueError("Invalid processing frequency")
        self.timing=VisualTiming(self.profile.get("visual_max_age_sec",1.5),
            self.profile.get("visual_future_tolerance_sec",.1))
        self.bridge=CvBridge()
        self.frames=[deque(maxlen=2),deque(maxlen=2)]
        self.last_pair_stamp=-1.
        self.queue=LatestStereoPair()
        self.lock=threading.Lock()
        self.counts=dict(left_received=0,right_received=0,synchronized=0,replaced_pending=0,
            started=0,tracked=0,anchors=0,rejected=0,invalid_input=0,sync_dropped=0)
        self.last={"reason":"initializing"}
        self.samples=deque(maxlen=256)
        self.fatal=None
        self.failure_streak=0
        self.next_probe=0.
        cv2.setNumThreads(1)
        self.encoding='bgr8' if self.cfg['backend']=='roma' else 'mono8'
        if self.cfg['backend']=='roma':
            from .roma_tracker import RomaStereoTracker
            self.tracker=RomaStereoTracker(self.profile)
            self.backend=self.tracker
        else:
            self.backend=LearnedMatcher(root,self.cfg)
            # Warm CUDA kernels before accepting timed sensor data.
            width,height=self.profile["output_image_size"]
            warm=self.backend.extract(np.random.default_rng(0).integers(0,256,(height,width),dtype=np.uint8))
            self.backend.match(warm,warm)
            del warm
            self.tracker=LearnedStereoTracker(self.profile,self.backend)
        self.resources=self.backend.resource_snapshot()
        self.pub=self.create_publisher(Odometry,self.profile["visual_odometry_topic"],3)
        self.origin_pub=self.create_publisher(Odometry,"/fusion/visual_epoch_origin",3)
        self.mapping_pub=(self.create_publisher(PointCloud2,self.profile["stereo_mapping_topic"],2)
            if self.profile.get("stereo_mapping_topic") else None)
        self.status_pub=self.create_publisher(String,"/fusion/learned_status",3)
        self.create_subscription(String,"/fusion/status",self.fusion_feedback,3)
        qos=QoSProfile(depth=1,reliability=ReliabilityPolicy.RELIABLE)
        for side,topic in enumerate(["/fusion/left","/fusion/right"]):
            self.create_subscription(Image,topic,lambda msg,side=side:self.image(msg,side),qos)
        self.output.mkdir(parents=True,exist_ok=True)
        self.file_logger=logging.getLogger("fusion_learned_metrics_"+str(id(self)))
        self.file_logger.setLevel(logging.INFO);self.file_logger.propagate=False
        self.file_handler=RotatingFileHandler(self.output/"learned_metrics.jsonl",
            maxBytes=8*2**20,backupCount=2,encoding="utf-8")
        self.file_logger.addHandler(self.file_handler)
        self.worker=threading.Thread(target=self.work,name="learned_stereo_worker",daemon=True)
        self.worker.start()
        self.create_timer(1.,self.status)
        self.get_logger().info("Learned frontend ready: "+self.cfg["backend"]+
            "; one pending pair, max keypoints="+str(self.cfg.get("max_keypoints",1024)))

    def fusion_feedback(self,msg):
        try:
            status=json.loads(msg.data)
            if status.get("vision_reason") in ("visual_lidar_disagreement","pose_jump"):
                self.next_probe=max(self.next_probe,time.monotonic()+self.cfg.get("failure_probe_period_sec",1.))
        except (ValueError,TypeError):pass

    def increment(self,name,amount=1):
        with self.lock:self.counts[name]+=amount

    def image(self,msg,side):
        self.increment("left_received" if side==0 else "right_received")
        if ((msg.width,msg.height)!=tuple(self.profile["output_image_size"])
            or msg.encoding not in (("bgr8",) if self.encoding=='bgr8' else ("mono8","8UC1"))
            or len(msg.data)>self.cfg.get("max_normalized_image_bytes",1048576)):
            self.increment("invalid_input");return
        if len(self.frames[side])==self.frames[side].maxlen:self.increment("sync_dropped")
        self.frames[side].append((msg,time.monotonic()))
        while self.frames[0] and self.frames[1]:
            (left,wall0),(right,wall1)=self.frames[0][0],self.frames[1][0]
            a,b=stamp_sec(left),stamp_sec(right)
            if abs(a-b)>self.profile.get("stereo_max_skew",.01):
                self.frames[0 if a<b else 1].popleft();self.increment("sync_dropped");continue
            self.frames[0].popleft();self.frames[1].popleft()
            if a<=self.last_pair_stamp:self.increment("sync_dropped");continue
            self.last_pair_stamp=a
            self.increment("synchronized")
            if self.queue.put((left,right,min(wall0,wall1))):self.increment("replaced_pending")

    def age(self,msg,queued):
        return self.timing.check(stamp_sec(msg),self.get_clock().now().nanoseconds/1e9,
            queued,time.monotonic())

    def record(self,metrics):
        record=dict(metrics,backend=self.cfg["backend"])
        resources=self.backend.resource_snapshot()
        with self.lock:
            self.last=record
            self.samples.append({k:v for k,v in record.items() if k.endswith("_sec") and isinstance(v,(int,float))})
            self.resources=resources
        self.file_logger.info(json.dumps(record,allow_nan=False))

    def publish_stereo(self,left,queued):
        """Publish current metric depth; the mapper obtains the fused body pose.

        A temporal tracking failure must not discard valid stereo depth, and a
        failed stereo pair must never reuse the previous pair's point cloud.
        """
        sample=getattr(self.tracker,'current_stereo',None)
        if (self.mapping_pub is None or self.queue.stopped or sample is None or
                sample[0]!=stamp_sec(left) or not self.age(left,queued).valid):return 0
        points=self.tracker.geometry.mapping_points(sample[1])
        header=copy.deepcopy(left.header);header.frame_id='base_link'
        if len(points):self.mapping_pub.publish(xyz_cloud(points,header))
        return len(points)

    def reject(self,reason,left,queued,start=None,metrics=None):
        retained=self.tracker.reject(stamp_sec(left))
        self.increment("rejected")
        self.failure_streak+=1
        if self.failure_streak>=self.cfg.get("backoff_after_failures",3):
            self.next_probe=time.monotonic()+self.cfg.get("failure_probe_period_sec",1.)
        age=self.age(left,queued)
        data=dict(metrics or {})
        data.update(reason=reason,tracking_valid=False,reference_retained=retained,
            reference_reset=not retained,epoch=self.tracker.epoch,sensor_stamp_sec=stamp_sec(left),
            sensor_age_sec=age.sensor_age_sec,wall_since_input_sec=max(0.,time.monotonic()-queued))
        # An invalid sample closes the downstream quality gate promptly. Its
        # large covariance prevents the held pose becoming a motion observation.
        if age.valid:
            msg=Odometry();msg.header=copy.deepcopy(left.header)
            msg.header.frame_id="learned_epoch_"+str(self.tracker.epoch);msg.child_frame_id="base_link"
            set_pose(msg.pose.pose,np.eye(4) if self.tracker.last_pose is None else self.tracker.last_pose[1])
            msg.pose.covariance=np.diag(np.full(6,1e6)).reshape(-1).tolist()
            msg.twist.covariance=list(msg.pose.covariance)
            if not self.queue.stopped:self.pub.publish(msg)
        if start is not None:data["processing_sec"]=time.perf_counter()-start
        self.record(data)

    def work(self):
        earliest=0.
        try:
            while True:
                pair=self.queue.take(max(earliest,self.next_probe))
                if pair is None:return
                left,right,queued=pair
                earliest=time.monotonic()+self.period
                self.increment("started")
                begin=time.perf_counter()
                if not self.age(left,queued).valid:
                    self.reject(self.age(left,queued).reason,left,queued,begin);continue
                try:
                    a=self.bridge.imgmsg_to_cv2(left,desired_encoding=self.encoding)
                    b=self.bridge.imgmsg_to_cv2(right,desired_encoding=self.encoding)
                    quality=[assess_image(cv2.cvtColor(x,cv2.COLOR_BGR2GRAY) if x.ndim==3 else x,
                                          texture_required=False) for x in (a,b)]
                    if not all(q.valid for q in quality):
                        self.reject(next(q.reason for q in quality if not q.valid),left,queued,begin);continue
                    result=self.tracker.process(stamp_sec(left),a,b)
                except (TrackingFailure,ValueError,cv2.error) as exc:
                    metrics=dict(getattr(exc,'metrics',{}))
                    metrics['mapping_stereo_points']=self.publish_stereo(left,queued)
                    self.reject(str(exc)[:160],left,queued,begin,metrics);continue
                freshness=self.age(left,queued)
                if not freshness.valid:
                    self.reject(freshness.reason,left,queued,begin);continue
                msg=Odometry();msg.header=copy.deepcopy(left.header)
                msg.header.frame_id="learned_epoch_"+str(result.epoch)
                msg.child_frame_id="base_link"
                set_pose(msg.pose.pose,result.base_pose)
                q=max(.1,min(min(x.score for x in quality),result.metrics.get("pnp_ratio",1.),
                    result.metrics.get("stereo_motion_ratio",1.)))
                variance=np.asarray(self.profile["vision_pose_variance"])*min(25.,1./q**2)
                msg.pose.covariance=np.diag(np.full(6,1e6) if result.anchor else variance).reshape(-1).tolist()
                msg.twist.covariance=np.diag(np.full(6,1e6)).reshape(-1).tolist()
                if self.queue.stopped:return
                # Explicit gauge event: an ordinary rejected/held identity pose
                # must never be mistaken for a newly validated stereo origin.
                if result.anchor:self.origin_pub.publish(msg)
                self.pub.publish(msg)
                result.metrics['mapping_stereo_points']=self.publish_stereo(left,queued)
                self.increment("anchors" if result.anchor else "tracked")
                if result.anchor and result.metrics["reason"] not in ("initialized","tracking_gap"):
                    self.failure_streak+=1
                    if self.failure_streak>=self.cfg.get("backoff_after_failures",3):
                        self.next_probe=time.monotonic()+self.cfg.get("failure_probe_period_sec",1.)
                else:self.failure_streak=0;self.next_probe=0.
                metrics=dict(result.metrics,epoch=result.epoch,sensor_stamp_sec=stamp_sec(left),
                    sensor_age_sec=self.age(left,queued).sensor_age_sec,
                    wall_since_input_sec=time.monotonic()-queued,
                    processing_sec=time.perf_counter()-begin,quality_score=q)
                self.record(metrics)
        except Exception as exc:
            if not self.queue.stopped:
                with self.lock:self.fatal=type(exc).__name__+": "+str(exc)[:500]
                self.get_logger().error("Learned frontend failed: "+self.fatal)

    def status(self):
        with self.lock:
            data=dict(self.counts,last=dict(self.last),resources=dict(self.resources))
            samples=list(self.samples);fatal=self.fatal
        if fatal:raise RuntimeError(fatal)
        fields=sorted({key for sample in samples for key in sample})
        data["recent_timing"]={key:dict(zip(("p50","p95","p99"),map(float,
            np.percentile([s[key] for s in samples if key in s],[50,95,99])))) for key in fields}
        data["timing_samples"]=len(samples)
        data["failure_streak"]=self.failure_streak
        data["probing_at_reduced_rate"]=self.next_probe>time.monotonic()
        data["pending_pair_capacity"]=1
        data["stereo_sync_capacity_per_camera"]=2
        payload=json.dumps(data,allow_nan=False)
        self.status_pub.publish(String(data=payload))
        temp=self.output/"learned_metrics.tmp"
        temp.write_text(payload)
        temp.replace(self.output/"learned_metrics.json")

    def close(self):
        self.queue.close()
        self.worker.join(timeout=5.)
        self.file_handler.close()
        self.file_logger.removeHandler(self.file_handler)


def main():
    rclpy.init();node=None
    try:
        node=LearnedOdometry();rclpy.spin(node)
    except KeyboardInterrupt:pass
    finally:
        if node is not None:
            node.close();node.destroy_node()
        if rclpy.ok():rclpy.shutdown()
