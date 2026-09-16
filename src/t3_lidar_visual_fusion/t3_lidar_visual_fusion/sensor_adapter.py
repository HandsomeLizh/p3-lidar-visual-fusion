"""Normalize sensor messages without generating depth from stereo."""
from collections import deque
import copy
import json
from pathlib import Path
import numpy as np
import cv2
import yaml
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data,QoSProfile,ReliabilityPolicy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, Imu, PointCloud2, PointField
from std_msgs.msg import String
from .ros_utils import cloud_arrays, stamp_sec
from .core import rigid


class SensorAdapter(Node):
    def __init__(self):
        super().__init__("fusion_sensor_adapter")
        self.declare_parameter("profile_path", "")
        self.declare_parameter("normalized_camera_input", False)
        self.cfg = yaml.safe_load(Path(self.get_parameter("profile_path").value).read_text())
        self.bridge = CvBridge()
        self.images = [deque(maxlen=self.cfg.get("stereo_queue_depth",2)), deque(maxlen=self.cfg.get("stereo_queue_depth",2))]
        self.last_image_stamp = -1.
        self.last_cloud_stamp = -1.
        self.last_imu_stamp = -1.
        self.counts = {"pairs":0, "clouds":0, "rejected":0, "left_received":0, "right_received":0, "sync_dropped":0, "imu_received":0, "imu_forwarded":0, "imu_rejected":0}
        self.pubs = [self.create_publisher(Image,"/fusion/left",3), self.create_publisher(Image,"/fusion/right",3)]
        self.cloud_pub = self.create_publisher(PointCloud2,"/fusion/lidar",3)
        self.imu_pub = self.create_publisher(Imu,"/fusion/imu",100)
        self.status = self.create_publisher(String,"/fusion/sensor_status",10)
        reliability=self.cfg.get("image_input_reliability","best_effort")
        if reliability not in ("reliable","best_effort"):
            raise ValueError("image_input_reliability must be reliable or best_effort")
        image_qos=QoSProfile(depth=1,reliability=(
            ReliabilityPolicy.RELIABLE if reliability=="reliable" else ReliabilityPolicy.BEST_EFFORT))
        for i,key in enumerate(["left_topic","right_topic"] if self.cfg.get("visual_source")!="none" and not self.get_parameter("normalized_camera_input").value else []):
            self.create_subscription(Image,self.cfg[key],lambda m,i=i:self.image(m,i),image_qos)
        cloud_reliability=self.cfg.get("lidar_input_reliability","best_effort")
        if cloud_reliability not in ("reliable","best_effort"):
            raise ValueError("lidar_input_reliability must be reliable or best_effort")
        cloud_qos=QoSProfile(depth=2,reliability=(
            ReliabilityPolicy.RELIABLE if cloud_reliability=="reliable" else ReliabilityPolicy.BEST_EFFORT))
        self.create_subscription(PointCloud2,self.cfg["lidar_topic"],self.cloud,cloud_qos)
        self.imu_mode=self.cfg.get("imu_mode","auto" if self.cfg.get("use_imu",False) else "off")
        if isinstance(self.imu_mode,bool):self.imu_mode="auto" if self.imu_mode else "off"
        if self.imu_mode=="auto" or self.cfg.get("use_imu",False):
            self.create_subscription(Imu,self.cfg["imu_topic"],self.imu,qos_profile_sensor_data)
        self.create_timer(2.,lambda:self.status.publish(String(data=json.dumps(self.counts))))
        self.get_logger().info("Adapter ready: stereo + LiDAR; imu_mode="+self.imu_mode)

    def image(self,msg,side):
        self.counts["left_received" if side==0 else "right_received"]+=1
        if (msg.width,msg.height) != tuple(self.cfg["input_image_size"]):
            self.counts["rejected"] += 1
            self.get_logger().error("Image dimensions differ from calibrated input_image_size",throttle_duration_sec=5)
            return
        if len(msg.data)>self.cfg.get("max_image_bytes",24000000):
            self.counts["rejected"]+=1;return
        try:
            if msg.encoding in ("mono8","8UC1"):
                mono=self.bridge.imgmsg_to_cv2(msg,desired_encoding="passthrough")
            elif msg.encoding in ("mono16","16UC1"):
                raw=self.bridge.imgmsg_to_cv2(msg,desired_encoding="passthrough")
                mono=np.clip(raw.astype(np.float32)*self.cfg.get("mono16_scale",255./65535.),0,255).astype(np.uint8)
            else:
                mono=self.bridge.imgmsg_to_cv2(msg,desired_encoding="mono8")
            mono=cv2.resize(mono,tuple(self.cfg["output_image_size"]),interpolation=cv2.INTER_AREA)
            out=self.bridge.cv2_to_imgmsg(mono,encoding="mono8");out.header=copy.deepcopy(msg.header)
            out.header.frame_id="camera_left_optical" if side==0 else "camera_right_optical"
        except Exception as e:
            self.counts["rejected"]+=1
            self.get_logger().error("Image conversion failed: "+str(e),throttle_duration_sec=5)
            return
        # Synchronize compact mono images, not 20 MB raw BGRA payloads.
        if len(self.images[side])==self.images[side].maxlen:self.counts["sync_dropped"]+=1
        self.images[side].append(out)
        while self.images[0] and self.images[1]:
            left,right=self.images[0][0],self.images[1][0]
            a,b=stamp_sec(left),stamp_sec(right)
            if abs(a-b)>self.cfg["stereo_max_skew"]:
                self.images[0 if a<b else 1].popleft();self.counts["sync_dropped"]+=1;continue
            self.images[0].popleft();self.images[1].popleft()
            if a<=self.last_image_stamp:continue
            self.last_image_stamp=a
            right.header.stamp=copy.deepcopy(left.header.stamp)
            self.pubs[0].publish(left);self.pubs[1].publish(right)
            self.counts["pairs"]+=1

    def cloud(self,msg):
        try:
            t=stamp_sec(msg)
            if t<=self.last_cloud_stamp: raise ValueError("Non-monotonic LiDAR header")
            expected=self.cfg.get("lidar_input_frame","")
            if expected and msg.header.frame_id!=expected:
                raise ValueError("Unexpected LiDAR frame: "+msg.header.frame_id)
            if len(msg.data)>self.cfg.get("max_cloud_bytes",16000000):
                raise ValueError("PointCloud2 payload exceeds configured input byte cap")
            xyz=cloud_arrays(msg)
            names={f.name for f in msg.fields}
            if self.cfg["instantaneous_cloud"] or self.cfg.get("cloud_motion_compensated",False):
                times=np.zeros(len(xyz))
            else:
                field=self.cfg["point_time_field"]
                if field not in names: raise ValueError("Real LiDAR requires calibrated per-point timing")
                times=cloud_arrays(msg,(field,))[:,0]*self.cfg["point_time_scale"]
                if not np.isfinite(times).all() or times.min()<0 or times.max()>self.cfg["max_scan_duration"]:
                    raise ValueError("Per-point time must be scan-start relative seconds")
            intensity=cloud_arrays(msg,("intensity",))[:,0] if "intensity" in names else np.zeros(len(xyz))
            ranges=np.linalg.norm(xyz,axis=1)
            valid=np.isfinite(xyz).all(axis=1)&(ranges>=self.cfg["min_range"])&(ranges<=self.cfg["max_range"])
            xyz,times,intensity=xyz[valid],times[valid],intensity[valid]
            cap=self.cfg.get("max_input_points",160000)
            if len(xyz)>cap:
                ix=np.linspace(0,len(xyz)-1,cap,dtype=int);xyz,times,intensity=xyz[ix],times[ix],intensity[ix]
            if len(xyz)<100: raise ValueError("Insufficient valid LiDAR points")
            dtype=np.dtype({"names":["x","y","z","intensity","ring","time"],
                            "formats":["<f4","<f4","<f4","<f4","<u2","<f4"],
                            "offsets":[0,4,8,12,16,20],"itemsize":24})
            data=np.zeros(len(xyz),dtype=dtype)
            for j,n in enumerate(["x","y","z"]):data[n]=xyz[:,j]
            data["intensity"],data["time"]=intensity,times
            out=PointCloud2()
            out.header=copy.deepcopy(msg.header);out.header.frame_id="lidar"
            out.height,out.width=1,len(data)
            out.fields=[PointField(name=n,offset=dtype.fields[n][1],datatype=(PointField.UINT16 if n=="ring" else PointField.FLOAT32),count=1) for n in dtype.names]
            out.point_step=24;out.row_step=len(data)*24;out.is_dense=True;out.data=data.tobytes()
            self.cloud_pub.publish(out);self.last_cloud_stamp=t;self.counts["clouds"]+=1
        except (ValueError,TypeError) as e:
            self.counts["rejected"]+=1
            self.get_logger().error(str(e),throttle_duration_sec=5)

    def imu(self,msg):
        self.counts["imu_received"]+=1
        vals=[msg.linear_acceleration.x,msg.linear_acceleration.y,msg.linear_acceleration.z,
              msg.angular_velocity.x,msg.angular_velocity.y,msg.angular_velocity.z]
        expected=self.cfg.get("imu_input_frame","")
        t=stamp_sec(msg)
        if (not np.isfinite(vals).all() or not np.isfinite(t) or t<=self.last_imu_stamp
            or (expected and msg.header.frame_id!=expected)
            or msg.angular_velocity_covariance[0]<0 or msg.linear_acceleration_covariance[0]<0):
            self.counts["imu_rejected"]+=1;return
        # Preserve SI measurements and timestamps. Extrinsics are applied in the
        # backend; relabel only after validating the configured physical frame.
        out=copy.deepcopy(msg);out.header.frame_id="imu"
        self.imu_pub.publish(out);self.last_imu_stamp=t;self.counts["imu_forwarded"]+=1


def main():
    rclpy.init()
    node=SensorAdapter()
    try:rclpy.spin(node)
    except KeyboardInterrupt:pass
    finally:
        node.destroy_node()
        if rclpy.ok():rclpy.shutdown()
