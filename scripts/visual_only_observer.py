#!/usr/bin/env python3
import json,time,sys
from pathlib import Path
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud2,Imu
from std_msgs.msg import String
out=Path(sys.argv[1]);rclpy.init();n=Node("pure_visual_observer")
poses=[];counts={"lidar":0,"imu":0};states=[]
def pose(m):
 p,q=m.pose.pose.position,m.pose.pose.orientation
 row=dict(source="learned_raw",stamp_sec=m.header.stamp.sec+m.header.stamp.nanosec*1e-9,
  frame=m.header.frame_id,child=m.child_frame_id,pose=[p.x,p.y,p.z,q.x,q.y,q.z,q.w],received_monotonic_sec=time.monotonic())
 poses.append(row)
 with (out/"visual_raw.jsonl").open("a") as f:f.write(json.dumps(row)+"\n")
n.create_subscription(Odometry,"/fusion/learned_raw",pose,100)
n.create_subscription(PointCloud2,"/fusion/lidar",lambda m:counts.__setitem__("lidar",counts["lidar"]+1),10)
n.create_subscription(Imu,"/fusion/imu",lambda m:counts.__setitem__("imu",counts["imu"]+1),100)
n.create_subscription(String,"/fusion/sensor_status",lambda m:states.append(json.loads(m.data)),20)
(out/"observer.ready").write_text("ready\n")
deadline=time.monotonic()+1000
while time.monotonic()<deadline and not (out/"observer.stop").exists():rclpy.spin_once(n,timeout_sec=.1)
(out/"verification.json").write_text(json.dumps(dict(raw_poses=poses,sensor_counts=counts,sensor_states=states,
 visual_only=True,no_lidar_or_imu_observed=not any(counts.values())),indent=2))
n.destroy_node();rclpy.shutdown()
