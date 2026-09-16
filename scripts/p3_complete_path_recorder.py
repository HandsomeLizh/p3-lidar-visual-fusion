from pathlib import Path
import json,time
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile,DurabilityPolicy,ReliabilityPolicy
from nav_msgs.msg import Path as RosPath
out=Path("/home/yanfa/P3/lidar_visual_fusion/results/p3_roma_long_80m_20260915")
rclpy.init();n=Node("p3_complete_path_recorder");latest=[None]
n.create_subscription(RosPath,"/T3/semantic/trajectory",lambda m:latest.__setitem__(0,m),
    QoSProfile(depth=1,reliability=ReliabilityPolicy.RELIABLE,durability=DurabilityPolicy.TRANSIENT_LOCAL))
deadline=time.monotonic()+600
while time.monotonic()<deadline:
 rclpy.spin_once(n,timeout_sec=.2)
 m=latest[0]
 if m and m.poses:
  p=m.poses[-1];stamp=p.header.stamp.sec+p.header.stamp.nanosec*1e-9
  if stamp>=1598.7317:
   rows=[]
   for pose in m.poses:
    t=pose.header.stamp.sec+pose.header.stamp.nanosec*1e-9
    p,q=pose.pose.position,pose.pose.orientation
    rows.append([t,p.x,p.y,p.z,q.x,q.y,q.z,q.w])
   (out/"observed_path.tum").write_text("\n".join(" ".join(format(v,".15g") for v in row) for row in rows)+"\n")
   (out/"path_receipt.json").write_text(json.dumps(dict(poses=len(rows),frame=m.header.frame_id,last_stamp_sec=stamp,source="/T3/semantic/trajectory",full_path=True),indent=2))
   print("Captured final P3 path",len(rows),stamp,flush=True);break
else:raise RuntimeError("Final P3 path was not published")
n.destroy_node();rclpy.shutdown()
