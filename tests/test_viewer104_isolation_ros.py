"""Exercise real ROS remaps without opening windows or publishing to a vehicle."""
import argparse,importlib.util,json,os,struct,sys,time
from pathlib import Path

import rclpy,yaml
from rclpy.node import Node
from rclpy.qos import QoSProfile,ReliabilityPolicy,DurabilityPolicy
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Header


def main():
 p=argparse.ArgumentParser();p.add_argument('--viewer-root',type=Path,required=True);args=p.parse_args()
 assert os.environ['ROS_DOMAIN_ID']=='98' and os.environ['ROS_LOCALHOST_ONLY']=='1'
 sys.path.insert(0,str(args.viewer_root));sys.path.insert(0,str(args.viewer_root/'scripts'))
 import viewer,p3_visual_monitor
 from t3_lidar_visual_fusion.ros_utils import xyz_cloud
 import numpy as np
 commands=viewer.display_commands(args.viewer_root/'results/isolation_test')
 rosargs=commands['monitor'][commands['monitor'].index('--ros-args'):]
 rclpy.init(args=rosargs);monitor=p3_visual_monitor.visual_monitor.Monitor()
 observer=Node('isolation_observer',use_global_arguments=False)
 received={'private':[],'legacy':[]}
 qos=QoSProfile(depth=1,reliability=ReliabilityPolicy.RELIABLE,durability=DurabilityPolicy.TRANSIENT_LOCAL)
 def receive(key):
  return lambda msg:received[key].append(struct.unpack_from('<f',msg.data,0)[0])
 observer.create_subscription(PointCloud2,'/viewer104_237/rviz_cloud',receive('private'),qos)
 observer.create_subscription(PointCloud2,'/T3/demo/rviz_cloud',receive('legacy'),qos)
 legacy=observer.create_publisher(PointCloud2,'/T3/demo/rviz_cloud',qos)
 try:
  assert monitor.get_fully_qualified_name()=='/viewer104_237/t3_104_237_monitor'
  assert monitor.goal_pub.topic_name=='/Car/T4/rviz_goal'
  config=yaml.safe_load((args.viewer_root/'config/p3_visual_window.rviz').read_text())
  display_topics=[d.get('Topic',{}).get('Value') for d in config['Visualization Manager']['Displays'] if isinstance(d.get('Topic'),dict)]
  assert monitor.cloud_pub.topic_name in display_topics
  assert monitor.marker_pub.topic_name in display_topics
  assert not any(t and t.startswith('/T3/demo/') for t in display_topics)
  header=Header(frame_id='map')
  until=time.monotonic()+4
  while time.monotonic()<until:
   legacy.publish(xyz_cloud(np.array([[42.,0.,0.]]),header))
   monitor.cloud_pub.publish(xyz_cloud(np.array([[7.,0.,0.]]),header))
   rclpy.spin_once(observer,timeout_sec=.05)
  assert received['private'] and set(received['private'])=={7.},received
  assert received['legacy'] and set(received['legacy'])=={42.},received
  pubs=monitor.get_publisher_names_and_types_by_node(monitor.get_name(),monitor.get_namespace())
  assert not any(t.startswith('/T3/demo/') for t,_ in pubs),pubs
  print(json.dumps(dict(passed=True,domain=98,private_cloud_messages=len(received['private']),
   legacy_cloud_messages=len(received['legacy']),private_node=monitor.get_fully_qualified_name(),
   window_config_topics=display_topics,vehicle_commands_published=0),indent=2))
 finally:
  observer.destroy_node();monitor.destroy_node();rclpy.shutdown()


if __name__=='__main__':main()
