"""Bounded cross-host receive check; no goals or vehicle commands."""
import collections,json,time,sys,io
from pathlib import Path
import numpy as np,rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile,ReliabilityPolicy,DurabilityPolicy
from nav_msgs.msg import Odometry,Path as PathMessage
from sensor_msgs.msg import PointCloud2,CompressedImage
from grid_map_msgs.msg import GridMap
from std_msgs.msg import String,UInt8MultiArray
from PIL import Image as PILImage
sys.path.insert(0,str(Path(__file__).resolve().parent/'visual'))
from viewer_wire import decode_grid,PREFIX
rclpy.init();node=Node('receive_probe',namespace='/viewer104_237');counts=collections.Counter();last={};first={};latest={}
qos=QoSProfile(depth=1,reliability=ReliabilityPolicy.RELIABLE)
retained=QoSProfile(depth=1,reliability=ReliabilityPolicy.RELIABLE,durability=DurabilityPolicy.TRANSIENT_LOCAL)
topics={'pose':('/T3/semantic/current_pose',Odometry,qos),
 'global_grid':(PREFIX+'/global_grid_zlib',UInt8MultiArray,retained),
 'local_grid':(PREFIX+'/local_grid_zlib',UInt8MultiArray,retained),
 'cloud':('/T3/mapping/stereo_map',PointCloud2,retained),
 'lidar_cloud':('/T3/mapping/lidar_map',PointCloud2,retained),
 'left':(PREFIX+'/left/compressed',CompressedImage,qos),'right':(PREFIX+'/right/compressed',CompressedImage,qos),
 'display_cloud':('/viewer104_237/rviz_cloud',PointCloud2,retained)}
def callback(key):
 def receive(msg):
  if key in ('global_grid','local_grid'):msg=decode_grid(msg)
  counts[key]+=1;last[key]=time.monotonic();first.setdefault(key,last[key]);latest[key]=msg
 return receive
for key,(topic,kind,policy) in topics.items():node.create_subscription(kind,topic,callback(key),policy)
try:
 until=time.monotonic()+20
 while time.monotonic()<until:rclpy.spin_once(node,timeout_sec=.1)
 report=dict(counts=dict(counts),rate_hz={k:(n-1)/(last[k]-first[k]) for k,n in counts.items() if n>1},
  idle_sec={k:time.monotonic()-v for k,v in last.items()},
  publishers={k:[dict(name=e.node_name,namespace=e.node_namespace) for e in node.get_publishers_info_by_topic(v[0])] for k,v in topics.items()},
  source='104',receiver='237',domain=59,vehicle_commands_published=0)
 for key in ['global_grid','local_grid']:
  if key in latest:
   m=latest[key];layers=list(m.layers)
   report[key]=dict(layers=layers,known_height_cells=(int(np.isfinite(np.asarray(m.data[layers.index('elevation')].data)).sum()) if 'elevation' in layers else 0))
 for key in ['cloud','lidar_cloud','display_cloud']:
  if key in latest:report[key+'_points']=latest[key].width*latest[key].height
 for key in ['left','right']:
  if key in latest:
   bitmap=PILImage.open(io.BytesIO(bytes(latest[key].data)))
   report[key+'_size']=[bitmap.width,bitmap.height,latest[key].format]
 required=['pose','left','right','display_cloud']
 fresh=lambda k: counts[k]>=3 and time.monotonic()-last.get(k,0)<5
 usable_grid=any(fresh(k) and report.get(k,{}).get('known_height_cells',0)>0 for k in ['global_grid','local_grid'])
 usable_cloud=any(fresh(k) and report.get(k+'_points',0)>0 for k in ['cloud','lidar_cloud'])
 report['passed']=all(fresh(k) for k in required) and usable_grid and usable_cloud
 target=Path(__file__).resolve().parent/'results/receive_check.json';target.parent.mkdir(exist_ok=True);target.write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
finally:node.destroy_node();rclpy.shutdown()
raise SystemExit(0 if report['passed'] else 1)
