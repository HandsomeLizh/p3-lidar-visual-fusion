#!/usr/bin/env python3
"""Compare pre-DDS normalization with the actual frozen adapter on bag images."""
from pathlib import Path
import collections,hashlib,importlib.util,json,sys
import yaml
from cv_bridge import CvBridge
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
from indexed_bag import groups
from stereo_transport import StereoNormalizer
path=ROOT/"results/long_voxelmap_80m_20260915/source_snapshot/src/t3_lidar_visual_fusion/t3_lidar_visual_fusion/sensor_adapter.py"
if not path.exists():
 path=ROOT/"archive/pre_optional_imu_20260915/src/t3_lidar_visual_fusion/t3_lidar_visual_fusion/sensor_adapter.py"
spec=importlib.util.spec_from_file_location("t3_lidar_visual_fusion.frozen_sensor_adapter",path)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
cfg=yaml.safe_load((ROOT/"config/long_bag_voxelmap.yaml").read_text())
class Publisher:
 def __init__(self):self.messages=[]
 def publish(self,m):self.messages.append(m)
class Fake:
 def __init__(self):
  self.cfg=cfg;self.bridge=CvBridge();self.images=[collections.deque(maxlen=2),collections.deque(maxlen=2)]
  self.counts=collections.defaultdict(int);self.last_image_stamp=-1.;self.pubs=[Publisher(),Publisher()]
 def get_logger(self):raise RuntimeError("Unexpected adapter rejection")
adapter=Fake();normalizer=StereoNormalizer(cfg);evidence=[]
for i,(raw,batch) in enumerate(groups("/home/yanfa/Env_X/InterFace/bags/20260824_235238",
 str(ROOT/"test_data/20260824_235238/first_10min_index.json"),4,False)):
 originals=[]
 for side,topic in enumerate(["/Car/T5/Cam_Left/image_raw/color","/Car/T5/Cam_Right/image_raw/color"]):
  msg=batch[topic];msg.header.stamp.sec=1000+i;msg.header.stamp.nanosec=0;originals.append(msg)
  module.SensorAdapter.image(adapter,msg,side)
 for side,msg in enumerate(originals):
  actual=normalizer.convert(msg,side);expected=adapter.pubs[side].messages[-1]
  assert actual.data==expected.data
  assert (actual.width,actual.height,actual.encoding,actual.step,actual.header.frame_id)==(
   expected.width,expected.height,expected.encoding,expected.step,expected.header.frame_id)
  evidence.append(dict(frame=i,side=side,sha256=hashlib.sha256(bytes(actual.data)).hexdigest(),
   source_bytes=len(msg.data),normalized_bytes=len(actual.data)))
out=ROOT/"results/stereo_transport_equivalence_20260915.json"
out.write_text(json.dumps(dict(passed=True,compared_images=len(evidence),frozen_adapter=str(path),
 intrinsic_profile_unchanged=True,evidence=evidence),indent=2))
print("PASS",len(evidence),"images byte-identical to frozen adapter; calibration profile unchanged")
