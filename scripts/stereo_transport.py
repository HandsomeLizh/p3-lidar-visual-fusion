"""Normalize the exact stereo pixels used by the current visual adapter."""
import copy
import cv2
import numpy as np
from cv_bridge import CvBridge
class StereoNormalizer:
 def __init__(self,profile):self.cfg=profile;self.bridge=CvBridge()
 def convert(self,msg,side):
  if (msg.width,msg.height)!=tuple(self.cfg["input_image_size"]):raise ValueError("Image dimensions differ from calibration")
  if len(msg.data)>self.cfg.get("max_image_bytes",24000000):raise ValueError("Image payload exceeds limit")
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
  return out
