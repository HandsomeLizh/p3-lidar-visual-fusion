#!/usr/bin/env python3
"""Use the existing P3 display implementation; adapt fusion diagnostic fields."""
import json
import sys
from pathlib import Path
P3 = Path("/home/yanfa/P3/roma_t3_algorithm_bundle_20260825")
sys.path.insert(0, str(P3 / "integration_demo"))
import visual_monitor

def timing(self, message):
    data = json.loads(message.data)
    last = data.get("last", {})
    elapsed = last.get("processing_sec", data.get("processing_sec"))
    with self.lock:
        self.timing = dict(pose_processing_sec=elapsed,
                          outcome=("completed" if last.get("tracking_valid") else last.get("reason", data.get("state", ""))),
                          visual_reason=last.get("reason", data.get("state", "")))

original_subscription = visual_monitor.Monitor.create_subscription
def subscription(self, message_type, topic, callback, qos, *args, **kwargs):
    if topic == "/T3/mapping/lidar_status":
        from rclpy.qos import QoSProfile, ReliabilityPolicy
        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE)
    return original_subscription(self, message_type, topic, callback, qos, *args, **kwargs)
visual_monitor.Monitor.create_subscription = subscription

def cloud_status(self, message):
    data = json.loads(message.data)
    data["voxel_count"] = data.get("map_points", data.get("voxel_count", 0))
    with self.lock:
        self.cloud_status = data

visual_monitor.Monitor.on_status = cloud_status
visual_monitor.Monitor.on_timing = timing
if __name__ == "__main__":
    visual_monitor.main()
