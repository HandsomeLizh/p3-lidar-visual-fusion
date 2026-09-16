#!/usr/bin/env python3
"""Check selected ROS domain for an already-running primary mapper."""
import time
import rclpy
from rclpy.node import Node


def check(capture=False):
    rclpy.init()
    node=Node("fusion_start_preflight")
    try:
        deadline=time.monotonic()+2.
        while time.monotonic()<deadline:rclpy.spin_once(node,timeout_sec=.1)
        outputs=["/T3/semantic/current_pose","/Car/T3/localization/odometry"]
        if capture:outputs += ["/Car/T5/OS1/points","/Car/T5/Cam_Left/image_raw/color"]
        conflicts={t:[p.node_name for p in node.get_publishers_info_by_topic(t)] for t in outputs}
        conflicts={t:v for t,v in conflicts.items() if v}
        if conflicts:raise RuntimeError("Publishers already active in this domain: "+str(conflicts)+
            ". Reuse an existing sensor source without --capture; run only one primary mapper in the planner's domain.")
    finally:node.destroy_node();rclpy.shutdown()


if __name__=="__main__":check()
