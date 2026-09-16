"""Exercise real ROS messages when the map exceeds configured memory bounds."""
import tempfile
import unittest
from pathlib import Path
import numpy as np
import rclpy
import yaml
from std_msgs.msg import Header
from t3_lidar_visual_fusion.terrain_mapper import TerrainMapper

ROOT=Path(__file__).resolve().parents[1]

class MapLimitTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix="map_limit_",dir=ROOT/"build")
        path=Path(self.tmp.name)
        cfg=yaml.safe_load((ROOT/"config/bag_20260824_lidar.yaml").read_text())
        cfg.update(global_max_cells=1,map_window=4.0,semantic_topic="")
        self.profile=path/"profile.yaml"
        self.profile.write_text(yaml.safe_dump(cfg))
        rclpy.init(args=["--ros-args","-p","profile_path:="+str(self.profile),
                        "-p","output_dir:="+str(path/"output")])
        self.node=TerrainMapper()
        self.node.last_header=Header(frame_id="map")
        self.node.last_header.stamp.sec=1
        self.node.last_pose=np.eye(4)
        xx,yy=np.meshgrid(np.arange(0,2,.2),np.arange(0,2,.2))
        pts=np.column_stack([xx.ravel(),yy.ravel(),np.zeros(xx.size)])
        self.node.grid.update_elevation_only(points_map=pts)
        self.node.overview.update(pts)
        self.node.dirty=True

    def tearDown(self):
        n=self.node
        n.grid.close();n.delivery.close();n.cloud.close();n.tum.close()
        n.destroy_node()
        rclpy.shutdown()
        self.tmp.cleanup()

    def test_extent_limit_keeps_local_mapping_alive(self):
        self.node.global_available=True
        self.node.publish()
        self.assertFalse(self.node.global_available)
        self.assertIsNotNone(self.node.latest_window)
        self.assertTrue(np.isfinite(self.node.latest_window[0].elevation).any())
        # A later update remains publishable after the limit was hit.
        self.node.dirty=True
        self.node.publish()
        self.node.save()

    def test_memory_pressure_invalidates_global_without_crash(self):
        self.node.pressure=True
        self.node.global_available=True
        self.node.publish_global()
        self.assertFalse(self.node.global_available)
        self.node.pressure=False
        self.node.cfg["global_max_cells"]=1000000
        self.node.publish_global()
        self.assertTrue(self.node.global_available)

if __name__=="__main__":
    unittest.main()
