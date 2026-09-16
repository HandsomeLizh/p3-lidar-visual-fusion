"""Bad sensor packets and delayed sources must not corrupt map freshness."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
import rclpy
import yaml
from nav_msgs.msg import Odometry
from std_msgs.msg import Header
from t3_lidar_visual_fusion.terrain_mapper import TerrainMapper
from t3_lidar_visual_fusion.ros_utils import cloud_arrays,xyz_cloud,stamp_sec

ROOT=Path(__file__).resolve().parents[1]


class MappingFaultTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(dir=ROOT/'build')
        directory=Path(self.tmp.name)
        cfg=yaml.safe_load((ROOT/'config/hardware104.yaml').read_text())
        cfg.update(mapping_pose_settle_sec=0.,map_window=8.,tile_cells=16,
                   min_range=.1,base_from_lidar=np.eye(4).tolist(),
                   instantaneous_cloud=True,deskew={'enabled':False})
        profile=directory/'profile.yaml';profile.write_text(yaml.safe_dump(cfg))
        rclpy.init(args=['--ros-args','-p','profile_path:='+str(profile),
                        '-p','output_dir:='+str(directory/'map')])
        self.node=TerrainMapper()
        self.pose=Odometry();self.pose.header=Header(frame_id='odom')
        self.pose.header.stamp.sec=10;self.pose.child_frame_id='base_link'
        self.pose.pose.pose.orientation.w=1.
        self.pose.pose.covariance=(np.eye(6)*.001).ravel().tolist()
        self.node.odom(self.pose)

    def tearDown(self):
        n=self.node
        n.dense_writer.close(flush_revision=n.last_dense_revision if n.last_dense_revision>=0 else None)
        n.grid.close();n.delivery.close();n.cloud.close();n.tum.close()
        n.destroy_node();rclpy.shutdown();self.tmp.cleanup()

    def cloud(self,t=10.,points=((1.,0.,0.),)):
        h=Header(frame_id='lidar');h.stamp.sec,h.stamp.nanosec=divmod(round(t*1e9),10**9)
        return xyz_cloud(points,h)

    def process(self,msg):
        self.node.enqueue(msg,'lidar',np.eye(4));self.node.process()

    def test_malformed_field_then_valid_scan_keeps_mapper_alive(self):
        bad=self.cloud();bad.fields[0].datatype=99
        with self.assertRaisesRegex(ValueError,'datatype'):cloud_arrays(bad)
        self.process(bad)
        self.assertEqual(self.node.stats['dropped_scans'],1)
        self.assertIsNone(self.node.last_header)
        self.process(self.cloud(10.05))
        self.assertEqual(self.node.stats['mapped_scans'],1)

    def test_truncated_stride_and_field_overflow_are_rejected(self):
        msg=self.cloud();msg.data.pop()
        with self.assertRaisesRegex(ValueError,'Truncated'):cloud_arrays(msg)
        msg=self.cloud();msg.fields[2].offset=msg.point_step
        with self.assertRaisesRegex(ValueError,'exceeds'):cloud_arrays(msg)

    def test_missing_raw_scan_time_does_not_escape_timer(self):
        self.node.cfg['instantaneous_cloud']=False
        self.process(self.cloud())
        self.assertEqual(self.node.stats['dropped_scans'],1)
        self.assertFalse(self.node.pending)
        self.node.cfg['instantaneous_cloud']=True
        self.process(self.cloud(10.05))
        self.assertEqual(self.node.stats['mapped_scans'],1)

    def test_empty_or_nan_scan_cannot_refresh_map(self):
        self.process(self.cloud())
        before=stamp_sec(self.node.last_header)
        self.process(self.cloud(10.02,points=()))
        self.process(self.cloud(10.04,points=((np.nan,0.,0.),)))
        self.assertEqual(self.node.stats['mapped_scans'],1)
        self.assertEqual(stamp_sec(self.node.last_header),before)

    def test_invalid_pose_does_not_replace_valid_pose_or_quality(self):
        for frame,cov in [('camera',np.eye(6)),('odom',-np.eye(6)),
                          ('odom',np.full((6,6),np.nan))]:
            msg=copy.deepcopy(self.pose);msg.header.frame_id=frame
            msg.pose.pose.position.x=5.;msg.pose.covariance=cov.ravel().tolist()
            self.node.odom(msg)
        self.assertEqual(self.node.stats['invalid_poses'],3)
        np.testing.assert_array_equal(self.node.last_pose,np.eye(4))
        np.testing.assert_array_equal(self.node.pose_quality_at(10.)[3],np.eye(6)*.001)

    def test_pose_and_quality_snapshot_survive_later_revision(self):
        self.node.enqueue(self.cloud(),'lidar',np.eye(4))
        sample=self.node.take_ready(self.node.pending)
        msg=copy.deepcopy(self.pose);msg.pose.pose.position.x=2.
        msg.pose.covariance=(np.eye(6)*.02).ravel().tolist();self.node.odom(msg)
        self.assertEqual(sample[-2][0,3],0.)
        self.assertEqual(sample[-1][1],.001)
        self.assertEqual(self.node.pose_quality_at(10.)[1],.02)

    def test_delayed_second_source_never_rewinds_map_stamp(self):
        self.process(self.cloud())
        self.node.enqueue(self.cloud(9.95),'tof',np.eye(4));self.node.process()
        self.assertEqual(self.node.stats['mapped_scans'],2)
        self.assertEqual(stamp_sec(self.node.last_header),10.)
        self.node.report_status()
        status=json.loads((self.node.output/'map_statistics.json').read_text())
        self.assertEqual(status['last_map_stamp'],10.)
        self.assertGreaterEqual(status['pose_receipt_idle_sec'],0.)
        self.assertIn('lidar',status['sensor_receipt_idle_sec'])
        self.assertIsNotNone(status['map_age_sec'])


if __name__=='__main__':unittest.main()
