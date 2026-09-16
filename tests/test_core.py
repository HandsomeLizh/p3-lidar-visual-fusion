import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from t3_lidar_visual_fusion.core import VisionGate,PoseBuffer
from t3_lidar_visual_fusion.ros_utils import cloud_arrays
from sensor_msgs.msg import PointCloud2,PointField


def pose(x=0.,yaw=0.):
    p=np.eye(4);p[0,3]=x;p[:3,:3]=Rotation.from_euler("z",yaw).as_matrix();return p


class GuardTests(unittest.TestCase):
    def test_blackout_reset_does_not_bridge_motion(self):
        g=VisionGate(recovery_frames=3,max_gap=.5)
        for i in range(5): r=g.accept(i*.1,pose(i*.05))
        before=r.transform.copy()
        self.assertTrue(g.enabled)
        self.assertIsNone(g.accept(.5,pose(500),False,"dark").transform)
        for i in range(3):r=g.accept(.6+i*.1,pose(100+i*.05,1.2))
        self.assertTrue(r.reseed)
        np.testing.assert_allclose(r.transform,before)
        r=g.accept(.9,pose(100.15,1.2))
        self.assertLess(np.linalg.norm(r.transform[:3,3]-before[:3,3]),.06)

    def test_jump_nan_and_time_reversal_close_gate(self):
        for bad_stamp,bad_pose in [(.2,pose(100)),(-.1,pose()),(.2,np.full((4,4),np.nan))]:
            g=VisionGate(recovery_frames=2)
            g.accept(0.,pose());g.accept(.1,pose(.01))
            r=g.accept(bad_stamp,bad_pose)
            self.assertIsNone(r.transform);self.assertFalse(g.enabled)

    def test_timeout_requires_recovery(self):
        g=VisionGate(recovery_frames=3,max_gap=.5)
        for i in range(4):g.accept(i*.1,pose(i*.01))
        self.assertIsNone(g.accept(2.,pose(.2)).transform)
        self.assertFalse(g.enabled)
        self.assertIsNone(g.accept(2.1,pose(.21)).transform)
        self.assertTrue(g.accept(2.2,pose(.22)).reseed)


class PoseTests(unittest.TestCase):
    def test_interpolation_rotates_across_wrap(self):
        b=PoseBuffer();b.append(1,pose(0,np.deg2rad(179)));b.append(2,pose(2,np.deg2rad(-179)))
        m=b.at(1.5,max_gap=2)
        self.assertAlmostEqual(m[0,3],1)
        self.assertAlmostEqual(abs(Rotation.from_matrix(m[:3,:3]).as_euler("xyz")[2]),np.pi)
        self.assertIsNone(b.at(3,tolerance=.1))
        self.assertFalse(b.append(1.1,pose()))

    def test_large_gap_not_interpolated(self):
        b=PoseBuffer();b.append(0,pose());b.append(10,pose(5))
        self.assertIsNone(b.at(5,max_gap=.5))


class CloudTests(unittest.TestCase):
    def test_big_endian_row_padding(self):
        m=PointCloud2();m.height=2;m.width=2;m.point_step=16;m.row_step=40;m.is_bigendian=True
        m.fields=[PointField(name=n,offset=i*4,datatype=7,count=1) for i,n in enumerate(["x","y","z"])]
        raw=bytearray(80)
        for r in range(2):
            for c in range(2):
                off=r*40+c*16;raw[off:off+12]=np.array([r,c,r+c],dtype=">f4").tobytes()
        m.data=bytes(raw)
        np.testing.assert_array_equal(cloud_arrays(m),[[0,0,0],[0,1,1],[1,0,1],[1,1,2]])
        m.data=bytes(raw[:40])
        with self.assertRaises(ValueError):cloud_arrays(m)


if __name__=="__main__":unittest.main()
