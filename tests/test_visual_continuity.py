import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from t3_lidar_visual_fusion.visual_continuity import VisualContinuity


def pose(x=0.,yaw=0.):
    t=np.eye(4);t[:3,:3]=Rotation.from_euler('z',yaw).as_matrix();t[0,3]=x;return t


class VisualContinuityTests(unittest.TestCase):
    def test_hundred_frames_without_lidar_retain_reference_frame_and_uncertainty(self):
        track=VisualContinuity();cov=np.eye(6)*.0001
        track.observe(10.,'epoch_a',pose(),cov)
        anchor=pose(8.,.6);self.assertTrue(track.anchor(10.,anchor,cov))
        for i in range(1,151):
            t=10.+i*.2;raw=pose(i*.01,i*.001)
            track.observe(t,'epoch_a',raw,cov)
            stamp,estimated,uncertainty=track.estimate()
            self.assertEqual(stamp,t);np.testing.assert_allclose(estimated,anchor@raw,atol=1e-10)
            self.assertGreater(np.linalg.eigvalsh(uncertainty)[0],.0001)
        self.assertEqual(len(track.raw),128)
        self.assertLess(np.linalg.eigvalsh(uncertainty[:3,:3])[-1],.1)

    def test_epoch_change_and_gap_require_new_cotimed_reference(self):
        track=VisualContinuity();cov=np.eye(6)*.01
        track.observe(1.,'a',pose(),cov);track.anchor(1.,pose(10),cov)
        track.observe(2.,'b',pose(),cov)
        self.assertIsNone(track.estimate());self.assertFalse(track.anchor(1.,pose(10),cov))
        self.assertTrue(track.anchor(2.,pose(11),cov))
        track.observe(20.,'b',pose(.2),cov)
        self.assertIsNone(track.estimate())

    def test_delayed_reference_and_invalid_inputs(self):
        track=VisualContinuity();cov=np.eye(6)*.01
        track.observe(1.,'a',pose(),cov);track.observe(2.,'a',pose(.2),cov)
        self.assertTrue(track.anchor(1.,pose(5),cov))
        np.testing.assert_allclose(track.estimate()[1][:3,3],[5.2,0,0])
        self.assertFalse(track.anchor(.5,pose(),cov))
        with self.assertRaises(ValueError):track.observe(2.,'a',pose(),cov)
        with self.assertRaises(ValueError):track.observe(3.,'a',pose(),cov*np.nan)


if __name__=='__main__':unittest.main()
