import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from t3_lidar_visual_fusion.visual_continuity import VisualContinuity


def pose(x=0.,yaw=0.):
    t=np.eye(4);t[:3,:3]=Rotation.from_euler('z',yaw).as_matrix();t[0,3]=x;return t


class VisualContinuityTests(unittest.TestCase):
    def test_origin_requires_tracked_motion_and_cotimed_qualified_reference(self):
        track=VisualContinuity();cov=np.eye(6)*.001
        track.remember_origin(10.,'a',pose())
        self.assertIsNone(track.estimate())
        self.assertFalse(track.anchor(10.,pose(8.),cov))
        track.observe(11.,'a',pose(.1),cov)
        self.assertIsNone(track.estimate())
        self.assertFalse(track.anchor(9.,pose(8.),cov))
        self.assertTrue(track.anchor(10.,pose(8.),cov))
        _,estimated,uncertainty=track.estimate()
        np.testing.assert_allclose(estimated,pose(8.1),atol=1e-10)
        self.assertGreater(np.linalg.eigvalsh(uncertainty)[0],.001)
        for i in range(2,151):track.observe(10.+i,'a',pose(i*.1),cov)
        np.testing.assert_allclose(track.estimate()[1],pose(23.),atol=1e-10)
        self.assertLessEqual(len(track.raw),128)

    def test_delayed_origin_event_and_epoch_gap_do_not_bridge_unobserved_motion(self):
        track=VisualContinuity();cov=np.eye(6)*.001
        track.observe(11.,'a',pose(.1),cov)
        track.remember_origin(10.,'a',pose())
        self.assertTrue(track.anchor(10.,pose(8.),cov))
        track.observe(12.,'b',pose(.2),cov)
        self.assertFalse(track.anchor(10.,pose(8.),cov))
        track.remember_origin(12.5,'b',pose())
        track.observe(30.,'b',pose(.3),cov)
        self.assertFalse(track.anchor(12.5,pose(8.),cov))
        self.assertIsNone(track.estimate())
        with self.assertRaises(ValueError):track.remember_origin(31.,'b',pose(1.))
        with self.assertRaises(ValueError):track.remember_origin(31.,'odom',pose())

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

    def test_weak_reference_does_not_destroy_usable_visual_fallback(self):
        track=VisualContinuity();cov=np.eye(6)*.0001
        track.observe(1.,'a',pose(),cov);track.anchor(1.,pose(10.),cov)
        track.observe(2.,'a',pose(.1),cov)
        before=track.estimate()
        self.assertFalse(track.anchor(2.,pose(10.1),np.eye(6)*3.9))
        after=track.estimate()
        np.testing.assert_allclose(after[1],before[1]);np.testing.assert_allclose(after[2],before[2])
        quality=track.status()['estimate_quality']
        self.assertEqual(quality['reference_stamp_sec'],1.)
        self.assertAlmostEqual(quality['position_variance'],np.linalg.eigvalsh(before[2][:3,:3])[-1])
        self.assertAlmostEqual(quality['rotation_variance'],np.linalg.eigvalsh(before[2][3:,3:])[-1])
        self.assertTrue(track.anchor(2.,pose(10.1),cov*.5))


if __name__=='__main__':unittest.main()
