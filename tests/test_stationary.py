"""Independent image, range, and motion evidence must agree before zero motion."""
import unittest
from unittest.mock import patch
import cv2
import numpy as np
from t3_lidar_visual_fusion.stationary import StationaryDetector


class StationaryTests(unittest.TestCase):
    def setUp(self):
        self.detector=StationaryDetector()
        rng=np.random.default_rng(741)
        self.image=cv2.GaussianBlur(rng.integers(30,220,(256,320),np.uint8),(3,3),0)
        az,el=np.meshgrid(np.linspace(-np.pi,np.pi,128,endpoint=False),np.linspace(-.4,.4,16))
        self.points=np.column_stack([3*np.cos(el.ravel())*np.cos(az.ravel()),
                                     3*np.cos(el.ravel())*np.sin(az.ravel()),3*np.sin(el.ravel())])
        self.stamp=100.

    def frame(self,shift=0.,right_shift=None,velocity=None,valid=True,points=None,blank=False):
        self.stamp+=1.
        for side,s in enumerate([shift,shift if right_shift is None else right_shift]):
            image=cv2.warpAffine(self.image,np.float32([[1,0,s],[0,1,0]]),(320,256),borderMode=cv2.BORDER_REFLECT)
            if blank:image[:]=0
            self.detector.image(self.stamp,image,side,valid)
        self.detector.cloud(self.stamp,self.points if points is None else points)
        return self.detector.check(self.stamp,np.zeros(6) if velocity is None else np.asarray(velocity))

    def confirm(self):
        result=[self.frame() for _ in range(4)]
        self.assertEqual(result,[False,False,False,True])

    def test_static_requires_consecutive_stereo_cloud_and_motion(self):
        self.confirm()
        self.assertEqual(self.detector.status()['state'],'stationary')

    def test_subpixel_creep_accumulates_against_image_anchor(self):
        decisions=[self.frame(shift=i*.12) for i in range(12)]
        self.assertFalse(any(decisions),decisions)

    def test_slow_start_and_slow_turn_release_immediately(self):
        for velocity in [[.004,0,0,0,0,0],[0,0,0,0,0,.003]]:
            self.detector=StationaryDetector();self.confirm()
            self.assertFalse(self.frame(velocity=velocity))
            self.assertEqual(self.detector.status()['state'],'moving')

    def test_one_moving_camera_releases_even_with_zero_visual_estimate(self):
        self.confirm();self.assertFalse(self.frame(right_shift=.75))

    def test_blackout_saturation_and_texture_loss_are_unknown(self):
        for args in [dict(valid=False),dict(blank=True)]:
            self.detector=StationaryDetector();self.confirm()
            self.assertFalse(self.frame(**args));self.assertEqual(self.detector.status()['state'],'unknown')

    def test_changed_lidar_cannot_be_overridden_by_frozen_images(self):
        self.confirm();self.assertFalse(self.frame(points=self.points*1.1))

    def test_missing_or_stale_lidar_does_not_freeze(self):
        self.confirm();self.detector.cloud_evidence.clear()
        self.assertFalse(self.detector.check(self.stamp+1,np.zeros(6)))
        self.assertEqual(self.detector.status()['state'],'unknown')

    def test_expired_stationary_status_is_unknown(self):
        self.confirm()
        with patch('t3_lidar_visual_fusion.stationary.time.monotonic',return_value=self.detector.last_wall+4):
            self.assertEqual(self.detector.status()['state'],'unknown')

    def test_nonfinite_motion_is_never_stationary(self):
        self.confirm();self.assertFalse(self.frame(velocity=[np.nan,0,0,0,0,0]))


if __name__=='__main__':unittest.main()
