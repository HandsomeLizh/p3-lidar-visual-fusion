"""Only the configured body volume is suppressed; nearby ground stays measured."""
import unittest
import numpy as np
from t3_lidar_visual_fusion.self_filter import SelfFilter


class SelfFilterTests(unittest.TestCase):
    def test_body_boundary_and_external_surfaces(self):
        f=SelfFilter(True,[-1.47,-1.,.13],[.53,1.,1.03])
        p=np.array([[-.4,.2,.7],[-1.47,-1.,.13],[.53,1.,1.03],
            [.55,0.,.5],[-1.49,0.,.5],[0.,1.01,.5],[-.4,.2,-.4],[-.4,.2,1.05]])
        np.testing.assert_equal(f.keep_mask(p),[False,False,False,True,True,True,True,True])
        self.assertEqual(f.keep_mask(np.empty((0,3))).shape,(0,))

    def test_disabled_legacy_profile(self):
        np.testing.assert_equal(SelfFilter().keep_mask([[0.,0.,.5],[2.,0.,1.]]),[True,True])

    def test_reject_bad_bounds(self):
        for lo,hi in [(None,None),([0.,0.],[1.,1.]),([0.,0.,0.],[0.,1.,1.]),
                      ([0.,0.,0.],[1.,np.inf,1.]),([0.,np.nan,0.],[1.,1.,1.])]:
            with self.subTest(lo=lo,hi=hi),self.assertRaises(ValueError):SelfFilter(True,lo,hi)

if __name__=='__main__':unittest.main()
