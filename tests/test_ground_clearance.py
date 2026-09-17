import unittest
import numpy as np
from t3_lidar_visual_fusion.ground_clearance import GroundClearance


class GroundTests(unittest.TestCase):
    def scene(self):
        x,y=np.meshgrid(np.linspace(-3,3,25),np.linspace(-3,3,25))
        return np.c_[x.ravel(),y.ravel(),(.12*x+.03*y).ravel()]

    def test_slope_and_rock_remain_ceiling_excluded_and_input_unchanged(self):
        ground=self.scene();points=np.r_[ground,[[1,0,.72],[1,0,2.5]]];before=points.copy()
        selector=GroundClearance(enabled=True,clearance_m=1.2)
        selected=selector.select(points,[0,0,.65],stamp=1.)
        np.testing.assert_equal(points,before)
        self.assertIn(len(ground),selected);self.assertNotIn(len(ground)+1,selected)
        np.testing.assert_allclose(selector.plane[:2],[.12,.03],atol=1e-5)
        self.assertEqual(len(selector.select_from_reference(points,[0,0,.65],1.1)),len(points)-1)
        self.assertEqual(len(selector.select_from_reference(points,[0,0,.65],4.)),0)
        self.assertEqual(len(selector.select_from_reference(points,[5,0,.65],1.2)),0)
        # A valid reference rejecting every point must not permit fitting a
        # replacement plane through overhead returns.
        self.assertTrue(selector.reference_available([0,0,.65],1.1))
        self.assertEqual(len(selector.select_from_reference([[1,0,2.5]],[0,0,.65],1.1)),0)
        self.assertFalse(selector.reference_available([0,0,.65],4.))

    def test_no_ground_cannot_fabricate_terrain(self):
        selector=GroundClearance(enabled=True)
        self.assertEqual(len(selector.select(np.array([[1.,0.,2.]]),[0,0,.65],stamp=1.)),0)
        self.assertIsNone(selector.plane)
        self.assertEqual(len(selector.select_from_reference([[1,0,0]],[0,0,.65],1.)),0)


if __name__=='__main__':unittest.main()
