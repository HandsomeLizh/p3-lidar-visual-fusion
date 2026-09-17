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

    def test_uncropped_support_restores_cropped_terrain_without_adding_points(self):
        ground=self.scene()
        # Nearby ground is outside the camera crop. The view contains only a
        # farther measured surface, a low rock, a ceiling and a distant point.
        view=np.array([[4.,0.,.48],[4.5,0.,1.14],[4.,0.,2.5],[9.,0.,1.08]])
        full=np.r_[ground,view];before=full.copy()
        selector=GroundClearance(enabled=True,clearance_m=1.2)
        self.assertEqual(len(selector.select(view,[0,0,.65],stamp=1.)),0)
        selected=selector.select(view,[0,0,.65],stamp=1.,support_points=full)
        np.testing.assert_equal(selected,[0,1])
        np.testing.assert_equal(full,before)
        self.assertEqual(selector.stats['input_points'],len(view))
        self.assertEqual(selector.stats['ground_support_points'],len(full))
        np.testing.assert_allclose(selector.plane[:2],[.12,.03],atol=1e-5)

    def test_sparse_support_and_overhead_returns_do_not_open_ground_gate(self):
        selector=GroundClearance(enabled=True,clearance_m=1.2)
        view=np.array([[4.,0.,0.]])
        sparse=np.array([[1.,0.,0.],[1.,1.,0.]])
        self.assertEqual(len(selector.select(view,[0,0,.65],support_points=sparse)),0)
        overhead=self.scene()+[0.,0.,3.]
        self.assertEqual(len(selector.select(view,[0,0,.65],support_points=overhead)),0)
        self.assertIsNone(selector.plane)


if __name__=='__main__':unittest.main()
