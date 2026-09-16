"""Deferred motion-integrity regressions; no images or true poses are fabricated as inputs to deployment."""
import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from t3_lidar_visual_fusion.motion_consistency import MotionConsistency, increment_covariance
from t3_lidar_visual_fusion.core import PoseBuffer


def pose(x=0.,y=0.,yaw=0.):
    p=np.eye(4);p[:3,3]=[x,y,0.];p[:3,:3]=Rotation.from_euler("z",yaw).as_matrix();return p


class ConsistencyTests(unittest.TestCase):
    def test_independent_world_origins_do_not_cause_rejection(self):
        check=MotionConsistency(min_span=.1)
        world_change=pose(100.,-300.,1.2)
        for i in range(8):
            actual=pose(i*.05,yaw=i*.01)
            result=check.check(i*.1,world_change@actual,actual)
        self.assertTrue(result.ready)

    def test_wrong_direction_with_equal_motion_magnitude_is_rejected(self):
        check=MotionConsistency(min_span=.1)
        check.check(0,pose(),pose())
        result=check.check(1,pose(-.5),pose(.5))
        self.assertEqual(result.reason,"visual_lidar_disagreement")

    def test_small_repeated_error_is_caught_over_multiple_frames(self):
        check=MotionConsistency(min_span=.5,translation_floor=.15,translation_ratio=.1)
        reasons=[]
        for i in range(12):
            reasons.append(check.check(i*.1,pose(i*.045),pose(i*.015)).reason)
        self.assertIn("visual_lidar_disagreement",reasons)

    def test_yaw_ambiguity_rejected(self):
        check=MotionConsistency(min_span=.1)
        check.check(0,pose(),pose())
        self.assertEqual(check.check(1,pose(yaw=.6),pose()).reason,"visual_lidar_disagreement")

    def test_exact_lidar_sample_survives_a_large_adjacent_gap(self):
        buf=PoseBuffer();buf.append(1.,pose(0));buf.append(5.,pose(1))
        np.testing.assert_allclose(buf.at(5.,max_gap=.2),pose(1))
        self.assertIsNone(buf.at(3.,max_gap=.2))

    def test_uncertain_lidar_lateral_motion_does_not_veto_good_vision(self):
        cov=np.diag([1e-4,.04,1e-4,1e-5,1e-5,1e-5])
        check=MotionConsistency(min_span=.1)
        check.check(0.,pose(),pose(),cov)
        result=check.check(1.,pose(.1),pose(.1,.5),cov)
        self.assertTrue(result.ready)
        self.assertAlmostEqual(result.translation_error,.5)
        # The same disagreement in a measured direction is still rejected.
        check.reset();check.check(0.,pose(),pose(),cov)
        result=check.check(1.,pose(.6),pose(.1),cov)
        self.assertEqual(result.reason,"visual_lidar_disagreement")

    def test_directional_margin_is_invariant_to_map_rotation(self):
        world=pose(100.,-300.,1.1)
        rotate=np.zeros((6,6));rotate[:3,:3]=world[:3,:3];rotate[3:,3:]=world[:3,:3]
        cov=rotate@np.diag([1e-4,.04,1e-4,1e-5,1e-5,1e-5])@rotate.T
        check=MotionConsistency(min_span=.1)
        check.check(0.,pose(),world,cov)
        self.assertTrue(check.check(1.,pose(.1),world@pose(.1,.5),cov).ready)
        check.reset();check.check(0.,pose(),world,cov)
        self.assertEqual(check.check(1.,pose(.6),world@pose(.1),cov).reason,"visual_lidar_disagreement")

    def test_orientation_uncertainty_propagates_to_relative_translation(self):
        cov=np.diag([.01]*3+[.001,.001,.04])
        result=increment_covariance(pose(),pose(2.),cov,cov)
        self.assertAlmostEqual(result[1,1],.02+4*.04)
        self.assertGreaterEqual(np.linalg.eigvalsh(result)[0],0.)


if __name__=="__main__":unittest.main()
