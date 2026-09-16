"""Deferred camera geometry regressions. Synthetic geometry is not an accuracy benchmark."""
import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from t3_lidar_visual_fusion.stereo_geometry import StereoGeometry,TrackingFailure,project
from t3_lidar_visual_fusion.core import inverse
from t3_lidar_visual_fusion.learned_tracker import LearnedStereoTracker


def profile():
    left=np.eye(4);right=np.eye(4);right[0,3]=.25
    return dict(input_image_size=[640,480],output_image_size=[640,480],
        camera_k=[[500.,0.,320.],[0.,500.,240.],[0.,0.,1.]],
        distortion=[0.,0.,0.,0.],base_from_camera_left=left,base_from_camera_right=right,
        learned_visual=dict(max_image_side=640,min_pnp_inliers=25,min_pnp_ratio=.45,
            min_pnp_coverage=.15,min_stereo_checks=12,stereo_motion_depth_ratio=.01))


def scene():
    rng=np.random.default_rng(11)
    return np.column_stack((rng.uniform(-2.,2.,120),rng.uniform(-1.5,1.5,120),rng.uniform(4.,8.,120)))


class StereoGeometryTests(unittest.TestCase):
    def test_triangulation_keeps_metric_scale(self):
        geometry=StereoGeometry(profile());xyz=scene()
        left,_=project(geometry.p0,xyz);right,_=project(geometry.p1,xyz)
        pairs=np.column_stack((np.arange(len(xyz)),np.arange(len(xyz))))
        actual,quality=geometry.triangulate(left,right,pairs)
        self.assertEqual(quality["stereo_points"],len(xyz))
        np.testing.assert_allclose(actual,xyz,atol=1e-6)

    def test_wrong_disparity_and_epipolar_matches_are_removed(self):
        geometry=StereoGeometry(profile());xyz=scene()
        left,_=project(geometry.p0,xyz);right,_=project(geometry.p1,xyz)
        right[:15,0]=left[:15,0]+5
        right[15:30,1]+=10
        pairs=np.column_stack((np.arange(len(xyz)),np.arange(len(xyz))))
        actual,_=geometry.triangulate(left,right,pairs)
        self.assertFalse(np.isfinite(actual[:30]).any())
        self.assertTrue(np.isfinite(actual[30:]).all())

    def test_pnp_transform_direction_with_mismatched_features(self):
        geometry=StereoGeometry(profile());reference=scene()
        current_from_reference=np.eye(4)
        current_from_reference[:3,:3]=Rotation.from_euler("y",.025).as_matrix()
        current_from_reference[:3,3]=[-.15,.02,.03]
        current=reference@current_from_reference[:3,:3].T+current_from_reference[:3,3]
        old_uv,_=project(geometry.p0,reference);new_uv,_=project(geometry.p0,current)
        pairs=np.column_stack((np.arange(len(reference)),np.arange(len(reference))))
        pairs[:20,1]=np.roll(pairs[:20,1],7)
        estimate=geometry.estimate(reference,current,old_uv,new_uv,pairs)
        np.testing.assert_allclose(estimate.current_from_reference,current_from_reference,atol=1e-5)
        # World pose is composed with inverse(current_from_reference).
        np.testing.assert_allclose(inverse(estimate.current_from_reference)@current_from_reference,np.eye(4),atol=1e-5)

    def test_pnp_2d_fit_does_not_override_wrong_current_stereo_depth(self):
        geometry=StereoGeometry(profile());reference=scene()
        uv,_=project(geometry.p0,reference)
        pairs=np.column_stack((np.arange(len(reference)),np.arange(len(reference))))
        with self.assertRaisesRegex(TrackingFailure,"stereo_motion_disagreement"):
            geometry.estimate(reference,reference*1.5,uv,uv,pairs)

    def test_body_extrinsic_is_not_applied_twice(self):
        config=profile()
        mount=np.eye(4);mount[:3,:3]=Rotation.from_euler("xyz",[.2,.1,-.4]).as_matrix()
        mount[:3,3]=[.5,.1,.8]
        config["base_from_camera_left"]=mount
        right=np.eye(4);right[0,3]=.25
        config["base_from_camera_right"]=mount@right
        geometry=StereoGeometry(config)
        expected_base=np.eye(4)
        expected_base[:3,:3]=Rotation.from_euler("z",.03).as_matrix()
        expected_base[:3,3]=[.1,.02,-.01]
        current_from_reference=inverse(expected_base@geometry.base_from_rect)@geometry.base_from_rect
        reference=scene()
        current=reference@current_from_reference[:3,:3].T+current_from_reference[:3,3]
        fixtures=[project(p,xyz)[0] for xyz in [reference,current] for p in [geometry.p0,geometry.p1]]
        class KnownCorrespondences:
            def __init__(self):self.index=0
            def extract(self,image):
                result={"pixels":fixtures[self.index]};self.index+=1;return result
            def match(self,first,second):
                ids=np.arange(len(first["pixels"]));return np.column_stack((ids,ids))
        tracker=LearnedStereoTracker(config,KnownCorrespondences())
        blank=np.zeros((480,640),dtype=np.uint8)
        first=tracker.process(100.,blank,blank)
        result=tracker.process(100.1,blank,blank)
        self.assertTrue(first.anchor)
        self.assertFalse(result.anchor)
        np.testing.assert_allclose(result.base_pose,expected_base,atol=1e-5)


if __name__=="__main__":unittest.main()
