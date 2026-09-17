"""Deferred camera geometry regressions. Synthetic geometry is not an accuracy benchmark."""
import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from t3_lidar_visual_fusion.stereo_geometry import StereoGeometry,TrackingFailure,project,spatial_density_mask,radial_continuity_weights
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
    def test_radial_gap_cuts_remote_returns_and_fades_near_boundary(self):
        radii=np.array([2.,2.2,2.4,2.6,2.8,4.,4.2])
        angle=np.deg2rad(1.)
        points=np.column_stack([radii*np.cos(angle),radii*np.sin(angle),np.zeros(len(radii))])
        weights=radial_continuity_weights(points,max_gap_m=.25,feather_m=.4)
        np.testing.assert_allclose(weights,[1.,1.,1.,.5,0.,0.,0.],atol=1e-12)
        np.testing.assert_allclose(radial_continuity_weights(points[::-1],feather_m=.4)[::-1],weights)
        # A nearer viewpoint with dense intervening observations admits the
        # formerly remote returns; this is not a fixed maximum-range crop.
        filled=np.r_[radii,np.arange(2.9,4.,.1)]
        connected=np.column_stack([filled*np.cos(angle),filled*np.sin(angle),np.zeros(len(filled))])
        np.testing.assert_allclose(radial_continuity_weights(connected),1.)

    def test_radial_gap_is_directional_and_does_not_start_at_sensor_origin(self):
        def ray(radii,deg):
            radii=np.asarray(radii);a=np.deg2rad(deg)
            return np.column_stack([radii*np.cos(a),radii*np.sin(a),np.zeros(len(radii))])
        first=ray([3.,3.2,4.],1.)
        second=ray(np.arange(3.,4.1,.2),7.)
        only_remote=ray([9.,9.2],13.)
        points=np.r_[first,second,only_remote,[[np.nan,0.,0.]]]
        w=radial_continuity_weights(points,feather_m=0.)
        np.testing.assert_array_equal(w[:3],[1.,1.,0.])
        np.testing.assert_array_equal(w[3:3+len(second)],1.)
        np.testing.assert_array_equal(w[-3:],0.)
        higher=points.copy();higher[:,2]+=2.
        np.testing.assert_array_equal(radial_continuity_weights(higher,feather_m=0.),w)

    def test_radial_gap_rejects_invalid_configuration_and_accepts_empty_scan(self):
        self.assertEqual(len(radial_continuity_weights(np.empty((0,3)))),0)
        for options in [dict(max_gap_m=0.),dict(sector_deg=.01),dict(feather_m=-1.),
                        dict(seed_range_m=np.inf)]:
            with self.assertRaises(ValueError):radial_continuity_weights(np.empty((0,3)),**options)

    def test_soft_camera_boundary_has_smooth_falloff_and_round_corners(self):
        geometry=StereoGeometry(profile())
        pixels=np.array([[600.,240.],[640.,240.],[672.,240.],[704.,240.],[704.,528.]])
        rect=np.column_stack((pixels,np.ones(len(pixels))))@geometry.inverse_k.T*2.
        weights=geometry.mapping_frustum_weights(rect,image_feather_fraction=.1)
        np.testing.assert_allclose(weights,[1.,1.,.5,0.,0.],atol=1e-9)
        # A repeat does not change the sampling pattern or add new data.
        positions=np.column_stack((np.arange(2000)*.2+.05,np.zeros(2000),np.zeros(2000)))
        varying=np.full(2000,.3)
        a=spatial_density_mask(positions,varying,.2)
        b=spatial_density_mask(positions[::-1],varying,.2)[::-1]
        np.testing.assert_array_equal(a,b)
        self.assertLess(abs(a.mean()-.3),.04)
        other_height=positions.copy();other_height[:,2]+=1.
        np.testing.assert_array_equal(a,spatial_density_mask(other_height,varying,.2))
        np.testing.assert_array_equal(spatial_density_mask(positions,np.ones(2000),.2),True)
        np.testing.assert_array_equal(spatial_density_mask(positions,np.zeros(2000),.2),False)

    def test_range_fades_from_ten_to_fifteen_in_body_horizontal_distance(self):
        cfg=profile()
        mount=np.eye(4);mount[:3,:3]=[[0.,0.,1.],[-1.,0.,0.],[0.,-1.,0.]]
        right=mount.copy();right[:3,3]=[0.,-.25,0.]
        cfg['base_from_camera_left']=mount;cfg['base_from_camera_right']=right
        geometry=StereoGeometry(cfg)
        base=np.array([[r,0.,0.] for r in (3.,10.,11.25,12.5,13.75,15.,16.)])
        options=dict(match_stereo_depth_limit=False,full_density_range_m=10.,range_feather_m=5.)
        weights=geometry.mapping_frustum_weights(base,**options)
        np.testing.assert_allclose(weights,[1.,1.,.84375,.5,.15625,0.,0.],atol=1e-9)
        self.assertTrue(np.all(np.diff(weights)<=0.))
        with self.assertRaises(ValueError):geometry.mapping_frustum_weights(base,range_feather_m=5.)
        with self.assertRaises(ValueError):geometry.mapping_frustum_weights(base,full_density_range_m=10.)

    def test_lidar_crop_requires_both_images_and_near_depth(self):
        geometry=StereoGeometry(profile())
        points=np.array([[0.,0.,2.],[-1.2,0.,2.],[0.,1.1,2.],
                         [0.,0.,-2.],[0.,0.,7.],[np.nan,0.,2.]])
        original=points.copy()
        keep=geometry.mapping_frustum_mask(points)
        np.testing.assert_array_equal(keep,[True,False,False,False,False,False])
        # Point 1 is inside the left view, outside the right. No invented points.
        left,_=project(geometry.p0,points[:2]);right,_=project(geometry.p1,points[:2])
        self.assertGreater(left[1,0],0);self.assertLess(right[1,0],0)
        self.assertTrue(geometry.mapping_frustum_mask(points,match_stereo_depth_limit=False)[4])
        np.testing.assert_array_equal(points,original)

    def test_lidar_crop_uses_body_mount_and_rectification(self):
        cfg=profile();mount=np.eye(4)
        mount[:3,:3]=Rotation.from_euler('xyz',[.2,-.4,1.1]).as_matrix()
        mount[:3,3]=[.6,.12,.8]
        baseline=np.eye(4);baseline[:3,3]=[.25,.01,.015]
        baseline[:3,:3]=Rotation.from_euler('y',.03).as_matrix()
        cfg['base_from_camera_left']=mount;cfg['base_from_camera_right']=mount@baseline
        geometry=StereoGeometry(cfg)
        rect=np.array([[0.,0.,2.],[0.,0.,-2.],[3.,0.,2.],[0.,0.,20.]])
        body=rect@geometry.base_from_rect[:3,:3].T+geometry.base_from_rect[:3,3]
        np.testing.assert_array_equal(geometry.mapping_frustum_mask(body),[True,False,False,False])
        self.assertEqual(len(geometry.mapping_frustum_mask(np.empty((0,3)))),0)
        with self.assertRaises(ValueError):geometry.mapping_frustum_mask(body,image_margin_px=240.)

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

    def test_repeated_pixels_with_stereo_vertical_noise_have_zero_motion(self):
        geometry=StereoGeometry(profile());xyz=scene()
        left,_=project(geometry.p0,xyz);right,_=project(geometry.p1,xyz)
        right[:,1]+=.7  # Within the accepted epipolar tolerance.
        pairs=np.column_stack((np.arange(len(xyz)),np.arange(len(xyz))))
        points,_=geometry.triangulate(left,right,pairs)
        estimate=geometry.estimate(points,points,left,left,pairs)
        np.testing.assert_allclose(estimate.current_from_reference,np.eye(4),atol=1e-6)

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
        np.testing.assert_allclose(result.stereo_points,current,atol=1e-5)

    def test_mapping_points_use_rectified_mount_and_reject_uncertain_depth(self):
        config=profile()
        mount=np.eye(4);mount[:3,:3]=Rotation.from_euler('xyz',[.2,.1,-.4]).as_matrix()
        mount[:3,3]=[.5,.1,.8]
        right=np.eye(4);right[:3,:3]=Rotation.from_euler('y',.03).as_matrix()
        right[:3,3]=[.25,.01,.015]
        config['base_from_camera_left']=mount
        config['base_from_camera_right']=mount@right
        geometry=StereoGeometry(config)
        rectified=np.array([[.1,.2,2.],[.2,-.1,3.]])
        points=np.vstack([rectified,[0.,0.,40.],[0.,0.,-2.],[np.nan,0.,2.]])
        actual=geometry.mapping_points(points)
        # Recover raw-camera coordinates from the calibrated rectification, then
        # apply the physical mount. A raw-camera transform alone must differ.
        rect_from_left=inverse(geometry.base_from_rect)@mount
        in_left=rectified@rect_from_left[:3,:3]
        expected=in_left@mount[:3,:3].T+mount[:3,3]
        np.testing.assert_allclose(actual,expected,atol=1e-9)
        self.assertGreater(np.linalg.norm(actual-(rectified@mount[:3,:3].T+mount[:3,3])),.01)

    def test_metric_depth_survives_temporal_failure_but_not_bad_stereo(self):
        config=profile();geometry=StereoGeometry(config);xyz=scene()
        views=[project(p,xyz)[0] for p in (geometry.p0,geometry.p1)]
        class Correspondences:
            def __init__(self):self.index=0;self.matches=0
            def extract(self,image):
                result={'pixels':views[self.index%2]};self.index+=1;return result
            def match(self,first,second):
                self.matches+=1
                # First pair seeds a reference, second pair has metric stereo
                # but no temporal matches, third pair has no stereo matches.
                if self.matches>=3:return np.empty((0,2),dtype=int)
                ids=np.arange(len(xyz));return np.column_stack((ids,ids))
        tracker=LearnedStereoTracker(config,Correspondences())
        blank=np.zeros((480,640),dtype=np.uint8)
        self.assertTrue(tracker.process(100.,blank,blank).anchor)
        with self.assertRaisesRegex(TrackingFailure,'no_temporal_matches'):
            tracker.process(101.,blank,blank)
        self.assertEqual(tracker.current_stereo[0],101.)
        np.testing.assert_allclose(tracker.current_stereo[1],xyz,atol=1e-6)
        with self.assertRaisesRegex(TrackingFailure,'insufficient_stereo_points'):
            tracker.process(102.,blank,blank)
        self.assertIsNone(tracker.current_stereo)


if __name__=="__main__":unittest.main()
