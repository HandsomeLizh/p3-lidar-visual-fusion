"""Geometry and bounded-state checks; no network/model or ROS needed."""
from pathlib import Path
from types import SimpleNamespace
from contextlib import nullcontext
import unittest
from unittest.mock import patch
import numpy as np
from scipy.spatial.transform import Rotation
import yaml

from t3_lidar_visual_fusion.core import inverse
from t3_lidar_visual_fusion.roma_tracker import RomaStereoTracker, camera_pose_to_body, bound_history
from t3_lidar_visual_fusion.stereo_geometry import StereoGeometry, project, TrackingFailure
from t3_lidar_visual_fusion.learned_tracker import LearnedStereoTracker

ROOT=Path(__file__).resolve().parents[1]


class RomaAdapterTests(unittest.TestCase):
    def setUp(self):
        self.profile=yaml.safe_load((ROOT/'config/simulation_live.yaml').read_text())
        self.geometry=StereoGeometry(self.profile)

    def test_actual_mount_forward_yaw_and_height_have_correct_axes(self):
        mount=self.geometry.base_from_rect
        for xyz,yaw in [([1.,0.,0.],0.),([0.,1.,0.],0.),([0.,0.,.2],0.),
                        ([0.,0.,0.],.4),([.1,.05,.01],-.2)]:
            with self.subTest(xyz=xyz,yaw=yaw):
                expected=np.eye(4);expected[:3,:3]=Rotation.from_euler('z',yaw).as_matrix()
                expected[:3,3]=xyz
                raw_camera=inverse(mount)@expected@mount
                np.testing.assert_allclose(camera_pose_to_body(raw_camera,mount),expected,atol=1e-10)
        # Omitting the downward-pitched mount really would leak forward motion into z.
        expected=np.eye(4);expected[0,3]=1.
        raw=inverse(mount)@expected@mount
        self.assertGreater(abs(raw[2,3]),.8)

    def test_full_sparse_pnp_with_live_mount_has_no_synthetic_downward_drift(self):
        config=self.profile.copy();config['learned_visual']=dict(config['learned_visual'],
            image_stationary_enabled=False,min_pnp_coverage=.05,min_stereo_coverage=.05)
        geometry=StereoGeometry(config)
        width,height=geometry.size
        xx,yy=np.meshgrid(np.linspace(width*.2,width*.8,15),np.linspace(height*.2,height*.8,10))
        rays=np.c_[xx.ravel(),yy.ravel(),np.ones(xx.size)]@geometry.inverse_k.T
        points=rays*np.random.default_rng(7).uniform(4.,9.,(len(rays),1))
        expected=np.eye(4);expected[:3,:3]=Rotation.from_euler('z',.03).as_matrix();expected[:3,3]=[.2,.04,0.]
        current_from_reference=inverse(expected@geometry.base_from_rect)@geometry.base_from_rect
        current=points@current_from_reference[:3,:3].T+current_from_reference[:3,3]
        fixtures=[project(p,xyz)[0] for xyz in [points,current] for p in [geometry.p0,geometry.p1]]
        class Matches:
            def __init__(self):self.index=0
            def extract(self,image):
                item={'pixels':fixtures[self.index]};self.index+=1;return item
            def match(self,a,b):
                ids=np.arange(len(a['pixels']));return np.c_[ids,ids]
        tracker=LearnedStereoTracker(config,Matches())
        blank=np.zeros((height,width),dtype=np.uint8)
        tracker.process(100.,blank,blank)
        result=tracker.process(101.,blank,blank)
        np.testing.assert_allclose(result.base_pose,expected,atol=1e-5)

    def test_trim_preserves_current_geometry_and_remaps_history(self):
        class Point:
            def __init__(self,i):self.id=i;self.observations={i:np.zeros(2)}
            def remove_observation(self,i):self.observations.pop(i,None)
        removed=[]
        slam=SimpleNamespace(enable_backend=False,keyframes=[],global_mps={},poses=[],
            _pose_reference_kf_ids=[],_pose_relative_to_kf=[],_kf_id_to_pose_idx={},
            relocalizer=SimpleNamespace(remove=removed.append),trim_keyframe_images=lambda n:None)
        for i in range(600):
            pose=np.eye(4);pose[0,3]=i*.01
            slam.poses.append(pose);slam._pose_reference_kf_ids.append(i//5);slam._pose_relative_to_kf.append(np.eye(4))
            if i%5==0:
                point=Point(i);slam.global_mps[i]=point
                slam.keyframes.append(SimpleNamespace(id=i,mappoints=[point],pose=pose.copy()))
                slam._kf_id_to_pose_idx[i]=len(slam.poses)-1
            bound_history(slam,keyframe_limit=4,pose_limit=8)
            self.assertLessEqual(len(slam.keyframes),4)
            self.assertLessEqual(len(slam.global_mps),4)
            self.assertLessEqual(len(slam.poses),12)
            for frame in slam.keyframes:
                np.testing.assert_array_equal(slam.poses[slam._kf_id_to_pose_idx[frame.id]],frame.pose)
            np.testing.assert_array_equal(slam.poses[-1],pose)
        self.assertEqual(len(removed),116)

    def fake_frontend(self,body_poses,success=True):
        obj=RomaStereoTracker.__new__(RomaStereoTracker)
        obj.geometry=self.geometry;obj.cfg={};obj.epoch=1;obj.last_pose=None
        obj.correction=np.eye(4);obj.poisoned=False;obj.recovery_gap=30.;obj.keyframe_limit=12
        obj.torch=SimpleNamespace(inference_mode=nullcontext)
        obj.stationary=SimpleNamespace(image=lambda *a:None,check=lambda *a:True,
            status=lambda:{},zero_updates=0)
        w,h=self.geometry.size
        xx,yy=np.meshgrid(np.linspace(.1*w,.9*w,12),np.linspace(.1*h,.9*h,8))
        points=np.c_[xx.ravel(),yy.ravel()]
        poses=iter(body_poses)
        calls=[0]
        def process(left,right):
            calls[0]+=1
            return inverse(self.geometry.base_from_rect)@next(poses)@self.geometry.base_from_rect
        def status():
            return SimpleNamespace(reason='tracked',inlier_count=96,correspondence_count=260,
                success=success,initialized=calls[0]==1,keyframe_created=False)
        obj.slam=SimpleNamespace(process_frame=process,get_tracking_status=status,_kf_pts3d=np.zeros((96,3)),
            get_feature_match_snapshot=lambda:dict(temporal_inlier_mask=np.ones(96,dtype=bool),temporal_current_points=points))
        return obj,np.zeros((h,w,3),dtype=np.uint8)

    @patch('t3_lidar_visual_fusion.roma_tracker.bound_history')
    def test_stationary_creep_does_not_reappear_when_motion_resumes(self,trim):
        first=np.eye(4);creep=np.eye(4);creep[:3,3]=[.001,0.,-.002]
        moved=creep.copy();moved[0,3]+=.1
        tracker,image=self.fake_frontend([first,creep,moved])
        self.assertTrue(tracker.process(100.,image,image).anchor)
        np.testing.assert_allclose(tracker.process(101.,image,image).base_pose,first,atol=1e-10)
        tracker.stationary.check=lambda *args:False
        expected=np.eye(4);expected[0,3]=.1
        np.testing.assert_allclose(tracker.process(102.,image,image).base_pose,expected,atol=1e-10)

    @patch('t3_lidar_visual_fusion.roma_tracker.bound_history')
    def test_held_failure_and_large_jump_never_become_valid_poses(self,trim):
        tracker,image=self.fake_frontend([np.eye(4)],success=False)
        with self.assertRaises(TrackingFailure):tracker.process(100.,image,image)
        self.assertIsNone(tracker.last_pose)
        jump=np.eye(4);jump[0,3]=20.
        tracker,image=self.fake_frontend([np.eye(4),jump])
        tracker.process(100.,image,image)
        with self.assertRaisesRegex(TrackingFailure,'motion_jump'):tracker.process(101.,image,image)
        self.assertTrue(tracker.poisoned)
        np.testing.assert_allclose(tracker.last_pose[1],np.eye(4),atol=1e-10)


if __name__=='__main__':unittest.main()
