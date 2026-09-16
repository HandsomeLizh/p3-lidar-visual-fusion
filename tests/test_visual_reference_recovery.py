"""Metric stereo recovery with known correspondences, including rejected depth."""
import copy
from types import SimpleNamespace
import unittest
import numpy as np
from sensor_msgs.msg import Image
from t3_lidar_visual_fusion.learned_tracker import LearnedStereoTracker
from t3_lidar_visual_fusion.learned_odometry import LearnedOdometry
from t3_lidar_visual_fusion.stereo_geometry import StereoGeometry,TrackingFailure,project
from t3_lidar_visual_fusion.visual_continuity import VisualContinuity
from test_stereo_geometry import profile,scene


class Fixture:
    def __init__(self):
        self.cfg=profile();self.cfg['learned_visual'].update(max_tracking_gap_sec=3.,max_recovery_gap_sec=30.)
        self.geometry=StereoGeometry(self.cfg);self.xyz=scene();self.features=[];self.index=0
        self.tracker=LearnedStereoTracker(self.cfg,self)
        self.blank=np.zeros((480,640),np.uint8)

    def extract(self,image):
        f=self.features[self.index];self.index+=1;return f

    def match(self,first,second):
        if first['frame']==getattr(self,'unmatchable_reference',None):return np.empty((0,2),int)
        if first['frame']!=second['frame'] and second['missing']:return np.empty((0,2),int)
        ids=np.arange(len(first['pixels']));return np.column_stack([ids,ids])

    def frame(self,stamp,x=0.,bad_depth=False,missing=False,record_rejection=True):
        current=self.xyz-[x,0,0]
        self.features.extend(dict(pixels=project(p,current*(1.5 if bad_depth and side else 1.))[0],
                                  frame=stamp,missing=missing)
                             for side,p in enumerate([self.geometry.p0,self.geometry.p1]))
        try:return self.tracker.process(stamp,self.blank,self.blank)
        except TrackingFailure:
            if record_rejection:self.tracker.reject(stamp)
            raise


class ReferenceRecoveryTests(unittest.TestCase):
    def test_backup_reference_recovers_same_global_pose(self):
        f=Fixture();f.frame(100.);f.frame(101.,.05)
        f.unmatchable_reference=101.
        recovered=f.frame(102.,.1)
        self.assertEqual(recovered.metrics['reference_candidates_tried'],2)
        self.assertEqual(recovered.metrics['reference_stamp_sec'],100.)
        self.assertEqual(recovered.epoch,1)
        np.testing.assert_allclose(recovered.base_pose[:3,3],[.1,0,0],atol=1e-5)
        self.assertEqual(f.tracker.backup[0],100.)

    def test_six_bad_depth_frames_keep_epoch_and_recover_without_lidar(self):
        f=Fixture();f.frame(100.);first=f.frame(101.,.05)
        bridge=VisualContinuity(max_gap=30.);cov=np.eye(6)*.001
        bridge.observe(101.,str(first.epoch),first.base_pose,cov)
        anchor=first.base_pose.copy();anchor[0,3]+=5.;bridge.anchor(101.,anchor,cov)
        reference=f.tracker.keyframe
        for i in range(6):
            with self.assertRaisesRegex(TrackingFailure,'stereo_motion_disagreement') as caught:
                f.frame(102.+i,.08+i*.02,bad_depth=True)
            self.assertIn('stereo_motion_ratio',caught.exception.metrics)
            self.assertLess(caught.exception.metrics['stereo_motion_ratio'],.6)
            self.assertIs(f.tracker.keyframe,reference)
            self.assertEqual(f.tracker.epoch,1)
        recovered=f.frame(109.,.24)
        self.assertFalse(recovered.anchor);self.assertEqual(recovered.epoch,1)
        self.assertTrue(recovered.metrics['recovered_reference'])
        np.testing.assert_allclose(recovered.base_pose[:3,3],[.24,0,0],atol=1e-5)
        bridge.observe(109.,str(recovered.epoch),recovered.base_pose,cov)
        np.testing.assert_allclose(bridge.estimate()[1][:3,3],[5.24,0,0],atol=1e-5)

    def test_gap_tries_old_geometry_and_expired_reference_starts_unanchored_epoch(self):
        f=Fixture();f.frame(100.);f.frame(101.,.05)
        result=f.frame(113.,.3)
        self.assertEqual(result.epoch,1);self.assertTrue(result.metrics['recovered_reference'])
        bridge=VisualContinuity(max_gap=30.);cov=np.eye(6)*.001
        bridge.observe(113.,'1',result.base_pose,cov);bridge.anchor(113.,result.base_pose,cov)
        expired=f.frame(144.,.32)
        self.assertTrue(expired.anchor);self.assertEqual(expired.epoch,2)
        self.assertEqual(expired.metrics['reason'],'recovery_reference_expired')
        new=f.frame(145.,.34);bridge.observe(145.,'2',new.base_pose,cov)
        self.assertIsNone(bridge.estimate())

    def test_missing_overlap_never_publishes_a_made_up_recovery(self):
        f=Fixture();f.frame(100.);f.frame(101.,.05)
        for i in range(10):
            with self.assertRaisesRegex(TrackingFailure,'no_temporal_matches'):
                f.frame(102.+i,.1,missing=True)
        self.assertEqual(f.tracker.epoch,1)
        np.testing.assert_allclose(f.tracker.last_pose[1][:3,3],[.05,0,0],atol=1e-5)
        for i in range(12):f.frame(113.+i,.1+i*.01)
        self.assertIsNotNone(f.tracker.backup)
        self.assertEqual(len(f.tracker.keyframe[1]['pixels']),120)
        self.assertEqual(len(f.tracker.backup[1]['pixels']),120)

    def test_image_rejection_retains_reference_but_sends_invalid_sample(self):
        f=Fixture();f.frame(100.);f.frame(101.,.05);messages=[];records=[]
        fake=SimpleNamespace(tracker=f.tracker,cfg=f.cfg['learned_visual'],failure_streak=0,next_probe=0.,
            age=lambda left,queued:SimpleNamespace(valid=True,sensor_age_sec=.1),
            increment=lambda name:None,queue=SimpleNamespace(stopped=False),
            pub=SimpleNamespace(publish=messages.append),record=records.append)
        left=Image();left.header.stamp.sec=102
        LearnedOdometry.reject(fake,'underexposed',left,0.)
        self.assertTrue(records[-1]['reference_retained'])
        self.assertEqual(messages[-1].header.frame_id,'learned_epoch_1')
        self.assertEqual(messages[-1].pose.covariance[0],1e6)
        result=f.frame(103.,.1)
        self.assertEqual(result.epoch,1);self.assertFalse(result.anchor)
        left.header.stamp.sec=150;LearnedOdometry.reject(fake,'underexposed',left,0.)
        self.assertFalse(records[-1]['reference_retained'])
        self.assertIsNone(f.tracker.keyframe)


if __name__=='__main__':unittest.main()
