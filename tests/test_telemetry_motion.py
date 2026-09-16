"""Velocity feedback cannot import truth poses, stale samples, or repeated evidence."""
from types import SimpleNamespace
import unittest
import numpy as np
from t3_lidar_visual_fusion.telemetry_motion import VelocityGate, TelemetryMotion


class VelocityInputTests(unittest.TestCase):
    def test_reject_bad_frame_time_speed_and_nonfinite(self):
        cases = [(10., 'odom', [0.]*6, 10.), (9., 'base_link', [0.]*6, 10.),
                 (11., 'base_link', [0.]*6, 10.), (10., 'base_link', [8.,0,0,0,0,0], 10.),
                 (10., 'base_link', [0,0,0,0,0,3.], 10.),
                 (10., 'base_link', [np.nan,0,0,0,0,0], 10.)]
        for args in cases:
            gate=VelocityGate({})
            self.assertFalse(gate.accept(*args))
            self.assertIsNone(gate.take(10.))

    def test_bounded_coalescing_preserves_latest_and_never_repeats(self):
        gate=VelocityGate({})
        for i in range(100):
            stamp=10.+i*.001
            self.assertTrue(gate.accept(stamp,'base_link',[i*.001,0,0,0,0,0],stamp))
        stamp,v,c=gate.take(10.1)
        self.assertAlmostEqual(stamp,10.099)
        self.assertAlmostEqual(v[0],.099)
        self.assertEqual(gate.counts['coalesced'],99)
        self.assertTrue(np.all(np.linalg.eigvalsh(c)>0))
        self.assertIsNone(gate.take(10.11))
        self.assertIsNone(gate.take(10.2))

    def test_outage_drops_pending_then_recovers_on_new_data(self):
        gate=VelocityGate({})
        gate.accept(10.,'base_link',[.2,0,0,0,0,0],10.)
        self.assertIsNone(gate.take(11.))
        self.assertFalse(gate.accept(10.,'base_link',[0.]*6,11.))
        self.assertTrue(gate.accept(11.1,'base_link',[0.]*6,11.1))
        self.assertIsNotNone(gate.take(11.11))

    def test_absolute_pose_is_never_read(self):
        class Message:
            header=SimpleNamespace(stamp=SimpleNamespace(sec=10,nanosec=0))
            child_frame_id='base_link'
            twist=SimpleNamespace(twist=SimpleNamespace(
                linear=SimpleNamespace(x=.1,y=.2,z=.3),angular=SimpleNamespace(x=.01,y=.02,z=.03)))
            @property
            def pose(self):
                raise AssertionError('UE absolute pose must not be read')
        gate=VelocityGate({})
        node=SimpleNamespace(gate=gate,get_clock=lambda:SimpleNamespace(now=lambda:SimpleNamespace(nanoseconds=10_000_000_000)))
        TelemetryMotion.receive(node,Message())
        _,v,_=gate.take(10.)
        np.testing.assert_allclose(v,[.1,.2,.3,.01,.02,.03])

    def test_bad_uncertainty_is_rejected(self):
        for values in [[0.]*6,[.1]*3,[float('nan')]*6]:
            with self.assertRaises(ValueError):VelocityGate({'standard_deviation':values})

    def test_explicit_unit_conversion_precedes_speed_limits(self):
        gate=VelocityGate({'angular_scale':np.pi/180,'linear_scale':.01})
        self.assertTrue(gate.accept(10.,'base_link',[20.,0,0,0,0,90.],10.))
        _,v,_=gate.take(10.)
        np.testing.assert_allclose(v,[.2,0,0,0,0,np.pi/2])

    def test_proper_mount_rotation_applies_to_linear_and_angular(self):
        gate=VelocityGate({'base_from_feedback_rotation':[[0,-1,0],[1,0,0],[0,0,1]]})
        gate.accept(10.,'base_link',[.2,0,0,.1,0,0],10.)
        _,v,_=gate.take(10.)
        np.testing.assert_allclose(v,[0,.2,0,0,.1,0])
        with self.assertRaises(ValueError):
            VelocityGate({'base_from_feedback_rotation':[[-1,0,0],[0,1,0],[0,0,1]]})


if __name__=='__main__':unittest.main()
