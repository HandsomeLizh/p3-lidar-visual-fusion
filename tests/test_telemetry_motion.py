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

    def test_restores_all_six_flu_wire_fields_after_legacy_decoder(self):
        # Independent fixture for the current C++ decoder's three sign changes.
        gate=VelocityGate({'input_encoding':'legacy_driver_from_flu_wire','angular_scale':np.pi/180})
        self.assertTrue(gate.accept(10.,'base_link',[.2,-.04,-.03,-10.,-20.,-30.],10.))
        _,v,_=gate.take(10.)
        np.testing.assert_allclose(v,np.r_[.2,.04,-.03,np.deg2rad([10.,-20.,30.])])

    def test_legacy_decode_happens_before_mounting_rotation(self):
        gate=VelocityGate({'input_encoding':'legacy_driver_from_flu_wire',
            'base_from_feedback_rotation':[[0,-1,0],[1,0,0],[0,0,1]]})
        gate.accept(10.,'base_link',[.2,-.04,-.03,-.1,-.2,-.3],10.)
        _,v,_=gate.take(10.)
        np.testing.assert_allclose(v,[-.04,.2,-.03,.2,.1,.3])

    def test_unknown_input_encoding_is_rejected(self):
        with self.assertRaises(ValueError):VelocityGate({'input_encoding':'unrecognized'})

    def test_actor_body_polar_and_axial_fields_after_legacy_decoder(self):
        gate=VelocityGate({'input_encoding':'legacy_driver_from_actor_body_wire',
            'angular_scale':np.pi/180,'standard_deviation':[.1,.2,.3,.01,.02,.03]})
        # Raw Actor v=(-.2,-.03,.04), w=(10,20,30). The C++ decoder
        # flips linear Y and angular X/Z before publishing these fields.
        self.assertTrue(gate.accept(10.,'base_link',[-.2,.03,.04,-10.,20.,-30.],10.))
        _,v,cov=gate.take(10.)
        np.testing.assert_allclose(v,np.r_[.2,.04,.03,np.deg2rad([10.,-30.,20.])])
        np.testing.assert_allclose(np.diag(cov),np.array([.1,.3,.2,.01,.03,.02])**2)

    def test_actor_body_forward_left_and_right_turns_ignore_pose(self):
        class Message:
            child_frame_id='base_link'
            @property
            def pose(self):raise AssertionError('Body feedback must not use UE pose')
        for turn in (-6.,6.):
            gate=VelocityGate({'input_encoding':'legacy_driver_from_actor_body_wire','angular_scale':np.pi/180})
            msg=Message();msg.header=SimpleNamespace(stamp=SimpleNamespace(sec=10,nanosec=0))
            msg.twist=SimpleNamespace(twist=SimpleNamespace(
                linear=SimpleNamespace(x=-.2,y=0.,z=0.),angular=SimpleNamespace(x=0.,y=turn,z=0.)))
            node=SimpleNamespace(gate=gate,get_clock=lambda:SimpleNamespace(now=lambda:SimpleNamespace(nanoseconds=10_000_000_000)))
            TelemetryMotion.receive(node,msg)
            _,v,_=gate.take(10.)
            np.testing.assert_allclose(v,[.2,0.,0.,0.,0.,np.deg2rad(turn)])

    def test_actor_body_encoding_rejects_world_mode(self):
        with self.assertRaises(ValueError):
            VelocityGate({'input_encoding':'legacy_driver_from_actor_body_wire','velocity_frame':'world'})

    def test_world_velocity_uses_current_attitude_and_rotates_uncertainty(self):
        r=np.array([[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]])
        gate=VelocityGate({'velocity_frame':'world','standard_deviation':[.1,.2,.3,.01,.02,.03]})
        self.assertTrue(gate.accept(10.,'base_link',[0.,.2,0.,0.,0.,.1],10.,world_from_body=r))
        _,v,cov=gate.take(10.)
        np.testing.assert_allclose(v,[.2,0.,0.,0.,0.,.1],atol=1e-12)
        np.testing.assert_allclose(np.diag(cov),np.array([.2,.1,.3,.02,.01,.03])**2)
        for bad in [None,np.zeros((3,3)),np.diag([-1.,1.,1.])]:
            self.assertFalse(gate.accept(11.,'base_link',[0.]*6,11.,world_from_body=bad))

    def test_world_input_reads_only_orientation_not_position(self):
        class Pose:
            orientation=SimpleNamespace(x=0.,y=0.,z=np.sqrt(.5),w=np.sqrt(.5))
            @property
            def position(self):raise AssertionError('Absolute position leaked into velocity conversion')
        message=SimpleNamespace(header=SimpleNamespace(stamp=SimpleNamespace(sec=10,nanosec=0)),
            child_frame_id='base_link',pose=SimpleNamespace(pose=Pose()),
            twist=SimpleNamespace(twist=SimpleNamespace(linear=SimpleNamespace(x=0.,y=.2,z=0.),
                angular=SimpleNamespace(x=0.,y=0.,z=.1))))
        gate=VelocityGate({'velocity_frame':'world'})
        node=SimpleNamespace(gate=gate,get_clock=lambda:SimpleNamespace(now=lambda:SimpleNamespace(nanoseconds=10_000_000_000)))
        TelemetryMotion.receive(node,message)
        _,v,_=gate.take(10.);np.testing.assert_allclose(v,[.2,0.,0.,0.,0.,.1],atol=1e-12)

    def test_p4_actor_orientation_is_a_proper_body_correction(self):
        c=np.array([[-1.,0.,0.],[0.,0.,1.],[0.,1.,0.]])
        gate=VelocityGate({'velocity_frame':'world','orientation_child_from_base_rotation':c.tolist()})
        self.assertTrue(gate.proper_rotation(c))
        # The reported child frame is rotated by C; corrected body is identity.
        self.assertTrue(gate.accept(10.,'base_link',[.2,0.,0.,0.,0.,.1],10.,world_from_body=c@gate.orientation_child_from_base))
        _,v,_=gate.take(10.);np.testing.assert_allclose(v,[.2,0.,0.,0.,0.,.1])


if __name__=='__main__':unittest.main()
