#!/usr/bin/env python3
"""Physical motion, covariance units, gauge invariance and epoch discontinuities."""
import sys
import time
import unittest
from pathlib import Path
import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src/t3_lidar_visual_fusion'))
from t3_lidar_visual_fusion.body_motion import BodyMotion, body_velocity


def exp_motion(velocity, dt):
    rho, phi=np.asarray(velocity[:3])*dt,np.asarray(velocity[3:])*dt
    theta=np.linalg.norm(phi);x,y,z=phi
    skew=np.array([[0.,-z,y],[z,0.,-x],[-y,x,0.]])
    a=(1.-np.cos(theta))/theta**2 if theta>1e-5 else .5-theta**2/24.
    b=(theta-np.sin(theta))/theta**3 if theta>1e-5 else 1./6.-theta**2/120.
    value=np.eye(4);value[:3,:3]=cv2.Rodrigues(phi)[0]
    value[:3,3]=(np.eye(3)+a*skew+b*skew@skew)@rho
    return value


class MotionTests(unittest.TestCase):
    def test_constant_three_dimensional_turn(self):
        velocity=np.array([.5,-.03,.02,.01,-.02,.13])
        for dt in [.1,1.,4.]:
            np.testing.assert_allclose(body_velocity(np.eye(4),exp_motion(velocity,dt),dt),velocity,atol=1e-9)

    def test_stationary_covariance_has_velocity_units(self):
        covariance=np.diag([.01]*3+[.005]*3)
        for dt in [.2,1.,4.]:
            tracker=BodyMotion(max_gap=6.,variance_floor=[1e-10]*6)
            self.assertIsNone(tracker.update(1000.,'a',np.eye(4),covariance))
            velocity,var,interval=tracker.update(1000.+dt,'a',np.eye(4),covariance)
            np.testing.assert_allclose(velocity,0.,atol=1e-10)
            np.testing.assert_allclose(var,2.*covariance/dt**2,rtol=1e-7,atol=1e-10)
            self.assertAlmostEqual(interval,dt)

    def test_epoch_and_gap_do_not_create_motion(self):
        tracker=BodyMotion(max_gap=6.)
        cov=np.eye(6)*.01
        self.assertIsNone(tracker.update(0.,'a',np.eye(4),cov))
        offset=np.eye(4);offset[:3,3]=[1000.,-700.,300.]
        self.assertIsNone(tracker.update(1.,'b',offset,cov))
        later=offset@exp_motion([.2,0.,0.,0.,0.,.1],1.)
        np.testing.assert_allclose(tracker.update(2.,'b',later,cov)[0],[.2,0.,0.,0.,0.,.1],atol=1e-9)
        self.assertIsNone(tracker.update(20.,'b',later,cov))
        self.assertIsNone(tracker.update(20.,'b',later,cov))
        tracker.reset()
        self.assertIsNone(tracker.update(21.,'b',later,cov))

    def test_body_twist_and_covariance_are_gauge_invariant(self):
        covariance=np.diag([.01,.02,.03,.004,.005,.006])
        first=exp_motion([1.,.2,.1,.1,.2,.3],1.)
        second=first@exp_motion([.2,-.01,.005,.01,.02,.08],3.)
        gauge=exp_motion([30.,-20.,10.,.2,.3,.4],1.)
        jac=np.zeros((6,6));jac[:3,:3]=gauge[:3,:3];jac[3:,3:]=gauge[:3,:3]
        a,b=BodyMotion(max_gap=6.),BodyMotion(max_gap=6.)
        a.update(1.,'a',first,covariance);b.update(1.,'b',gauge@first,jac@covariance@jac.T)
        va,ca,_=a.update(4.,'a',second,covariance)
        vb,cb,_=b.update(4.,'b',gauge@second,jac@covariance@jac.T)
        np.testing.assert_allclose(va,vb,atol=1e-9)
        np.testing.assert_allclose(ca,cb,rtol=1e-5,atol=1e-8)
        self.assertGreater(np.linalg.eigvalsh(ca).min(),0.)


if __name__=='__main__': unittest.main()
