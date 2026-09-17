"""Independent analytic-motion checks for the offline diagnostic, not ROS."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

SPEC = importlib.util.spec_from_file_location('audit', Path(__file__).resolve().parents[1] / 'scripts/audit_simulation_twist.py')
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


def arc(rate):
    t = np.linspace(0., 12., 241)
    speed, initial = .2, .43
    angle = initial + rate*t
    p = np.c_[speed/rate*(np.sin(angle)-np.sin(initial)),
              -speed/rate*(np.cos(angle)-np.cos(initial)), t*0.]
    r = Rotation.from_rotvec(np.c_[t*0., t*0., angle]).as_matrix()
    return dict(t=t, p=p,
                q=Rotation.from_matrix(r @ AUDIT.C).as_quat(),
                v=np.tile([speed, 0., 0.], (len(t), 1)) @ AUDIT.C,
                w=np.tile([0., 0., np.rad2deg(rate)], (len(t), 1)) @ AUDIT.C)


class AuditTests(unittest.TestCase):
    def test_both_turn_signs_against_analytic_arc(self):
        for rate in (-.15, .15):
            report = AUDIT.audit(arc(rate))
            self.assertLess(report['free_twist_integration']['endpoint_error_m'], 1e-5)
            w = report['windows']['1.0']
            self.assertGreater(w['turning_windows'], 5)
            self.assertEqual(w['yaw_same_sign_windows'], w['turning_windows'])
            self.assertLess(w['gyro_rotation_error_deg']['maximum'], 1e-10)

    def test_wrong_linear_sign_is_not_hidden_by_reference_attitude(self):
        d = arc(.15)
        d['v'] *= -1.
        report = AUDIT.audit(d)
        self.assertGreater(report['free_twist_integration']['endpoint_error_m'], 3.)
        self.assertGreater(report['reference_attitude_each_sample_diagnostic_only']['endpoint_error_m'], 3.)
        self.assertGreater(report['windows']['1.0']['linear_hypotheses']['declared_actor_body']['direction_error_deg']['median'], 179.)

    def test_receive_gap_is_not_integrated_as_continuous_motion(self):
        d = arc(.15)
        d = {k: value[np.r_[np.arange(80), np.arange(120, len(d['t']))]] for k, value in d.items()}
        report = AUDIT.audit(d)
        self.assertNotIn('free_twist_integration', report)
        self.assertGreater(report['windows']['1.0']['skipped_gap_windows'], 0)

    def test_single_raw_metadata_cannot_be_treated_as_decoded_trajectory(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'meta.json'
            path.write_text('{"timestamp":"2026-09-17 00:00:00.000","velocity":{"x":1}}')
            with self.assertRaisesRegex(ValueError, 'raw metadata'):
                AUDIT.load_record(path)


if __name__ == '__main__':
    unittest.main()
