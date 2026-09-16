"""Uncalibrated feedback can assist stationary checks but cannot enable EKF input."""
from pathlib import Path
import sys
import tempfile
import unittest
import yaml

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from generate_config import generate


class TelemetryConfigTests(unittest.TestCase):
    def test_disabled_unconfirmed_and_confirmed_integration(self):
        cfg=yaml.safe_load((ROOT/'config/simulation_live.yaml').read_text())
        with tempfile.TemporaryDirectory(dir=ROOT/'build',prefix='telemetry_config_') as directory:
            tmp=Path(directory);profile=tmp/'profile.yaml'
            cfg['telemetry_motion'].update(enabled=True,fuse_velocity=False,calibration_confirmed=False)
            profile.write_text(yaml.safe_dump(cfg));generate(profile,tmp/'generated')
            read=lambda:yaml.safe_load((tmp/'generated/ekf.yaml').read_text())['ekf_filter_node']['ros__parameters']
            self.assertNotIn('twist0',read())
            cfg['telemetry_motion']['fuse_velocity']=True
            profile.write_text(yaml.safe_dump(cfg))
            with self.assertRaises(ValueError):generate(profile,tmp/'generated')
            cfg['telemetry_motion']['calibration_confirmed']=True
            profile.write_text(yaml.safe_dump(cfg));generate(profile,tmp/'generated')
            self.assertEqual(read()['twist0_config'],[False]*6+[True]*6+[False]*3)
            self.assertNotIn('/car/odom',read().values())


if __name__=='__main__':unittest.main()
