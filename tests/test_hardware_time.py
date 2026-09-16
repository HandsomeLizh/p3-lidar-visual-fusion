import unittest
import numpy as np
from t3_lidar_visual_fusion.hardware_time import SharedSensorClock


class ClockTests(unittest.TestCase):
    def ready(self):
        c=SharedSensorClock()
        for i in range(130):
            t=100.+i*.01;c.observe_imu(t,1_700_000_000.+t+.001+(i%7)*.0002)
        self.assertTrue(c.status()['ready']);return c

    def test_shared_clock_keeps_lidar_imu_intervals(self):
        c=self.ready();a=c.convert(101.30,1_700_000_101.5);b=c.convert(101.4,1_700_000_101.5)
        self.assertAlmostEqual(b-a,.1,places=6)
        self.assertLess(c.jitter,.002)

    def test_sensor_reset_latches_failure(self):
        c=self.ready()
        with self.assertRaisesRegex(ValueError,'reset'):c.observe_imu(1.,1_700_000_200.)
        with self.assertRaises(ValueError):c.convert(200.,1_700_000_200.)
        self.assertFalse(c.status()['ready'])

    def test_delayed_packet_does_not_move_clock(self):
        c=self.ready();offset=c.offset
        c.observe_imu(101.3,1_700_000_101.5)
        self.assertEqual(c.offset,offset)
        with self.assertRaisesRegex(ValueError,'Stale'):c.convert(100.,1_700_000_103.)

    def test_drift_blocks_instead_of_reanchoring(self):
        c=self.ready();offset=c.offset
        with self.assertRaisesRegex(ValueError,'drift'):
            for i in range(150):
                t=101.3+i*.01;c.observe_imu(t,1_700_000_000.+t+.2)
        self.assertEqual(c.offset,offset)

    def test_noisy_reception_stays_uninitialized(self):
        c=SharedSensorClock()
        for i in range(150):
            t=100.+i*.01;c.observe_imu(t,1_700_000_000.+t+(i%2)*.15)
        self.assertIsNone(c.offset)

    def test_system_clock_does_not_add_offset(self):
        c=SharedSensorClock(mode='system')
        self.assertEqual(c.observe_imu(1700000000.,1700000000.001),1700000000.)


if __name__=='__main__':unittest.main()
