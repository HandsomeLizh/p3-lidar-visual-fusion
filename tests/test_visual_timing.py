"""Deferred regressions for delayed vision and simulated-clock behavior."""
import unittest
from t3_lidar_visual_fusion.visual_timing import VisualTiming


class VisualTimingTests(unittest.TestCase):
    def test_newly_delivered_old_result_is_still_expired(self):
        result=VisualTiming(1.5).check(100.,105.,20.,20.)
        self.assertEqual(result.reason,"visual_measurement_expired")

    def test_small_bounded_lag_keeps_original_sensor_age(self):
        result=VisualTiming(1.5).check(100.,101.,20.,20.1)
        self.assertTrue(result.valid)
        self.assertEqual(result.sensor_age_sec,1.)

    def test_paused_sim_clock_does_not_allow_unbounded_queue_wait(self):
        result=VisualTiming(1.5).check(100.,100.,20.,22.)
        self.assertEqual(result.reason,"visual_queue_expired")

    def test_missing_clock_and_wrong_epoch_are_rejected(self):
        self.assertEqual(VisualTiming().check(100.,0.,20.,20.).reason,
                         "visual_clock_not_ready")
        self.assertEqual(VisualTiming().check(1000.,100.,20.,20.).reason,
                         "visual_stamp_in_future")

    def test_future_tolerance_is_not_general_clock_retiming(self):
        timing=VisualTiming(1.5,.1)
        self.assertTrue(timing.check(100.05,100.,20.,20.).valid)
        self.assertFalse(timing.check(100.5,100.,20.,20.).valid)

    def test_invalid_clock_values_are_json_safe(self):
        result=VisualTiming().check(float("nan"),100.,20.,20.)
        self.assertEqual(result.reason,"invalid_visual_time")
        self.assertIsNone(result.dictionary()["sensor_age_sec"])


if __name__=="__main__":unittest.main()
