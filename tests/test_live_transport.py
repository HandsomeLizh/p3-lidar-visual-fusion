import sys
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from live_transport import SensorInbox


class Transport(unittest.TestCase):
    def setUp(self):
        self.wall = 100.0
        self.mono = 10.0
        self.q = SensorInbox({'visual_max_age_sec': 4., 'stereo_max_skew': .01,
            'max_image_bytes': 10, 'max_cloud_bytes': 20,
            'live_transport': {'imu_queue_samples': 4}}, lambda: self.wall, lambda: self.mono)

    def msg(self, stamp=100., size=1):
        ns = round(stamp*1e9)
        return NS(header=NS(stamp=NS(sec=ns//10**9, nanosec=ns % 10**9)), data=bytes(size))

    def test_stereo_waits_and_matches_without_rewriting(self):
        self.q.offer('left', self.msg(99.95))
        self.q.offer('right', self.msg(99.955))
        group = self.q.take()
        self.assertEqual([s.key for s in group], ['left', 'right'])
        self.assertEqual([s.stamp_ns for s in group], [99950000000, 99955000000])

    def test_no_wrong_pair(self):
        self.q.offer('left', self.msg(99.))
        self.q.offer('right', self.msg(100.))
        self.assertIsNone(self.q.stereo)
        self.q.offer('left', self.msg(100.))
        self.assertEqual(self.q.take()[0].stamp_ns, 100000000000)

    def test_bounds_with_blocked_consumer(self):
        for i in range(2000):
            self.wall += .01
            self.q.offer('left', self.msg(self.wall))
            self.q.offer('right', self.msg(self.wall))
            self.q.offer('lidar', self.msg(self.wall))
            self.q.offer('imu', self.msg(self.wall))
        snapshot = self.q.snapshot()
        self.assertEqual(snapshot['pending'], dict(left=0, right=0, stereo=1, lidar=1, imu=4))
        self.assertEqual(snapshot['counts']['stereo_replaced'], 1999)
        self.assertEqual(snapshot['counts']['lidar_replaced'], 1999)
        self.assertEqual(snapshot['counts']['imu_overflow'], 1996)

    def test_unmatched_camera_memory_bound(self):
        for i in range(50):
            self.wall += .01
            self.q.offer('left', self.msg(self.wall))
        self.assertEqual(len(self.q.images['left']), 2)
        self.assertEqual(self.q.counts['left_unmatched'], 48)

    def test_stale_lidar_is_rejected(self):
        self.assertFalse(self.q.offer('lidar', self.msg(80.)))
        self.assertTrue(self.q.offer('lidar', self.msg(100.)))

    def test_future_and_zero_rejected(self):
        self.assertFalse(self.q.offer('left', self.msg(101.)))
        self.assertFalse(self.q.offer('imu', self.msg(0.)))
        self.assertTrue(self.q.offer('left', self.msg(100.)))

    def test_duplicate_and_backwards_never_reset_origin(self):
        self.assertTrue(self.q.offer('lidar', self.msg(100.)))
        self.assertFalse(self.q.offer('lidar', self.msg(100.)))
        self.assertFalse(self.q.offer('lidar', self.msg(99.)))

    def test_queued_frames_expire_even_if_wall_clock_moves_back(self):
        self.q.offer('left', self.msg(100.))
        self.q.offer('right', self.msg(100.))
        group = self.q.take()
        self.mono += 18.
        self.wall -= 10.
        self.assertFalse(all(self.q.fresh(s) for s in group))

    def test_stale_partner_rejects_whole_pair(self):
        self.q.offer('left', self.msg(100.))
        self.mono += 4.
        self.q.offer('right', self.msg(100.))
        self.assertFalse(all(self.q.fresh(s) for s in self.q.take()))

    def test_size_limit(self):
        self.assertFalse(self.q.offer('left', self.msg(size=11)))
        self.assertFalse(self.q.offer('lidar', self.msg(size=21)))

    def test_imu_fifo_and_age(self):
        self.assertFalse(self.q.offer('imu', self.msg(99.)))
        for t in (99.8, 99.9, 100.):
            self.q.offer('imu', self.msg(t))
        self.assertEqual([self.q.take()[0].stamp_ns for _ in range(3)], [99800000000, 99900000000, 100000000000])

    def test_new_data_after_long_gap(self):
        self.q.offer('lidar', self.msg(100.))
        self.q.take()
        self.wall += 20.
        self.mono += 20.
        self.assertTrue(self.q.offer('lidar', self.msg(120.)))
        self.assertEqual(self.q.counts['lidar_input_gap'], 1)
        self.assertTrue(self.q.fresh(self.q.take()[0]))

    def test_shutdown_wakes_worker(self):
        result = []
        worker = threading.Thread(target=lambda: result.append(self.q.take()))
        worker.start()
        self.q.close()
        worker.join(1.)
        self.assertFalse(worker.is_alive())
        self.assertEqual(result, [None])


if __name__ == '__main__':
    unittest.main()
