import unittest
from t3_lidar_visual_fusion.measurement_order import VisualConstraintQueue


class MeasurementOrderTests(unittest.TestCase):
    def test_faster_visual_waits_for_cotimed_lidar(self):
        q = VisualConstraintQueue(.8)
        q.append(10.2, 1., 'a'); q.append(10.4, 1.1, 'b')
        self.assertEqual(q.ready(10., 1.2, True), [])
        self.assertEqual(q.ready(10.2, 1.3, True), ['a'])
        self.assertEqual(q.ready(10.4, 1.4, True), ['b'])

    def test_wait_is_bounded_and_lidar_failure_releases_visual(self):
        q = VisualConstraintQueue(.8)
        q.append(10.2, 1., 'a')
        self.assertEqual(q.ready(10., 1.81, True), ['a'])
        self.assertEqual(q.expired, 1)
        q.append(10.4, 2., 'b')
        self.assertEqual(q.ready(10., 2.01, False), ['b'])

    def test_quality_reset_and_capacity_are_bounded(self):
        q = VisualConstraintQueue(.8, 2)
        for i in range(3):q.append(i, 0., i)
        self.assertEqual(q.dropped, 1)
        q.clear()
        self.assertEqual(q.ready(3., 1., True), [])
        self.assertEqual(q.dropped, 3)

    def test_disabled_preserves_existing_profiles(self):
        q = VisualConstraintQueue()
        q.append(10., 1., 'a')
        self.assertEqual(q.ready(0., 1., True), ['a'])


if __name__ == '__main__':unittest.main()
