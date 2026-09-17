"""Bounded frontend ingress and timestamp provenance, without a model or ROS graph."""
from collections import deque
from types import SimpleNamespace
import json
import threading
import unittest
from sensor_msgs.msg import Image
from t3_lidar_visual_fusion.learned_odometry import LearnedOdometry, LatestStereoPair


class IngressTests(unittest.TestCase):
    def fixture(self):
        f=SimpleNamespace(lock=threading.Lock(),ingress_events=deque(maxlen=64),
            frames=[deque(maxlen=2),deque(maxlen=2)],queue=LatestStereoPair(),
            profile={'output_image_size':[4,4]},cfg={'backend':'fixture'},encoding='mono8',
            last_pair_stamp=-1.,counts={},clock_ns=0,
            processing_timing={'processing_start_ros_sec':100.},samples=deque(maxlen=256))
        f.increment=lambda name:f.counts.update({name:f.counts.get(name,0)+1})
        f.get_clock=lambda:SimpleNamespace(now=lambda:SimpleNamespace(nanoseconds=f.clock_ns))
        return f

    def send(self,f,stamp,side,received):
        m=Image();m.width=m.height=4;m.step=4;m.encoding='mono8';m.data=bytes(16)
        m.header.stamp.sec=stamp;f.clock_ns=round(received*1e9)
        LearnedOdometry.image(f,m,side)

    def test_sensor_time_and_two_callback_times_remain_distinct(self):
        f=self.fixture();self.send(f,10,0,10.2);self.send(f,10,1,10.7)
        self.assertIsNotNone(f.queue.pending)
        left,right,queued,timing=f.queue.take()
        self.assertEqual(left.header.stamp.sec,right.header.stamp.sec)
        self.assertEqual(left.header.stamp.sec,10)
        self.assertEqual(timing['left_callback_ros_sec'],10.2)
        self.assertEqual(timing['right_callback_ros_sec'],10.7)
        self.assertEqual(timing['pair_ready_ros_sec'],10.7)

    def test_burst_replaces_pending_work_and_keeps_bounded_callback_evidence(self):
        f=self.fixture()
        for stamp in range(1,41):
            self.send(f,stamp,0,stamp+.1);self.send(f,stamp,1,stamp+.2)
        self.assertEqual(f.counts['replaced_pending'],39)
        self.assertEqual(len(f.ingress_events),64)
        self.assertEqual(f.queue.take()[0].header.stamp.sec,40)
        records=[]
        f.backend=SimpleNamespace(resource_snapshot=lambda:{})
        f.file_logger=SimpleNamespace(info=lambda text:records.append(json.loads(text)))
        LearnedOdometry.record(f,{'sensor_stamp_sec':40.})
        self.assertEqual(len(records[0]['ingress_events']),64)
        self.assertEqual(records[0]['ingress_events'][-1]['sensor_stamp_sec'],40.)
        self.assertEqual(records[0]['message_timing']['processing_start_ros_sec'],100.)
        self.assertEqual(len(f.ingress_events),0)


if __name__=='__main__':unittest.main()
