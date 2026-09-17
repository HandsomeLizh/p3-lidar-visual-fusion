"""Bounded live ingress. No ROS dependencies; capture stamps are never rewritten."""
from collections import Counter, deque
from dataclasses import dataclass
import threading
import time


@dataclass
class Sample:
    key: str
    message: object
    stamp_ns: int
    received: float
    initial_age: float


class SensorInbox:
    """Receive callbacks only validate/enqueue. One worker owns conversion/publishing.

    At most two unmatched frames per camera, one complete stereo pair, one LiDAR
    cloud and a finite IMU FIFO wait here. A worker may hold one additional group.
    """
    def __init__(self, profile, wall=time.time, monotonic=time.monotonic):
        self.profile, self.wall, self.monotonic = profile, wall, monotonic
        settings = profile.get('live_transport', {})
        image_age = max(.1, float(profile.get('visual_max_age_sec', 4.0)) - .5)
        self.max_age = dict(left=image_age, right=image_age,
                            lidar=float(settings.get('lidar_max_age_sec', image_age)),
                            imu=float(settings.get('imu_max_age_sec', .5)))
        self.future = float(profile.get('visual_future_tolerance_sec', .1))
        self.skew_ns = int(float(profile.get('stereo_max_skew', .01)) * 1e9)
        self.imu_capacity = int(settings.get('imu_queue_samples', 256))
        if not 2 <= self.imu_capacity <= 4096 or any(not 0 < v <= 30 for v in self.max_age.values()):
            raise ValueError('Invalid bounded live transport configuration')
        self.cv = threading.Condition()
        self.counts = Counter()
        self.last_seen, self.last_forwarded, self.last_received = {}, {}, {}
        self.last_forwarded_wall, self.last_forwarded_age = {}, {}
        self.max_input_gap = {}
        self.images = {'left': {}, 'right': {}}
        self.stereo = None
        self.cloud = None
        self.imu = deque()
        self.closed = False

    def count(self, key, n=1):
        with self.cv:
            self.counts[key] += n

    def fresh(self, sample):
        # Monotonic elapsed age protects against a backwards system clock step.
        age = max(self.wall() - sample.stamp_ns / 1e9,
                  sample.initial_age + self.monotonic() - sample.received)
        return -self.future <= age <= self.max_age[sample.key]

    def offer(self, key, message):
        ns = message.header.stamp.sec * 10**9 + message.header.stamp.nanosec
        now, wall = self.monotonic(), self.wall()
        sample = Sample(key, message, ns, now, wall - ns / 1e9)
        with self.cv:
            if self.closed:
                return False
            self.counts[key + '_received'] += 1
            self.last_received[key] = now
            if ns <= 0 or ns <= self.last_seen.get(key, -1):
                self.counts[key + '_old_or_duplicate'] += 1
                return False
            if not self.fresh(sample):
                self.counts[key + '_stale'] += 1
                return False
            maximum = self.profile.get('max_image_bytes', 24000000) if key in self.images else self.profile.get('max_cloud_bytes', 16000000)
            if key != 'imu' and len(message.data) > maximum:
                self.counts[key + '_oversized'] += 1
                return False
            previous = self.last_seen.get(key)
            if previous is not None:
                gap = (ns - previous) / 1e9
                self.max_input_gap[key] = max(gap, self.max_input_gap.get(key, 0.0))
                if gap > self.profile.get('vision_gate', {}).get('max_gap', 6.0):
                    self.counts[key + '_input_gap'] += 1
            self.last_seen[key] = ns
            if key in self.images:
                other = 'right' if key == 'left' else 'left'
                matches = [stamp for stamp in self.images[other] if abs(stamp - ns) <= self.skew_ns]
                if matches:
                    stamp = min(matches, key=lambda t: abs(t - ns))
                    partner = self.images[other].pop(stamp)
                    if self.stereo is not None:
                        self.counts['stereo_replaced'] += 1
                    pair = {key: sample, other: partner}
                    self.stereo = [pair['left'], pair['right']]
                    # Frames older than this complete pair cannot form a later pair.
                    for side in self.images:
                        obsolete = [s for s in self.images[side] if s <= pair[side].stamp_ns]
                        for s in obsolete:
                            del self.images[side][s]
                            self.counts[side + '_unmatched'] += 1
                else:
                    self.images[key][ns] = sample
                    while len(self.images[key]) > 2:
                        del self.images[key][min(self.images[key])]
                        self.counts[key + '_unmatched'] += 1
            elif key == 'lidar':
                if self.cloud is not None:
                    self.counts['lidar_replaced'] += 1
                self.cloud = sample
            elif key == 'imu':
                if len(self.imu) >= self.imu_capacity:
                    self.imu.popleft()
                    self.counts['imu_overflow'] += 1
                self.imu.append(sample)
            self.cv.notify()
        return True

    def take(self):
        with self.cv:
            while not self.closed:
                candidates = []
                if self.stereo:
                    candidates.append((self.stereo[0].stamp_ns, 'stereo'))
                if self.cloud:
                    candidates.append((self.cloud.stamp_ns, 'cloud'))
                if self.imu:
                    candidates.append((self.imu[0].stamp_ns, 'imu'))
                if candidates:
                    _, kind = min(candidates)
                    if kind == 'stereo':
                        result, self.stereo = self.stereo, None
                    elif kind == 'cloud':
                        result, self.cloud = [self.cloud], None
                    else:
                        result = [self.imu.popleft()]
                    return result
                self.cv.wait(.1)
            return None

    def forwarded(self, sample):
        with self.cv:
            self.last_forwarded[sample.key] = sample.stamp_ns
            self.last_forwarded_wall[sample.key] = self.monotonic()
            self.last_forwarded_age[sample.key] = self.wall() - sample.stamp_ns / 1e9
            self.counts[sample.key + '_forwarded'] += 1

    def snapshot(self):
        with self.cv:
            now = self.monotonic()
            return dict(counts=dict(self.counts), last_forwarded_stamp_ns=dict(self.last_forwarded),
                sensor_idle_seconds={k: now-v for k, v in self.last_received.items()},
                forwarded_idle_seconds={k: now-v for k, v in self.last_forwarded_wall.items()},
                last_forwarded_age_sec=dict(self.last_forwarded_age),
                max_input_gap_sec=dict(self.max_input_gap),
                pending=dict(left=len(self.images['left']), right=len(self.images['right']),
                    stereo=int(self.stereo is not None), lidar=int(self.cloud is not None), imu=len(self.imu)))

    def close(self):
        with self.cv:
            self.closed = True
            self.images = {'left': {}, 'right': {}}
            self.stereo = self.cloud = None
            self.imu.clear()
            self.cv.notify_all()
