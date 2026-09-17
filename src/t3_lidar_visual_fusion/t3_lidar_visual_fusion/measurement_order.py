"""Bound a faster visual stream's lead over delayed, qualified LiDAR poses."""
from collections import deque
import math


class VisualConstraintQueue:
    def __init__(self, wait_sec=0., capacity=32):
        if not math.isfinite(wait_sec) or wait_sec < 0 or capacity < 1:
            raise ValueError('Invalid visual constraint queue bounds')
        self.wait_sec = float(wait_sec)
        self.capacity = int(capacity)
        self.pending = deque()
        self.released = self.expired = self.dropped = 0
        self.last_wait_sec = 0.

    def append(self, stamp, received, value):
        if len(self.pending) == self.capacity:
            self.pending.popleft(); self.dropped += 1
        self.pending.append((stamp, received, value))

    def clear(self):
        self.dropped += len(self.pending)
        self.pending.clear()

    def ready(self, lidar_stamp, now, lidar_active):
        result = []
        while self.pending:
            stamp, received, value = self.pending[0]
            wait = now - received
            ahead = stamp > lidar_stamp + 1e-6
            if ahead and lidar_active and wait < self.wait_sec:
                break
            self.pending.popleft()
            self.released += 1
            self.expired += int(ahead and lidar_active and wait >= self.wait_sec)
            self.last_wait_sec = max(0., wait)
            result.append(value)
        return result

    def status(self):
        return dict(pending=len(self.pending), released=self.released,
                    expired=self.expired, dropped=self.dropped,
                    last_wait_sec=self.last_wait_sec, max_wait_sec=self.wait_sec)
