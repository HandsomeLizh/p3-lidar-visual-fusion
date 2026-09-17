"""One frozen clock offset shared by the Ouster LiDAR and its internal IMU.

This is a reception-based clock estimate, not PTP synchronization. Sensor
intervals are preserved exactly. A reset/drift invalidates the session instead
of silently changing the time base of an existing map.
"""
from collections import deque
import math
import numpy as np


class SharedSensorClock:
    def __init__(self, mode="device_uptime", warmup_seconds=1., min_samples=80,
                 max_jitter=.03, max_drift=.10, max_age=.75):
        if mode not in ("device_uptime", "system"):
            raise ValueError("Unknown hardware clock mode")
        self.mode=mode;self.warmup_seconds=warmup_seconds;self.min_samples=min_samples
        self.max_jitter=max_jitter;self.max_drift=max_drift;self.max_age=max_age
        self.samples=deque(maxlen=400);self.last_imu=None;self.offset=0. if mode=="system" else None
        self.jitter=None;self.failure=None;self.drift_samples=deque(maxlen=100)

    def observe_imu(self, stamp, arrival):
        if self.failure:raise ValueError(self.failure)
        if not math.isfinite(stamp) or not math.isfinite(arrival) or stamp<=0:
            raise ValueError("Invalid hardware timestamp")
        if self.last_imu is not None and stamp<=self.last_imu:
            if stamp<self.last_imu-1.:
                self.failure="Sensor clock reset; restart mapping with a new time base"
            raise ValueError(self.failure or "Out-of-order IMU")
        self.last_imu=stamp
        if self.mode=="system":return self.convert(stamp,arrival)
        offset=arrival-stamp
        if self.offset is None:
            self.samples.append((stamp,offset))
            if len(self.samples)<self.min_samples or stamp-self.samples[0][0]<self.warmup_seconds:return None
            values=np.asarray([s[1] for s in self.samples])
            low,high=np.quantile(values,[.05,.95]);self.jitter=float(high-low)
            if self.jitter>self.max_jitter:return None
            self.offset=float(low)
        self.drift_samples.append(offset-self.offset)
        if len(self.drift_samples)==self.drift_samples.maxlen:
            drift=float(np.quantile(self.drift_samples,.05))
            if abs(drift)>self.max_drift:
                self.failure="Sensor/host clock drift exceeded limit; restart time alignment"
                raise ValueError(self.failure)
        return self.convert(stamp,arrival)

    def convert(self,stamp,arrival):
        if self.failure:raise ValueError(self.failure)
        if self.offset is None:return None
        mapped=stamp+self.offset
        if not math.isfinite(mapped) or mapped<=0 or arrival-mapped>self.max_age or mapped-arrival>.05:
            raise ValueError("Stale/future hardware measurement")
        return mapped

    def adopt(self, status):
        """Use the IMU worker's frozen estimate without fitting a second clock."""
        if self.failure:raise ValueError(self.failure)
        if status.get('failure'):
            self.failure=str(status['failure']);raise ValueError(self.failure)
        if not status.get('ready'):return False
        offset=status.get('offset_seconds')
        if status.get('mode')!=self.mode or not isinstance(offset,(int,float)) or not math.isfinite(offset):
            raise ValueError('Invalid shared hardware clock estimate')
        if self.offset is not None and offset!=self.offset:
            self.failure='Shared hardware clock changed; restart mapping with a new time base'
            raise ValueError(self.failure)
        self.offset=float(offset);self.jitter=status.get('reception_jitter_seconds')
        return True

    def status(self):
        return dict(mode=self.mode,ready=self.offset is not None and self.failure is None,
                    offset_seconds=self.offset,reception_jitter_seconds=self.jitter,
                    synchronization="shared_fixed_reception_estimate" if self.mode=="device_uptime" else "source_system_clock",
                    failure=self.failure)
