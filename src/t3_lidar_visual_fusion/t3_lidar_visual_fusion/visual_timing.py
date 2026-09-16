"""Age limits for visual measurements; timestamps stay in the sensor time base."""
from dataclasses import dataclass, asdict
import math


@dataclass(frozen=True)
class TimingDecision:
    valid: bool
    reason: str
    sensor_age_sec: float | None
    queue_wait_sec: float | None

    def dictionary(self):
        return asdict(self)


class VisualTiming:
    def __init__(self, max_age_sec=1.5, future_tolerance_sec=.1):
        self.max_age_sec=float(max_age_sec)
        self.future_tolerance_sec=float(future_tolerance_sec)
        if not math.isfinite(self.max_age_sec) or self.max_age_sec<=0:
            raise ValueError("Visual age budget must be positive and finite")
        if not math.isfinite(self.future_tolerance_sec) or self.future_tolerance_sec<0:
            raise ValueError("Future timestamp tolerance must be nonnegative and finite")

    def check(self, stamp, now_ros, queued_wall, now_wall):
        values=(stamp,now_ros,queued_wall,now_wall)
        if not all(math.isfinite(v) for v in values):
            return TimingDecision(False,"invalid_visual_time",None,None)
        age=now_ros-stamp
        wait=now_wall-queued_wall
        if stamp<=0:
            reason="invalid_visual_stamp"
        elif now_ros<=0:
            reason="visual_clock_not_ready"
        elif wait<0:
            reason="invalid_visual_wall_clock"
        elif age < -self.future_tolerance_sec:
            reason="visual_stamp_in_future"
        elif age>self.max_age_sec:
            reason="visual_measurement_expired"
        elif wait>self.max_age_sec:
            reason="visual_queue_expired"
        else:
            return TimingDecision(True,"timely",age,wait)
        return TimingDecision(False,reason,age,wait)
