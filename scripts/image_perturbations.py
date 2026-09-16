"""Deterministic photometric interventions, indexed by original stereo batch."""
import copy
import numpy as np


class ImagePerturbations:
    def __init__(self, schedule):
        self.schedule = schedule
        previous_end = 0
        for block in schedule:
            start, end = block["start"], block["end"]
            if not isinstance(start, int) or not isinstance(end, int) or start < previous_end or end <= start:
                raise ValueError("Perturbations must be ordered, nonoverlapping, zero-based [start,end) intervals")
            if block["kind"] not in ("blackout", "overexposure", "diagonal_glare", "low_light", "grayscale_control"):
                raise ValueError("Unknown photometric intervention")
            previous_end = end

    def block(self, frame):
        return next((b for b in self.schedule if b["start"] <= frame < b["end"]), None)

    def kind(self, frame):
        block = self.block(frame)
        return block["kind"] if block else "clean"

    def apply(self, message, frame):
        block = self.block(frame)
        if block is None or block["kind"] == "grayscale_control":
            # This frontend already uses grayscale. The control is byte-identical.
            return message
        if message.encoding != "mono8" or message.step != message.width:
            raise ValueError("Perturbations require compact calibrated mono8 input")
        source = np.frombuffer(message.data, np.uint8).reshape(message.height, message.width)
        kind = block["kind"]
        if kind == "blackout":
            changed = np.zeros_like(source)
        elif kind == "overexposure":
            changed = np.clip(source.astype(np.float32)*8.+220., 0., 255.).astype(np.uint8)
        elif kind == "low_light":
            changed = np.rint(source.astype(np.float32)*.025).astype(np.uint8)
        else:
            y, x = np.mgrid[0:message.height, 0:message.width].astype(np.float32)
            x /= max(1, message.width-1); y /= max(1, message.height-1)
            distance = (x-y-.05)/np.sqrt(2.)
            band = np.exp(-.5*(distance/.16)**2)
            changed = np.clip(source.astype(np.float32)+300.*band, 0., 255.).astype(np.uint8)
        output = copy.deepcopy(message); output.data = changed.tobytes()
        return output
