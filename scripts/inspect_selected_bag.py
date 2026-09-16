#!/usr/bin/env python3
"""Read-only sensor/reference inspection. Never imports ROS or a learned model."""
import csv
import hashlib
import json
import math
import sqlite3
import struct
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
BAG = Path("/home/yanfa/P3/roma_t3_algorithm_bundle_20260825/recordings/short_20260915_190151/bag")
TOPICS = ("/Car/T5/Cam_Left/image_raw/color",
          "/Car/T5/Cam_Right/image_raw/color", "/Car/T5/OS1/points")
BASE_NS = 1000000000000

class CDR:
    def __init__(self, blob):
        if blob[:2] not in (b"\x00\x00", b"\x00\x01"):
            raise ValueError("Expected CDR v1 serialization")
        self.blob, self.offset = blob, 4
        self.endian = "<" if blob[1] == 1 else ">"

    def number(self, fmt, alignment):
        self.offset += (-(self.offset - 4)) % alignment
        value = struct.unpack_from(self.endian + fmt, self.blob, self.offset)[0]
        self.offset += struct.calcsize(fmt)
        return value

    def string(self):
        length = self.number("I", 4)
        if length < 1 or self.offset + length > len(self.blob):
            raise ValueError("Invalid CDR string")
        result = self.blob[self.offset:self.offset + length - 1].decode()
        self.offset += length
        return result

    def header(self):
        return self.number("i", 4) * 10**9 + self.number("I", 4), self.string()

def database(path):
    connection = sqlite3.connect("file:" + quote(str(path.resolve())) + "?mode=ro", uri=True)
    connection.execute("PRAGMA query_only=ON")
    connection.execute("PRAGMA cache_size=-2048")
    return connection

def stats(values):
    a = np.asarray(values, dtype=float)
    return dict(count=len(a), min=float(a.min()), median=float(np.median(a)),
                p95=float(np.percentile(a, 95)), max=float(a.max())) if len(a) else {}

def main():
    metadata_path = BAG / "metadata.yaml"
    meta = yaml.safe_load(metadata_path.read_text())["rosbag2_bagfile_information"]
    sensors = {t: [] for t in TOPICS}
    reference = []
    parts = []
    for relative in meta["relative_file_paths"]:
        path = BAG / relative
        stat = path.stat()
        parts.append(dict(name=relative, bytes=stat.st_size, mtime_ns=stat.st_mtime_ns))
        with database(path) as con:
            ids = {n: i for i, n in con.execute("SELECT id,name FROM topics")}
            for topic in TOPICS:
                if topic not in ids:
                    raise ValueError("Missing sensor: " + topic)
                for mid, received, prefix in con.execute(
                    "SELECT id,timestamp,substr(data,1,2048) FROM messages "
                    "WHERE topic_id=? ORDER BY timestamp", (ids[topic],)):
                    reader = CDR(prefix)
                    stamp, frame = reader.header()
                    height, width = reader.number("I", 4), reader.number("I", 4)
                    row = dict(part=relative, message_id=mid, header_ns=stamp,
                               receive_ns=received, frame=frame, height=height, width=width)
                    if topic != TOPICS[2]:
                        row["encoding"] = reader.string()
                    else:
                        count = reader.number("I", 4)
                        row["fields"] = [dict(name=reader.string(), offset=reader.number("I", 4),
                            datatype=reader.number("B", 1), count=reader.number("I", 4))
                            for _ in range(count)]
                    sensors[topic].append(row)
            if "/car/pose" not in ids:
                raise ValueError("No independent simulator reference")
            for received, data in con.execute(
                "SELECT timestamp,data FROM messages WHERE topic_id=? ORDER BY timestamp",
                (ids["/car/pose"],)):
                reader = CDR(data)
                stamp, frame = reader.header()
                pose = [reader.number("d", 8) for _ in range(7)]
                if not all(math.isfinite(x) for x in pose):
                    raise ValueError("Nonfinite reference")
                reference.append((stamp, *pose, received))
    for topic, rows in sensors.items():
        stamps = [row["header_ns"] for row in rows]
        if any(b <= a for a, b in zip(stamps, stamps[1:])):
            raise ValueError("Nonmonotonic sensor clock: " + topic)
    stamps = [r["header_ns"] for r in sensors[TOPICS[2]]]
    if any([r["header_ns"] for r in sensors[t]] != stamps for t in TOPICS):
        raise ValueError("Exact stereo/LiDAR triplets required by selected player")
    first = stamps[0]
    dt = np.diff(np.asarray(stamps, dtype=np.int64)) / 1e9
    ref = sorted(reference)
    if any(b[0] <= a[0] for a, b in zip(ref, ref[1:])):
        raise ValueError("Duplicate/nonmonotonic reference timestamps")
    positions = np.array([r[1:4] for r in ref])
    image_samples = []
    for index in sorted(set(np.linspace(0, len(stamps) - 1, 5, dtype=int).tolist())):
        row = sensors[TOPICS[0]][index]
        with database(BAG / row["part"]) as con:
            blob = con.execute("SELECT data FROM messages WHERE id=?", (row["message_id"],)).fetchone()[0]
        reader = CDR(blob)
        reader.header()
        height, width = reader.number("I", 4), reader.number("I", 4)
        encoding = reader.string()
        reader.number("B", 1)
        step = reader.number("I", 4)
        length = reader.number("I", 4)
        if encoding != "bgra8" or length != height * step or step < width * 4:
            raise ValueError("Unexpected selected image format")
        array = np.ndarray((height, width, 4), dtype=np.uint8, buffer=blob,
                           offset=reader.offset, strides=(step, 4, 1))
        small = array[::16, ::16, :3].astype(float)
        gray = small[:, :, 0] * .114 + small[:, :, 1] * .587 + small[:, :, 2] * .299
        image_samples.append(dict(frame=index, mean=float(gray.mean()), std=float(gray.std()),
            dark_fraction=float((gray <= 8).mean()), saturated_fraction=float((gray >= 247).mean())))
        del blob, array, small, gray
    calibration = BAG.parent / "calibration_spreadsheets_parallel_24p9.yaml"
    lidar_calibration = BAG.parent / "pointcloud_extrinsic_os1_mapping_only.yaml"
    profile_path = ROOT / "config/simulation_xfeat.yaml"
    profile = yaml.safe_load(profile_path.read_text())
    camera = yaml.safe_load(calibration.read_text())
    lidar = yaml.safe_load(lidar_calibration.read_text())
    axes = np.diag([1., -1., -1., 1.])
    calibration_checks = {
        "image_size": profile["input_image_size"] == [camera["raw_stereo"]["width"], camera["raw_stereo"]["height"]],
        "left_K": bool(np.allclose(profile["camera_k"], camera["raw_stereo"]["left"]["k"], atol=1e-10, rtol=0)),
        "right_K": bool(np.allclose(profile["camera_k"], camera["raw_stereo"]["right"]["k"], atol=1e-10, rtol=0)),
        "left_mount": bool(np.allclose(profile["base_from_camera_left"],
            axes @ np.array(camera["mounting"]["transform_vehicle_left"]), atol=1e-10, rtol=0)),
        "right_mount": bool(np.allclose(profile["base_from_camera_right"],
            axes @ np.array(camera["mounting"]["transform_vehicle_right"]), atol=1e-10, rtol=0)),
        "lidar_mount": bool(np.allclose(profile["base_from_lidar"],
            np.array(lidar["sensors"]["os1"]["transform_base_sensor"]) @ (axes if profile.get("lidar_input_basis")=="ros_flu" else np.eye(4)), atol=1e-10, rtol=0)),
    }
    if not all(calibration_checks.values()):
        raise ValueError("Calibration snapshot differs: " + str(calibration_checks))
    output = ROOT / "test_data/short_20260915_190151"
    output.mkdir(parents=True, exist_ok=True)
    with (output / "reference.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["replay_stamp_sec", "x", "y", "z", "qx", "qy", "qz", "qw", "record_replay_sec"])
        for r in ref:
            writer.writerow([f"{(BASE_NS + r[0] - first) / 1e9:.9f}", *r[1:8],
                             f"{(BASE_NS + r[8] - first) / 1e9:.9f}"])
    with (output / "sensor_frames.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["frame", "raw_header_ns", "replay_stamp_sec",
                         "left_record_delay_sec", "right_record_delay_sec", "lidar_record_delay_sec"])
        for i, stamp in enumerate(stamps):
            writer.writerow([i, stamp, f"{(BASE_NS + stamp - first) / 1e9:.9f}",
                *[(sensors[t][i]["receive_ns"] - stamp) / 1e9 for t in TOPICS]])
    report = dict(bag=str(BAG), inspected_utc=datetime.now(timezone.utc).isoformat(),
        read_only_inspection=True, playback_started=False, algorithm_tests_started=False,
        metadata_sha256=hashlib.sha256(metadata_path.read_bytes()).hexdigest(),
        parts=parts, bag_size_gib=sum(p["bytes"] for p in parts) / 2**30,
        record_duration_sec=meta["duration"]["nanoseconds"] / 1e9,
        sensor_duration_sec=(stamps[-1] - first) / 1e9,
        frames=len(stamps), header_time_scale=1.0, replay_base_ns=BASE_NS,
        first_raw_header_ns=first, last_raw_header_ns=stamps[-1],
        interval_sec=stats(dt), input_hz=(len(stamps)-1)/((stamps[-1]-first)/1e9),
        sensors={t:dict(count=len(rows), first=rows[0], last=rows[-1],
            record_delay_sec=stats([(r["receive_ns"]-r["header_ns"])/1e9 for r in rows]))
            for t, rows in sensors.items()},
        reference=dict(topic="/car/pose", count=len(ref), frame="odom",
            travel_m=float(np.linalg.norm(np.diff(positions, axis=0), axis=1).sum()),
            net_displacement_m=float(np.linalg.norm(positions[-1]-positions[0])),
            height_range_m=float(np.ptp(positions[:, 2])),
            interval_sec=stats(np.diff(np.array([r[0] for r in ref], dtype=np.int64))/1e9),
            qualification="Independent simulator telemetry. Request-anchored sensor clock and telemetry receive clock; absolute exposure-time offset is not certified. Position diagnostic only; reference body quaternion convention has not been verified."),
        calibration_checks=calibration_checks,
        calibration_files={str(p):hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in [calibration, lidar_calibration, profile_path]},
        image_samples=image_samples,
        limitations=["58 frames at approximately 0.25 Hz cannot establish maximum sustainable FPS.",
                     "Five image samples do not certify extreme-lighting coverage.",
                     "No independent ground surface truth is present for elevation-map accuracy."])
    (output / "selection.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k:report[k] for k in ["bag", "frames", "sensor_duration_sec", "interval_sec",
                     "input_hz", "reference", "calibration_checks", "image_samples"]}, indent=2))

if __name__ == "__main__":
    main()
