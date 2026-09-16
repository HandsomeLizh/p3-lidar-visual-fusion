"""Bounded playback of validated legacy batches using bag receive-time intervals."""
import json
import sqlite3
from collections import OrderedDict
from pathlib import Path
from urllib.parse import quote

def groups(directory, index_path, limit=0, lidar_only=False):
    from rclpy.serialization import deserialize_message
    from sensor_msgs.msg import Image, PointCloud2
    directory = Path(directory).resolve()
    index_path = Path(index_path)
    if index_path.stat().st_size > 32 * 2**20:
        raise ValueError("Frame index exceeds 32 MiB")
    index = json.loads(index_path.read_text())
    if Path(index["bag"]).resolve() != directory:
        raise ValueError("Frame index belongs to another bag")
    for part in index["parts"]:
        path = (directory / part["name"]).resolve()
        if not path.is_relative_to(directory):
            raise ValueError("Invalid bag part path")
        st = path.stat()
        if st.st_size != part["bytes"] or st.st_mtime_ns != part["mtime_ns"]:
            raise ValueError("Bag changed after indexing: " + str(path))
    cache = OrderedDict()
    last = None
    try:
        for number, frame in enumerate(index["frames"]):
            if limit > 0 and number >= limit:
                break
            stamp = int(frame["record_ns"])
            if last is not None and stamp <= last:
                raise ValueError("Nonmonotonic index")
            last = stamp
            messages = {}
            total = 0
            for topic, item in frame["messages"].items():
                if lidar_only and topic != "/Car/T5/OS1/points":
                    continue
                part = item["part"]
                path = (directory / part).resolve()
                if not path.is_relative_to(directory):
                    raise ValueError("Invalid message part")
                if part not in cache:
                    while len(cache) >= 2:
                        _, old = cache.popitem(last=False)
                        old.close()
                    con = sqlite3.connect("file:" + quote(str(path)) + "?mode=ro", uri=True)
                    con.execute("PRAGMA query_only=ON")
                    con.execute("PRAGMA cache_size=-2048")
                    cache[part] = con
                con = cache[part]
                cache.move_to_end(part)
                row = con.execute("SELECT t.name,m.data FROM messages m JOIN topics t ON t.id=m.topic_id WHERE m.id=?",
                                  (item["id"],)).fetchone()
                if row is None or row[0] != topic:
                    raise ValueError("Frame index topic mismatch")
                total += len(row[1])
                if total > 96 * 2**20:
                    raise ValueError("Batch exceeds 96 MiB")
                kind = PointCloud2 if topic == "/Car/T5/OS1/points" else Image
                messages[topic] = deserialize_message(row[1], kind)
            if len(messages) != (1 if lidar_only else 3):
                raise ValueError("Incomplete indexed sensor batch")
            yield stamp, messages
    finally:
        for con in cache.values():
            con.close()
