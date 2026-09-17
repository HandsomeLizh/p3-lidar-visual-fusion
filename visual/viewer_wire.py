"""Bounded, lossless ROS GridMap transport for the remote display only."""
import zlib
from array import array

from grid_map_msgs.msg import GridMap
from rclpy.serialization import serialize_message, deserialize_message
from std_msgs.msg import UInt8MultiArray

PREFIX = '/viewer104_transport'
MAX_RAW_BYTES = 64 * 1024 * 1024
MAX_WIRE_BYTES = 16 * 1024 * 1024


def encode_grid(message):
    # Keep every layer used by the viewer, including goal/obstacle checks.
    # The original full GridMap for planners is never changed.
    view = GridMap(header=message.header, info=message.info,
                   outer_start_index=message.outer_start_index,
                   inner_start_index=message.inner_start_index)
    indices = [i for i, name in enumerate(message.layers)
               if name in ('elevation', 'occupancy', 'obstacle')]
    view.layers = [message.layers[i] for i in indices]
    view.basic_layers = [name for name in message.basic_layers if name in view.layers]
    view.data = [message.data[i] for i in indices]
    raw = serialize_message(view)
    if len(raw) > MAX_RAW_BYTES:
        raise ValueError('Display grid exceeds decoded byte limit')
    packed = zlib.compress(raw, 1)
    if len(packed) > MAX_WIRE_BYTES:
        raise ValueError('Display grid exceeds wire byte limit')
    return UInt8MultiArray(data=array('B', packed)), len(raw)


def decode_grid(message):
    if not 0 < len(message.data) <= MAX_WIRE_BYTES:
        raise ValueError('Invalid compressed grid length')
    decoder = zlib.decompressobj()
    raw = decoder.decompress(bytes(message.data), MAX_RAW_BYTES + 1)
    if (len(raw) > MAX_RAW_BYTES or not decoder.eof or decoder.unused_data
            or decoder.unconsumed_tail):
        raise ValueError('Incomplete or oversized compressed grid')
    try:
        return deserialize_message(raw, GridMap)
    except Exception as error:
        raise ValueError('Malformed serialized grid') from error
