import copy
from array import array
import numpy as np
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import PointCloud2, PointField
from .core import rigid


def stamp_sec(msg):
    header = getattr(msg, "header", msg)
    return header.stamp.sec + 1e-9*header.stamp.nanosec


def transform_from_pose(pose):
    q = np.array([pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w])
    p = np.array([pose.position.x, pose.position.y, pose.position.z])
    if not np.isfinite(q).all() or abs(np.linalg.norm(q)-1) > 0.01 or not np.isfinite(p).all():
        raise ValueError("Invalid position/quaternion")
    t = np.eye(4)
    t[:3, :3] = Rotation.from_quat(q).as_matrix()
    t[:3, 3] = p
    return t


def set_pose(pose, t):
    t = rigid(t)
    pose.position.x, pose.position.y, pose.position.z = map(float, t[:3, 3])
    q = Rotation.from_matrix(t[:3, :3]).as_quat()
    pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w = map(float, q)


def cloud_arrays(msg, names=("x", "y", "z")):
    fields = {f.name: f for f in msg.fields}
    scalars = {1:"i1",2:"u1",3:"i2",4:"u2",5:"i4",6:"u4",7:"f4",8:"f8"}
    endian = ">" if msg.is_bigendian else "<"
    if msg.point_step <= 0 or msg.row_step < msg.width*msg.point_step:
        raise ValueError("Invalid cloud strides")
    if len(msg.data) < msg.row_step*msg.height:
        raise ValueError("Truncated PointCloud2")
    if any(n not in fields or fields[n].count != 1 for n in names):
        raise ValueError("Missing scalar fields: " + str(names))
    dtype = np.dtype({"names":list(names), "formats":[endian+scalars[fields[n].datatype] for n in names],
                      "offsets":[fields[n].offset for n in names], "itemsize":msg.point_step})
    a = np.ndarray((msg.height, msg.width), dtype=dtype, buffer=msg.data, strides=(msg.row_step, msg.point_step))
    return np.column_stack([a[n].reshape(-1) for n in names]).astype(float)


def xyz_cloud(points, header, extra=None):
    points = np.asarray(points, dtype="<f4").reshape(-1, 3)
    if extra is not None:
        points = np.column_stack([points, np.asarray(extra,dtype="<f4")]).astype("<f4")
    names = ["x","y","z"] + (["intensity"] if extra is not None else [])
    msg = PointCloud2()
    msg.header = copy.deepcopy(header)
    msg.height, msg.width = 1, len(points)
    msg.fields = [PointField(name=n, offset=4*i, datatype=PointField.FLOAT32, count=1) for i,n in enumerate(names)]
    msg.point_step = len(names)*4
    msg.row_step = msg.width*msg.point_step
    msg.is_bigendian = False
    msg.is_dense = bool(np.isfinite(points).all())
    msg.data = array('B',points.tobytes())
    return msg


def typed(values, code):
    a = array(code)
    dtype = {"f":"float32","b":"int8","B":"uint8","I":"uint32"}[code]
    a.frombytes(np.asarray(values, dtype=dtype).tobytes())
    return a
