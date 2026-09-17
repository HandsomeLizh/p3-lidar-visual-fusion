"""Display-only geometry and continuous elevation colors (no ROS dependency)."""
import numpy as np


UNKNOWN_COLOR = (24, 28, 34)
HEIGHT_STOPS = np.asarray([[30, 66, 160], [30, 175, 195], [249, 149, 53]], dtype=float)


class GridSourceCache:
    """Keep one bounded payload per source; an invalid global must not erase local."""
    def __init__(self, max_global_lag=5.):
        self.maps = {'global': None, 'local': None}
        self.max_global_lag = max_global_lag

    def update(self, source, payload):
        if source not in self.maps:
            raise ValueError('Unknown grid source: ' + source)
        self.maps[source] = payload
        global_map, local_map = self.maps['global'], self.maps['local']
        if global_map is not None and (local_map is None or
                global_map['stamp'] + self.max_global_lag >= local_map['stamp']):
            return 'global', global_map
        return ('local', local_map) if local_map is not None else ('', None)


def localization_status(age, outcome='', has_path=False, phase=None):
    """End-of-input is distinct from missing telemetry, without a source badge."""
    if phase in ('completed', 'stopped'):
        return '数据输入已结束 · 地图已保留，可继续旋转和缩放', '#b9c9dc'
    if phase == 'error':
        return '数据输入异常停止 · 已保留现有地图，请检查日志', '#ffbe6b'
    healthy = age is not None and age < 15 and outcome in ('', 'completed')
    label = '定位更新正常' if healthy else (
        '地图与轨迹已载入 · 等待定位更新' if has_path and age is None else '等待 / 检查定位数据')
    if age is not None:
        label += f'   最近更新 {age:.1f} 秒前'
    return label, '#6de4cc' if healthy else '#ffbe6b'


def height_colors(values, low, high):
    """Blue -> cyan -> orange, linearly interpolated, not discrete height bins."""
    values = np.asarray(values)
    normalized = np.clip((np.nan_to_num(values, nan=low) - low) / max(high-low, 1e-6), 0, 1)
    rgb = np.stack([np.interp(normalized, [0, .5, 1], HEIGHT_STOPS[:, i])
                    for i in range(3)], axis=-1).astype(np.uint8)
    rgb[~np.isfinite(values)] = UNKNOWN_COLOR
    return rgb


def expand_height_limits(values, previous=None):
    finite = np.asarray(values)[np.isfinite(values)]
    if not finite.size:
        return previous or (-1., 1.)
    low, high = float(np.floor(finite.min())), float(np.ceil(finite.max()))
    if high <= low:
        high = low + 1.
    if previous is not None:
        low, high = min(low, previous[0]), max(high, previous[1])
    return low, high


def display_sample(points, voxel=0.15, maximum=100000):
    points = np.asarray(points).reshape(-1, 3)
    points = points[np.isfinite(points).all(axis=1)]
    if len(points):
        _, indices = np.unique(np.floor(points / voxel).astype(np.int64), axis=0, return_index=True)
        points = points[np.sort(indices)]
        if len(points) > maximum:
            points = points[np.linspace(0, len(points) - 1, maximum, dtype=np.int64)]
    return points


def planning_points_in_map(message,lookup_transform):
    """Bounded display geometry; every point must use the declared path frame."""
    frame=message.header.frame_id.lstrip('/')
    if not frame:raise ValueError('Path frame is missing')
    poses=message.poses
    if len(poses)>100000:raise ValueError('Path exceeds display input limit')
    if not poses:return np.empty((0,3))
    if any(p.header.frame_id and p.header.frame_id.lstrip('/')!=frame for p in poses):
        raise ValueError('Path has mixed point frames')
    indices=np.linspace(0,len(poses)-1,min(len(poses),10000),dtype=int)
    points=np.asarray([[poses[i].pose.position.x,poses[i].pose.position.y,
                        poses[i].pose.position.z] for i in indices],dtype=float)
    if not np.isfinite(points).all():raise ValueError('Path contains nonfinite coordinates')
    if frame!='map':
        stamp_ns=message.header.stamp.sec*10**9+message.header.stamp.nanosec
        if not stamp_ns and frame!='odom':raise ValueError('Moving-frame path has no timestamp')
        transform=np.asarray(lookup_transform(frame,stamp_ns),dtype=float)
        if (transform.shape!=(4,4) or not np.isfinite(transform).all() or
                not np.allclose(transform[3],[0,0,0,1],atol=1e-8) or
                not np.allclose(transform[:3,:3].T@transform[:3,:3],np.eye(3),atol=1e-5) or
                abs(np.linalg.det(transform[:3,:3])-1.)>1e-5):raise ValueError('Invalid path transform')
        points=points@transform[:3,:3].T+transform[:3,3]
    return points
