"""Private P3 map display with explicit, one-shot operator goals for P4."""
import json
import io
import math
import os
from pathlib import Path as FilePath
import signal
import threading
import time
import zlib
from array import array
import tkinter as tk

import numpy as np
from PIL import Image as PILImage, ImageTk
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from rclpy.signals import SignalHandlerOptions
from geometry_msgs.msg import Point, TransformStamped, PoseStamped
from tf2_ros import StaticTransformBroadcaster
from grid_map_msgs.msg import GridMap
from nav_msgs.msg import Odometry, Path
from sensor_msgs.msg import Image, CompressedImage, PointCloud2, PointField
from std_msgs.msg import String, UInt8MultiArray
from visualization_msgs.msg import Marker, MarkerArray
from t3_lidar_visual_fusion.ros_utils import cloud_arrays as pointcloud2_xyz_array
from visual_style import display_sample, expand_height_limits, height_colors, localization_status, UNKNOWN_COLOR
from viewer_wire import decode_grid


def write_visual_state(name, value):
    directory = os.environ.get('T3_VISUAL_RUNTIME')
    if not directory:
        return
    path = FilePath(directory) / name
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value))
    temporary.replace(path)


def colored_cloud(header, points, limits):
    rgb = height_colors(points[:, 2], *limits).astype(np.uint32)
    packed = np.empty(len(points), dtype=[('x', '<f4'), ('y', '<f4'), ('z', '<f4'), ('rgb', '<u4')])
    for i, axis in enumerate('xyz'):
        packed[axis] = points[:, i]
    packed['rgb'] = (rgb[:, 0] << 16) | (rgb[:, 1] << 8) | rgb[:, 2]
    fields = [PointField(name=axis, offset=4*i, datatype=PointField.FLOAT32, count=1)
              for i, axis in enumerate('xyz')]
    fields.append(PointField(name='rgb', offset=12, datatype=PointField.UINT32, count=1))
    return PointCloud2(header=header, height=1, width=len(points), fields=fields,
                      is_bigendian=False, point_step=16, row_step=len(points)*16,
                      data=array('B',packed.tobytes()), is_dense=True)


class Monitor(Node):
    def __init__(self):
        super().__init__('t3_visual_monitor')
        self.lock = threading.Lock()
        self.path = None
        self.pose_at = 0.
        self.timing = {}
        self.fusion_health = None
        self.fusion_health_at = 0.
        self.cloud_status = {}
        self.grid = None
        self.thumbnails = {}
        self.display_points = 0
        self.counter = 0
        self.height_limits = None
        self.full_height_limits = None
        self.contrast_enabled = False
        self.contrast_requested = False
        self.cloud_bounds = None
        self.cloud_points = None
        self.cloud_header = None
        self.elevation_data = None
        self.obstacle_data = None
        self.last_color_key = None
        self.last_grid_token = None
        self.local_grid_message = None
        self.local_grid_at = 0.
        self.global_grid_at = 0.
        self.global_grid_valid = False
        self.grid_source = 'none'
        self.grid_message_count = {'global': 0, 'local': 0}
        self.grid_display_at = 0.
        self.retain_stale_grid_sec = float(self.declare_parameter('retain_stale_grid_sec', 0.).value)
        compressed_prefix = str(self.declare_parameter('compressed_display_prefix', '').value).rstrip('/')
        self.grid_stamps = {}
        self.grid_frame = 'map'
        self.planning_paths = {}
        self.planning_revision = 0
        self.goal_requested = None
        self.last_goal = None
        self.goal_status = '点击“设置目标”，在已观测的可通行栅格上拖动指定朝向'
        self.create_timer(.1, self.send_requested_goal)
        self.create_timer(.25, self.apply_color_mode)
        self.create_timer(.5, self.check_grid_freshness)
        # Keep the fixed map frame available even before the first estimate.
        # This is only a display anchor, never map->base_link or an estimated pose.
        self.display_anchor = StaticTransformBroadcaster(self)
        anchor = TransformStamped()
        anchor.header.frame_id, anchor.child_frame_id = 'map', 't3_display_origin'
        anchor.transform.rotation.w = 1.
        self.display_anchor.sendTransform(anchor)
        live = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE)
        retained = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                              durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.cloud_pub = self.create_publisher(PointCloud2, '/T3/demo/rviz_cloud', retained)
        self.marker_pub = self.create_publisher(MarkerArray, '/T3/demo/markers', retained)
        # Goals must never be latched/replayed when a planning node reconnects.
        self.goal_pub = self.create_publisher(PoseStamped, '/Car/T4/rviz_goal', live)
        for label,topic in [('global','/Car/T4/planning/global_route'),('local','/Car/T4/planning/local_path')]:
            self.create_subscription(Path,topic,lambda msg,key=label:self.on_planning_path(key,msg),retained)
        self.create_subscription(Path, '/T3/semantic/trajectory', self.on_path, retained)
        self.create_subscription(Odometry, '/T3/semantic/current_pose', self.on_pose, live)
        self.create_subscription(PointCloud2, '/T3/mapping/lidar_map', self.on_cloud, retained)
        self.cloud_sources={}
        self.create_subscription(PointCloud2, '/T3/mapping/stereo_map',
                                 lambda msg:self.on_cloud(msg,'stereo'),retained)
        if compressed_prefix:
            for source in ('global', 'local'):
                self.create_subscription(UInt8MultiArray, compressed_prefix+'/'+source+'_grid_zlib',
                                         lambda msg, s=source: self.on_compressed_grid(s, msg), retained)
        else:
            self.create_subscription(GridMap, '/T3/mapping/global_grid_map', self.on_grid, retained)
            self.create_subscription(GridMap, '/Car/T3/mapping/grid_map', self.on_local_grid, retained)
        self.create_subscription(String, '/T3/mapping/lidar_status', self.on_status, retained)
        self.create_subscription(String, '/Car/T3/metrics/frame_timing', self.on_timing, live)
        self.create_subscription(String, '/fusion/status', self.on_fusion_health, live)
        for side in ('Left', 'Right'):
            if compressed_prefix:
                self.create_subscription(CompressedImage, compressed_prefix+'/'+side.lower()+'/compressed',
                                         lambda msg, key=side: self.on_compressed_image(key, msg), live)
            else:
                self.create_subscription(Image, f'/Car/T5/Cam_{side}/image_raw/color',
                                         lambda msg, key=side: self.on_image(key, msg), live)

    def on_compressed_grid(self, source, message):
        try:
            decoded = decode_grid(message)
            (self.on_grid if source == 'global' else self.on_local_grid)(decoded)
        except (ValueError, TypeError, zlib.error) as error:
            self.get_logger().warning('Compressed display grid rejected: '+str(error), throttle_duration_sec=5.)

    def on_compressed_image(self, side, message):
        if len(message.data) > 1024*1024:
            return
        try:
            bitmap = PILImage.open(io.BytesIO(bytes(message.data)))
            if bitmap.width*bitmap.height > 320*240:
                return
            bitmap = bitmap.convert('RGB')
            with self.lock:
                self.thumbnails[side] = bitmap
        except (ValueError, OSError):
            return

    def on_pose(self, _):
        with self.lock:
            self.pose_at = time.monotonic()
            self.counter += 1

    def on_timing(self, message):
        with self.lock:
            self.timing = json.loads(message.data)

    def on_status(self, message):
        with self.lock:
            self.cloud_status = json.loads(message.data)

    def on_fusion_health(self,message):
        try:
            value=json.loads(message.data)
            if not isinstance(value,dict) or not isinstance(value.get('localization_valid'),bool):return
            with self.lock:
                self.fusion_health=value;self.fusion_health_at=time.monotonic()
        except (ValueError,TypeError):return

    def localization_unavailable(self):
        return (self.fusion_health is not None and
            (not self.fusion_health.get('localization_valid') or time.monotonic()-self.fusion_health_at>3.))

    def on_cloud(self, message, source='lidar'):
        original = pointcloud2_xyz_array(message)
        finite = original[np.isfinite(original).all(axis=1)]
        self.cloud_sources[source]=(display_sample(finite),message.header.frame_id)
        finite=np.vstack([points for points,frame in self.cloud_sources.values() if frame==message.header.frame_id])
        # Each source is already voxel sampled and bounded. Joining it should
        # not repeat an expensive voxel sort over the entire LiDAR preview.
        points = finite if len(finite)<=100000 else finite[np.linspace(0,len(finite)-1,100000,dtype=int)]
        self.full_height_limits = expand_height_limits(finite[:, 2], self.full_height_limits)
        self.cloud_points, self.cloud_header = points, message.header
        self.update_color_limits()
        self.cloud_pub.publish(colored_cloud(message.header, points, self.height_limits))
        if len(finite):
            self.cloud_bounds = (finite.min(axis=0), finite.max(axis=0))
            self.write_bounds()
        self.recolor_grid()
        with self.lock:
            self.display_points = len(points)

    def write_bounds(self):
        if self.cloud_bounds is None:
            return
        low, high = (value.copy() for value in self.cloud_bounds)
        if self.path and self.path.poses:
            path = np.asarray([(p.pose.position.x, p.pose.position.y, p.pose.position.z)
                               for p in self.path.poses])
            low = np.minimum(low, path.min(axis=0))
            high = np.maximum(high, path.max(axis=0) + [0, 0, 3])
        write_visual_state('view_bounds.json', {'min': low.tolist(), 'max': high.tolist()})

    def on_image(self, side, message):
        channels = {'bgra8': 4, 'rgba8': 4, 'bgr8': 3, 'rgb8': 3, 'mono8': 1}.get(message.encoding)
        if channels is None:
            return
        raw = np.frombuffer(message.data, dtype=np.uint8).reshape(message.height, message.step)
        pixels = raw[:, :message.width * channels].reshape(message.height, message.width, channels)[::8, ::8]
        if channels == 1:
            pixels = np.repeat(pixels, 3, axis=2)
        else:
            pixels = pixels[:, :, :3]
            if message.encoding.startswith('bgr'):
                pixels = pixels[:, :, ::-1]
        thumbnail = PILImage.fromarray(np.ascontiguousarray(pixels))
        with self.lock:
            self.thumbnails[side] = thumbnail

    def on_grid(self, message):
        self.grid_message_count['global'] += 1
        if not self.grid_ordered(message,'global'):return
        try:
            ready=self.grid_payload_ready(message)
            if ready:
                self.render_grid(message)
                self.global_grid_at=time.monotonic();self.global_grid_valid=True
                self.grid_display_at=self.global_grid_at
                self.grid_source='global';return
        except (ValueError,IndexError,TypeError) as error:
            self.get_logger().warning('Global display grid rejected: '+str(error),throttle_duration_sec=5.)
        self.global_grid_valid=False
        self.show_local_grid()

    def on_local_grid(self,message):
        self.grid_message_count['local'] += 1
        if not self.grid_ordered(message,'local'):return
        try:ready=self.grid_payload_ready(message,160000)
        except (ValueError,IndexError,TypeError) as error:
            ready=False
            self.get_logger().warning('Local display grid rejected: '+str(error),throttle_duration_sec=5.)
        self.local_grid_message=message if ready else None
        self.local_grid_at=time.monotonic() if ready else 0.
        if not self.global_grid_valid:self.show_local_grid()

    def grid_ordered(self,message,source):
        if not message.layers:return True  # Explicit invalidation may have no stamp.
        stamp=message.header.stamp.sec+message.header.stamp.nanosec*1e-9
        if stamp<self.grid_stamps.get(source,-float('inf')):return False
        self.grid_stamps[source]=stamp
        return True

    @staticmethod
    def grid_payload_ready(message,limit=1000000):
        if 'elevation' not in message.layers or message.outer_start_index or message.inner_start_index:return False
        if message.header.frame_id not in ('map','odom'):raise ValueError('Unsupported grid frame')
        if len(message.layers)!=len(message.data) or not 1<=len(message.data)<=16:raise ValueError('Malformed grid layers')
        layer=message.data[list(message.layers).index('elevation')]
        if len(layer.layout.dim)!=2:raise ValueError('Malformed grid dimensions')
        ny,nx=(int(d.size) for d in layer.layout.dim)
        geometry=[message.info.resolution,message.info.length_x,message.info.length_y,
                  message.info.pose.position.x,message.info.pose.position.y]
        if nx<=0 or ny<=0 or nx*ny>limit or not np.isfinite(geometry).all() or min(geometry[:3])<=0:
            raise ValueError('Grid geometry exceeds display limits')
        if any(len(d.data)!=nx*ny for d in message.data):raise ValueError('Malformed grid layers')
        return bool(np.isfinite(np.asarray(layer.data)).any())

    def check_grid_freshness(self):
        if not (self.global_grid_valid and time.monotonic()-self.global_grid_at<=8.):
            self.global_grid_valid=False
            self.show_local_grid()
        write_visual_state('grid_display_status.json', dict(source=self.grid_source,
            messages=dict(self.grid_message_count),
            known_cells=self.grid[2] if self.grid else 0,
            receipt_age_sec=time.monotonic()-self.grid_display_at if self.grid_display_at else None,
            source_stamps=dict(self.grid_stamps), fresh=self.grid_is_fresh()))

    def grid_is_fresh(self):
        return ((self.grid_source=='global' and time.monotonic()-self.global_grid_at<=8.) or
                (self.grid_source=='local' and time.monotonic()-self.local_grid_at<=5.))

    def show_local_grid(self):
        if self.local_grid_message is not None and time.monotonic()-self.local_grid_at<=5.:
            self.render_grid(self.local_grid_message);self.grid_source='local'
            self.grid_display_at=self.local_grid_at;return
        self.local_grid_message=None
        with self.lock:
            if self.grid is not None and time.monotonic()-self.grid_display_at<=self.retain_stale_grid_sec:
                self.grid_source='stale';self.goal_requested=None;return
            self.elevation_data=None;self.obstacle_data=None;self.grid=None
            self.last_grid_token=None;self.grid_source='none'
            self.goal_requested=None

    def render_grid(self, message):
        if 'elevation' not in message.layers or message.outer_start_index or message.inner_start_index:
            with self.lock:
                self.elevation_data=None;self.obstacle_data=None;self.grid=None
                self.last_grid_token=None
            return
        layer = message.data[list(message.layers).index('elevation')]
        ny, nx = (int(d.size) for d in layer.layout.dim)
        if nx<=0 or ny<=0 or nx*ny>1000000 or not np.isfinite(message.info.resolution) or message.info.resolution<=0:
            raise ValueError('Grid geometry exceeds display limits')
        if len(message.data)>16 or any(len(d.data)!=nx*ny for d in message.data):
            raise ValueError('Malformed grid layers')
        token=(message.header.frame_id,nx,ny,message.info.resolution,message.info.length_x,message.info.length_y,
               message.info.pose.position.x,message.info.pose.position.y,tuple(message.layers),
               tuple(zlib.crc32(memoryview(d.data)) for d in message.data))
        if token==self.last_grid_token:return
        elevation = np.flip(np.asarray(layer.data, dtype=np.float32).reshape(ny, nx), axis=(0, 1))
        valid = np.isfinite(elevation)
        if not valid.any():
            with self.lock:
                self.elevation_data=None;self.obstacle_data=None;self.grid=None
                self.last_grid_token=None
            return
        obstacle=np.zeros(elevation.shape,dtype=bool)
        occupancy=np.full(elevation.shape,np.nan,dtype=np.float32)
        if 'occupancy' in message.layers:
            data=message.data[list(message.layers).index('occupancy')]
            occupancy=np.flip(np.asarray(data.data,dtype=np.float32).reshape(ny,nx),axis=(0,1))
            obstacle=np.isfinite(occupancy)&(occupancy>=.65)
        if 'obstacle' in message.layers:
            data=message.data[list(message.layers).index('obstacle')]
            explicit=np.flip(np.asarray(data.data,dtype=np.float32).reshape(ny,nx),axis=(0,1))
            obstacle=np.isfinite(explicit)&(explicit>.5)
        previous = self.height_limits
        self.full_height_limits = expand_height_limits(elevation, self.full_height_limits)
        geometry = (message.info.pose.position.x - message.info.length_x / 2,
                    message.info.pose.position.y - message.info.length_y / 2,
                    message.info.resolution, nx, ny)
        with self.lock:
            self.elevation_data = (elevation, geometry, int(valid.sum()))
            self.obstacle_data=(obstacle,occupancy)
            self.terrain_classification_available=('occupancy' in message.layers or 'obstacle' in message.layers)
            self.grid_frame=message.header.frame_id
            self.last_grid_token=token
        self.update_color_limits()
        self.recolor_grid()
        if previous != self.height_limits and self.cloud_points is not None:
            self.cloud_pub.publish(colored_cloud(self.cloud_header, self.cloud_points, self.height_limits))

    def update_color_limits(self):
        self.height_limits = self.full_height_limits or (-1., 1.)
        if not self.contrast_enabled:
            return
        values = self.elevation_data[0] if self.elevation_data else (
            self.cloud_points[:, 2] if self.cloud_points is not None else np.array([]))
        values = values[np.isfinite(values)]
        if values.size:
            lo, hi = np.percentile(values, [5, 95])
            # Do not stretch tiny numerical noise into the full color range.
            if hi-lo < .5:
                center = (lo+hi)/2
                lo, hi = center-.25, center+.25
            self.height_limits = float(lo), float(hi)

    def apply_color_mode(self):
        with self.lock:
            requested = self.contrast_requested
        if requested == self.contrast_enabled:
            return
        self.contrast_enabled = requested
        self.update_color_limits()
        self.recolor_grid()
        if self.cloud_points is not None:
            self.cloud_pub.publish(colored_cloud(self.cloud_header, self.cloud_points, self.height_limits))

    def recolor_grid(self):
        if self.elevation_data is None:
            return
        color_key=(id(self.elevation_data),self.height_limits)
        if color_key==self.last_color_key:return
        self.last_color_key=color_key
        elevation, geometry, known = self.elevation_data
        lo, hi = self.height_limits
        rgb = height_colors(elevation, lo, hi)
        if self.obstacle_data is not None:
            rgb[self.obstacle_data[0]]=0
        bitmap = PILImage.fromarray(np.flip(rgb, axis=0))
        with self.lock:
            self.grid = (bitmap, geometry, known, lo, hi, self.contrast_enabled)

    def on_planning_path(self, label, message):
        # P3 publishes map->odom identity; both are the agreed planning frames.
        points=[]
        if message.header.frame_id in ('map','odom'):
            poses=message.poses
            for i in np.linspace(0,len(poses)-1,min(len(poses),10000),dtype=int) if poses else []:
                p=poses[i].pose.position
                if np.isfinite([p.x,p.y,p.z]).all():points.append((p.x,p.y,p.z))
        with self.lock:
            self.planning_paths[label]=np.asarray(points,dtype=float).reshape(-1,3)
            self.planning_revision+=1

    def request_goal(self,x,y,yaw):
        with self.lock:
            if not self.grid_is_fresh():
                self.goal_status='地图更新中断，恢复后才能设置目标';return False
            if self.localization_unavailable():
                self.goal_status='定位暂不可用，未发送目标';return False
            if not np.isfinite([x,y,yaw]).all() or self.elevation_data is None or self.grid_frame not in ('map','odom'):
                self.goal_status='地图尚未就绪，未发送目标';return False
            elevation,(ox,oy,res,nx,ny),_=self.elevation_data
            col,row=int(math.floor((x-ox)/res)),int(math.floor((y-oy)/res))
            if not (0<=row<ny and 0<=col<nx) or not np.isfinite(elevation[row,col]):
                self.goal_status='目标位于未观测区域，未发送';return False
            if (getattr(self,'terrain_classification_available',True) and
                    (self.obstacle_data is None or not np.isfinite(self.obstacle_data[1][row,col]))):
                self.goal_status='目标栅格通行性未知，未发送';return False
            if self.obstacle_data is not None and self.obstacle_data[0][row,col]:
                self.goal_status='目标位于黑色障碍格，未发送';return False
            self.goal_requested=(float(x),float(y),float(yaw),float(elevation[row,col]),time.monotonic())
            self.goal_status='正在发送目标到 P4'
            return True

    def send_requested_goal(self):
        with self.lock:
            goal,self.goal_requested=self.goal_requested,None
            if goal is not None and (self.localization_unavailable() or not self.grid_is_fresh()):
                self.goal_status='定位或地图暂不可用，未发送目标';return
        if goal is None:return
        x,y,yaw,z,requested=goal
        if time.monotonic()-requested>1. or self.goal_pub.get_subscription_count()==0:
            with self.lock:self.goal_status='P4 目标接收器未连接或请求过期，未发送'
            return
        message=PoseStamped();message.header.stamp=self.get_clock().now().to_msg();message.header.frame_id='map'
        message.pose.position.x=x;message.pose.position.y=y;message.pose.position.z=z
        message.pose.orientation.z=math.sin(yaw/2);message.pose.orientation.w=math.cos(yaw/2)
        self.goal_pub.publish(message)
        with self.lock:
            self.last_goal=(x,y,yaw);self.planning_revision+=1
            self.goal_status=f'已发送目标 ({x:.2f}, {y:.2f})，等待 P4 规划 / 执行'
        write_visual_state('last_operator_goal.json',dict(x=x,y=y,yaw=yaw,frame='map',topic='/Car/T4/rviz_goal'))

    def on_path(self, message):
        if not message.poses:
            return
        with self.lock:
            self.path = message
        self.write_bounds()
        trace = Marker()
        trace.header = message.header
        trace.ns, trace.id, trace.type = 'trajectory', 0, Marker.LINE_STRIP
        trace.pose.orientation.w = 1.
        trace.scale.x = .30
        trace.color.r, trace.color.g, trace.color.b, trace.color.a = 1., .65, .05, 1.
        trace.points = [Point(x=p.pose.position.x, y=p.pose.position.y, z=p.pose.position.z + 2.)
                        for p in message.poses[-5000:]]
        arrow = Marker()
        arrow.header = message.header
        arrow.ns, arrow.id, arrow.type = 'rover', 1, Marker.ARROW
        import copy
        arrow.pose = copy.deepcopy(message.poses[-1].pose)
        arrow.pose.position.z += 2.
        arrow.scale.x, arrow.scale.y, arrow.scale.z = 3., .7, .7
        arrow.color.r, arrow.color.g, arrow.color.b, arrow.color.a = .15, 1., 1., 1.
        self.marker_pub.publish(MarkerArray(markers=[trace, arrow]))


class Window:
    def __init__(self, node, stop):
        self.node, self.stop = node, stop
        self.grid_spacing = 10.
        self.manual_view = None
        self.current_view = None
        self.drag_at = None
        self.goal_mode=False
        self.goal_start=None
        self.follow_vehicle=True
        self.last_draw_key = None
        self.last_legend_key = None
        self.last_image_ids = {}
        self.root = tk.Tk()
        # The Qt host owns placement and focus. Without override_redirect, Tk's
        # first camera-image geometry change remaps its wrapper as a managed
        # top-level window; Mutter then reparents it OUT of the Qt container.
        # This must be set before the first map, not after embedding.
        if os.environ.get('T3_VISUAL_RUNTIME'):
            self.root.overrideredirect(True)
        self.root.title('定位与建图 · 轨迹 / 位姿 / 双目')
        self.root.geometry('785x985+1125+38')
        self.root.configure(bg='#111823')
        self.root.protocol('WM_DELETE_WINDOW', stop.set)
        tk.Label(self.root, text='定位与建图', font=('Noto Sans CJK SC', 23, 'bold'),
                 bg='#111823', fg='#f1f5f9', anchor='w').pack(fill='x', padx=20, pady=(12, 2))
        self.status = tk.Label(self.root, text='等待定位数据', font=('Noto Sans CJK SC', 12),
                               bg='#111823', fg='#6de4cc', anchor='w')
        self.status.pack(fill='x', padx=20)
        self.pose = tk.Label(self.root, text='X —     Y —     Z —', font=('DejaVu Sans Mono', 16),
                             bg='#111823', fg='white', anchor='w', pady=8)
        self.pose.pack(fill='x', padx=20)
        self.detail = tk.Label(self.root, text='航向 —    处理耗时 —', font=('Noto Sans CJK SC', 12),
                               bg='#111823', fg='#b9c9dc', anchor='w')
        self.detail.pack(fill='x', padx=20)
        map_header = tk.Frame(self.root, bg='#111823')
        map_header.pack(fill='x', padx=20, pady=(12, 4))
        tk.Label(map_header, text='俯视图 · 橙线：轨迹 · 青色：车头',
                 font=('Noto Sans CJK SC', 11), bg='#111823', fg='#ffd18a', anchor='w').pack(side='left')
        tk.Button(map_header, text='全图', command=self.reset_view).pack(side='right')
        tk.Button(map_header, text='跟车 32 m', command=self.follow_view).pack(side='right',padx=4)
        self.goal_button=tk.Button(map_header,text='设置目标',command=self.toggle_goal)
        self.goal_button.pack(side='right',padx=6)
        tk.Label(self.root, text='平时左拖平移 · 设置目标后按下选点、拖动朝向、松开发送给 P4',
                 bg='#111823', fg='#b9c9dc', anchor='w').pack(fill='x', padx=20)
        self.goal_label=tk.Label(self.root,text='',bg='#111823',fg='#f0cd78',anchor='w',wraplength=720)
        self.goal_label.pack(fill='x',padx=20)
        self.canvas = tk.Canvas(self.root, height=470, bg='#111823', highlightthickness=1, highlightbackground='#33465b')
        self.canvas.pack(fill='both', expand=True, padx=16)
        self.canvas.bind('<Button-4>', lambda event: self.zoom(event, 1.3))
        self.canvas.bind('<Button-5>', lambda event: self.zoom(event, 1/1.3))
        self.canvas.bind('<MouseWheel>', lambda event: self.zoom(event, 1.3 if event.delta > 0 else 1/1.3))
        self.canvas.bind('<ButtonPress-1>', self.press)
        self.canvas.bind('<B1-Motion>', self.motion)
        self.canvas.bind('<ButtonRelease-1>', self.release)
        self.root.bind('<Escape>',lambda _:self.cancel_goal())
        legend_row = tk.Frame(self.root, bg='#111823')
        legend_row.pack(fill='x', padx=20, pady=(6, 0))
        tk.Label(legend_row, text='高程（m）', bg='#111823', fg='#cdd8e6').pack(side='left')
        self.legend = tk.Canvas(legend_row, width=270, height=42, bg='#111823', highlightthickness=0)
        self.legend.pack(side='left', padx=8)
        tk.Label(legend_row, text='  未观测', bg='#181c22', fg='#cdd8e6').pack(side='left', padx=10)
        tk.Label(legend_row, text=' 障碍 ', bg='black', fg='white').pack(side='left',padx=5)
        tk.Label(legend_row, text='蓝低 → 青 → 橙高\n高度相对于地图原点', bg='#111823', fg='#cdd8e6').pack(side='left')
        self.contrast_var = tk.BooleanVar(value=False)
        tk.Checkbutton(self.root, text='色差增强（5–95% 色域；两端饱和，不改变高度）',
                       variable=self.contrast_var, command=self.request_contrast,
                       bg='#111823', fg='#cdd8e6', selectcolor='#223349', activebackground='#111823',
                       activeforeground='white').pack(anchor='w', padx=16)
        self.map_label = tk.Label(self.root, text='高程栅格与点云等待更新', font=('Noto Sans CJK SC', 10),
                                  bg='#111823', fg='#b9c9dc', anchor='w')
        self.map_label.pack(fill='x', padx=20, pady=5)
        row = tk.Frame(self.root, bg='#111823')
        row.pack(fill='x', padx=16, pady=(0, 8))
        self.images = {}
        for side, name in [('Left', '左相机'), ('Right', '右相机')]:
            frame = tk.Frame(row, bg='#111823')
            frame.pack(side='left', expand=True, fill='both')
            tk.Label(frame, text=name, bg='#111823', fg='#b9c9dc').pack()
            label = tk.Label(frame, text='等待图像', bg='#1a2635', fg='white', width=1)
            label.pack(fill='both', expand=True)
            self.images[side] = label
        self.photo_refs = []
        self.root.after(200, self.refresh)
        self.root.after(200, self.check_stop)
        self.root.after(100, self.announce_window)

    def toggle_goal(self):
        self.goal_mode=not self.goal_mode;self.goal_start=None
        # Freeze the viewport during a goal gesture even if the car is moving.
        if self.follow_vehicle:
            self.manual_view=self.current_view if self.goal_mode else None
        self.goal_button.config(text='取消选点' if self.goal_mode else '设置目标')
        self.canvas.config(cursor='crosshair' if self.goal_mode else '')

    def cancel_goal(self):
        self.goal_mode=True;self.toggle_goal();self.canvas.delete('goal_drag')

    def pixel_to_map(self,x,y):
        if self.current_view is None:return None
        center,scale=self.current_view
        return center+np.array([x-self.canvas.winfo_width()/2,self.canvas.winfo_height()/2-y])/scale

    def press(self,event):
        if self.goal_mode:
            self.goal_start=self.pixel_to_map(event.x,event.y)
            if (self.follow_vehicle and self.goal_start is not None and
                    np.max(abs(self.goal_start-self.current_view[0]))>16.):self.goal_start=None
        else:self.drag_at=(event.x,event.y)

    def motion(self,event):
        if not self.goal_mode:return self.pan(event)
        if self.goal_start is None:return
        center,scale=self.current_view
        start=(self.canvas.winfo_width()/2+(self.goal_start[0]-center[0])*scale,
               self.canvas.winfo_height()/2-(self.goal_start[1]-center[1])*scale)
        self.canvas.delete('goal_drag')
        self.canvas.create_line(*start,event.x,event.y,fill='#ff71dc',width=3,arrow=tk.LAST,tags='goal_drag')

    def release(self,event):
        self.drag_at=None
        if not self.goal_mode or self.goal_start is None:return
        end=self.pixel_to_map(event.x,event.y);delta=end-self.goal_start
        yaw=math.atan2(delta[1],delta[0]) if np.linalg.norm(delta)>.05 else 0.
        self.node.request_goal(*self.goal_start,yaw)
        self.cancel_goal();self.redraw()

    def announce_window(self):
        # Tk's client wrapper, not any unrelated desktop window or WM decoration.
        import ctypes as c
        xlib = c.CDLL('libX11.so.6')
        xlib.XOpenDisplay.restype = c.c_void_p
        xlib.XOpenDisplay.argtypes = [c.c_char_p]
        xlib.XQueryTree.argtypes = [c.c_void_p, c.c_ulong, c.POINTER(c.c_ulong),
                                   c.POINTER(c.c_ulong), c.POINTER(c.POINTER(c.c_ulong)), c.POINTER(c.c_uint)]
        xlib.XFree.argtypes = [c.c_void_p]
        xlib.XCloseDisplay.argtypes = [c.c_void_p]
        display = xlib.XOpenDisplay(None)
        if not display:
            return
        root, parent, children, count = c.c_ulong(), c.c_ulong(), c.POINTER(c.c_ulong)(), c.c_uint()
        success = xlib.XQueryTree(display, self.root.winfo_id(), c.byref(root), c.byref(parent),
                                 c.byref(children), c.byref(count))
        if children:
            xlib.XFree(children)
        xlib.XCloseDisplay(display)
        if success:
            write_visual_state('monitor_window.json', {'pid': os.getpid(), 'window': parent.value})

    def check_stop(self):
        if self.stop.is_set():
            self.root.destroy()
        else:
            self.root.after(200, self.check_stop)

    def reset_view(self):
        self.follow_vehicle=False
        self.manual_view = None
        self.redraw()

    def follow_view(self):
        self.follow_vehicle=True;self.manual_view=None;self.redraw()

    def request_contrast(self):
        with self.node.lock:
            self.node.contrast_requested = self.contrast_var.get()

    def zoom(self, event, factor):
        if self.current_view is None:
            return
        if self.goal_mode:return
        self.follow_vehicle=False
        center, scale = self.current_view
        offset = np.array([event.x-self.canvas.winfo_width()/2,
                           self.canvas.winfo_height()/2-event.y])
        next_scale = float(np.clip(scale*factor, .1, 800.))
        self.manual_view = (center+offset/scale-offset/next_scale, next_scale)
        self.redraw()

    def pan(self, event):
        if self.current_view is not None and self.drag_at is not None:
            self.follow_vehicle=False
            center, scale = self.current_view
            self.manual_view = (center + np.array([self.drag_at[0]-event.x, event.y-self.drag_at[1]])/scale, scale)
            self.drag_at = (event.x, event.y)
            self.redraw()

    def redraw(self):
        with self.node.lock:
            path, grid = self.node.path, self.node.grid
        if path and path.poses:
            q = path.poses[-1].pose.orientation
            yaw = math.atan2(2*(q.w*q.z+q.x*q.y), 1-2*(q.y*q.y+q.z*q.z))
            self.draw_map(path, yaw, grid)

    def refresh(self):
        if self.stop.is_set():
            self.root.destroy()
            return
        with self.node.lock:
            path, pose_at = self.node.path, self.node.pose_at
            timing, grid = dict(self.node.timing), self.node.grid
            cloud, count = dict(self.node.cloud_status), self.node.display_points
            thumbnails = dict(self.node.thumbnails)
            goal_status=self.node.goal_status
            unavailable=self.node.localization_unavailable()
            output_source=(self.node.fusion_health or {}).get('output_source')
            output_preference=(self.node.fusion_health or {}).get('pose_source_preference')
        self.goal_label.config(text=('选点模式：松开鼠标将交给 P4 规划并行驶；Esc 取消' if self.goal_mode else goal_status))
        age = time.monotonic() - pose_at if pose_at else None
        outcome = timing.get('outcome', '')
        phase = None
        directory = os.environ.get('T3_VISUAL_RUNTIME')
        if directory:
            # Only this output's completed verification report establishes EOF.
            # A live telemetry timeout must never be mislabeled as normal EOF.
            report = FilePath(directory).parent / 'history_verification.json'
            try:
                ended = json.loads(report.read_text())
                phase = 'error' if ended.get('error') else 'stopped' if ended.get('interrupted') else 'completed'
            except (OSError, ValueError):
                pass
        label, color = localization_status(age, outcome, bool(path), phase)
        if phase is None and unavailable:
            label,color='定位暂不可用 · 正在恢复','#ffb366'
        elif phase is None and output_source=='visual' and age is not None:
            name='视觉优先定位' if output_preference=='visual' else '视觉接续定位'
            label,color=f'{name} · 最近更新 {age:.1f} 秒前','#53e0e5'
        self.status.config(text=label, fg=color)
        if path and path.poses:
            last = path.poses[-1].pose
            p, q = last.position, last.orientation
            yaw = math.atan2(2*(q.w*q.z+q.x*q.y), 1-2*(q.y*q.y+q.z*q.z))
            self.pose.config(text=f'X {p.x:8.2f}   Y {p.y:8.2f}   Z {p.z:7.2f} m')
            elapsed = timing.get('pose_processing_sec', timing.get('processing_time_sec'))
            cost = f'{elapsed:.2f} s' if isinstance(elapsed, (int, float)) else '—'
            self.detail.config(text=f'航向 {math.degrees(yaw):.1f}°   处理耗时 {cost}   轨迹点 {len(path.poses)}   坐标 {path.header.frame_id}')
            self.draw_map(path, yaw, grid)
        source_label={'global':'全局高程图','local':'局部高程图（全局暂不可用）',
                      'stale':'地图更新中断 · 保留上次画面，暂停选点','none':'等待高程图'}.get(self.node.grid_source,'等待高程图')
        age_text=(f' · 已知 {grid[2]:,} 格 · 接收于 {time.monotonic()-self.node.grid_display_at:.1f} 秒前' if grid else '')
        self.map_label.config(text=f"{source_label}{age_text}\n完整点云 {cloud.get('voxel_count', 0):,} 点 · 显示 {count:,} 点 · 绿：全局规划 / 紫：局部路径",
                              fg='#ffb366' if self.node.grid_source=='stale' else '#cdd8e6')
        directory = os.environ.get('T3_VISUAL_RUNTIME')
        if directory:
            settings = FilePath(directory) / 'view_settings.json'
            try:
                value = float(json.loads(settings.read_text())['reference_grid_m'])
                if value in (1., 5., 10., 20.):
                    self.grid_spacing = value
            except (OSError, KeyError, ValueError):
                pass
        if grid:
            lo, hi = grid[3:5]
            legend_key = (lo, hi, grid[5])
            if legend_key != self.last_legend_key:
                self.last_legend_key = legend_key
                self.legend.delete('all')
                gradient = height_colors(np.linspace(lo, hi, 260), lo, hi)
                for x, rgb in enumerate(gradient):
                    self.legend.create_line(x+5, 0, x+5, 14, fill='#%02x%02x%02x' % tuple(rgb))
                for x, value, anchor in [(5, lo, 'w'), (135, (lo+hi)/2, 'center'), (265, hi, 'e')]:
                    prefix = ('≤' if x == 5 else '≥' if x == 265 else '') if grid[5] else ''
                    self.legend.create_text(x, 29, text=f'{prefix}{value:.1f}', fill='#cdd8e6', anchor=anchor)
        for side, bitmap in thumbnails.items():
            if self.last_image_ids.get(side) == id(bitmap):
                continue
            self.last_image_ids[side] = id(bitmap)
            bitmap = bitmap.copy()
            bitmap.thumbnail((340, 190))
            photo = ImageTk.PhotoImage(bitmap)
            self.images[side].configure(image=photo, text='', width=bitmap.width, height=bitmap.height)
            self.images[side].image = photo
        self.root.after(500, self.refresh)

    def draw_map(self, path, yaw, grid):
        canvas = self.canvas
        width, height = max(canvas.winfo_width(), 100), max(canvas.winfo_height(), 100)
        manual = None if self.manual_view is None else (tuple(self.manual_view[0]), self.manual_view[1])
        with self.node.lock:
            routes=dict(self.node.planning_paths);revision=self.node.planning_revision;goal=self.node.last_goal
        draw_key = (id(path), id(grid), width, height, manual, self.grid_spacing,revision,self.follow_vehicle)
        if draw_key == self.last_draw_key:
            return
        self.last_draw_key = draw_key
        points = np.asarray([(p.pose.position.x, p.pose.position.y) for p in path.poses])
        low, high = points.min(axis=0)-6, points.max(axis=0)+6
        if grid:
            ox,oy,res,nx,ny=grid[1]
            low=np.minimum(low,[ox,oy]);high=np.maximum(high,[ox+res*nx,oy+res*ny])
        center = (low+high)/2
        scale = min((width-70)/max(high[0]-low[0], 20), (height-60)/max(high[1]-low[1], 20))
        if self.follow_vehicle:
            center=points[-1].copy();scale=min(width-70,height-60)/32.
        if self.manual_view is not None:
            center, scale = self.manual_view
        self.current_view = (np.asarray(center).copy(), scale)
        xmin, xmax = center[0]-(width/2)/scale, center[0]+(width/2)/scale
        ymin, ymax = center[1]-(height/2)/scale, center[1]+(height/2)/scale
        def project(x, y):
            return width/2+(x-center[0])*scale, height/2-(y-center[1])*scale
        canvas.delete('all')
        if grid:
            bitmap, (ox, oy, res, nx, ny), known, lo, hi, *_ = grid
            crop = ((xmin-ox)/res, ny-(ymax-oy)/res, (xmax-ox)/res, ny-(ymin-oy)/res)
            # PIL affine sampling draws only the current viewport, not a second full map.
            visible = bitmap.transform((width, height), PILImage.EXTENT, crop,
                                       resample=PILImage.NEAREST, fillcolor=UNKNOWN_COLOR)
            self.map_photo = ImageTk.PhotoImage(visible)
            canvas.create_image(0, 0, image=self.map_photo, anchor='nw')
            # These lines are actual map-cell edges, aligned to the map origin.
            # Below 8 pixels per cell they would alias; keep only the reference grid.
            if res*scale >= 8:
                for column in range(max(0, math.ceil((xmin-ox)/res)), min(nx, math.floor((xmax-ox)/res))+1):
                    x, _ = project(ox+column*res, 0)
                    canvas.create_line(x, 0, x, height, fill='#2c3e50')
                for row in range(max(0, math.ceil((ymin-oy)/res)), min(ny, math.floor((ymax-oy)/res))+1):
                    _, y = project(0, oy+row*res)
                    canvas.create_line(0, y, width, y, fill='#2c3e50')
        step = self.grid_spacing
        label_every = max(1, int(math.ceil(45 / (step*scale))))
        for x in np.arange(math.ceil(xmin/step)*step, xmax, step):
            px, _ = project(x, 0)
            canvas.create_line(px, 0, px, height, fill='#536477')
            if round(x/step) % label_every == 0:
                canvas.create_text(px+3, height-12, text=f'{x:.0f}', fill='#e1e8f0', anchor='w')
        for y in np.arange(math.ceil(ymin/step)*step, ymax, step):
            _, py = project(0, y)
            canvas.create_line(0, py, width, py, fill='#536477')
            if round(y/step) % label_every == 0:
                canvas.create_text(4, py-8, text=f'{y:.0f}', fill='#e1e8f0', anchor='w')
        xy = [project(*point) for point in points]
        for label,color in [('global','#70ff78'),('local','#ee7cff')]:
            route=routes.get(label)
            if route is not None and len(route)>1:
                coords=[v for p in route for v in project(p[0],p[1])]
                canvas.create_line(*coords,fill='#111823',width=6)
                canvas.create_line(*coords,fill=color,width=3)
        if goal:
            gx,gy=project(goal[0],goal[1])
            canvas.create_oval(gx-6,gy-6,gx+6,gy+6,outline='#ff71dc',width=3)
            canvas.create_line(gx,gy,gx+22*math.cos(goal[2]),gy-22*math.sin(goal[2]),fill='#ff71dc',width=3,arrow=tk.LAST)
            canvas.create_text(gx+9,gy+12,text='目标',fill='#ff71dc',anchor='w')
        if len(xy)>1:
            coords = [v for pair in xy for v in pair]
            canvas.create_line(*coords, fill='#111823', width=7)
            canvas.create_line(*coords, fill='#ffbd55', width=3)
        sx, sy = xy[0]
        canvas.create_oval(sx-4, sy-4, sx+4, sy+4, fill='white', outline='')
        canvas.create_text(sx, sy+15, text='起点', fill='white')
        px, py = xy[-1]
        ux, uy = math.cos(yaw), -math.sin(yaw)
        arrow = [px+17*ux, py+17*uy, px-10*ux-8*uy, py-10*uy+8*ux,
                 px-6*ux, py-6*uy, px-10*ux+8*uy, py-10*uy-8*ux]
        canvas.create_polygon(*arrow, fill='#5ffff1', outline='white', width=2)
        canvas.create_text(px+22, py-15, text='当前位置', fill='#8ffff5', anchor='w')
        if self.follow_vehicle:
            half=16*scale;left,right=width/2-half,width/2+half;top,bottom=height/2-half,height/2+half
            # Square 32 x 32 m viewport without stretching map geometry.
            for box in [(0,0,left,height),(right,0,width,height),(left,0,right,top),(left,bottom,right,height)]:
                canvas.create_rectangle(*box,fill='#111823',outline='')
            canvas.create_rectangle(left,top,right,bottom,outline='#637a91')
        canvas.create_text(width-12, 15, text='X →   Y ↑   单位：米', fill='#d9e5f3', anchor='e')
        resolution = grid[1][2] if grid else None
        cell_text = f' · 地图单元 {resolution:g} m' if resolution is not None else ''
        canvas.create_rectangle(8, 4, 315, 30, fill='#111823', outline='')
        view_text='跟车 32 × 32 m' if self.follow_vehicle else '全图 / 自由视角'
        canvas.create_text(14, 17, text=f'{view_text} · 格 {step:g} m{cell_text}', fill='white', anchor='w')
        target = 110/scale
        magnitude = 10**math.floor(math.log10(target))
        distance = min([1, 2, 5, 10], key=lambda value: abs(value*magnitude-target))*magnitude
        end = width-20
        begin = end-distance*scale
        sy = height-35
        canvas.create_line(begin, sy, end, sy, fill='white', width=3)
        for x in [begin, end]:
            canvas.create_line(x, sy-4, x, sy+4, fill='white', width=2)
        canvas.create_text((begin+end)/2, sy-14, text=f'{distance:g} m', fill='white')


def main():
    stop = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop.set())
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    node = Monitor()
    def spin():
        try:
            while not stop.is_set():
                rclpy.spin_once(node, timeout_sec=.1)
        except Exception as error:
            print(f'Visualization receiver stopped: {error}', flush=True)
            stop.set()
    thread = threading.Thread(target=spin, daemon=True)
    thread.start()
    try:
        Window(node, stop).root.mainloop()
    finally:
        stop.set()
        thread.join(timeout=5)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
