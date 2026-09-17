"""Independent LiDAR/ToF maps driven exclusively by fused body odometry."""
import copy
import json
import time
import sqlite3
import threading
from functools import wraps
from collections import deque
from pathlib import Path
import numpy as np
import yaml
import rclpy
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor,ExternalShutdownException
from rclpy.qos import QoSProfile,DurabilityPolicy,ReliabilityPolicy,qos_profile_sensor_data
from nav_msgs.msg import Odometry,OccupancyGrid
from sensor_msgs.msg import PointCloud2,Image
from std_msgs.msg import String,UInt64
from std_srvs.srv import Trigger
from grid_map_msgs.msg import GridMap
from grid_map_msgs.srv import GetGridMap
from t3_interfaces.msg import IncrementalSemanticMap
from cv_bridge import CvBridge
from .core import PoseBuffer,rigid,inverse
from .ros_utils import stamp_sec,transform_from_pose,cloud_arrays,xyz_cloud,typed
from .disk_map import DiskElevationMap
from .bounded_cloud import BoundedCloudStore
from .visibility_cleanup import VisibilityCleanup
from .overview import Overview
from .resources import memory_sample
from .compact_delivery import CompactDelivery
from .legacy.dense_grid_store import LiveDenseGlobalMapWriter
from .legacy.grid_map_message import make_grid_map_message
from .legacy.voxel_cloud_store import VoxelCloudStore
from .stereo_mapping import StereoConfirmation
from .ground_clearance import GroundClearance
from .self_filter import SelfFilter


def input_locked(method):
    @wraps(method)
    def call(self,*args,**kwargs):
        with self.input_lock:return method(self,*args,**kwargs)
    return call


class TerrainMapper(Node):
    def __init__(self,concurrent_inputs=False):
        super().__init__("fusion_terrain_mapper")
        self.declare_parameter("profile_path","");self.declare_parameter("output_dir","")
        self.cfg=yaml.safe_load(Path(self.get_parameter("profile_path").value).read_text())
        self.output=Path(self.get_parameter("output_dir").value);self.output.mkdir(parents=True,exist_ok=True)
        self.poses=PoseBuffer(self.cfg.get("pose_buffer_samples",1200))
        self.input_lock=threading.RLock()
        self.input_node=Node('fusion_map_inputs') if concurrent_inputs else self
        self.pending=deque(maxlen=self.cfg.get("map_pending_scans",4))
        self.stereo_pending=deque(maxlen=2)
        self.last_processed_source=None
        self.internal=self.output/"_internal";self.internal.mkdir(exist_ok=True)
        if self.cfg.get('map_output','terrain') not in ('terrain','elevation_only'):
            raise ValueError('map_output must be terrain or elevation_only')
        self.height_only=self.cfg.get('map_output')=='elevation_only'
        self.self_filter=SelfFilter(**self.cfg.get('self_filter',{}))
        self.ground_clearance=GroundClearance(**self.cfg.get('ground_clearance',{}))
        self.stereo_ground=GroundClearance(**self.cfg.get('ground_clearance',{}))
        self.grid=DiskElevationMap(self.internal/"elevation_tiles.sqlite",resolution=self.cfg["map_resolution"],
            tile_cells=self.cfg.get("tile_cells",128),max_tiles=self.cfg.get("max_resident_tiles",64),
            cache_mib=self.cfg.get("tile_cache_mib",128),max_window_cells=self.cfg.get("max_query_cells",250000),
            elevation_fusion=self.cfg.get('elevation_fusion'))
        self.cloud=BoundedCloudStore(self.internal/"lidar_voxels.sqlite",self.cfg["cloud_voxel_size"],
            self.cfg["cloud_preview_points"],self.cfg.get("preview_voxel_size",.3))
        self.cleanup=VisibilityCleanup(self.cfg.get('dynamic_map',{}))
        self.pose_quality=deque(maxlen=self.cfg.get('pose_buffer_samples',1200))
        self.delivery=CompactDelivery(self.output/"global_grid_map.sqlite3",height_only=self.height_only)
        self.overview=Overview(self.cfg.get("overview_side_cells",256),self.cfg.get("overview_resolution",1.))
        self.dense_writer=LiveDenseGlobalMapWriter(self.output/"global_grid_map.npz",frame_id="map",height_only=self.height_only)
        self.last_dense_revision=-1
        self.cached_global=None;self.last_global_revision=-1
        self.pressure=False;self.storage_paused=False;self.last_global_wall=0.
        self.last_checkpoint_wall=time.monotonic();self.global_available=False;self.latest_window=None
        self.last_pose=None;self.last_header=None;self.last_stamps={};self.dirty=False
        self.last_input_wall={};self.last_pose_wall=None
        self.stats={"mapped_scans":0,"dropped_scans":0,"map_points":0,"lidar_scans":0,"tof_scans":0}
        self.stats['self_filter']=dict(enabled=self.self_filter.enabled,frame='base_link',
            min_xyz_m=self.self_filter.minimum.tolist() if self.self_filter.enabled else None,
            max_xyz_m=self.self_filter.maximum.tolist() if self.self_filter.enabled else None,sources={})
        self.stereo_cfg=self.cfg.get('stereo_mapping',{})
        self.stereo_confirmation=StereoConfirmation(self.cfg['map_resolution'],self.stereo_cfg)
        self.fusion_health={};self.fusion_health_wall=-float('inf')
        self.stereo_health_generation=0;self.stereo_processed_generation=0
        self.stats.update(stereo_scans=0,stereo_cells=0,stereo_rejected=0,
                          stereo_enabled=bool(self.stereo_cfg.get('enabled',False)),
                          stereo_reason='waiting' if self.stereo_cfg.get('enabled',False) else 'disabled')
        self.bridge=CvBridge();self.mask=None
        self.camera_from_base=inverse(rigid(self.cfg["base_from_camera_left"]))
        self.tum=(self.output/"trajectory_map.tum").open("w")
        self.tum_last_offset=None
        self.input_node.create_subscription(Odometry,"/T3/semantic/current_pose",self.odom,100)
        self.input_node.create_subscription(PointCloud2,self.cfg.get("mapping_lidar_topic","/fusion/lidar"),lambda m:self.enqueue(m,"lidar",self.cfg["base_from_lidar"]),QoSProfile(depth=1,reliability=ReliabilityPolicy.RELIABLE))
        if self.stereo_cfg.get('enabled',False):
            self.input_node.create_subscription(PointCloud2,self.stereo_cfg.get('topic','/fusion/stereo_points'),
                self.enqueue_stereo,QoSProfile(depth=1,reliability=ReliabilityPolicy.RELIABLE))
            self.input_node.create_subscription(String,'/fusion/status',self.fusion_status,2)
        for source in self.cfg.get("tof_sources",[]):
            rigid(source["base_from_sensor"])
            self.input_node.create_subscription(PointCloud2,source["topic"],lambda m,s=source:self.enqueue(m,s["name"],s["base_from_sensor"]),qos_profile_sensor_data)
        if self.cfg.get("semantic_topic") and not self.height_only:
            self.input_node.create_subscription(Image,self.cfg["semantic_topic"],self.semantic,1)
        qos=QoSProfile(depth=1,reliability=ReliabilityPolicy.RELIABLE,durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.grid_pubs=[self.create_publisher(GridMap,t,qos) for t in
            ["/T3/mapping/elevation_map","/T3/mapping/grid_map","/Car/T3/mapping/grid_map"]]
        self.global_pubs=[self.create_publisher(GridMap,t,qos) for t in
            ["/T3/mapping/global_grid_map","/Car/T3/mapping/global_grid_map"]]
        self.overview_pubs=[self.create_publisher(OccupancyGrid,t,qos) for t in
            ([] if self.height_only else ["/T3/mapping/global_overview","/Car/T3/mapping/global_overview"])]
        self.revision_pubs=[self.create_publisher(UInt64,t,qos) for t in
            ["/T3/mapping/global_map_revision","/Car/T3/mapping/global_map_revision"]]
        self.cloud_pub=self.create_publisher(PointCloud2,"/T3/mapping/lidar_map",qos)
        self.stereo_cloud_pub=(self.create_publisher(PointCloud2,'/T3/mapping/stereo_map',qos)
                               if self.stereo_cfg.get('enabled',False) else None)
        self.elevation_pub=self.create_publisher(PointCloud2,"/T3/mapping/elevation_cloud",qos)
        self.incremental_pub=None if self.height_only else self.create_publisher(IncrementalSemanticMap,"/T3/semantic/incremental_map",qos)
        self.occupancy_pub=None if self.height_only else self.create_publisher(OccupancyGrid,"/T3/mapping/traversability",qos)
        self.status_pub=self.create_publisher(String,"/fusion/map_status",10)
        self.create_timer(.1,self.process)
        self.create_timer(self.cfg["map_publish_period"],self.publish)
        self.create_timer(self.cfg.get("global_publish_period",20.),self.publish_global)
        self.create_timer(2.,self.health)
        for topic in ["/T3/mapping/get_grid_map","/Car/T3/mapping/get_grid_map"]:
            self.create_service(GetGridMap,topic,self.query)
        for topic in ["/T3/mapping/save","/T3/mapping/save_grid_map","/Car/T3/mapping/save_grid_map"]:
            self.create_service(Trigger,topic,self.save_service)
        self.get_logger().info("Terrain mapper ready: elevation GridMap and persistent LiDAR XYZ")

    @input_locked
    def odom(self,msg):
        try:
            t=transform_from_pose(msg.pose.pose)
            covariance=np.asarray(msg.pose.covariance).reshape(6,6)
            if (msg.header.frame_id not in ('map','odom') or msg.child_frame_id!='base_link'
                    or not np.isfinite(covariance).all()
                    or not np.allclose(covariance,covariance.T,rtol=0,atol=1e-6)
                    or np.linalg.eigvalsh(covariance).min()<-1e-8):
                raise ValueError('Invalid mapping pose frame/covariance')
            appended=self.poses.append(stamp_sec(msg),t)
            revised=not appended and self.poses.replace_latest(stamp_sec(msg),t)
            if appended or revised:
                if revised:self.pose_quality.pop()
                self.pose_quality.append((stamp_sec(msg),float(np.max(np.diag(covariance)[:3])),
                                          float(np.max(np.diag(covariance)[3:])),covariance.copy()))
                self.last_pose=t
                self.last_pose_wall=time.monotonic()
                if appended:
                    self.tum_last_offset=self.tum.tell()
                elif self.tum_last_offset is not None:
                    self.tum.seek(self.tum_last_offset);self.tum.truncate()
                p=msg.pose.pose;q=p.orientation
                self.tum.write(f"{stamp_sec(msg):.9f} {p.position.x:.9f} {p.position.y:.9f} {p.position.z:.9f} {q.x:.9f} {q.y:.9f} {q.z:.9f} {q.w:.9f}\n")
        except ValueError:
            self.stats['invalid_poses']=self.stats.get('invalid_poses',0)+1

    def pose_quality_at(self,stamp):
        """Match PoseBuffer.at's time support; caller holds input_lock.

        The convex covariance sum bounds the linearly interpolated endpoint
        errors even with unknown endpoint correlation. Rotation uses the same
        small-angle fixed-frame convention as the height Jacobian below.
        """
        if not self.pose_quality:return None
        times=np.array([q[0] for q in self.pose_quality])
        i=int(np.searchsorted(times,stamp))
        if i<len(times) and abs(times[i]-stamp)<1e-8:return self.pose_quality[i]
        if i==0 or i==len(times):
            q=self.pose_quality[0 if i==0 else -1]
            return q if abs(q[0]-stamp)<=self.cfg['pose_tolerance'] else None
        a,b=self.pose_quality[i-1],self.pose_quality[i]
        if b[0]-a[0]>self.cfg['pose_max_gap']:return None
        for q in (a,b):
            c=q[3]
            if not np.isfinite(c).all() or np.linalg.eigvalsh((c+c.T)/2).min()<-1e-8:
                return (stamp,float('nan'),float('nan'),c)
        u=(stamp-a[0])/(b[0]-a[0])
        covariance=(1-u)*a[3]+u*b[3]
        return (stamp,float(np.max(np.diag(covariance)[:3])),
                float(np.max(np.diag(covariance)[3:])),covariance)

    @input_locked
    def fusion_status(self,msg):
        try:
            value=json.loads(msg.data)
            if not isinstance(value,dict):return
            self.fusion_health=value;self.fusion_health_wall=time.monotonic()
            if not value.get('vision_enabled') or not value.get('localization_valid'):
                self.stereo_health_generation+=1
        except (ValueError,TypeError):pass

    @input_locked
    def enqueue_stereo(self,msg):
        if (msg.header.frame_id!='camera_left_optical' or
                msg.width*msg.height>int(self.stereo_cfg.get('max_points',512)) or
                len(msg.data)>int(self.stereo_cfg.get('max_points',512))*32):
            self.stats['stereo_rejected']+=1;self.stats['stereo_reason']='invalid_input';return
        self.enqueue(msg,'stereo',self.cfg['base_from_camera_left'])

    @input_locked
    def process_stereo(self,msg,pose,transform,stamp,quality):
        if self.stereo_processed_generation!=self.stereo_health_generation:
            self.stereo_confirmation.clear();self.stereo_processed_generation=self.stereo_health_generation
        self.stats['stereo_pose_variance']=(dict(position=quality[1],rotation=quality[2])
                                           if quality is not None else None)
        reason=None
        if not self.fusion_health.get('vision_enabled') or not self.fusion_health.get('localization_valid'):
            reason='visual_or_localization_unqualified'
        elif time.monotonic()-self.fusion_health_wall>1.5:reason='fusion_status_stale'
        elif quality is None or abs(quality[0]-stamp)>self.cfg['pose_tolerance']:reason='missing_cotimed_pose_quality'
        elif not 0<=quality[1]<=self.stereo_cfg.get('max_position_variance',.04):reason='position_uncertainty'
        elif not 0<=quality[2]<=self.stereo_cfg.get('max_rotation_variance',.01):reason='rotation_uncertainty'
        if reason:
            self.stereo_confirmation.clear();self.stats['stereo_rejected']+=1
            reasons=self.stats.setdefault('stereo_rejection_reasons',{})
            reasons[reason]=reasons.get(reason,0)+1
            self.stats['stereo_reason']=reason;return
        data=cloud_arrays(msg,('x','y','z','position_variance'))
        valid=np.isfinite(data).all(axis=1)&(data[:,3]>0)&(data[:,3]<=self.stereo_cfg.get('max_point_std_m',.15)**2)
        data=data[valid];xyz=data[:,:3]
        ranges=np.linalg.norm(xyz,axis=1)
        use=(ranges>=self.stereo_cfg.get('min_depth_m',.5))&(ranges<=self.stereo_cfg.get('max_depth_m',8.))
        xyz,variance=xyz[use],data[use,3]
        base=xyz@transform[:3,:3].T+transform[:3,3]
        keep=self.body_keep_mask(base,'stereo')
        base,variance=base[keep],variance[keep]
        if not len(base):
            self.stereo_confirmation.clear();self.stats['stereo_reason']='no_points_after_mapping_filters'
            return
        offset=base@pose[:3,:3].T;points=offset+pose[:3,3]
        # Odometry orientation covariance is about the fixed frame axes.
        jacobian=np.zeros((len(points),6));jacobian[:,2]=1.
        jacobian[:,3]=offset[:,1];jacobian[:,4]=-offset[:,0]
        covariance=quality[3]
        if not np.isfinite(covariance).all() or np.linalg.eigvalsh((covariance+covariance.T)/2).min()<-1e-8:
            self.stats['stereo_rejected']+=1;self.stats['stereo_reason']='invalid_pose_covariance';return
        variance=variance+np.maximum(0.,np.einsum('ni,ij,nj->n',jacobian,covariance,jacobian))
        use=variance<=self.stereo_cfg.get('max_height_std_m',.25)**2
        terrain=self.ground_clearance.select_from_reference(points,pose[:3,3],stamp)
        if (self.ground_clearance.enabled
                and not self.ground_clearance.reference_available(pose[:3,3],stamp)):
            # Visual fallback can establish its own observed supporting surface;
            # do not require a live LiDAR plane indefinitely.
            terrain=self.stereo_ground.select(points,(pose@transform)[:3,3],stamp=stamp)
        terrain_mask=np.zeros(len(points),dtype=bool);terrain_mask[terrain]=True
        use &= terrain_mask
        self.stats['stereo_terrain_points']=int(use.sum())
        if not use.any():self.stereo_confirmation.clear()
        evidence=self.stereo_confirmation.observe(points[use],variance[use],stamp)
        accepted=self.grid.update_stereo(evidence,
            stamp=stamp,
            variance_floor=self.stereo_cfg.get('variance_floor_m2',.0025),
            preview_cap=int(self.stereo_cfg.get('preview_points',20000)))
        self.stats['stereo_scans']+=1;self.stats['stereo_cells']+=len(accepted)
        self.stats['stereo_reason']=('mapped' if len(accepted) else
            'known_range_cells_or_height_conflict' if len(evidence) else
            'confirming' if use.any() else 'depth_or_pose_uncertainty')
        if len(accepted):
            self.advance_map_stamp(msg)
            self.dirty=True;self.stats['mapped_scans']+=1

    def advance_map_stamp(self,msg):
        # Independently queued sensors can finish out of timestamp order.
        # Keep the newest integrated acquisition time, never a fresh wall time.
        if self.last_header is None or stamp_sec(msg)>stamp_sec(self.last_header):
            self.last_header=copy.deepcopy(msg.header);self.last_header.frame_id='map'

    @input_locked
    def semantic(self,msg):
        try:
            mask=self.bridge.imgmsg_to_cv2(msg,desired_encoding="passthrough")
            if mask.ndim==2 and np.issubdtype(mask.dtype,np.integer):
                self.mask=(stamp_sec(msg),np.asarray(mask).copy())
        except Exception:pass

    @input_locked
    def enqueue(self,msg,name,transform):
        self.last_input_wall[name]=time.monotonic()
        if self.storage_paused:
            self.stats["dropped_scans"]+=1
            return
        if len(msg.data)>self.cfg.get("max_cloud_bytes",16000000):
            self.stats["dropped_scans"]+=1
            return
        stamp=stamp_sec(msg)
        if stamp<=self.last_stamps.get(name,-1):return
        self.last_stamps[name]=stamp
        queue=self.stereo_pending if name=='stereo' else self.pending
        if len(queue)==queue.maxlen:
            # Preserve the oldest scan until its pose-settle interval elapses.
            # Evicting it on every arrival can prevent any scan from ever being
            # mapped when input_hz * settle_seconds exceeds the queue capacity.
            self.stats["dropped_scans"]+=1
            self.stats["backpressure_dropped_scans"]=self.stats.get("backpressure_dropped_scans",0)+1
            # Keep the waiting head so the settle period cannot starve mapping,
            # but retain the latest sample in the tail instead of stale backlog.
            if len(queue)>1 and queue[-1][2]==name:
                queue[-1]=(time.monotonic(),msg,name,rigid(transform))
            return
        queue.append((time.monotonic(),msg,name,rigid(transform)))

    def add_semantics(self,points_base,points_map,stamp):
        # A task-2 label image contributes labels only at calibrated projections.
        # Unseen labels remain unknown; no geometric class is fabricated.
        with self.input_lock:sample=self.mask
        if sample is None or abs(sample[0]-stamp)>self.cfg["semantic_tolerance"]:return
        mask=sample[1]
        pc=points_base@self.camera_from_base[:3,:3].T+self.camera_from_base[:3,3]
        k=np.asarray(self.cfg["camera_k"]).reshape(3,3).copy()
        k[0]*=mask.shape[1]/self.cfg["input_image_size"][0]
        k[1]*=mask.shape[0]/self.cfg["input_image_size"][1]
        valid=np.isfinite(pc).all(axis=1)&(pc[:,2]>.1)
        ix=np.flatnonzero(valid);uv=pc[ix]@k.T;uv=uv[:,:2]/uv[:,2,None]
        uv=np.rint(uv).astype(int)
        keep=(uv[:,0]>=0)&(uv[:,0]<mask.shape[1])&(uv[:,1]>=0)&(uv[:,1]<mask.shape[0])
        ix,uv=ix[keep],uv[keep]
        labels=mask[uv[:,1],uv[:,0]].astype(int)
        valid=(labels>0)&(labels<len(self.grid.class_names))
        xy=points_map[ix[valid],:2];labels=labels[valid]
        self.grid.add_semantics(xy,labels)

    def process(self):
        queues=(self.pending,self.stereo_pending) if self.last_processed_source=='stereo' else (self.stereo_pending,self.pending)
        for queue in queues:
            if self.process_queue(queue):break

    @input_locked
    def take_ready(self,queue):
        if not queue:return False
        if self.storage_paused:
            self.stats["dropped_scans"]+=len(queue);queue.clear();return False
        # Select the newest settled sample with a usable co-timed pose. Keep
        # unrelated sources and the not-yet-settled tail; never reset its wait
        # on each arrival. This bounds latency without starving mapping.
        now=time.monotonic();settle=self.cfg.get('mapping_pose_settle_sec',0.)
        source=queue[0][2];chosen=0
        for i,(received,message,name,_) in enumerate(queue):
            if name=='lidar' and not (self.cfg['instantaneous_cloud'] or self.cfg.get('cloud_motion_compensated',False) or self.cfg.get('deskew',{}).get('enabled',False)):continue
            if name==source and now-received>=settle and self.poses.at(stamp_sec(message),
                    tolerance=self.cfg['pose_tolerance'],max_gap=self.cfg['pose_max_gap']) is not None:
                chosen=i
        if chosen:
            discard=[i for i in range(chosen) if queue[i][2]==source]
            for i in reversed(discard):del queue[i]
            chosen-=len(discard)
            self.stats['dropped_scans']+=len(discard)
            self.stats['superseded_scans']=self.stats.get('superseded_scans',0)+len(discard)
        queued,msg,name,t_base_sensor=queue[chosen]
        # A bounded settle interval lets delayed visual corrections join the
        # same measurement-time pose before a scan is permanently integrated.
        if time.monotonic()-queued<self.cfg.get("mapping_pose_settle_sec",0.):
            return False
        stamp=stamp_sec(msg)
        point_times=None
        compensated=(self.cfg.get("cloud_motion_compensated",False) or self.cfg.get("deskew",{}).get("enabled",False))
        if name=="lidar" and not self.cfg["instantaneous_cloud"] and not compensated:
            try:
                point_times=cloud_arrays(msg,("time",))[:,0]
                if (not np.isfinite(point_times).all() or (point_times<0.).any()
                        or (point_times>self.cfg.get('max_scan_duration',.2)).any()):
                    raise ValueError('Invalid per-point scan time')
            except (ValueError,TypeError) as error:
                del queue[chosen];self.stats['dropped_scans']+=1
                self.get_logger().warn('Map input rejected: '+str(error),throttle_duration_sec=5)
                return False
        pose=self.poses.at(stamp,tolerance=self.cfg["pose_tolerance"],max_gap=self.cfg["pose_max_gap"])
        end_pose=pose if point_times is None or not len(point_times) else self.poses.at(
            stamp+float(np.max(point_times)),tolerance=self.cfg["pose_tolerance"],max_gap=self.cfg["pose_max_gap"])
        if pose is None or end_pose is None:
            end_stamp=stamp+(0. if point_times is None or not len(point_times) else float(np.max(point_times)))
            # A missing interval already behind the latest pose cannot arrive
            # later in this monotonic buffer; do not block newer clouds on it.
            past_gap=bool(self.poses.samples and self.poses.samples[-1][0]>end_stamp+self.cfg["pose_tolerance"])
            if past_gap or time.monotonic()-queued>self.cfg["mapping_wait_timeout"]:
                del queue[chosen];self.stats["dropped_scans"]+=1
                if name=='stereo':self.stats['stereo_reason']='missing_cotimed_pose'
            return False
        del queue[chosen]
        self.last_processed_source=name
        self.stats['last_queue_wait_sec']=time.monotonic()-queued
        self.stats.setdefault('queue_wait_sec_by_source',{})[name]=self.stats['last_queue_wait_sec']
        return msg,name,t_base_sensor,stamp,point_times,pose,self.pose_quality_at(stamp)

    def body_keep_mask(self,base,source):
        keep=self.self_filter.keep_mask(base)
        state=self.stats['self_filter']['sources'].setdefault(source,
            dict(scans=0,input_points=0,removed_points=0,empty_scans=0))
        removed=len(base)-int(np.count_nonzero(keep))
        state['scans']+=1;state['input_points']+=len(base);state['removed_points']+=removed
        state['last_input_points']=len(base);state['last_removed_points']=removed
        if len(base) and removed==len(base):state['empty_scans']+=1
        return keep

    def process_queue(self,queue):
        sample=self.take_ready(queue)
        if not sample:return False
        msg,name,t_base_sensor,stamp,point_times,pose,quality=sample
        process_started=time.monotonic()
        try:
            if name=='stereo':
                self.process_stereo(msg,pose,t_base_sensor,stamp,quality)
                self.stats['last_stereo_update_sec']=time.monotonic()-process_started
                return True
            xyz=cloud_arrays(msg)
            good=np.isfinite(xyz).all(axis=1)
            ranges=np.linalg.norm(xyz,axis=1)
            valid=good&(ranges>=self.cfg["min_range"])&(ranges<=self.cfg["max_range"])
            xyz=xyz[valid]
            if not len(xyz):raise ValueError('Cloud contains no usable range points')
            base=xyz@t_base_sensor[:3,:3].T+t_base_sensor[:3,3]
            # Use the calibrated body frame, before ground fitting or either
            # map store. Keep raw input and the localization stream unchanged.
            keep=self.body_keep_mask(base,name)
            base,xyz=base[keep],xyz[keep]
            if not len(base):return True
            points=base@pose[:3,:3].T+pose[:3,3]
            if point_times is not None:
                times=point_times[valid][keep]
                bins=np.floor(times/.005).astype(int)
                for key in np.unique(bins):
                    use=bins==key
                    with self.input_lock:
                        sample=self.poses.at(stamp+float(np.mean(times[use])),
                            tolerance=self.cfg["pose_tolerance"],max_gap=self.cfg["pose_max_gap"])
                    if sample is None:raise ValueError("Missing pose during LiDAR scan")
                    points[use]=base[use]@sample[:3,:3].T+sample[:3,3]
            stage_started=time.monotonic()
            stages={'transform':stage_started-process_started}
            sensor_origin=(pose@t_base_sensor)[:3,3]
            terrain_indices=self.ground_clearance.select(points,sensor_origin,stamp=stamp)
            terrain_points=points[terrain_indices];terrain_base=base[terrain_indices]
            self.stats['ground_clearance']=dict(self.ground_clearance.stats)
            if self.grid.temporal_elevation and quality is None:
                raise ValueError('Missing co-timed pose covariance for elevation fusion')
            if name=="lidar":
                removed=np.empty((0,3))
                dynamic=self.cfg.get('dynamic_map',{})
                clear_ok=(quality is not None and abs(quality[0]-stamp)<=self.cfg['pose_tolerance']
                    and 0<=quality[1]<=dynamic.get('max_position_variance',.01)
                    and 0<=quality[2]<=dynamic.get('max_rotation_variance',.0025))
                # A single rigid scan and LiDAR-only evidence are required for
                # this clearance path. Calibrated multi-source maps need their
                # own retained source evidence before replacing height columns.
                if (dynamic.get('enabled',False) and clear_ok and not self.cfg.get('tof_sources')
                        and not self.stereo_cfg.get('enabled',False)
                        and self.cfg.get('instantaneous_cloud',False)):
                    removed=self.cleanup.update(self.cloud,xyz,pose@t_base_sensor,stamp,self.cfg['map_resolution'])
                else:
                    # An uncertain pose breaks clearance confirmation; later
                    # good scans must establish fresh free-space evidence.
                    self.cleanup.votes.clear()
                    self.cleanup.stats=dict(checked=0,free_evidence=0,removed=0,pending=0)
                self.cloud.append(points);self.stats["lidar_scans"]+=1
                if len(removed):
                    self.grid.rebuild_cells(np.floor(removed[:,:2]/self.cfg['map_resolution']).astype(int),self.cloud,
                        lambda p:self.ground_clearance.filter_columns(p,sensor_origin))
                    if not self.height_only:self.refresh_overview(removed)
                self.stats['cleanup']=dict(self.cleanup.stats,pose_qualified=clear_ok,
                    removed_total=self.stats.get('cleanup',{}).get('removed_total',0)+len(removed))
            else:self.stats["tof_scans"]+=1
            stages['cloud_store']=time.monotonic()-stage_started;stage_started=time.monotonic()
            # Bound scan-density bias while retaining min/median/max height
            # evidence for each XY cell. The existing robust map rejects outliers.
            if self.grid.temporal_elevation:
                self.grid.update_elevation_only(points_map=terrain_points,stamp=stamp,
                    covariance=quality[3],pose_origin=pose[:3,3])
                self.stats['elevation_fusion']=dict(self.grid.fusion_stats)
                if not self.height_only:self.add_semantics(terrain_base,terrain_points,stamp)
            elif len(terrain_points):
                keys=np.floor(terrain_points[:,:2]/self.cfg["map_resolution"]).astype(np.int64)
                order=np.lexsort((terrain_points[:,2],keys[:,1],keys[:,0]));ordered=keys[order]
                starts=np.r_[0,np.flatnonzero(np.any(np.diff(ordered,axis=0),axis=1))+1]
                ends=np.r_[starts[1:],len(order)]
                positions=np.column_stack([starts,starts+(ends-starts)//2,ends-1]).ravel()
                # Positions are already sorted; keep each sample once for
                # one- and two-point cells, preserving min/median/max order.
                positions=positions[np.r_[True,positions[1:]!=positions[:-1]]]
                selected=order[positions]
                if len(selected)>self.cfg["max_elevation_points"]:
                    selected=selected[np.linspace(0,len(selected)-1,self.cfg["max_elevation_points"],dtype=int)]
                self.grid.update_elevation_only(points_map=terrain_points[selected])
                if not self.height_only:self.add_semantics(terrain_base[selected],terrain_points[selected],stamp)
            stages['elevation']=time.monotonic()-stage_started;stage_started=time.monotonic()
            if not self.height_only:self.overview.update(terrain_points)
            stages['overview']=time.monotonic()-stage_started
            self.advance_map_stamp(msg)
            self.dirty=True;self.stats["mapped_scans"]+=1;self.stats["map_points"]=self.cloud.count
            self.stats["last_map_update_sec"]=time.monotonic()-process_started
            self.stats['map_stage_sec']=stages
        except (ValueError,RuntimeError,OSError,sqlite3.Error) as e:
            with self.input_lock:self.stats["dropped_scans"]+=1
            if isinstance(e,(OSError,sqlite3.Error)):self.storage_paused=True
            self.get_logger().error("Map input rejected: "+str(e),throttle_duration_sec=5)
        return True

    def terrain_layers(self,m):
        if self.height_only:
            return m.layers(['elevation','elevation_variance','height_range','roughness','observation_count']),None
        layers=m.all_layers()
        elevation=m.elevation
        if min(elevation.shape)<2:
            slope=np.full_like(elevation,np.nan)
        else:
            dy,dx=np.gradient(elevation,self.cfg["map_resolution"])
            slope=np.arctan(np.hypot(dx,dy))
        step=np.full_like(elevation,np.nan)
        for axis in [0,1]:
            delta=np.abs(np.diff(elevation,axis=axis))
            a=[slice(None),slice(None)];b=a.copy();a[axis]=slice(1,None);b[axis]=slice(None,-1)
            step[tuple(a)]=np.fmax(step[tuple(a)],delta);step[tuple(b)]=np.fmax(step[tuple(b)],delta)
        # Height spread detects multiple surfaces/vertical obstacles in one XY cell.
        cost=np.fmax(slope/np.deg2rad(self.cfg.get("max_slope_deg",35.)),
                     np.fmax(step,m.height_range)/self.cfg.get("max_step_m",.3))
        hard_obstacle=np.isfinite(elevation)&((m.height_range>=self.cfg.get('max_step_m',.3))|
            (step>=self.cfg.get('max_step_m',.3))|(slope>=np.deg2rad(self.cfg.get('max_slope_deg',35.))))
        known=(np.isfinite(elevation)&np.isfinite(slope)&np.isfinite(step))|hard_obstacle
        traversability=np.where(known,np.clip(1-cost,0,1),np.nan).astype(np.float32)
        occupancy=np.full(elevation.shape,-1,np.int8)
        occupancy[known]=np.rint((1-traversability[known])*100).astype(np.int8)
        layers.update(slope=slope.astype(np.float32),step=step,traversability=traversability,
                      occupancy=np.where(known,occupancy.astype(np.float32)/100.,np.nan),
                      obstacle=np.where(known,(occupancy>=self.cfg.get('obstacle_threshold',65)).astype(np.float32),np.nan))
        return layers,occupancy

    def refresh_overview(self, removed):
        if not self.overview.ready:return
        cells=np.unique(self.overview._indices(removed[:,:2]),axis=0)
        for col,row in cells:
            if not (0<=row<self.overview.size and 0<=col<self.overview.size):continue
            self.overview.minimum[row,col]=np.inf;self.overview.maximum[row,col]=-np.inf;self.overview.count[row,col]=0
            absolute=np.rint(self.overview.origin/self.overview.resolution).astype(int)+[col,row]
            for points in self.cloud.column_points(absolute,self.overview.resolution):
                self.overview.minimum[row,col]=min(self.overview.minimum[row,col],float(points[:,2].min()))
                self.overview.maximum[row,col]=max(self.overview.maximum[row,col],float(points[:,2].max()))
                self.overview.count[row,col]+=len(points)

    def occupancy_message(self,geometry,data):
        msg=OccupancyGrid();msg.header=copy.deepcopy(self.last_header)
        msg.info.resolution=float(geometry.resolution);msg.info.width=geometry.width;msg.info.height=geometry.height
        msg.info.origin.position.x=float(geometry.origin_x);msg.info.origin.position.y=float(geometry.origin_y)
        msg.info.origin.orientation.w=1.;msg.data=typed(data,"b")
        return msg

    def publish(self):
        if not self.dirty or self.last_header is None or self.last_pose is None:return
        if self.pressure and time.monotonic()-getattr(self,"last_publish_wall",0)<2*self.cfg["map_publish_period"]:return
        try:
            publication_started=time.monotonic()
            with self.input_lock:pose=self.last_pose.copy()
            m=self.grid.extract_window(center_x=float(pose[0,3]),center_y=float(pose[1,3]),
                 length_x=self.cfg["map_window"],length_y=self.cfg["map_window"])
            layers,occupancy=self.terrain_layers(m)
            local_header=copy.deepcopy(self.last_header);local_header.frame_id="odom"
            grid_msg=make_grid_map_message(header=local_header,geometry=m.geometry,layers=layers)
            for pub in self.grid_pubs:pub.publish(grid_msg)
            rr,cc=np.where(np.isfinite(m.elevation))
            pts=np.column_stack([m.geometry.origin_x+(cc+.5)*m.geometry.resolution,
                                 m.geometry.origin_y+(rr+.5)*m.geometry.resolution,m.elevation[rr,cc]])
            self.elevation_pub.publish(xyz_cloud(pts,self.last_header,m.elevation[rr,cc]))
            preview_cap=self.cfg["cloud_preview_points"]//4 if self.pressure else self.cfg["cloud_preview_points"]
            self.cloud_pub.publish(xyz_cloud(self.cloud.preview(preview_cap),self.last_header))
            if self.stereo_cloud_pub is not None:
                self.stereo_cloud_pub.publish(xyz_cloud(
                    np.asarray(list(self.grid.stereo_preview.values())).reshape(-1,3),self.last_header))
            if self.incremental_pub is not None:
                inc=IncrementalSemanticMap();inc.header=copy.deepcopy(self.last_header)
                inc.update_id=self.grid.update_id%(2**32);inc.resolution=float(m.geometry.resolution)
                inc.width=m.geometry.width;inc.height=m.geometry.height
                inc.origin.position.x=float(m.geometry.origin_x);inc.origin.position.y=float(m.geometry.origin_y)
                inc.origin.orientation.w=1.
                inc.occupancy=typed(occupancy,"b");inc.semantic=typed(m.semantic,"B")
                inc.semantic_confidence=typed(m.semantic_confidence,"f")
                inc.elevation=typed(m.elevation,"f");inc.elevation_variance=typed(m.elevation_variance,"f")
                inc.height_range=typed(m.height_range,"f");inc.roughness=typed(m.roughness,"f")
                inc.observation_count=typed(m.observation_count,"I");inc.class_names=list(m.class_names)
                self.incremental_pub.publish(inc)
                self.occupancy_pub.publish(self.occupancy_message(m.geometry,occupancy))
            self.latest_window=(m,layers)
            self.dirty=False;self.last_publish_wall=time.monotonic()
            self.tum.flush()
            self.stats["last_local_publish_sec"]=time.monotonic()-publication_started
        except (ValueError,RuntimeError,OSError,sqlite3.Error) as e:
            self.get_logger().error("Map publication failed: "+str(e),throttle_duration_sec=5)

    def publish_global(self):
        if self.last_header is None:return
        publication_started=time.monotonic()
        self.last_global_wall=time.monotonic()
        height,spread,occupancy=self.overview.layers()
        for pub in self.overview_pubs:pub.publish(self.occupancy_message(self.overview.geometry,occupancy))
        try:
            if self.pressure:raise ValueError("Memory pressure: full global message suspended")
            if self.cached_global is not None and self.last_global_revision==self.grid.update_id:
                for pub in self.global_pubs:pub.publish(self.cached_global)
                self.stats["last_global_publish_sec"]=time.monotonic()-publication_started
                return
            m=self.grid.extract_global(self.cfg.get("global_max_cells",1000000))
            if m is None:return
            # Keep the original full-extent, original-resolution global contract.
            # Publish the contract's core layers, not visualization color copies.
            all_layers,_=self.terrain_layers(m)
            layers=(all_layers if self.height_only else {key:all_layers[key] for key in
                ['elevation','elevation_variance','height_range','traversability','occupancy','obstacle']})
            grid_msg=make_grid_map_message(header=self.last_header,geometry=m.geometry,layers=layers)
            for pub in self.global_pubs:pub.publish(grid_msg)
            self.cached_global=grid_msg;self.last_global_revision=self.grid.update_id
            self.global_available=True
            # This detached window is owned by a bounded coalescing worker.
            # Compression and fsync must not block sensor or publication timers.
            self.dense_writer.enqueue(m,map_revision=self.grid.update_id,
                timestamp_text=time.strftime("%Y%m%d_%H%M%S"))
            self.last_dense_revision=self.grid.update_id
            self.stats["last_global_publish_sec"]=time.monotonic()-publication_started
        except ValueError as e:
            # Explicitly invalidate the latched global payload rather than leave a
            # stale/cropped map pretending to cover the latest revision.
            empty=GridMap();empty.header=copy.deepcopy(self.last_header)
            for pub in self.global_pubs:pub.publish(empty)
            self.global_available=False
            self.cached_global=None;self.last_global_revision=-1
            self.get_logger().warn(str(e)+"; use local map/query",throttle_duration_sec=20)
        except (RuntimeError,OSError,sqlite3.Error) as e:
            self.get_logger().error("Global map publication failed: "+str(e),throttle_duration_sec=5)

    def query(self,request,response):
        try:
            if request.frame_id not in ("map","odom"):
                raise ValueError("Query frame must be map or odom")
            if self.last_header is None:raise ValueError("No mapped scan yet")
            if self.pressure and request.length_x*request.length_y/self.cfg["map_resolution"]**2>self.cfg.get("pressure_query_cells",160000):
                raise ValueError("Query exceeds memory-pressure limit")
            m=self.grid.extract_window(center_x=request.position_x,center_y=request.position_y,
                length_x=request.length_x,length_y=request.length_y)
            layers,_=self.terrain_layers(m)
            names=list(request.layers) if request.layers else list(layers)
            if any(n not in layers for n in names):raise ValueError("Unknown requested layer")
            header=copy.deepcopy(self.last_header);header.frame_id=request.frame_id
            response.map=make_grid_map_message(header=header,geometry=m.geometry,layers={n:layers[n] for n in names})
        except (ValueError,RuntimeError,OSError,sqlite3.Error) as e:
            # Standard GetGridMap has no success field; empty map is failure.
            response.map=GridMap()
            self.get_logger().warn("Map query rejected: "+str(e),throttle_duration_sec=5)
        return response

    def health(self):
        sample=memory_sample(self.output)
        self.storage_paused=sample["disk_free_gib"]<self.cfg.get("min_disk_free_gib",5.)
        self.pressure=(sample.get("vmrss_mib",0)>self.cfg.get("mapper_soft_mib",1536)
                       or sample["system_available_mib"]<self.cfg.get("min_system_available_mib",2048))
        self.grid.tiles.max_tiles=(min(16,self.cfg.get("max_resident_tiles",64)) if self.pressure
                                   else self.cfg.get("max_resident_tiles",64))
        try:
            if not self.storage_paused:
                self.grid.tiles.trim()
                if time.monotonic()-self.last_checkpoint_wall>=self.cfg.get("checkpoint_period",5.):
                    self.save();self.last_checkpoint_wall=time.monotonic()
        except Exception as e:
            self.storage_paused=True
            self.get_logger().error("Map storage paused: "+str(e),throttle_duration_sec=5)
        self.report_status(sample)

    def report_status(self,sample=None):
        if sample is None:sample=memory_sample(self.output)
        with self.input_lock:
            now=time.monotonic()
            freshness=dict(sensor_receipt_idle_sec={k:now-v for k,v in self.last_input_wall.items()},
                pose_receipt_idle_sec=now-self.last_pose_wall if self.last_pose_wall is not None else None,
                pose_stamp=self.poses.samples[-1][0] if self.poses.samples else None)
        ros_now=self.get_clock().now().nanoseconds*1e-9
        freshness['pose_age_sec']=ros_now-freshness['pose_stamp'] if freshness['pose_stamp'] is not None else None
        freshness['map_age_sec']=ros_now-stamp_sec(self.last_header) if self.last_header else None
        data=dict(self.stats,**self.grid.memory_stats(),**self.cloud.memory_stats(),**sample,
            **freshness,
            stereo_pending_cells=len(self.stereo_confirmation.pending),stereo_preview_points=len(self.grid.stereo_preview),
            pending=len(self.pending)+len(self.stereo_pending),range_pending=len(self.pending),
            stereo_pending=len(self.stereo_pending),memory_pressure=self.pressure,map_writable=not self.storage_paused,
            global_grid_available=self.global_available,overview_resolution=self.overview.resolution,
            global_published_revision=self.last_global_revision,
            dense_persisted_revision=self.dense_writer.persisted_revision,
            dense_export_error=str(self.dense_writer.last_error) if self.dense_writer.last_error else None,
            map_output=self.cfg.get('map_output','terrain'),
            elevation_fusion_model='temporal_upper_v1' if self.grid.temporal_elevation else 'historical_mean',
            revision=self.grid.update_id,last_map_stamp=stamp_sec(self.last_header) if self.last_header else None)
        if rclpy.ok():self.status_pub.publish(String(data=json.dumps(data)))
        try:
            tmp=self.output/"map_statistics.json.tmp";tmp.write_text(json.dumps(data,indent=2))
            tmp.replace(self.output/"map_statistics.json")
        except OSError as e:
            self.storage_paused=True
            self.get_logger().error("Map statistics write failed: "+str(e),throttle_duration_sec=5)

    def save(self):
        # Checkpoint only: full exports run after the mapper stops. This avoids
        # map copies and long-running SQLite read snapshots during a demo.
        checkpoint_started=time.monotonic()
        with self.input_lock:self.tum.flush()
        self.delivery.checkpoint(self.grid);self.cloud.checkpoint()
        if rclpy.ok():
            for pub in self.revision_pubs:pub.publish(UInt64(data=self.delivery.revision))
        if not self.height_only:self.overview.save(self.internal/"overview.npz")
        metadata=dict(format="t3_fusion_disk_map_v1",frame_id="map",map_to_odom="identity",
            revision=self.grid.update_id,resolution=self.cfg["map_resolution"],tile_cells=self.grid.tile_cells,
            elevation_database="_internal/elevation_tiles.sqlite",
            cloud_database="_internal/lidar_voxels.sqlite",profile=self.cfg,
            last_pose=self.last_pose.tolist() if self.last_pose is not None else None,
            last_stamp=stamp_sec(self.last_header) if self.last_header else None)
        tmp=self.output/"map_metadata.json.tmp";tmp.write_text(json.dumps(metadata,indent=2))
        tmp.replace(self.output/"map_metadata.json")
        if self.latest_window:
            m,layers=self.latest_window
            tmp=self.output/"local_elevation_latest.npz.tmp"
            with tmp.open("wb") as stream:
                extra=(layers if self.height_only else dict(elevation=m.elevation,variance=m.elevation_variance,
                    slope=layers['slope'],step=layers['step'],traversability=layers['traversability']))
                np.savez_compressed(stream,origin=np.array([m.geometry.origin_x,m.geometry.origin_y]),
                    resolution=m.geometry.resolution,**extra)
            tmp.replace(self.output/"local_elevation_latest.npz")
        self.stats["last_checkpoint_sec"]=time.monotonic()-checkpoint_started
        return self.cloud.count

    def save_service(self,request,response):
        try:
            # ROS timers stop with /clock at bag EOF. Explicitly process queued
            # scans against already received poses before checkpointing.
            for _ in range(len(self.pending)+len(self.stereo_pending)):
                before=len(self.pending)+len(self.stereo_pending)
                self.process()
                if len(self.pending)+len(self.stereo_pending)==before:break
            # A live checkpoint may still have inputs awaiting poses. The
            # benchmark retries until pending=0 and checks scan accounting.
            self.publish()
            self.publish_global()
            response.message=f"Checkpointed {self.save()} voxels and elevation tiles to {self.output}; export_map.sh after stop"
            if self.last_dense_revision>=0:self.dense_writer.flush(self.last_dense_revision)
            self.report_status()
            response.success=True
        except Exception as e:response.success=False;response.message=str(e)
        return response


def main():
    rclpy.init();node=TerrainMapper(concurrent_inputs=True)
    # Input callbacks stay responsive; all grid/cloud/SQLite operations remain
    # in the creating thread. No concurrent database writes or map copies.
    inputs=SingleThreadedExecutor();inputs.add_node(node.input_node)
    def receive():
        try:inputs.spin()
        except (ExternalShutdownException,KeyboardInterrupt):pass
    receiver=threading.Thread(target=receive,name='mapping_inputs');receiver.start()
    try:rclpy.spin(node)
    except KeyboardInterrupt:pass
    finally:
        inputs.shutdown();receiver.join();node.input_node.destroy_node()
        try:node.save();node.report_status()
        except Exception as e:print("Final checkpoint failed:",e,flush=True)
        try:node.dense_writer.close(flush_revision=node.last_dense_revision if node.last_dense_revision>=0 else None)
        except Exception as e:print("Dense map checkpoint failed:",e,flush=True)
        try:node.grid.close()
        except Exception as e:print("Tile checkpoint failed:",e,flush=True)
        node.delivery.close();node.cloud.close();node.tum.close();node.destroy_node()
        if rclpy.ok():rclpy.shutdown()
