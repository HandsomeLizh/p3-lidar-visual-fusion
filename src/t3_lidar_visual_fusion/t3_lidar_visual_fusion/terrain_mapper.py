"""Independent LiDAR/ToF maps driven exclusively by fused body odometry."""
import copy
import json
import time
import sqlite3
from collections import deque
from pathlib import Path
import numpy as np
import yaml
import rclpy
from rclpy.node import Node
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


class TerrainMapper(Node):
    def __init__(self):
        super().__init__("fusion_terrain_mapper")
        self.declare_parameter("profile_path","");self.declare_parameter("output_dir","")
        self.cfg=yaml.safe_load(Path(self.get_parameter("profile_path").value).read_text())
        self.output=Path(self.get_parameter("output_dir").value);self.output.mkdir(parents=True,exist_ok=True)
        self.poses=PoseBuffer(self.cfg.get("pose_buffer_samples",1200))
        self.pending=deque(maxlen=self.cfg.get("map_pending_scans",4))
        self.internal=self.output/"_internal";self.internal.mkdir(exist_ok=True)
        self.grid=DiskElevationMap(self.internal/"elevation_tiles.sqlite",resolution=self.cfg["map_resolution"],
            tile_cells=self.cfg.get("tile_cells",128),max_tiles=self.cfg.get("max_resident_tiles",64),
            cache_mib=self.cfg.get("tile_cache_mib",128),max_window_cells=self.cfg.get("max_query_cells",250000))
        self.cloud=BoundedCloudStore(self.internal/"lidar_voxels.sqlite",self.cfg["cloud_voxel_size"],
            self.cfg["cloud_preview_points"],self.cfg.get("preview_voxel_size",.3))
        self.cleanup=VisibilityCleanup(self.cfg.get('dynamic_map',{}))
        self.pose_quality=deque(maxlen=self.cfg.get('pose_buffer_samples',1200))
        self.delivery=CompactDelivery(self.output/"global_grid_map.sqlite3")
        self.overview=Overview(self.cfg.get("overview_side_cells",256),self.cfg.get("overview_resolution",1.))
        self.dense_writer=LiveDenseGlobalMapWriter(self.output/"global_grid_map.npz",frame_id="map")
        self.last_dense_revision=-1
        self.cached_global=None;self.last_global_revision=-1
        self.pressure=False;self.storage_paused=False;self.last_global_wall=0.
        self.last_checkpoint_wall=time.monotonic();self.global_available=False;self.latest_window=None
        self.last_pose=None;self.last_header=None;self.last_stamps={};self.dirty=False
        self.stats={"mapped_scans":0,"dropped_scans":0,"map_points":0,"lidar_scans":0,"tof_scans":0}
        self.bridge=CvBridge();self.mask=None
        self.camera_from_base=inverse(rigid(self.cfg["base_from_camera_left"]))
        self.tum=(self.output/"trajectory_map.tum").open("w")
        self.tum_last_offset=None
        self.create_subscription(Odometry,"/T3/semantic/current_pose",self.odom,100)
        self.create_subscription(PointCloud2,self.cfg.get("mapping_lidar_topic","/fusion/lidar"),lambda m:self.enqueue(m,"lidar",self.cfg["base_from_lidar"]),QoSProfile(depth=2,reliability=ReliabilityPolicy.RELIABLE))
        for source in self.cfg.get("tof_sources",[]):
            rigid(source["base_from_sensor"])
            self.create_subscription(PointCloud2,source["topic"],lambda m,s=source:self.enqueue(m,s["name"],s["base_from_sensor"]),qos_profile_sensor_data)
        if self.cfg.get("semantic_topic"):
            self.create_subscription(Image,self.cfg["semantic_topic"],self.semantic,1)
        qos=QoSProfile(depth=1,reliability=ReliabilityPolicy.RELIABLE,durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.grid_pubs=[self.create_publisher(GridMap,t,qos) for t in
            ["/T3/mapping/elevation_map","/T3/mapping/grid_map","/Car/T3/mapping/grid_map"]]
        self.global_pubs=[self.create_publisher(GridMap,t,qos) for t in
            ["/T3/mapping/global_grid_map","/Car/T3/mapping/global_grid_map"]]
        self.overview_pubs=[self.create_publisher(OccupancyGrid,t,qos) for t in
            ["/T3/mapping/global_overview","/Car/T3/mapping/global_overview"]]
        self.revision_pubs=[self.create_publisher(UInt64,t,qos) for t in
            ["/T3/mapping/global_map_revision","/Car/T3/mapping/global_map_revision"]]
        self.cloud_pub=self.create_publisher(PointCloud2,"/T3/mapping/lidar_map",qos)
        self.elevation_pub=self.create_publisher(PointCloud2,"/T3/mapping/elevation_cloud",qos)
        self.incremental_pub=self.create_publisher(IncrementalSemanticMap,"/T3/semantic/incremental_map",qos)
        self.occupancy_pub=self.create_publisher(OccupancyGrid,"/T3/mapping/traversability",qos)
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

    def odom(self,msg):
        try:
            t=transform_from_pose(msg.pose.pose)
            appended=self.poses.append(stamp_sec(msg),t)
            revised=not appended and self.poses.replace_latest(stamp_sec(msg),t)
            if appended or revised:
                covariance=np.asarray(msg.pose.covariance).reshape(6,6)
                self.pose_quality.append((stamp_sec(msg),float(np.max(np.diag(covariance)[:3])),
                                          float(np.max(np.diag(covariance)[3:]))))
                self.last_pose=t
                if appended:
                    self.tum_last_offset=self.tum.tell()
                elif self.tum_last_offset is not None:
                    self.tum.seek(self.tum_last_offset);self.tum.truncate()
                p=msg.pose.pose;q=p.orientation
                self.tum.write(f"{stamp_sec(msg):.9f} {p.position.x:.9f} {p.position.y:.9f} {p.position.z:.9f} {q.x:.9f} {q.y:.9f} {q.z:.9f} {q.w:.9f}\n")
        except ValueError:pass

    def semantic(self,msg):
        try:
            mask=self.bridge.imgmsg_to_cv2(msg,desired_encoding="passthrough")
            if mask.ndim==2 and np.issubdtype(mask.dtype,np.integer):
                self.mask=(stamp_sec(msg),np.asarray(mask).copy())
        except Exception:pass

    def enqueue(self,msg,name,transform):
        if self.storage_paused:
            self.stats["dropped_scans"]+=1
            return
        if len(msg.data)>self.cfg.get("max_cloud_bytes",16000000):
            self.stats["dropped_scans"]+=1
            return
        stamp=stamp_sec(msg)
        if stamp<=self.last_stamps.get(name,-1):return
        self.last_stamps[name]=stamp
        if len(self.pending)==self.pending.maxlen:
            # Preserve the oldest scan until its pose-settle interval elapses.
            # Evicting it on every arrival can prevent any scan from ever being
            # mapped when input_hz * settle_seconds exceeds the queue capacity.
            self.stats["dropped_scans"]+=1
            self.stats["backpressure_dropped_scans"]=self.stats.get("backpressure_dropped_scans",0)+1
            return
        self.pending.append((time.monotonic(),msg,name,rigid(transform)))

    def add_semantics(self,points_base,points_map,stamp):
        # A task-2 label image contributes labels only at calibrated projections.
        # Unseen labels remain unknown; no geometric class is fabricated.
        if self.mask is None or abs(self.mask[0]-stamp)>self.cfg["semantic_tolerance"]:return
        mask=self.mask[1]
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
        if not self.pending:return
        if self.storage_paused:
            self.stats["dropped_scans"]+=len(self.pending);self.pending.clear();return
        queued,msg,name,t_base_sensor=self.pending[0]
        # A bounded settle interval lets delayed visual corrections join the
        # same measurement-time pose before a scan is permanently integrated.
        if time.monotonic()-queued<self.cfg.get("mapping_pose_settle_sec",0.):
            return
        stamp=stamp_sec(msg)
        point_times=None
        compensated=(self.cfg.get("cloud_motion_compensated",False) or self.cfg.get("deskew",{}).get("enabled",False))
        if name=="lidar" and not self.cfg["instantaneous_cloud"] and not compensated:
            point_times=cloud_arrays(msg,("time",))[:,0]
        pose=self.poses.at(stamp,tolerance=self.cfg["pose_tolerance"],max_gap=self.cfg["pose_max_gap"])
        end_pose=pose if point_times is None or not len(point_times) else self.poses.at(
            stamp+float(np.max(point_times)),tolerance=self.cfg["pose_tolerance"],max_gap=self.cfg["pose_max_gap"])
        if pose is None or end_pose is None:
            end_stamp=stamp+(0. if point_times is None or not len(point_times) else float(np.max(point_times)))
            # A missing interval already behind the latest pose cannot arrive
            # later in this monotonic buffer; do not block newer clouds on it.
            past_gap=bool(self.poses.samples and self.poses.samples[-1][0]>end_stamp+self.cfg["pose_tolerance"])
            if past_gap or time.monotonic()-queued>self.cfg["mapping_wait_timeout"]:
                self.pending.popleft();self.stats["dropped_scans"]+=1
            return
        self.pending.popleft()
        process_started=time.monotonic()
        try:
            xyz=cloud_arrays(msg)
            good=np.isfinite(xyz).all(axis=1)
            ranges=np.linalg.norm(xyz,axis=1)
            valid=good&(ranges>=self.cfg["min_range"])&(ranges<=self.cfg["max_range"])
            xyz=xyz[valid]
            base=xyz@t_base_sensor[:3,:3].T+t_base_sensor[:3,3]
            points=base@pose[:3,:3].T+pose[:3,3]
            if point_times is not None:
                times=point_times[valid]
                bins=np.floor(times/.005).astype(int)
                for key in np.unique(bins):
                    use=bins==key
                    sample=self.poses.at(stamp+float(np.mean(times[use])),
                        tolerance=self.cfg["pose_tolerance"],max_gap=self.cfg["pose_max_gap"])
                    if sample is None:raise ValueError("Missing pose during LiDAR scan")
                    points[use]=base[use]@sample[:3,:3].T+sample[:3,3]
            if name=="lidar":
                removed=np.empty((0,3))
                dynamic=self.cfg.get('dynamic_map',{})
                quality=min(self.pose_quality,key=lambda q:abs(q[0]-stamp)) if self.pose_quality else None
                clear_ok=(quality is not None and abs(quality[0]-stamp)<=self.cfg['pose_tolerance']
                    and 0<=quality[1]<=dynamic.get('max_position_variance',.01)
                    and 0<=quality[2]<=dynamic.get('max_rotation_variance',.0025))
                # A single rigid scan and LiDAR-only evidence are required for
                # this clearance path. Calibrated multi-source maps need their
                # own retained source evidence before replacing height columns.
                if (dynamic.get('enabled',False) and clear_ok and not self.cfg.get('tof_sources')
                        and self.cfg.get('instantaneous_cloud',False)):
                    removed=self.cleanup.update(self.cloud,xyz,pose@t_base_sensor,stamp,self.cfg['map_resolution'])
                else:
                    # An uncertain pose breaks clearance confirmation; later
                    # good scans must establish fresh free-space evidence.
                    self.cleanup.votes.clear()
                    self.cleanup.stats=dict(checked=0,free_evidence=0,removed=0,pending=0)
                self.cloud.append(points);self.stats["lidar_scans"]+=1
                if len(removed):
                    self.grid.rebuild_cells(np.floor(removed[:,:2]/self.cfg['map_resolution']).astype(int),self.cloud)
                    self.refresh_overview(removed)
                self.stats['cleanup']=dict(self.cleanup.stats,pose_qualified=clear_ok,
                    removed_total=self.stats.get('cleanup',{}).get('removed_total',0)+len(removed))
            else:self.stats["tof_scans"]+=1
            # Bound scan-density bias while retaining min/median/max height
            # evidence for each XY cell. The existing robust map rejects outliers.
            keys=np.floor(points[:,:2]/self.cfg["map_resolution"]).astype(np.int64)
            order=np.lexsort((keys[:,1],keys[:,0]));ordered=keys[order]
            if len(order):
                starts=np.r_[0,np.flatnonzero(np.any(np.diff(ordered,axis=0),axis=1))+1]
                ends=np.r_[starts[1:],len(order)]
                selected=[]
                for a,b in zip(starts,ends):
                    ix=order[a:b];ix=ix[np.argsort(points[ix,2])]
                    selected.extend(ix[np.unique([0,len(ix)//2,len(ix)-1])])
                selected=np.asarray(selected,dtype=int)
                if len(selected)>self.cfg["max_elevation_points"]:
                    selected=selected[np.linspace(0,len(selected)-1,self.cfg["max_elevation_points"],dtype=int)]
                self.grid.update_elevation_only(points_map=points[selected])
                self.add_semantics(base[selected],points[selected],stamp)
            self.overview.update(points)
            self.last_header=copy.deepcopy(msg.header);self.last_header.frame_id="map"
            self.dirty=True;self.stats["mapped_scans"]+=1;self.stats["map_points"]=self.cloud.count
            self.stats["last_map_update_sec"]=time.monotonic()-process_started
        except (ValueError,RuntimeError,OSError,sqlite3.Error) as e:
            self.stats["dropped_scans"]+=1
            if isinstance(e,(OSError,sqlite3.Error)):self.storage_paused=True
            self.get_logger().error("Map input rejected: "+str(e),throttle_duration_sec=5)

    def terrain_layers(self,m):
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
            m=self.grid.extract_window(center_x=float(self.last_pose[0,3]),center_y=float(self.last_pose[1,3]),
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
            layers={key:all_layers[key] for key in ['elevation','elevation_variance','height_range',
                                                    'traversability','occupancy','obstacle']}
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
            self.get_logger().warn(str(e)+"; use local map/query and global_overview",throttle_duration_sec=20)
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
        data=dict(self.stats,**self.grid.memory_stats(),**self.cloud.memory_stats(),**sample,
            pending=len(self.pending),memory_pressure=self.pressure,map_writable=not self.storage_paused,
            global_grid_available=self.global_available,overview_resolution=self.overview.resolution,
            global_published_revision=self.last_global_revision,
            dense_persisted_revision=self.dense_writer.persisted_revision,
            dense_export_error=str(self.dense_writer.last_error) if self.dense_writer.last_error else None,
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
        self.tum.flush()
        self.delivery.checkpoint(self.grid);self.cloud.checkpoint()
        if rclpy.ok():
            for pub in self.revision_pubs:pub.publish(UInt64(data=self.delivery.revision))
        self.overview.save(self.internal/"overview.npz")
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
                np.savez_compressed(stream,elevation=m.elevation,variance=m.elevation_variance,
                    origin=np.array([m.geometry.origin_x,m.geometry.origin_y]),resolution=m.geometry.resolution,
                    slope=layers["slope"],step=layers["step"],traversability=layers["traversability"])
            tmp.replace(self.output/"local_elevation_latest.npz")
        self.stats["last_checkpoint_sec"]=time.monotonic()-checkpoint_started
        return self.cloud.count

    def save_service(self,request,response):
        try:
            # ROS timers stop with /clock at bag EOF. Explicitly process queued
            # scans against already received poses before checkpointing.
            for _ in range(len(self.pending)):
                before=len(self.pending)
                self.process()
                if len(self.pending)==before:break
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
    rclpy.init();node=TerrainMapper()
    try:rclpy.spin(node)
    except KeyboardInterrupt:pass
    finally:
        try:node.save();node.report_status()
        except Exception as e:print("Final checkpoint failed:",e,flush=True)
        try:node.dense_writer.close(flush_revision=node.last_dense_revision if node.last_dense_revision>=0 else None)
        except Exception as e:print("Dense map checkpoint failed:",e,flush=True)
        try:node.grid.close()
        except Exception as e:print("Tile checkpoint failed:",e,flush=True)
        node.delivery.close();node.cloud.close();node.tum.close();node.destroy_node()
        if rclpy.ok():rclpy.shutdown()
