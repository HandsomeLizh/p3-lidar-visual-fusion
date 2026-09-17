"""Isolated height-only mapper, real ROS serialization and visual admission.

Synthetic sloped ground, a rock, overhead returns and calibrated stereo.
No vehicle controls or goal messages are published.
"""
import json,os,sys,tempfile,time
from pathlib import Path
import numpy as np,rclpy,yaml
from std_msgs.msg import Header,String
from nav_msgs.msg import Odometry
from grid_map_msgs.msg import GridMap
from t3_lidar_visual_fusion.terrain_mapper import TerrainMapper
from t3_lidar_visual_fusion.ros_utils import xyz_cloud

ROOT=Path(__file__).resolve().parents[1]


def main():
    assert os.environ['ROS_DOMAIN_ID']=='94' and os.environ['ROS_LOCALHOST_ONLY']=='1'
    with tempfile.TemporaryDirectory(dir=ROOT/'build') as temporary:
        out=Path(temporary);cfg=yaml.safe_load((ROOT/'config/hardware104.yaml').read_text())
        lidar=np.eye(4);lidar[2,3]=.65
        camera=np.eye(4);camera[:3,:3]=[[0,0,1],[-1,0,0],[0,-1,0]];camera[2,3]=.8
        cfg.update(base_from_lidar=lidar.tolist(),base_from_camera_left=camera.tolist(),
            mapping_pose_settle_sec=0.,instantaneous_cloud=True,deskew={'enabled':False},
            tile_cells=16,map_publish_period=1000.,global_publish_period=1000.,min_range=.1)
        profile=out/'profile.yaml';profile.write_text(yaml.safe_dump(cfg))
        rclpy.init(args=['--ros-args','-p','profile_path:='+str(profile),'-p','output_dir:='+str(out/'map')])
        mapper=TerrainMapper();received={'local':[],'global':[]}
        for key,topic in [('local','/Car/T3/mapping/grid_map'),('global','/Car/T3/mapping/global_grid_map')]:
            mapper.create_subscription(GridMap,topic,lambda m,k=key:received[k].append(m),10)
        os.environ['T3_VISUAL_RUNTIME']=str(out/'visual')
        sys.path.insert(0,str(ROOT/'visual'))
        from visual_monitor import Monitor
        monitor=Monitor()
        def header(t,frame):
            h=Header(frame_id=frame);h.stamp.sec,h.stamp.nanosec=divmod(round(t*1e9),10**9);return h
        def pose(t):
            m=Odometry();m.header=header(t,'odom');m.child_frame_id='base_link';m.pose.pose.orientation.w=1.
            m.pose.covariance=(np.eye(6)*.0001).ravel().tolist();mapper.odom(m)
        x,y=np.meshgrid(np.arange(-2.9,3.,.2),np.arange(-2.9,3.,.2))
        ground=np.c_[x.ravel(),y.ravel(),(.12*x+.03*y).ravel()]
        rock_ground=.12*1.1+.03*.1
        def scan(t,rock=True):
            pose(t);points=np.r_[ground,[[1.1,.1,rock_ground]]*70,
                [[1.1,.1,rock_ground+(.6 if rock else 0.)]]*30,[[1.1,.1,2.5]]]
            mapper.enqueue(xyz_cloud(points-lidar[:3,3],header(t,'lidar')),'lidar',lidar)
            mapper.process()
        def height(x,y):
            m=mapper.grid.extract_window(center_x=x+.001,center_y=y+.001,length_x=.2,length_y=.2)
            return float(m.elevation[0,0])
        try:
            scan(100.);scan(100.1)
            assert abs(height(1.1,.1)-(rock_ground+.6))<.02
            assert abs(height(2.1,.1)-(.12*2.1+.03*.1))<.02
            assert mapper.cloud.connection.execute('SELECT COUNT(*) FROM voxels WHERE z>2').fetchone()[0]>0
            # Confirm a visual cell outside the dense range fixture using its
            # fresh, actually observed supporting ground plane.
            for t in [100.2,100.3]:
                pose(t);mapper.fusion_status(String(data=json.dumps(dict(vision_enabled=True,localization_valid=True))))
                point=np.array([[3.7,.1,.12*3.7+.03*.1]])
                optical=(point-camera[:3,3])@camera[:3,:3]
                mapper.enqueue_stereo(xyz_cloud(optical,header(t,'camera_left_optical'),[.001],'position_variance'))
                mapper.process()
            assert mapper.stats['stereo_cells']>=1,mapper.stats
            scan(101.,False);scan(101.1,False)
            assert height(1.1,.1)>rock_ground+.5
            scan(101.2,False);assert abs(height(1.1,.1)-rock_ground)<.02
            mapper.publish();mapper.publish_global();mapper.save()
            until=time.monotonic()+.7
            while time.monotonic()<until:rclpy.spin_once(mapper,timeout_sec=.01)
            wanted={'elevation','elevation_variance','height_range','roughness','observation_count'}
            for name,messages in received.items():
                assert messages,name
                assert set(messages[-1].layers)==wanted,messages[-1].layers
            assert abs(received['local'][-1].info.length_x-32.)<1e-6
            assert mapper.incremental_pub is None and mapper.occupancy_pub is None and not mapper.overview_pubs
            mapper.dense_writer.flush(mapper.last_dense_revision)
            with np.load(out/'map/global_grid_map.npz') as saved:
                assert wanted<=set(saved.files) and 'occupancy' not in saved
            with np.load(out/'map/local_elevation_latest.npz') as saved:assert wanted<=set(saved.files)
            # The observer does not run its send timer; no goal publisher is invoked.
            monitor.on_grid(received['global'][-1])
            monitor.localization_unavailable=lambda:False
            assert monitor.request_goal(2.1,.1,0.),monitor.goal_status
            monitor.goal_requested=None
            assert not monitor.request_goal(100.,100.,0.)
            report=dict(passed=True,ground_slope_retained=True,minority_rock_retained=True,
                overhead_excluded_from_height_but_retained_in_cloud=True,confirmed_lower_surface=True,
                stereo_height_cells=mapper.stats['stereo_cells'],height_only_layers=sorted(wanted),
                local_window_m=32,known_height_goal_admitted=True,unknown_goal_rejected=True,
                vehicle_commands_published=0,goal_messages_published=0)
            result=ROOT/'results/hardware104_height_20260917/height_interfaces.json'
            result.parent.mkdir(parents=True,exist_ok=True);result.write_text(json.dumps(report,indent=2))
            print(json.dumps(report))
        finally:
            mapper.dense_writer.close();mapper.grid.close();mapper.delivery.close();mapper.cloud.close();mapper.tum.close()
            mapper.destroy_node();monitor.destroy_node();rclpy.try_shutdown()


if __name__=='__main__':main()
