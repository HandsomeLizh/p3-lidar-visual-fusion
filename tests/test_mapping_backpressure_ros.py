#!/usr/bin/env python3
"""Real ROS messages: bounded delayed mapping continues under sustained overload."""
import json
import os
from pathlib import Path
import tempfile
import time
import numpy as np
import yaml
import rclpy
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor
from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Header
from t3_lidar_visual_fusion.terrain_mapper import TerrainMapper
from t3_lidar_visual_fusion.ros_utils import xyz_cloud

ROOT=Path(__file__).resolve().parents[1]


def main():
    assert os.environ['ROS_DOMAIN_ID']=='59'
    result_dir=ROOT/'results/mapping_backpressure_regression_20260916';result_dir.mkdir(exist_ok=False)
    with tempfile.TemporaryDirectory(prefix='mapping_pressure_',dir=ROOT/'build') as temporary:
        output=Path(temporary)
        cfg=yaml.safe_load((ROOT/'config/fusion_motion_candidate.yaml').read_text())
        cfg.update(mapping_pose_settle_sec=.45,map_pending_scans=4,base_from_lidar=np.eye(4).tolist(),
                   map_window=8.,map_publish_period=1000.,semantic_topic='',mapping_wait_timeout=3.)
        path=output/'profile.yaml';path.write_text(yaml.safe_dump(cfg))
        rclpy.init(args=['--ros-args','-p','profile_path:='+str(path),'-p','output_dir:='+str(output/'map')])
        mapper=TerrainMapper();driver=Node('mapping_backpressure_test_driver')
        executor=SingleThreadedExecutor();executor.add_node(mapper);executor.add_node(driver)
        poses=driver.create_publisher(Odometry,'/T3/semantic/current_pose',100)
        clouds=driver.create_publisher(PointCloud2,'/fusion/lidar',100)
        received=[];original=mapper.enqueue
        def enqueue(msg,name,transform):
            received.append(msg.header.stamp.sec+msg.header.stamp.nanosec*1e-9)
            original(msg,name,transform)
        mapper.enqueue=enqueue
        peak_pending=0
        def drain(seconds):
            nonlocal peak_pending
            deadline=time.monotonic()+seconds
            while time.monotonic()<deadline:
                executor.spin_once(timeout_sec=.003)
                peak_pending=max(peak_pending,len(mapper.pending))
        def pose(header,x):
            msg=Odometry();msg.header.stamp=header.stamp;msg.header.frame_id='odom';msg.child_frame_id='base_link'
            msg.pose.pose.position.x=x;msg.pose.pose.orientation.w=1.
            poses.publish(msg)
        xx,yy=np.meshgrid(np.linspace(2.,2.8,12),np.linspace(-.4,.4,12))
        points=np.column_stack([xx.ravel(),yy.ravel(),np.zeros(xx.size)])
        try:
            drain(.6)
            # The map must use the late same-stamp correction, not the earlier
            # pose that arrived before the configured settle interval.
            header=Header();header.stamp=driver.get_clock().now().to_msg();header.frame_id='lidar'
            pose(header,0.);drain(.025);clouds.publish(xyz_cloud(points,header));drain(.15)
            assert mapper.stats['mapped_scans']==0
            pose(header,3.);drain(.55)
            assert mapper.stats['mapped_scans']==1
            assert float(mapper.cloud.preview()[:,0].min())>=4.99
            baseline_received=len(received);baseline_mapped=mapper.stats['mapped_scans']
            until=time.monotonic()+3.;next_input=time.monotonic();sent=0
            while time.monotonic()<until:
                if time.monotonic()>=next_input:
                    header=Header();header.stamp=driver.get_clock().now().to_msg();header.frame_id='lidar'
                    pose(header,3.+sent*.005)
                    clouds.publish(xyz_cloud(points,header));sent+=1;next_input+=.05
                drain(.005)
            during=mapper.stats['mapped_scans']-baseline_mapped
            assert during>=5, 'Oldest-frame eviction can starve all mapping during sustained input'
            drain(2.)
            assert not mapper.pending
            assert len(received)-baseline_received>=sent*.95
            assert mapper.stats['mapped_scans']+mapper.stats['dropped_scans']==len(received)
            assert mapper.stats.get('backpressure_dropped_scans',0)>0
            assert peak_pending<=4
            mapper.publish();mapper.save()
            result=dict(passed=True,input_target_hz=20.,settle_seconds=.45,queue_limit=4,
                        sent_during_stream=sent,received=len(received),mapped_during_stream=during,
                        peak_pending=peak_pending,late_pose_correction_used=True,
                        stats=mapper.stats,qualification='Actual ROS transport and terrain mapper with synthetic poses/clouds; validates bounded queue and continuing map updates, not full sensor-pipeline throughput')
            (result_dir/'verification.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
        finally:
            executor.remove_node(mapper);executor.remove_node(driver)
            mapper.dense_writer.close(flush_revision=mapper.last_dense_revision if mapper.last_dense_revision>=0 else None)
            mapper.grid.close();mapper.delivery.close();mapper.cloud.close();mapper.tum.close()
            mapper.destroy_node();driver.destroy_node();rclpy.shutdown()


if __name__=='__main__':main()
