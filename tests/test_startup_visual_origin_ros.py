"""Startup LiDAR loss must not strand an independently tracked visual epoch."""
import argparse,json,os,tempfile,time
from pathlib import Path
import numpy as np
import rclpy
from rclpy.executors import SingleThreadedExecutor
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image
from rosgraph_msgs.msg import Clock
import yaml
from t3_lidar_visual_fusion.adaptive_guard import AdaptiveGuard


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--workspace',type=Path,required=True)
    parser.add_argument('--report',type=Path,required=True)
    parser.add_argument('--expect-unavailable',action='store_true')
    parser.add_argument('--assert-dark-recovery',action='store_true')
    args=parser.parse_args()
    assert os.environ.get('ROS_DOMAIN_ID')=='89' and os.environ.get('ROS_LOCALHOST_ONLY')=='1'
    with tempfile.TemporaryDirectory() as folder:
        cfg=yaml.safe_load((args.workspace/'config/simulation_lidar_camera_fov.yaml').read_text())
        cfg['stationary']['enabled']=False;cfg['telemetry_motion']['enabled']=False
        cfg['vision_gate']['recovery_frames']=2
        if args.assert_dark_recovery:cfg['visual_continuity']['max_gap']=120.
        profile=Path(folder)/'profile.yaml';profile.write_text(yaml.safe_dump(cfg))
        rclpy.init(args=['--ros-args','-p','profile_path:='+str(profile),'-p','use_sim_time:=true'])
        guard=AdaptiveGuard();driver=rclpy.create_node('startup_origin_fixture')
        executor=SingleThreadedExecutor();executor.add_node(guard);executor.add_node(driver)
        pubs={topic:driver.create_publisher(kind,topic,20) for topic,kind in [
            ('/fusion/lio_raw',Odometry),('/fusion/learned_raw',Odometry),('/fusion/ekf',Odometry),
            ('/fusion/visual_epoch_origin',Odometry),('/fusion/left',Image),('/fusion/right',Image),('/clock',Clock)]}
        poses=[];driver.create_subscription(Odometry,'/T3/semantic/current_pose',poses.append,100)
        def spin(seconds=.03):
            end=time.monotonic()+seconds
            while time.monotonic()<end:executor.spin_once(timeout_sec=.001)
        def odom(stamp,x=0.,variance=.001,frame='odom'):
            m=Odometry();m.header.stamp.sec,m.header.stamp.nanosec=divmod(round(stamp*1e9),10**9)
            m.header.frame_id=frame;m.child_frame_id='base_link';m.pose.pose.orientation.w=1.
            m.pose.pose.position.x=x;m.pose.covariance=(np.eye(6)*variance).ravel().tolist();return m
        def prepare(stamp):
            pubs['/clock'].publish(Clock(clock=odom(stamp+.01).header.stamp));spin()
            m=Image();m.header.stamp=odom(stamp).header.stamp;m.height=192;m.width=256;m.step=256;m.encoding='mono8'
            yy,xx=np.indices((192,256));m.data=np.where((xx//12+yy//12)%2,50,200).astype(np.uint8).tobytes()
            for topic in ['/fusion/left','/fusion/right']:pubs[topic].publish(m)
            spin()
        try:
            spin(.5);prepare(1000.)
            pubs['/fusion/lio_raw'].publish(odom(1000.));spin()
            origin=odom(1000.,variance=1e6,frame='learned_epoch_1')
            pubs['/fusion/visual_epoch_origin'].publish(origin)
            pubs['/fusion/learned_raw'].publish(origin);spin()
            pubs['/fusion/ekf'].publish(odom(1000.));spin()
            initial=len(poses);assert initial==1
            # The only EKF reference is this first accepted origin. All later
            # LiDAR scans fail, while vision observes genuine 1 cm increments.
            for i in range(1,21):
                t=1000.+i;prepare(t)
                pubs['/fusion/lio_raw'].publish(odom(t,4.5,1e6));spin()
                pubs['/fusion/learned_raw'].publish(odom(t,i*.01,.01,'learned_epoch_1'));spin(.05)
            count=len(poses)-initial
            report={'visual_outputs_after_startup_lidar_loss':count,
                    'continuity':guard.visual_continuity.status(),'last_x':poses[-1].pose.pose.position.x}
            if args.expect_unavailable:
                assert count==0,report
            else:
                assert count>=18 and guard.output_qualified,report
                np.testing.assert_allclose(poses[-1].pose.pose.position.x,.2,atol=1e-8)
                offset=0.
                if args.assert_dark_recovery:
                    before_dark=len(poses)
                    for t in [1021.,1050.,1080.,1110.]:
                        prepare(t)
                        pubs['/fusion/lio_raw'].publish(odom(t,4.5,1e6));spin()
                        pubs['/fusion/learned_raw'].publish(odom(t,.2,1e6,'learned_epoch_1'));spin()
                    assert len(poses)==before_dark
                    for t in [1111.,1112.,1113.,1114.]:
                        prepare(t)
                        pubs['/fusion/lio_raw'].publish(odom(t,4.5,1e6));spin()
                        pubs['/fusion/learned_raw'].publish(odom(t,.22,.01,'learned_epoch_1'));spin(.05)
                    assert len(poses)>before_dark and guard.output_qualified
                    np.testing.assert_allclose(poses[-1].pose.pose.position.x,.22,atol=1e-8)
                    report.update(dark_interval_sec=90.,no_output_during_dark=True,
                        outputs_after_dark=len(poses)-before_dark,recovered_same_epoch=True)
                    offset=100.
                # Invalid visual samples and an unrelated new epoch cannot
                # borrow the previous origin or a rejected EKF reference.
                before=len(poses);prepare(1021.+offset)
                pubs['/fusion/lio_raw'].publish(odom(1021.+offset,4.5,1e6));spin()
                pubs['/fusion/learned_raw'].publish(odom(1021.+offset,.2,1e6,'learned_epoch_1'));spin()
                assert len(poses)==before
                prepare(1022.+offset)
                origin=odom(1022.+offset,variance=1e6,frame='learned_epoch_2')
                pubs['/fusion/visual_epoch_origin'].publish(origin)
                pubs['/fusion/learned_raw'].publish(origin);spin()
                pubs['/fusion/ekf'].publish(odom(1022.+offset,5.));spin()
                assert len(poses)==before
                for i in range(1,5):
                    t=1022.+offset+i;prepare(t)
                    pubs['/fusion/lio_raw'].publish(odom(t,5.,1e6));spin()
                    pubs['/fusion/learned_raw'].publish(odom(t,i*.01,.01,'learned_epoch_2'));spin()
                assert len(poses)==before and guard.visual_continuity.reference is None
                report.update(invalid_visual_stopped=True,unreferenced_epoch_stopped=True,
                              rejected_ekf_did_not_anchor=True)
            report['passed']=True;args.report.write_text(json.dumps(report,indent=2));print(json.dumps(report))
        finally:
            executor.shutdown();guard.destroy_node();driver.destroy_node();rclpy.shutdown()


if __name__=='__main__':main()
