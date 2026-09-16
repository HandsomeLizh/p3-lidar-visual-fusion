"""Real guard and ROS transport: zero motion replaces one qualified observation."""
import os,tempfile,time,unittest
from pathlib import Path
import cv2,numpy as np,yaml,rclpy
from nav_msgs.msg import Odometry
from std_msgs.msg import Header
from t3_lidar_visual_fusion.adaptive_guard import AdaptiveGuard
from t3_lidar_visual_fusion.ros_utils import xyz_cloud

ROOT=Path(__file__).resolve().parents[1]

class StationaryGuardTests(unittest.TestCase):
    def test_stationary_then_slow_start_then_blackout(self):
        self.assertEqual(os.environ['ROS_DOMAIN_ID'],'59')
        with tempfile.TemporaryDirectory(dir=ROOT/'build',prefix='stationary_guard_') as temporary:
            path=Path(temporary);cfg=yaml.safe_load((ROOT/'config/simulation_live.yaml').read_text())
            cfg['stationary']={'enabled':True};cfg['vision_gate']['recovery_frames']=2
            profile=path/'profile.yaml';profile.write_text(yaml.safe_dump(cfg))
            rclpy.init(args=['--ros-args','-p','profile_path:='+str(profile),'-p','output_dir:='+str(path)])
            guard=AdaptiveGuard();observed=[]
            sub=guard.create_subscription(Odometry,'/fusion/vision_odom_guarded',observed.append,30)
            rng=np.random.default_rng(847);base=cv2.GaussianBlur(rng.integers(30,220,(256,320),np.uint8),(3,3),0)
            az,el=np.meshgrid(np.linspace(-np.pi,np.pi,128,endpoint=False),np.linspace(-.4,.4,16))
            points=np.column_stack([3*np.cos(el.ravel())*np.cos(az.ravel()),3*np.cos(el.ravel())*np.sin(az.ravel()),3*np.sin(el.ravel())])
            previous=None;x=0.;shift=0.
            def spin(duration):
                deadline=time.monotonic()+duration
                while time.monotonic()<deadline:rclpy.spin_once(guard,timeout_sec=.005)
            def frame(speed,black=False):
                nonlocal previous,x,shift
                now=guard.get_clock().now();stamp=now.nanoseconds/1e9
                if previous is not None:x+=speed*(stamp-previous)
                previous=stamp
                if speed>.003:shift+=1.
                image=cv2.warpAffine(base,np.float32([[1,0,shift],[0,1,0]]),(320,256),borderMode=cv2.BORDER_REFLECT)
                if black:image[:]=0
                for side in [0,1]:
                    m=guard.bridge.cv2_to_imgmsg(image,'mono8');m.header.stamp=now.to_msg();guard.image(m,side)
                header=Header(stamp=now.to_msg(),frame_id='lidar');guard.stationary_cloud(xyz_cloud(points,header))
                lidar=Odometry();lidar.header=Header(stamp=now.to_msg(),frame_id='odom');lidar.child_frame_id='base_link';lidar.pose.pose.orientation.w=1.
                lidar.pose.pose.position.x=x;lidar.pose.covariance=np.diag([.01]*3+[.005]*3).ravel().tolist();guard.lio(lidar)
                vision=Odometry();vision.header=Header(stamp=now.to_msg(),frame_id='learned_epoch_1');vision.child_frame_id='base_link';vision.pose=lidar.pose
                guard.vision(vision);spin(.06)
            try:
                spin(.4)
                for _ in range(9):frame(.001)
                self.assertGreater(guard.stationary.zero_updates,0)
                self.assertEqual(guard.stationary.status()['state'],'stationary')
                self.assertEqual(observed[-1].twist.twist.linear.x,0.)
                before=guard.stationary.zero_updates
                for _ in range(3):frame(.024)
                self.assertEqual(guard.stationary.zero_updates,before)
                self.assertAlmostEqual(observed[-1].twist.twist.linear.x,.024,places=5)
                self.assertNotEqual(guard.stationary.status()['state'],'stationary')
                count=len(observed);frame(0.,black=True)
                self.assertEqual(len(observed),count)
                self.assertEqual(guard.stationary.status()['state'],'unknown')
            finally:
                guard.destroy_subscription(sub);guard.destroy_node();rclpy.shutdown()

if __name__=='__main__':unittest.main()
