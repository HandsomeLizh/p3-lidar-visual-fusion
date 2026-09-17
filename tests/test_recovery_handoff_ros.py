"""A validated LiDAR submap rejoins continuous visual output without a jump."""
import json
import os
from pathlib import Path
import tempfile
import time
import numpy as np
import rclpy
from rclpy.executors import SingleThreadedExecutor
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image
from rosgraph_msgs.msg import Clock
from std_msgs.msg import String
import yaml
from t3_lidar_visual_fusion.adaptive_guard import AdaptiveGuard

ROOT = Path(__file__).resolve().parents[1]


def main():
    assert os.environ['ROS_DOMAIN_ID'] == '82'
    out = ROOT / 'results/rear_map_audit_20260916'
    out.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=ROOT / 'build', prefix='handoff_') as tmp:
        cfg = yaml.safe_load((ROOT / 'config/simulation_live.yaml').read_text())
        cfg['stationary']['enabled'] = False
        cfg['telemetry_motion']['enabled'] = False
        cfg['vision_gate']['recovery_frames'] = 3
        profile = Path(tmp) / 'profile.yaml'
        profile.write_text(yaml.safe_dump(cfg))
        rclpy.init(args=['--ros-args', '-p', 'profile_path:=' + str(profile), '-p', 'use_sim_time:=true'])
        guard = AdaptiveGuard()
        driver = rclpy.create_node('recovery_handoff_fixture')
        ex = SingleThreadedExecutor()
        ex.add_node(guard)
        ex.add_node(driver)
        pubs = {topic: driver.create_publisher(cls, topic, 30) for topic, cls in [
            ('/fusion/lio_raw', Odometry), ('/fusion/learned_raw', Odometry),
            ('/fusion/ekf', Odometry), ('/fusion/lidar_quality', String),
            ('/fusion/left', Image), ('/fusion/right', Image), ('/clock', Clock)]}
        poses = []
        references = []
        driver.create_subscription(Odometry, '/T3/semantic/current_pose', poses.append, 100)
        driver.create_subscription(Odometry, '/fusion/recovery_reference', references.append, 100)

        def drain(seconds=.025):
            end = time.monotonic() + seconds
            while time.monotonic() < end:
                ex.spin_once(timeout_sec=.001)

        def odom(t, x, variance, epoch='odom'):
            m = Odometry()
            ns = round(t * 1e9)
            m.header.stamp.sec, m.header.stamp.nanosec = divmod(ns, 1000000000)
            m.header.frame_id = epoch
            m.child_frame_id = 'base_link'
            m.pose.pose.position.x = float(x)
            m.pose.pose.orientation.w = 1.
            m.pose.covariance = (np.eye(6) * variance).ravel().tolist()
            return m

        yy, xx = np.indices((192, 256))
        pixels = np.where((xx // 12 + yy // 12) % 2, 50, 200).astype(np.uint8).tobytes()
        index = 0

        def frame(healthy=True, submap=0, reliable=True, lidar_offset=0., late_quality=False):
            nonlocal index
            t = 1000. + 1.3 * index
            x = .2 * index
            index += 1
            pubs['/clock'].publish(Clock(clock=odom(t + .01, 0, .001).header.stamp))
            drain()
            im = Image()
            im.header.stamp = odom(t, 0, .001).header.stamp
            im.height = 192
            im.width = 256
            im.encoding = 'mono8'
            im.step = 256
            im.data = pixels
            pubs['/fusion/left'].publish(im)
            pubs['/fusion/right'].publish(im)
            drain()
            q = String(data=json.dumps(dict(stamp_sec=t, submap_id=submap, reliable=reliable,
                bridge_position_variance=.05 if submap else 0.,
                bridge_rotation_variance=.02 if submap else 0.)))
            if not late_quality:
                pubs['/fusion/lidar_quality'].publish(q)
                drain()
            pubs['/fusion/lio_raw'].publish(odom(t, x + lidar_offset if healthy else 2.2,
                                                .001 if healthy else 1e6))
            drain()
            # LiDAR arrives before the same-time visual trajectory. The guard
            # must defer the handoff until it can check that reference.
            pubs['/fusion/learned_raw'].publish(odom(t, x, 1e6 if index == 1 else .001, 'learned_epoch_1'))
            drain()
            if late_quality:
                pubs['/fusion/lidar_quality'].publish(q)
                drain()
            pubs['/fusion/ekf'].publish(odom(t, x if healthy else 500., .001 if healthy else 1000.))
            drain()
            return t, x

        try:
            drain(.6)
            for _ in range(12):
                frame()
            assert poses and guard.visual_continuity.reference is not None
            before = len(poses)
            for _ in range(100):
                _, x = frame(healthy=False)
            assert x - 2.2 > 19.
            assert len(poses) - before >= 98 and guard.output_source == 'visual', (len(poses)-before, guard.filter_quality, guard.visual_continuity.status(), guard.lidar_gate.reason, guard.gate.reason)
            assert references and abs(references[-1].pose.pose.position.x - x) < 1e-6
            # An announced id alone must not enable an unqualified map.
            frame(submap=1, reliable=False)
            assert guard.lidar_applied_submap == 0
            # Even a reliable backend report cannot move the established pose
            # by ten metres if it disagrees with the accepted visual output.
            frame(submap=1, lidar_offset=10.)
            assert guard.lidar_applied_submap == 0
            for k in range(5):
                _, x = frame(submap=1, late_quality=(k % 2 == 0))
            assert guard.lidar_applied_submap == 1 and guard.lidar_usable, guard.lidar_gate.reason
            assert guard.output_qualified
            assert abs(poses[-1].pose.pose.position.x - x) < 1e-6
            steps = np.abs(np.diff([m.pose.pose.position.x for m in poses]))
            assert np.max(steps) < .401, float(np.max(steps))
            cov = np.array(poses[-1].pose.covariance).reshape(6, 6)
            assert np.linalg.eigvalsh(cov[:3, :3])[0] >= .05 - 1e-12
            assert np.linalg.eigvalsh(cov[3:, 3:])[0] >= .02 - 1e-12
            result = dict(passed=True, visual_only_failed_lidar_frames=100,
                visual_travel_after_last_lidar_m=20., handoff_to_confirmed_submap=True,
                unqualified_submap_blocked=True, inconsistent_submap_blocked=True,
                both_callback_orders=True, max_output_step_m=float(np.max(steps)),
                bridge_uncertainty_preserved=True,
                scope='Synthetic ROS frontend and quality inputs; isolated domain 82, real guard')
            (out / 'recovery_handoff_ros.json').write_text(json.dumps(result, indent=2))
            print(json.dumps(result))
        finally:
            ex.shutdown()
            guard.destroy_node()
            driver.destroy_node()
            rclpy.try_shutdown()


if __name__ == '__main__':
    main()
