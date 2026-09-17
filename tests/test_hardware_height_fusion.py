"""Height adaptation, pose quality, real steps and bounded persistent state."""
import tempfile
import unittest
from pathlib import Path
import numpy as np
from t3_lidar_visual_fusion.disk_map import DiskElevationMap
from t3_lidar_visual_fusion.elevation_fusion import surface_observations
from t3_lidar_visual_fusion.legacy.tiled_semantic_map import TiledSemanticMapManager


class TemporalElevationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.grids = []

    def tearDown(self):
        for grid in self.grids:grid.close()
        self.tmp.cleanup()

    def grid(self, name='height.db', **kwargs):
        grid = DiskElevationMap(self.root/name, tile_cells=8, max_tiles=2, cache_mib=1,
                                elevation_fusion={'enabled':True}, **kwargs)
        self.grids.append(grid)
        return grid

    def feed(self, grid, z, stamp, x=.1, covariance=None, copies=3):
        points = np.tile([x, .1, z], (copies, 1))
        grid.update_elevation_only(points_map=points, stamp=stamp,
                                   covariance=np.zeros((6,6)) if covariance is None else covariance,
                                   pose_origin=np.zeros(3))

    def value(self, grid, x=.1):
        # Avoid placing the lower edge exactly on an inexact decimal boundary.
        return grid.extract_window(center_x=x+.001, center_y=.101, length_x=.2, length_y=.2)

    def test_old_height_remains_correctable_after_long_observation(self):
        grid = self.grid()
        for i in range(1000):self.feed(grid, 0., 1. + i*.1)
        for i in range(20):self.feed(grid, .1, 101. + i*.1)
        m = self.value(grid)
        self.assertLess(abs(float(m.elevation[0,0])-.1), .01)
        self.assertGreaterEqual(float(m.elevation_variance[0,0]), .000399)
        self.assertLess(float(m.height_range[0,0]), .001)

    def test_single_bad_scan_rejected_and_persistent_step_confirmed(self):
        grid = self.grid();self.feed(grid, 0., 1.)
        self.feed(grid, .6, 1.1)
        self.assertEqual(float(self.value(grid).elevation[0,0]), 0.)
        self.assertEqual(grid.fusion_stats['pending'], 1)
        self.feed(grid, 0., 1.2)
        for i in range(2):
            self.feed(grid, .4, 2. + .1*i)
            self.assertEqual(float(self.value(grid).elevation[0,0]), 0.)
        self.feed(grid, .4, 2.2)
        self.assertAlmostEqual(float(self.value(grid).elevation[0,0]), .4, places=6)
        self.assertEqual(grid.fusion_stats['replaced'], 1)

    def test_duplicate_old_or_long_gap_frames_do_not_confirm_change(self):
        grid = self.grid();self.feed(grid, 0., 1.);self.feed(grid, .5, 1.1)
        for stamp in [1.1,1.05,1.1]:self.feed(grid, .5, stamp)
        self.assertEqual(float(self.value(grid).elevation[0,0]), 0.)
        self.feed(grid, .5, 3.)
        self.feed(grid, .5, 3.1)
        self.assertEqual(float(self.value(grid).elevation[0,0]), 0.)
        self.feed(grid, .5, 3.2)
        self.assertAlmostEqual(float(self.value(grid).elevation[0,0]), .5)

    def test_confidence_and_density_control(self):
        good, poor, dense = self.grid('good.db'), self.grid('poor.db'), self.grid('dense.db')
        for grid in [good,poor,dense]:self.feed(grid, 0., 1.)
        self.feed(good, .08, 1.1)
        self.feed(poor, .08, 1.1, covariance=np.diag([.04,.04,.04,0,0,0]))
        self.feed(dense, .08, 1.1, copies=1000)
        self.assertLess(float(self.value(poor).elevation[0,0]), float(self.value(good).elevation[0,0])*.3)
        np.testing.assert_equal(self.value(good).elevation, self.value(dense).elevation)
        np.testing.assert_equal(self.value(good).elevation_variance, self.value(dense).elevation_variance)
        self.feed(good, 10., 1.2, covariance=np.eye(6))
        self.assertEqual(good.fusion_stats['rejected'], 1)

    def test_real_spatial_steps_slopes_and_unseen_cells_remain(self):
        grid = self.grid()
        for stamp in [1.,1.1,1.2]:
            for x,z in [(.1,0.),(.3,.3),(.5,.35)]:self.feed(grid,z,stamp,x=x)
        for x,z in [(.1,0.),(.3,.3),(.5,.35)]:
            self.assertAlmostEqual(float(self.value(grid,x).elevation[0,0]),z,places=6)
        self.feed(grid,.02,30.,x=.1)
        self.assertAlmostEqual(float(self.value(grid,.3).elevation[0,0]),.3,places=6)
        self.assertTrue(np.isnan(self.value(grid,1.1).elevation[0,0]))

    def test_pose_rotation_variance_projection_and_outlier(self):
        points=np.array([[2.1,3.1,0.],[2.1,3.1,.01],[2.1,3.1,3.]])
        cov=np.diag([.1,.1,.02,.003,.004,.05])
        obs=surface_observations(points,.2,cov,[0,0,0],{})[0]
        self.assertAlmostEqual(obs[3], .03**2+.02+3.1**2*.003+2.1**2*.004+.005**2)
        self.assertLess(abs(obs[2]),.015)
        self.assertLess(obs[5],.02)
        cov[2,2]=-1
        with self.assertRaises(ValueError):surface_observations(points,.2,cov,[0,0,0],{})

    def test_upper_surface_is_preserved_instead_of_a_midair_average(self):
        grid=self.grid()
        for height in [.4,.6]:
            grid.update_elevation_only(points_map=np.array([[.1,.1,0.],[.1,.1,height]]),
                stamp=1.,covariance=np.zeros((6,6)),pose_origin=np.zeros(3))
            self.assertAlmostEqual(float(self.value(grid).elevation[0,0]),.4,places=6)
        self.assertEqual(grid.fusion_stats['rejected'],1)

    def test_conflict_cannot_reduce_uncertainty(self):
        grid=self.grid();cov=np.diag([0.,0.,.04,0.,0.,0.])
        self.feed(grid,0.,1.,covariance=cov)
        before=float(self.value(grid).elevation_variance[0,0])
        self.feed(grid,.6,1.1,covariance=cov)
        self.assertEqual(float(self.value(grid).elevation[0,0]),0.)
        self.assertGreaterEqual(float(self.value(grid).elevation_variance[0,0]),before)

    def test_correlated_pose_error_remains_after_repeated_scans(self):
        grid=self.grid();cov=np.diag([0.,0.,.04,0.,0.,0.])
        for i in range(100):self.feed(grid,0.,1.+i*.1,covariance=cov)
        self.assertGreaterEqual(float(self.value(grid).elevation_variance[0,0]),.04-1e-8)

    def test_minority_rock_and_isolated_outlier(self):
        for lower,upper,height in [(70,30,.6),(50,50,.6),(99,1,0.)]:
            points=np.array([[.1,.1,0.]]*lower+[[.1,.1,.6]]*upper)
            obs=surface_observations(points,.2,np.zeros((6,6)),[0,0,0],{})
            self.assertAlmostEqual(obs[0,2],height)

    def test_range_replaces_stereo_only_after_qualified_observation(self):
        grid=self.grid()
        grid.update_stereo([[.1,.1,.2,.01,2]],stamp=1.)
        self.feed(grid,5.,1.1,covariance=np.eye(6))
        self.assertAlmostEqual(float(self.value(grid).elevation[0,0]),.2,places=6)
        grid.close();self.grids.remove(grid);grid=self.grid()
        self.feed(grid,.4,1.2)
        self.assertAlmostEqual(float(self.value(grid).elevation[0,0]),.4,places=6)
        self.assertEqual(len(grid.update_stereo([[.1,.1,0.,.01,2]],stamp=1.3)),0)

    def test_confirmed_lower_surface_updates_without_retaining_old_extreme(self):
        grid=self.grid();self.feed(grid,.6,1.)
        for t in [2.,2.1]:
            self.feed(grid,0.,t)
            self.assertAlmostEqual(float(self.value(grid).elevation[0,0]),.6,places=6)
        self.feed(grid,0.,2.2)
        self.assertEqual(float(self.value(grid).elevation[0,0]),0.)
        self.assertLess(float(self.value(grid).height_range[0,0]),.001)

    def test_uncertain_scan_breaks_surface_change_confirmation(self):
        grid=self.grid();self.feed(grid,0.,1.)
        self.feed(grid,.5,1.1);self.feed(grid,.5,1.2)
        self.feed(grid,.5,1.3,covariance=np.eye(6))
        self.feed(grid,.5,1.4)
        self.assertEqual(float(self.value(grid).elevation[0,0]),0.)
        self.feed(grid,.5,1.5);self.feed(grid,.5,1.6)
        self.assertAlmostEqual(float(self.value(grid).elevation[0,0]),.5)

    def test_eviction_restart_snapshot_keep_height_confidence_and_candidate(self):
        grid=self.grid();self.feed(grid,0.,1.);self.feed(grid,.5,1.1)
        for x in [2.1,4.1,6.1,8.1]:self.feed(grid,.2,1.,x=x)
        self.assertLessEqual(len(grid.tiles.cache),2)
        grid.close();self.grids.remove(grid)
        grid=self.grid()
        self.feed(grid,.5,1.2);self.feed(grid,.5,1.3)
        m=self.value(grid)
        self.assertAlmostEqual(float(m.elevation[0,0]),.5)
        snapshot=grid.save_snapshot(self.root/'snapshot',timestamp_text='test',frame_id='map')
        restored=TiledSemanticMapManager.load_snapshot(snapshot['index'])
        restored_map=restored.extract_window(center_x=.1,center_y=.1,length_x=.2,length_y=.2)
        np.testing.assert_equal(restored_map.elevation,m.elevation)
        np.testing.assert_equal(restored_map.elevation_variance,m.elevation_variance)
        np.testing.assert_equal(restored_map.roughness,m.roughness)


if __name__=='__main__':unittest.main()
