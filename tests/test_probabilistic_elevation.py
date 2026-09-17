"""Behavioral regressions for repeated scans, two surfaces, uncertainty and storage."""
import tempfile
import unittest
from pathlib import Path
import numpy as np
from t3_lidar_visual_fusion.probabilistic_elevation import (
    ProbabilisticSurfaceGrid, scan_observations, height_variances)
from t3_lidar_visual_fusion.height_bias import HeightBiasCompensator
from t3_lidar_visual_fusion.disk_map import DiskElevationMap
from t3_lidar_visual_fusion.bounded_cloud import BoundedCloudStore


class HeightFilterTests(unittest.TestCase):
    def setUp(self):
        self.grid=ProbabilisticSurfaceGrid(fusion_config={'enabled':True},resolution=.2,
            length_x=2.,length_y=2.,origin_x=0.,origin_y=0.)

    def put(self,z,stamp,variance=.0004,pose_variance=0.,source='lidar'):
        return self.grid.fuse_height(0,0,z,variance,pose_variance,0.,stamp,source)

    def test_uncertain_measurement_gets_less_weight(self):
        self.put(0.,1.,variance=.0001)
        self.put(.08,2.,variance=.04)
        self.assertLess(self.grid.elevation_mean[0,0],.001)

    def test_duplicate_scan_cannot_change_height_or_confidence(self):
        self.put(0.,1.)
        before=self.grid.elevation_variance_layer().copy()
        self.assertEqual(self.put(.02,1.),'duplicate')
        self.assertEqual(self.put(.02,.5),'out_of_order')
        self.assertEqual(self.grid.elevation_mean[0,0],0.)
        np.testing.assert_array_equal(self.grid.elevation_variance_layer(),before)

    def test_pose_uncertainty_does_not_average_away(self):
        for i in range(100):self.put(.001*(i%2),float(i),variance=.0404,pose_variance=.04)
        self.assertGreaterEqual(self.grid.elevation_variance_layer()[0,0],.04-1e-8)

    def test_common_pose_error_does_not_make_each_noisy_sample_dominate(self):
        for i in range(30):self.put(0.,float(i),variance=.0404,pose_variance=.04)
        self.put(.08,30.,variance=.0404,pose_variance=.04)
        self.assertLess(self.grid.elevation_mean[0,0],.02)
        self.assertGreaterEqual(self.grid.elevation_variance_layer()[0,0],.04-1e-8)

    def test_more_precise_pose_can_correct_an_old_uncertain_height(self):
        self.put(0.,1.,variance=.0404,pose_variance=.04)
        self.put(.08,2.,variance=.0004,pose_variance=0.)
        self.assertGreater(self.grid.elevation_mean[0,0],.07)
        self.assertLess(self.grid.elevation_variance_layer()[0,0],.001)

    def test_worse_pose_receives_little_weight(self):
        self.put(0.,1.,variance=.0004,pose_variance=0.)
        self.put(.08,2.,variance=.0404,pose_variance=.04)
        self.assertLess(self.grid.elevation_mean[0,0],.002)

    def test_one_bad_return_does_not_lift_ground(self):
        self.put(0.,1.)
        self.assertEqual(self.put(.6,2.),'candidate')
        for i in range(3,12):self.put(0.,float(i))
        self.assertEqual(self.grid.elevation_mean[0,0],0.)
        self.assertEqual(self.put(.6,12.),'candidate')
        self.assertEqual(self.grid.height_candidate_count[0,0],1)

    def test_repeated_rock_becomes_upper_surface_without_fake_middle_height(self):
        self.put(0.,1.)
        for stamp in (2.,3.):
            self.assertEqual(self.put(.6,stamp),'candidate')
            self.put(0.,stamp+.01,source='stereo')
            self.assertEqual(self.grid.elevation_mean[0,0],0.)
        self.assertEqual(self.put(.6,4.),'surface_replaced')
        self.assertAlmostEqual(self.grid.elevation_mean[0,0],.6,places=6)
        for stamp in range(5,12):self.put(0.,float(stamp))
        self.assertAlmostEqual(self.grid.elevation_mean[0,0],.6,places=6)

    def test_same_exposure_different_sources_cannot_confirm_a_change(self):
        self.put(0.,1.)
        self.put(.6,2.)
        self.put(.6,2.01,source='stereo')
        self.assertEqual(self.grid.height_candidate_count[0,0],1)
        self.assertEqual(self.grid.elevation_mean[0,0],0.)

    def test_inconsistent_candidates_do_not_replace_surface(self):
        self.put(0.,1.)
        for stamp,z in enumerate((.6,.8,.4,.7),2):self.put(z,float(stamp))
        self.assertEqual(self.grid.elevation_mean[0,0],0.)

    def test_invalid_input_does_not_consume_timestamp(self):
        self.assertEqual(self.put(np.nan,1.),'invalid')
        self.assertEqual(self.put(0.,1.),'initialized')


class ScanNoiseTests(unittest.TestCase):
    def test_nearby_cell_is_never_starved_by_distant_coverage(self):
        points=np.column_stack((np.arange(-100,101)+.05,np.full(201,.05),np.zeros(201)))
        for phase in range(6):
            obs=scan_observations(points,.0004,0.,.2,20,priority_center=[0.,0.],
                                  priority_radius=3.,selection_phase=phase)
            selected={tuple(x) for x in obs['cells']}
            for point in points[np.linalg.norm(points[:,:2],axis=1)<2.9]:
                self.assertIn(tuple(np.floor(point[:2]/.2).astype(int)),selected)
            self.assertEqual(len(selected),20)

    def test_rotating_budget_eventually_visits_all_static_cells(self):
        points=np.column_stack((np.arange(30)+.05,np.full(30,.05),np.zeros(30)))
        seen=set()
        for phase in range(6):
            obs=scan_observations(points,.0004,0.,.2,5,priority_center=[-10.,0.],
                                  priority_radius=1.,selection_phase=phase)
            self.assertEqual(len(obs['cells']),5)
            seen.update(tuple(x) for x in obs['cells'])
        self.assertEqual(len(seen),30)

    def test_dense_repeat_does_not_create_independent_measurements(self):
        points=np.tile([.05,.05,.01],(100,1))
        obs=scan_observations(points,.0004,.01,.2,100)
        self.assertEqual(len(obs['height']),1)
        self.assertAlmostEqual(obs['variance'][0],.0104)

    def test_two_height_modes_and_isolated_extreme(self):
        points=np.array([[.05,.05,z] for z in [0.]*70+[.6]*30])
        obs=scan_observations(points,.0004,0.,.2,100)
        self.assertAlmostEqual(obs['height'][0],.6)
        points=np.array([[.05,.05,z] for z in [0.]*99+[10.]])
        obs=scan_observations(points,.0004,0.,.2,100)
        self.assertEqual(obs['height'][0],0.)

    def test_sparse_upper_return_is_retained_for_temporal_check(self):
        obs=scan_observations([[.05,.05,0.],[.05,.05,.6]],.0004,0.,.2,100)
        self.assertAlmostEqual(obs['height'][0],.6)

    def test_pose_rotation_error_grows_with_range(self):
        covariance=np.zeros((6,6));covariance[2,2]=.01;covariance[4,4]=.001
        sensor,pose=height_variances([[1.,0.,0.],[10.,0.,0.]],np.eye(4),covariance,'lidar',{}, {})
        np.testing.assert_allclose(pose,[.011,.11])
        self.assertGreater(sensor[1],sensor[0])

    def test_bad_pose_covariance_is_rejected(self):
        covariance=np.eye(6);covariance[0,0]=-1.
        with self.assertRaises(ValueError):height_variances([[1.,0.,0.]],np.eye(4),covariance,'lidar',{}, {})


class BiasTests(unittest.TestCase):
    def setUp(self):
        xx,yy=np.meshgrid(np.arange(-20,20),np.arange(-20,20))
        cells=np.column_stack((xx.ravel(),yy.ravel()))
        self.obs=dict(cells=cells,height=np.zeros(len(cells)),variance=np.full(len(cells),.001),
                      pose_variance=np.zeros(len(cells)),relief=np.zeros(len(cells)))
        self.priors=(np.zeros(len(cells)),np.full(len(cells),.001),np.zeros(len(cells)))
        self.bias=HeightBiasCompensator({})
        self.pose=np.eye(4)
        self.call(1.)

    def call(self,stamp,source='lidar'):
        return self.bias.update(self.obs,self.priors,stamp=stamp,source=source,pose=self.pose,resolution=.2)

    def test_common_height_bias_corrected_without_moving_old_map(self):
        self.obs['height']+=.2;self.pose[2,3]=.2
        offset,allowed=self.call(2.)
        self.assertTrue(allowed);self.assertAlmostEqual(offset,.2)
        np.testing.assert_array_equal(self.priors[0],0.)

    def test_small_real_obstacle_is_not_removed_as_drift(self):
        self.obs['height'][:50]=.6;self.pose[0,3]=.2
        offset,allowed=self.call(2.)
        self.assertTrue(allowed);self.assertEqual(offset,0.)

    def test_real_inclined_ground_is_not_flattened(self):
        self.obs['height']=self.obs['cells'][:,0]*.2*.15
        self.priors=(self.obs['height'].copy(),self.priors[1],self.priors[2])
        self.pose[0,3]=.2
        offset,allowed=self.call(2.)
        self.assertTrue(allowed);self.assertEqual(offset,0.)

    def test_tilt_is_quarantined_and_stereo_cannot_bypass_it(self):
        self.obs['height']=self.obs['cells'][:,0]*.2*.08;self.pose[0,3]=.2
        offset,allowed=self.call(2.)
        self.assertFalse(allowed);self.assertEqual(offset,0.)
        self.assertFalse(self.call(2.1,'stereo')[1])

    def test_stationary_uniform_environment_change_is_not_pose_compensation(self):
        self.obs['height']+=.2
        offset,allowed=self.call(2.)
        self.assertFalse(allowed);self.assertEqual(offset,0.)

    def test_limited_overlap_and_excessive_jump_do_not_compensate(self):
        self.obs['height']+=.8;self.pose[0,3]=.2
        self.assertFalse(self.call(2.)[1]);self.assertEqual(self.bias.offset,0.)
        self.obs={k:v[:20] for k,v in self.obs.items()};self.priors=tuple(v[:20] for v in self.priors)
        self.call(3.);self.assertEqual(self.bias.offset,0.)

    def test_bias_state_survives_restart(self):
        self.obs['height']+=.2;self.pose[2,3]=.2;self.call(2.)
        restored=HeightBiasCompensator({});restored.restore(self.bias.state())
        self.assertEqual(restored.offset,self.bias.offset)
        self.assertEqual(restored.last_stamp,2.)
        with self.assertRaises(ValueError):restored.restore({'offset_m':float('nan')})


class StorageTests(unittest.TestCase):
    def test_eviction_preserves_pending_change_and_rejects_duplicate_after_reload(self):
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'grid.db'
            def opened():return DiskElevationMap(path,resolution=.2,tile_cells=8,max_tiles=1,cache_mib=2,
                                                 fusion_config={'enabled':True})
            grid=opened()
            def put(z,stamp,cell=(-1,-1)):
                obs=dict(cells=np.array([cell]),height=np.array([z]),variance=np.array([.0004]),
                         pose_variance=np.array([0.]),relief=np.array([0.]))
                return grid.update_probabilistic(obs,stamp=stamp,source='lidar')
            put(0.,1.);put(.6,2.);put(0.,2.,(100,100));grid.close();grid=opened()
            self.assertEqual(put(.6,2.),{'duplicate':1})
            put(.6,3.);put(.6,4.)
            self.assertAlmostEqual(grid.height_priors([[-1,-1]])[0][0],.6,places=6)
            self.assertLessEqual(grid.tiles.nbytes,grid.tiles.tile_bytes)
            grid.close()

    def test_legacy_database_cannot_silently_change_fusion_model(self):
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'grid.db';grid=DiskElevationMap(path,tile_cells=8)
            grid.update_elevation_only(points_map=[[.05,.05,0.]]);grid.close()
            with self.assertRaises(ValueError):DiskElevationMap(path,tile_cells=8,fusion_config={'enabled':True})

    def test_verified_cloud_cleanup_can_lower_surface(self):
        with tempfile.TemporaryDirectory() as td:
            grid=DiskElevationMap(Path(td)/'grid.db',tile_cells=8,fusion_config={'enabled':True})
            cloud=BoundedCloudStore(Path(td)/'cloud.db',.05,1000,.1)
            try:
                grid.update_elevation_only(points_map=[[.05,.05,.6]])
                cloud.append(np.array([[.05,.05,0.],[.15,.15,0.]]))
                grid.rebuild_cells([[0,0]],cloud)
                self.assertEqual(grid.height_priors([[0,0]])[0][0],0.)
            finally:grid.close();cloud.close()


if __name__=='__main__':unittest.main()
