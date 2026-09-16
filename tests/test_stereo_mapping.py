"""Depth uncertainty, independent confirmation and range-sensor precedence."""
import tempfile
import unittest
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from t3_lidar_visual_fusion.stereo_mapping import sparse_map_points,StereoConfirmation
from t3_lidar_visual_fusion.stereo_geometry import StereoGeometry,project
from t3_lidar_visual_fusion.disk_map import DiskElevationMap
from t3_lidar_visual_fusion.compact_delivery import CompactDelivery
from t3_lidar_visual_fusion.legacy.compact_grid_store import load_compact_tile
from test_stereo_geometry import profile,scene


class SparseMapTests(unittest.TestCase):
    def test_rectified_points_return_to_original_optical_frame(self):
        cfg=profile();right=np.eye(4);right[:3,:3]=Rotation.from_euler('y',.035).as_matrix();right[0,3]=.25
        cfg['base_from_camera_right']=right
        g=StereoGeometry(cfg);rect=np.array([[.1,.1,2.],[-.3,.2,3.],[.1,.1,20.],[np.nan,0,1.]])
        xyz,var=sparse_map_points(g,rect,{'max_points':20})
        self.assertEqual(len(xyz),2);self.assertTrue((var>0).all());self.assertGreater(var[1],var[0])
        back=xyz@g.base_from_left[:3,:3].T+g.base_from_left[:3,3]
        expected=rect[:2]@g.base_from_rect[:3,:3].T+g.base_from_rect[:3,3]
        np.testing.assert_allclose(back,expected,atol=1e-7)

    def test_temporal_bad_depth_points_cannot_enter_map(self):
        g=StereoGeometry(profile());points=scene();uv,_=project(g.p0,points)
        bad=points.copy();bad[:12]*=1.2
        ids=np.arange(len(points));result=g.estimate(points,bad,uv,uv,np.c_[ids,ids])
        self.assertFalse(np.isin(np.arange(12),result.current_inliers).any())
        self.assertEqual(len(result.current_inliers),len(points)-12)

    def test_same_frame_is_not_confirmation_and_memory_is_bounded(self):
        confirm=StereoConfirmation(.2,{'pending_cells':4})
        points=np.array([[.1,.1,.2],[.11,.1,.2]])
        self.assertEqual(len(confirm.observe(points,[.01,.01],1.)),0)
        self.assertEqual(len(confirm.observe(points,[.01,.01],1.)),0)
        result=confirm.observe(points,[.01,.01],2.)
        self.assertEqual(len(result),1);self.assertEqual(result[0,4],2)
        self.assertGreaterEqual(result[0,3],.0025)
        confirm.observe(np.c_[np.arange(20),np.ones(20),np.ones(20)],np.full(20,.01),3.)
        self.assertLessEqual(len(confirm.pending),4)
        confirm.clear();self.assertEqual(len(confirm.pending),0)

    def test_weighted_height_persists_and_lidar_replaces_stereo(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'map.sqlite';grid=DiskElevationMap(path,tile_cells=8,max_tiles=2,cache_mib=2)
            try:
                grid.update_stereo([[.1,.1,.2,.01,2]])
                grid.update_stereo([[.1,.1,.3,.09,2]])
                tile=grid.tiles.get((0,0));self.assertLess(tile.elevation_mean[0,0],.22)
                self.assertGreater(tile.elevation_variance_layer()[0,0],0.)
                self.assertTrue(tile.stereo_owned[0,0]);grid.checkpoint()
                delivery=CompactDelivery(Path(directory)/'delivery.sqlite')
                try:
                    delivery.checkpoint(grid)
                    exported=load_compact_tile(Path(directory)/'delivery.sqlite',0,0)
                    self.assertGreater(exported['elevation_variance'][0,0],0.)
                    self.assertAlmostEqual(exported['elevation'][0,0],tile.elevation_mean[0,0],places=2)
                finally:delivery.close()
            finally:grid.close()
            grid=DiskElevationMap(path,tile_cells=8,max_tiles=2,cache_mib=2)
            try:
                self.assertTrue(grid.tiles.get((0,0)).stereo_owned[0,0])
                grid.update_elevation_only(points_map=[[.1,.1,.8]])
                self.assertEqual(len(grid.update_stereo([[.1,.1,.1,.01,2]])),0)
                tile=grid.tiles.get((0,0));self.assertFalse(tile.stereo_owned[0,0])
                self.assertAlmostEqual(tile.elevation_mean[0,0],.8)
                window=grid.extract_window(center_x=.8,center_y=.8,length_x=1.6,length_y=1.6)
                self.assertEqual(np.isfinite(window.elevation).sum(),1)
            finally:grid.close()


if __name__=='__main__':unittest.main()
