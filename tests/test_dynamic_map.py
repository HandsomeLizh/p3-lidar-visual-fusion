"""Repeated real-ray evidence clears only visible obsolete geometry."""
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
import numpy as np
from t3_lidar_visual_fusion.bounded_cloud import BoundedCloudStore
from t3_lidar_visual_fusion.disk_map import DiskElevationMap
from t3_lidar_visual_fusion.visibility_cleanup import VisibilityCleanup
from t3_lidar_visual_fusion.terrain_mapper import TerrainMapper


class DynamicMapTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        self.cloud=BoundedCloudStore(self.root/'cloud.db',.1,1000,.2)
        self.grid=DiskElevationMap(self.root/'height.db',resolution=.2,tile_cells=16,max_tiles=8,cache_mib=8)
        self.sensor=np.eye(4);self.sensor[2,3]=1.
        self.obstacle=np.array([[2.02,.02,1.]])
        ground=np.array([[2.02+x,.02+y,0.] for x in [-.4,-.2,0.,.2,.4] for y in [-.4,-.2,0.,.2,.4]])
        points=np.r_[ground,self.obstacle,[[100.,0.,2.]]]
        self.cloud.append(points);self.grid.update_elevation_only(points_map=points)
        self.free=np.array([[4.04,.04,0.],[4.04,.4,0.],[4.04,-.4,0.],[4.04,0.,.4]])
        self.clear=VisibilityCleanup({'confirmations':3,'max_candidates':1000})

    def tearDown(self):
        self.cloud.close();self.grid.close();self.tmp.cleanup()

    def layer(self):
        m=self.grid.extract_window(center_x=2.1,center_y=.1,length_x=2.,length_y=2.)
        layers,occ=TerrainMapper.terrain_layers(SimpleNamespace(cfg={'map_resolution':.2}),m)
        return m,layers,occ

    def test_new_obstacle_then_confirmed_removal_updates_grid_and_preview(self):
        m,layers,_=self.layer();row=round((.02-m.geometry.origin_y)/.2-.5);col=round((2.02-m.geometry.origin_x)/.2-.5)
        self.assertGreater(layers['obstacle'][row,col],.5)
        original=self.cloud.count
        for stamp in [1.,2.]:self.assertEqual(len(self.clear.update(self.cloud,self.free,self.sensor,stamp,.2)),0)
        removed=self.clear.update(self.cloud,self.free,self.sensor,3.,.2)
        np.testing.assert_allclose(removed,self.obstacle)
        self.assertEqual(self.cloud.count,original-1)
        self.grid.rebuild_cells(np.floor(removed[:,:2]/.2).astype(int),self.cloud)
        _,layers,_=self.layer()
        self.assertEqual(layers['obstacle'][row,col],0.)
        self.assertEqual(layers['elevation'][row,col],0.)
        self.assertFalse(np.any(np.all(np.isclose(self.cloud.preview(),self.obstacle),axis=1)))
        self.assertTrue(np.any(np.all(np.isclose(self.cloud.preview(),[100.,0.,2.]),axis=1)))
        self.grid.checkpoint();self.cloud.checkpoint()
        with self.cloud.connection as db:self.assertEqual(db.execute('SELECT COUNT(*) FROM voxels').fetchone()[0],self.cloud.count)

    def test_occlusion_missing_beam_and_unobserved_regions_never_clear(self):
        before=self.cloud.count
        for stamp in range(1,6):
            self.clear.update(self.cloud,self.free*.25,self.sensor,float(stamp),.2)
        self.assertEqual(self.cloud.count,before)
        missing=self.free.copy();missing[:,1]+=3.
        for stamp in range(6,11):self.clear.update(self.cloud,missing,self.sensor,float(stamp),.2)
        self.assertEqual(self.cloud.count,before)

    def test_hit_timeout_and_duplicate_stamp_reset_or_do_not_add_votes(self):
        self.clear.update(self.cloud,self.free,self.sensor,1.,.2)
        self.clear.update(self.cloud,self.free,self.sensor,1.,.2)
        hit=np.r_[self.free,self.obstacle-self.sensor[:3,3]]
        self.clear.update(self.cloud,hit,self.sensor,2.,.2)
        self.assertEqual(len(self.clear.update(self.cloud,self.free,self.sensor,3.,.2)),0)
        self.assertEqual(len(self.clear.update(self.cloud,self.free,self.sensor,20.,.2)),0)
        self.assertEqual(len(self.clear.update(self.cloud,self.free,self.sensor,21.,.2)),0)
        self.assertEqual(len(self.clear.update(self.cloud,self.free,self.sensor,22.,.2)),1)

    def test_hard_obstacle_is_known_even_without_neighbor_heights(self):
        z=np.array([[.2]],dtype=np.float32)
        m=SimpleNamespace(elevation=z,height_range=np.array([[.5]],dtype=np.float32),all_layers=lambda:{'elevation':z})
        layers,occ=TerrainMapper.terrain_layers(SimpleNamespace(cfg={'map_resolution':.2}),m)
        self.assertEqual(occ[0,0],100);self.assertEqual(layers['obstacle'][0,0],1.)


if __name__=='__main__':unittest.main()
