"""The same physical cell must not change terrain class at a moving crop edge."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from t3_lidar_visual_fusion.disk_map import DiskElevationMap
from t3_lidar_visual_fusion.terrain_mapper import TerrainMapper


class LocalGlobalTerrainTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.grid=DiskElevationMap(Path(self.tmp.name)/'map.db',resolution=.2,
                                  tile_cells=8,max_tiles=8,cache_mib=8)
        xx,yy=np.meshgrid(np.arange(-24,24),np.arange(-24,24))
        height=(xx%2)*.1
        height[(xx==5)&(yy==4)]=.8
        keep=~((xx==-2)&(yy==3))
        points=np.column_stack(((xx[keep]+.5)*.2,(yy[keep]+.5)*.2,height[keep]))
        self.grid.update_elevation_only(points_map=points)
        self.mapper=SimpleNamespace(grid=self.grid,cfg={'map_resolution':.2})
        self.mapper.terrain_layers=lambda m:TerrainMapper.terrain_layers(self.mapper,m)

    def tearDown(self):
        self.grid.close()
        self.tmp.cleanup()

    def window(self,**kwargs):
        return TerrainMapper.terrain_window(self.mapper,**kwargs)

    def assert_same_as_global(self,**kwargs):
        global_map=self.grid.extract_global(10000)
        global_layers,global_occupancy=self.mapper.terrain_layers(global_map)
        local,layers,occupancy=self.window(**kwargs)
        col=round((local.geometry.origin_x-global_map.geometry.origin_x)/.2)
        row=round((local.geometry.origin_y-global_map.geometry.origin_y)/.2)
        crop=np.s_[row:row+local.geometry.height,col:col+local.geometry.width]
        for name in ('elevation','height_range','slope','step','occupancy','traversability','obstacle'):
            with self.subTest(layer=name,window=kwargs):
                np.testing.assert_array_equal(layers[name],global_layers[name][crop])
        np.testing.assert_array_equal(occupancy,global_occupancy[crop])
        expected=self.grid.window_geometry(**kwargs)
        self.assertEqual(local.geometry,expected)
        self.assertEqual(occupancy.shape,(expected.height,expected.width))
        return local,layers,occupancy

    def test_moving_rectangular_windows_match_global_including_edges(self):
        for x,y,lx,ly in ((0.,0.,3.2,2.8),(-1.13,.76,2.2,1.4),(.37,-.53,1.7,2.3)):
            self.assert_same_as_global(center_x=x,center_y=y,length_x=lx,length_y=ly)

    def test_old_one_sided_edge_would_mark_free_ripple_as_obstacle(self):
        m,layers,_=self.assert_same_as_global(center_x=-1.03,center_y=-1.07,length_x=2.,length_y=2.)
        dy,dx=np.gradient(m.elevation,.2)
        old_slope_blocked=np.arctan(np.hypot(dx,dy))>=np.deg2rad(35)*.65
        self.assertTrue(old_slope_blocked[:,0].any())
        self.assertTrue(np.all(layers['obstacle'][:,0]==0.))

    def test_single_cell_reads_neighbors_and_preserves_unknown_and_rock(self):
        for x,y,expected in ((-1.1,-1.1,0.),(-.3,.7,np.nan),(1.1,.9,1.)):
            _,layers,_=self.assert_same_as_global(center_x=x,center_y=y,length_x=.2,length_y=.2)
            if np.isnan(expected):self.assertTrue(np.isnan(layers['obstacle'][0,0]))
            else:self.assertEqual(layers['obstacle'][0,0],expected)

    def test_halo_is_bounded_and_requested_cell_limit_still_applies(self):
        self.grid.max_window_cells=100
        geometry,padded=self.grid.extract_window_with_halo(center_x=0.,center_y=0.,length_x=2.,length_y=2.)
        self.assertEqual((geometry.width,geometry.height),(10,10))
        self.assertEqual(padded.elevation.shape,(12,12))
        with self.assertRaises(ValueError):
            self.grid.extract_window_with_halo(center_x=0.,center_y=0.,length_x=2.2,length_y=2.)


if __name__=='__main__':unittest.main()
