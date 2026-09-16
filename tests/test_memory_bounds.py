"""Deferred tests: run only after the other agent releases the remote."""
import tempfile
import unittest
from pathlib import Path
import numpy as np
from t3_lidar_visual_fusion.disk_map import DiskElevationMap
from t3_lidar_visual_fusion.bounded_cloud import BoundedCloudStore
from t3_lidar_visual_fusion.compact_delivery import CompactDelivery
from t3_lidar_visual_fusion.overview import Overview
from t3_lidar_visual_fusion.legacy.compact_grid_store import load_compact_metadata,load_compact_tile
from t3_lidar_visual_fusion.legacy.tiled_semantic_map import TiledSemanticMapManager


class MemoryBounds(unittest.TestCase):
    def test_eviction_reload_negative_coordinates_and_lossless_export(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            grid=DiskElevationMap(root/"tiles.sqlite",tile_cells=8,max_tiles=2,cache_mib=1)
            locations=np.array([[x*2.+.1,.1,3.+x/100.] for x in range(-20,21)])
            for point in locations:
                grid.update_elevation_only(points_map=point[None,:])
                self.assertLessEqual(len(grid.tiles.cache),2)
            self.assertGreater(grid.tile_count,2)
            for point in locations:
                m=grid.extract_window(center_x=point[0],center_y=point[1],length_x=.4,length_y=.4)
                self.assertAlmostEqual(float(np.nanmean(m.elevation)),point[2],places=5)
                self.assertLessEqual(len(grid.tiles.cache),2)
            saved=grid.save_snapshot(root/"snapshot",timestamp_text="fixture",frame_id="map")
            restored=TiledSemanticMapManager.load_snapshot(saved["index"])
            self.assertEqual(restored.tile_count,grid.tile_count)
            grid.close()
            reopened=DiskElevationMap(root/"tiles.sqlite",tile_cells=8,max_tiles=2,cache_mib=1)
            self.assertEqual(reopened.tile_count,restored.tile_count)
            reopened.close()

    def test_query_rejected_before_large_allocation(self):
        with tempfile.TemporaryDirectory() as folder:
            grid=DiskElevationMap(Path(folder)/"tiles.sqlite",max_window_cells=100)
            for args in [dict(center_x=0,center_y=0,length_x=10000,length_y=10000),
                         dict(center_x=float("nan"),center_y=0,length_x=1,length_y=1),
                         dict(center_x=0,center_y=0,length_x=0,length_y=1)]:
                with self.assertRaises(ValueError):grid.extract_window(**args)
            grid.close()

    def test_preview_bound_dedup_and_vertical_surfaces(self):
        with tempfile.TemporaryDirectory() as folder:
            store=BoundedCloudStore(Path(folder)/"cloud.sqlite",.1,preview_points=40,preview_voxel=.2)
            rng=np.random.default_rng(4)
            for _ in range(12):
                points=rng.uniform(-10,10,(200,3));store.append(points);store.append(points)
                self.assertLessEqual(len(store.preview()),40)
                self.assertEqual(len(np.unique(store.keys,axis=0)),len(store.keys))
            self.assertGreater(store.count,40)
            store.append([[20,20,0],[20,20,1]])
            rows=store.connection.execute("SELECT z FROM voxels WHERE ix=200 AND iy=200").fetchall()
            self.assertEqual(len(rows),2)
            store.close()

    def test_original_compact_reader_and_committed_revision(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            grid=DiskElevationMap(root/"tiles.sqlite",tile_cells=8,max_tiles=1,cache_mib=1)
            delivery=CompactDelivery(root/"global_grid_map.sqlite3")
            for i in range(8):
                grid.update_elevation_only(points_map=np.array([[i*2+.1,.1,1.25]]))
            delivery.checkpoint(grid)
            metadata=load_compact_metadata(root/"global_grid_map.sqlite3")
            self.assertEqual(metadata["map_revision"],grid.update_id)
            self.assertEqual(metadata["tile_count"],grid.tile_count)
            tile=load_compact_tile(root/"global_grid_map.sqlite3",0,0)
            self.assertAlmostEqual(float(np.nanmean(tile["elevation"])),1.25,places=2)
            grid.update_elevation_only(points_map=np.array([[.1,.1,1.25]]))
            self.assertLess(delivery.revision,grid.update_id)
            delivery.checkpoint(grid)
            self.assertEqual(delivery.revision,grid.update_id)
            delivery.close();grid.close()

    def test_overview_expands_without_growing_arrays(self):
        overview=Overview(32,1.)
        initial=overview.minimum.nbytes+overview.maximum.nbytes+overview.count.nbytes
        for radius in [1,50,500,5000]:
            overview.update(np.array([[-radius,-radius,1],[radius,radius,2]],float))
            self.assertEqual(overview.minimum.nbytes+overview.maximum.nbytes+overview.count.nbytes,initial)
            ix=overview._indices(np.array([[-radius,-radius],[radius,radius]]))
            self.assertTrue(((ix>=0)&(ix<32)).all())
        self.assertEqual(int(overview.count.sum()),8)


if __name__=="__main__":unittest.main()
