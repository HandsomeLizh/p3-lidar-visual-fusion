"""Regression for historical ground becoming black after height drift."""
import io
import json
import sqlite3
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
import numpy as np
from t3_lidar_visual_fusion.disk_map import DiskElevationMap
from t3_lidar_visual_fusion.bounded_cloud import BoundedCloudStore
from t3_lidar_visual_fusion.terrain_mapper import TerrainMapper
from t3_lidar_visual_fusion.legacy.semantic_grid import LayeredSemanticGridMap


def occupancy(window):
    return TerrainMapper.terrain_layers(SimpleNamespace(cfg={'map_resolution': .2}), window)[1]


class SurfaceReliefTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.grid = DiskElevationMap(self.root/'height.db', resolution=.2,
                                     tile_cells=16, max_tiles=8, cache_mib=8)

    def tearDown(self):
        self.grid.close()
        self.tmp.cleanup()

    def window(self):
        return self.grid.extract_window(center_x=0., center_y=0., length_x=4., length_y=4.)

    def test_common_height_drift_does_not_turn_ground_into_obstacles(self):
        xx, yy = np.meshgrid(np.arange(-1.99, 2., .2), np.arange(-1.99, 2., .2))
        ground = np.column_stack([xx.ravel(), yy.ravel(), np.zeros(xx.size)])
        rock = np.array([[.01, .01, .8], [.21, .01, .8], [.01, .21, .8], [.21, .21, .8]])
        old = LayeredSemanticGridMap(resolution=.2, length_x=4., length_y=4., origin_x=-2., origin_y=-2.)
        for dz in np.linspace(0., .36, 25):
            scan = np.r_[ground, rock] + [0., 0., dz]
            self.grid.update_elevation_only(points_map=scan)
            old.update_elevation_only(points_map=scan)
        window = self.window()
        interior = np.ones(window.elevation.shape, bool)
        interior[:2, :] = interior[-2:, :] = False
        interior[:, :2] = interior[:, -2:] = False
        interior[8:14, 8:14] = False
        legacy = SimpleNamespace(elevation=old.elevation_layer(),
                                 height_range=old.height_range_layer(), all_layers=lambda: {})
        self.assertTrue(np.all(occupancy(legacy)[interior] >= 65), 'Fixture must reproduce the old black-ground failure')
        self.assertTrue(np.all(occupancy(window)[interior] == 0))
        self.assertTrue(np.all(np.isfinite(window.elevation)))
        self.assertTrue(np.all(window.elevation_variance[interior] > .005), 'Temporal uncertainty must remain visible')
        self.assertTrue(np.all(occupancy(window)[10:12, 10:12] == 100), 'Real rocks must remain blocked')
        # Looking only at the lower face must not erase previously observed relief.
        self.grid.update_elevation_only(points_map=ground + [0., 0., .36])
        self.assertTrue(np.all(occupancy(self.window())[10:12, 10:12] == 100))

    def test_history_survives_driving_out_of_window_eviction_and_reopen(self):
        self.grid.tiles.max_tiles = 1
        start = np.array([[-.19, -.19, 0.], [-.09, -.09, .01], [-.19, -.19, .5]])
        self.grid.update_elevation_only(points_map=start)
        before = self.window()
        for x in range(10, 110, 10):
            self.grid.update_elevation_only(points_map=np.array([[float(x)+.01, .01, .1]]))
            self.grid.extract_window(center_x=float(x), center_y=0., length_x=4., length_y=4.)
        self.assertGreater(self.grid.tiles.evictions, 0)
        np.testing.assert_array_equal(self.window().elevation, before.elevation)
        np.testing.assert_array_equal(self.window().height_range, before.height_range)
        self.grid.close()
        self.grid = DiskElevationMap(self.root/'height.db', resolution=.2, tile_cells=16, max_tiles=1, cache_mib=8)
        np.testing.assert_array_equal(self.window().height_range, before.height_range)
        self.assertLessEqual(self.grid.tiles.nbytes, self.grid.tiles.tile_bytes)

    def test_cloud_rebuild_does_not_reintroduce_temporal_spread(self):
        cloud = BoundedCloudStore(self.root/'cloud.db', .05, 1000, .1)
        try:
            for z in [0., .1, .2, .3]:
                scan = np.array([[.05, .05, z], [.15, .15, z+.001]])
                cloud.append(scan)
                self.grid.update_elevation_only(points_map=scan)
            self.grid.rebuild_cells([[0, 0]], cloud)
            w = self.grid.extract_window(center_x=.1, center_y=.1, length_x=.2, length_y=.2)
            self.assertLess(float(w.height_range[0, 0]), .002)
            self.assertGreater(float(w.elevation_variance[0, 0]), .01)
        finally:
            cloud.close()

    def test_legacy_tiles_keep_conservative_evidence_on_load(self):
        self.grid.update_elevation_only(points_map=np.array([[.01, .01, 0.], [.01, .01, .6]]))
        self.grid.close()
        db = sqlite3.connect(self.root/'height.db')
        blob = db.execute('SELECT payload FROM tiles WHERE x=0 AND y=0').fetchone()[0]
        with np.load(io.BytesIO(blob)) as data:
            payload = io.BytesIO()
            np.savez_compressed(payload, **{k: data[k] for k in data.files if k != 'surface_height_range'})
        with db:
            db.execute('UPDATE tiles SET payload=? WHERE x=0 AND y=0', (payload.getvalue(),))
        db.close()
        self.grid = DiskElevationMap(self.root/'height.db', resolution=.2, tile_cells=16, max_tiles=8, cache_mib=8)
        self.assertAlmostEqual(float(self.window().height_range[10, 10]), .6, places=6)


if __name__ == '__main__':
    unittest.main()
