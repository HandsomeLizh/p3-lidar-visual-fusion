"""Public height-only exports omit semantic/occupancy fields and retain geometry."""
import io
import json
import sqlite3
import tempfile
import unittest
import zlib
from contextlib import closing
from pathlib import Path
import numpy as np
from t3_lidar_visual_fusion.disk_map import DiskElevationMap
from t3_lidar_visual_fusion.compact_delivery import CompactDelivery
from t3_lidar_visual_fusion.legacy.dense_grid_store import save_dense_global_grid_map, LiveDenseGlobalMapWriter


HEIGHT_LAYERS={'elevation','elevation_variance','height_range','roughness','observation_count'}
CLASSIFICATION={'occupancy','obstacle','traversability','semantic','semantic_id','semantic_confidence','color_bgr'}


class HeightOutputStorageTests(unittest.TestCase):
    def test_height_only_and_legacy_exports(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            grid=DiskElevationMap(root/'internal.db',tile_cells=8,max_tiles=2,cache_mib=1)
            try:
                grid.update_elevation_only(points_map=np.array([[.1,.1,.3],[.1,.1,.31],[.3,.1,.5]]))
                window=grid.extract_window(center_x=.201,center_y=.101,length_x=.4,length_y=.2)
                for height_only in [True,False]:
                    path=root/('height.db' if height_only else 'legacy.db')
                    delivery=CompactDelivery(path,height_only=height_only)
                    try:delivery.checkpoint(grid)
                    finally:delivery.close()
                    with closing(sqlite3.connect(path)) as db:
                        meta={k:json.loads(v) for k,v in db.execute('SELECT key,value FROM metadata')}
                        offset,blob=db.execute('SELECT elevation_offset_m,payload FROM tiles').fetchone()
                    with np.load(io.BytesIO(zlib.decompress(blob)),allow_pickle=False) as tile:
                        if height_only:
                            self.assertEqual(set(tile.files),HEIGHT_LAYERS)
                            self.assertEqual(set(meta['encoding']),HEIGHT_LAYERS)
                            self.assertEqual(meta['format'],'t3_compact_elevation_map')
                            self.assertNotIn('class_names',meta)
                        else:self.assertIn('occupancy',tile.files)
                        self.assertAlmostEqual(float(tile['elevation'][0,0])*.01+offset,.305,delta=.0051)
                    npz=root/('height.npz' if height_only else 'legacy.npz')
                    save_dense_global_grid_map(npz,window,frame_id='map',map_revision=1,timestamp_text='test',height_only=height_only)
                    with np.load(npz,allow_pickle=False) as saved:
                        self.assertTrue(HEIGHT_LAYERS<=set(saved.files))
                        if height_only:
                            self.assertFalse(CLASSIFICATION & set(saved.files))
                            self.assertEqual(str(saved['format']),'t3_dense_elevation_map')
                        else:self.assertIn('occupancy',saved.files)
                        np.testing.assert_equal(saved['elevation'],window.elevation)
                        np.testing.assert_equal(saved['elevation_variance'],window.elevation_variance)
                    with self.assertRaises(ValueError):CompactDelivery(path,height_only=not height_only)
            finally:grid.close()

    def test_live_writer_retains_height_only_mode(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);grid=DiskElevationMap(root/'map.db',tile_cells=8,max_tiles=1,cache_mib=1)
            grid.update_elevation_only(points_map=np.array([[.1,.1,0.]]))
            window=grid.extract_window(center_x=.101,center_y=.101,length_x=.2,length_y=.2)
            writer=LiveDenseGlobalMapWriter(root/'live.npz',frame_id='map',height_only=True)
            try:
                writer.enqueue(window,map_revision=grid.update_id,timestamp_text='test')
                writer.flush(grid.update_id,timeout_sec=5.)
                with np.load(root/'live.npz',allow_pickle=False) as saved:
                    self.assertFalse(CLASSIFICATION & set(saved.files))
                    self.assertEqual(str(saved['format']),'t3_dense_elevation_map')
            finally:writer.close();grid.close()


if __name__=='__main__':unittest.main()
