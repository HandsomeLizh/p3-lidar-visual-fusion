"""Source replacement removes real XYZ; no fabricated grid-center cloud points."""
import tempfile,unittest
from pathlib import Path
import numpy as np
from t3_lidar_visual_fusion.stereo_fill import StereoFillStore
from t3_lidar_visual_fusion.disk_map import DiskElevationMap
from t3_lidar_visual_fusion.mapping_view import MappingCameraView,spatial_density_mask
from t3_lidar_visual_fusion.legacy.voxel_cloud_store import VoxelCloudStore
from test_stereo_geometry import profile


class FillTests(unittest.TestCase):
    def test_actual_xyz_persistence_cap_and_later_range_replacement(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            grid=DiskElevationMap(root/'grid.sqlite',tile_cells=8)
            store=StereoFillStore(root/'points.sqlite',.2,2)
            observed=np.array([[.023,.041,.30],[.25,-.17,.20],[.46,.23,.15]])
            evidence=np.c_[(np.floor(observed[:,:2]/.2)+.5)*.2,observed[:,2],np.full(3,.0025),np.full(3,2)]
            accepted=grid.update_stereo(evidence)
            store.update(accepted,observed,[.001,.001,.001])
            self.assertEqual(store.count,3);self.assertEqual(len(store.preview()),2)
            saved=np.array(store.db.execute('SELECT x,y,z FROM points').fetchall())
            np.testing.assert_allclose(saved,observed)
            # A new range surface replaces the stereo height and actual point.
            replaced=grid.update_elevation_only(points_map=[[.027,.046,.07]])
            self.assertEqual(store.remove(replaced),1)
            self.assertEqual(store.count,2)
            self.assertEqual(len(grid.update_stereo(evidence[:1])),0)
            store.close();store=StereoFillStore(root/'points.sqlite',.2,2)
            self.assertEqual(store.count,2);self.assertEqual(len(store.preview()),2)
            self.assertFalse((np.linalg.norm(store.preview()-observed[0],axis=1)<1e-5).any())
            self.assertEqual(VoxelCloudStore.export_pcd(root/'points.sqlite',root/'fill.pcd'),2)
            payload=(root/'fill.pcd').read_bytes().split(b'DATA binary\n',1)[1]
            np.testing.assert_allclose(np.frombuffer(payload,dtype='<f4').reshape(-1,3),observed[1:],atol=1e-6)
            store.close();grid.close()

    def test_old_or_uncertain_range_cannot_remove_qualified_fill(self):
        with tempfile.TemporaryDirectory() as directory:
            grid=DiskElevationMap(Path(directory)/'grid.sqlite',tile_cells=8,elevation_fusion={'enabled':True})
            grid.update_stereo([[.1,.1,.2,.0025,2]],stamp=10.)
            for stamp,cov in [(9.,np.eye(6)*1e-6),(11.,np.eye(6))]:
                replaced=grid.update_elevation_only(points_map=[[.1,.1,.1]],stamp=stamp,covariance=cov,pose_origin=np.zeros(3))
                self.assertEqual(len(replaced),0)
                self.assertTrue(grid.tiles.get((0,0)).stereo_owned[0,0])
            grid.close()

    def test_feather_is_deterministic_and_preserves_columns(self):
        p=np.array([[i*.2+.01,-.19,z] for i in range(-100,101) for z in [0.,1.]])
        weight=np.full(len(p),.5)
        a=spatial_density_mask(p,weight,.2);b=spatial_density_mask(p,weight,.2)
        np.testing.assert_array_equal(a,b);np.testing.assert_array_equal(a[::2],a[1::2])
        self.assertTrue(a.any());self.assertFalse(a.all())
        self.assertFalse(spatial_density_mask(p,np.zeros(len(p)),.2).any())
        self.assertTrue(spatial_density_mask(p,np.ones(len(p)),.2).all())

    def test_fov_transforms_cotimed_world_points_back_to_base(self):
        cfg=profile();cfg['map_resolution']=.2
        cfg['mapping_camera_view']={'full_density_range_m':10.,'range_feather_m':5.,'image_feather_fraction':.15}
        view=MappingCameraView(cfg);g=view.geometry
        rect=np.array([[0.,0.,3.],[100.,0.,3.],[0.,0.,-1.]])
        base=rect@g.base_from_rect[:3,:3].T+g.base_from_rect[:3,3]
        w=view.weights(base);self.assertEqual(w[0],1.);self.assertEqual(w[1],0.);self.assertEqual(w[2],0.)
        pose=np.eye(4);pose[:3,3]=[100.,-37.,2.]
        np.testing.assert_array_equal(view.select(base+pose[:3,3],pose),[True,False,False])
        far=base[:1].copy();far[:,0]+=50
        self.assertEqual(view.weights(far)[0],0.)


if __name__=='__main__':unittest.main()
