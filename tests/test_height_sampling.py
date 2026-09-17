"""The faster sampling path must preserve per-cell height evidence and its cap."""
import unittest
import numpy as np
from t3_lidar_visual_fusion.terrain_mapper import height_sample_indices


class HeightSamplingTests(unittest.TestCase):
    def test_cells_keep_min_median_max_including_single_and_double_hits(self):
        rng=np.random.default_rng(34)
        points=[];expected=[]
        for x,count in zip(range(-6,7),[1,2,3,4,10,50,1,2,30,3,12,1,6]):
            z=rng.uniform(-2,4,count)
            cell=np.column_stack((np.full(count,x*.2+.03),np.full(count,.07),z))
            points.extend(cell)
            expected.extend(cell[np.argsort(z)[np.unique([0,count//2,count-1])]])
        points=np.array(points);rng.shuffle(points)
        actual=points[height_sample_indices(points,.2,1000)]
        np.testing.assert_allclose(actual,expected)
        limited=points[height_sample_indices(points,.2,7)]
        np.testing.assert_allclose(limited,actual[np.linspace(0,len(actual)-1,7,dtype=int)])

    def test_empty_and_flat_cells(self):
        self.assertEqual(len(height_sample_indices(np.empty((0,3)),.2,10)),0)
        points=np.array([[1.01,1.01,0.],[1.02,1.02,0.],[1.03,1.03,0.]])
        chosen=height_sample_indices(points,.2,10)
        self.assertEqual(len(np.unique(chosen)),3)
        self.assertTrue(np.all(points[chosen,2]==0.))


if __name__=='__main__':unittest.main()
