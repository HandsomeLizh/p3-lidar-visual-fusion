"""Equivalence against the retained scalar robust fusion and voxel ordering."""
import numpy as np
import pytest
from t3_lidar_visual_fusion.array_groups import unique_row_indices
from t3_lidar_visual_fusion.legacy.semantic_grid import LayeredSemanticGridMap


def test_voxel_first_sample_order_including_large_negative_indices():
    rng=np.random.default_rng(731)
    rows=rng.integers(-90,90,(2000,3),dtype=np.int64)
    rows=np.r_[rows,rows[::2],[[2**58,-2**58,0],[-2**58,2**58,0]]]
    _,reference=np.unique(rows,axis=0,return_index=True)
    np.testing.assert_array_equal(unique_row_indices(rows),reference)
    assert len(unique_row_indices(np.empty((0,3),dtype=np.int64)))==0


@pytest.mark.parametrize('mad_scale',[0.,.2,3.5])
def test_batched_heights_match_scalar_over_outliers_and_repeat_observations(mad_scale):
    cfg=dict(resolution=.2,length_x=8.,length_y=8.,elevation_mad_scale=mad_scale)
    fast,scalar=LayeredSemanticGridMap(**cfg),LayeredSemanticGridMap(**cfg)
    rng=np.random.default_rng(682)
    for _ in range(6):
        cells=rng.choice(1600,350,replace=False);points=[]
        for i,cell in enumerate(cells):
            count=1+i%6
            xy=np.array([cell%40,cell//40])*.2-4.+.1
            heights=rng.normal(0,.05,count)+(rng.random(count)<.2)*3.
            points.extend(np.column_stack([np.tile(xy,(count,1)),heights]))
        points=np.asarray(points);points=np.r_[points,[[np.nan,0,0],[0,0,np.inf],[99,99,0]]]
        result=fast.update_elevation_only(points_map=points)
        finite=points[np.isfinite(points).all(axis=1)]
        rr,cc,valid=scalar.xy_to_indices(finite[:,0],finite[:,1]);finite=finite[valid];rr=rr[valid];cc=cc[valid]
        linear=rr*40+cc;accepted=0
        for cell in np.unique(linear):accepted+=scalar._update_elevation_cell(cell//40,cell%40,finite[linear==cell,2])
        assert result['accepted_points']==accepted
        for name in ('elevation_count','elevation_mean','elevation_M2','elevation_min','elevation_max'):
            np.testing.assert_allclose(getattr(fast,name),getattr(scalar,name),rtol=1e-6,atol=1e-7)
