"""Equivalence against the retained scalar robust fusion and voxel ordering."""
import numpy as np
import pytest
import sqlite3
from t3_lidar_visual_fusion.array_groups import unique_row_indices
from t3_lidar_visual_fusion.legacy.semantic_grid import LayeredSemanticGridMap
from t3_lidar_visual_fusion.bounded_cloud import BoundedCloudStore
from t3_lidar_visual_fusion.overview import Overview


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


def test_cached_voxels_keep_first_point_bound_memory_and_allow_reinsertion(tmp_path):
    store=BoundedCloudStore(tmp_path/'voxels.db',.1,preview_points=200,preview_voxel=.1)
    store.recent_key_limit=64
    expected={};rng=np.random.default_rng(91)
    try:
        for scan in range(6):
            keys=rng.integers(-20,20,(300,3));keys=np.r_[keys,[[-10000,0,0],[10000,0,0]]]
            points=(keys+.1+rng.random(keys.shape)*.1)*.1
            points=np.r_[points,points[::2],[[np.nan,0,0]]]
            store.append(points)
            for point in points[np.isfinite(points).all(axis=1)]:
                expected.setdefault(tuple(np.floor(point/.1).astype(int)),tuple(point))
            before=store.count
            store.append(points)
            assert store.count==before and store.last_cached_voxels==64
            assert store.recent_keys.nbytes<=64*3*8
        rows=store.connection.execute('SELECT ix,iy,iz,x,y,z FROM voxels').fetchall()
        assert {tuple(r[:3]):tuple(r[3:]) for r in rows}==expected
        assert store.count==len(expected)
        point=np.array([[123.41,23.41,3.41]])
        store.append(point);store.append(point);assert store.last_cached_voxels==1
        row=store.connection.execute('SELECT rowid,ix,iy,iz,x,y,z FROM voxels WHERE ix=1234').fetchone()
        store.remove_rows([row]);assert store.count==len(expected)
        assert store.append(point)==1
        assert store.connection.execute('SELECT COUNT(*) FROM voxels WHERE ix=1234').fetchone()[0]==1
    finally:store.close()


def test_failed_voxel_transaction_does_not_cache_uncommitted_points(tmp_path):
    store=BoundedCloudStore(tmp_path/'failure.db',1.,preview_points=100)
    try:
        store.append([[0.,0.,0.]])
        store.connection.execute("CREATE TRIGGER fail_insert BEFORE INSERT ON voxels WHEN NEW.ix=2 BEGIN SELECT RAISE(ABORT,'fixture'); END")
        with pytest.raises(sqlite3.IntegrityError):store.append([[1.,0.,0.],[2.,0.,0.]])
        assert store.count==1
        np.testing.assert_array_equal(store.recent_keys,[[0,0,0]])
        store.connection.execute('DROP TRIGGER fail_insert')
        assert store.append([[1.,0.,0.],[2.,0.,0.]])==2
        assert store.count==3
    finally:store.close()


def test_batched_overview_matches_pointwise_min_max_count_and_saturation():
    overview=Overview(128,1.)
    overview.ready=True;overview.origin=np.array([-64.,-64.])
    minimum=overview.minimum.copy();maximum=overview.maximum.copy();count=overview.count.copy()
    rng=np.random.default_rng(83)
    for scan in range(5):
        points=np.c_[rng.uniform(-30,30,(40000,2)),rng.normal(0,3,40000)]
        points=np.r_[points,points[::3],[[np.nan,0,0],[0,0,np.inf]]]
        if scan==3:count[64,64]=overview.count[64,64]=2**31+1
        finite=points[np.isfinite(points).all(axis=1)];ix=overview._indices(finite[:,:2])
        addr=ix[:,1],ix[:,0]
        np.minimum.at(minimum,addr,finite[:,2]);np.maximum.at(maximum,addr,finite[:,2])
        if count.max()>2**31:count//=2
        np.add.at(count,addr,1)
        overview.update(points)
        np.testing.assert_array_equal(overview.minimum,minimum)
        np.testing.assert_array_equal(overview.maximum,maximum)
        np.testing.assert_array_equal(overview.count,count)
