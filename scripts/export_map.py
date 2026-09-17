#!/usr/bin/env python3
"""Stream complete map exports after a run has stopped."""
import argparse,json,time,sqlite3,shutil
from pathlib import Path
from control import STATE,alive
from t3_lidar_visual_fusion.disk_map import DiskElevationMap
from t3_lidar_visual_fusion.legacy.voxel_cloud_store import VoxelCloudStore
from t3_lidar_visual_fusion.legacy.dense_grid_store import save_dense_global_grid_map


def main():
    a=argparse.ArgumentParser();a.add_argument("run_directory");x=a.parse_args()
    out=Path(x.run_directory).resolve()
    if STATE.exists():
        state=json.loads(STATE.read_text())
        if Path(state.get("output","")).resolve()==out and any(alive(p) for p in state["processes"].values()):
            raise RuntimeError("Stop this run before offline export to avoid a long WAL snapshot during live mapping")
    meta=json.loads((out/"map_metadata.json").read_text());p=meta["profile"]
    with sqlite3.connect("file:"+str(out/meta["cloud_database"])+"?mode=ro",uri=True) as db:
        points=db.execute("SELECT COUNT(*) FROM voxels").fetchone()[0]
    if meta.get('stereo_fill_database'):
        with sqlite3.connect("file:"+str(out/meta['stereo_fill_database'])+"?mode=ro",uri=True) as db:
            points+=db.execute('SELECT COUNT(*) FROM voxels').fetchone()[0]
    with sqlite3.connect("file:"+str(out/meta["elevation_database"])+"?mode=ro",uri=True) as db:
        tile_bytes=db.execute("SELECT COALESCE(SUM(length(payload)),0) FROM tiles").fetchone()[0]
    required=points*12+tile_bytes+128*2**20+p.get("min_disk_free_gib",5.)*2**30
    if shutil.disk_usage(out).free<required:
        raise RuntimeError("Insufficient free disk for complete temporary export plus configured reserve")
    grid=DiskElevationMap(out/meta["elevation_database"],resolution=p["map_resolution"],
        tile_cells=p.get("tile_cells",128),max_tiles=8,cache_mib=32,elevation_fusion=p.get('elevation_fusion'))
    try:
        result=grid.save_snapshot(out/"_internal/grid_map_snapshot",
            timestamp_text=time.strftime("%Y%m%d_%H%M%S"),frame_id="map")
        print("Lossless snapshot:",result["index"],flush=True)
        n=VoxelCloudStore.export_pcd(out/meta["cloud_database"],out/"lidar_map.pcd")
        print("PCD:",n,"points",flush=True)
        if meta.get('stereo_fill_database'):
            n=VoxelCloudStore.export_pcd(out/meta['stereo_fill_database'],out/'stereo_fill.pcd')
            print('Stereo fill PCD:',n,'measured points',flush=True)
        try:
            m=grid.extract_global(p.get("global_max_cells",1000000))
            if m is not None:
                save_dense_global_grid_map(out/"global_grid_map.npz",m,frame_id="map",
                    height_only=p.get('map_output')=='elevation_only',
                    map_revision=grid.update_id,timestamp_text=time.strftime("%Y%m%d_%H%M%S"))
                print("Dense global map exported",flush=True)
        except ValueError as e:print(str(e)+"; complete data remains in SQLite and tiled snapshot",flush=True)
    finally:grid.close()


if __name__=="__main__":main()
