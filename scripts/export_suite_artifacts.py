#!/usr/bin/env python3
"""Stream map exports and prepare one bounded, current-revision saved display."""
import argparse
import csv
from io import BytesIO
import json
from pathlib import Path
import sqlite3
import subprocess
import sys
import time
import numpy as np
from control import alive

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser();parser.add_argument('suite',type=Path)
    parser.add_argument('--saved-case',default='repair_clean');args=parser.parse_args()
    suite=args.suite.resolve()
    if not suite.is_relative_to(ROOT/'results'):raise ValueError('Foreign suite')
    state=json.loads((ROOT/'run_state.json').read_text())
    if any(alive(p) for p in state['processes'].values()):raise RuntimeError('Finish timed runs before exporting')
    results=json.loads((suite/'suite_complete.json').read_text())
    exports=[]
    for case in results['cases']:
        output=Path(case['output'])
        if not case['passed']:continue
        existing=output/'offline_export_verification.json'
        if existing.exists():
            exports.append(json.loads(existing.read_text()));continue
        start=time.monotonic()
        with (output/'offline_export.log').open('w') as stream:
            subprocess.run([sys.executable,str(ROOT/'scripts/export_map.py'),str(output)],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,check=True,timeout=180)
        metadata=json.loads((output/'map_metadata.json').read_text());pcd=output/'lidar_map.pcd'
        with pcd.open('rb') as stream:
            count=None
            while True:
                line=stream.readline()
                if not line:raise ValueError('Invalid PCD')
                if line.startswith(b'POINTS '):count=int(line.split()[1])
                if line==b'DATA binary\n':offset=stream.tell();break
        assert count==json.loads((output/'map_statistics.json').read_text())['map_points']
        assert pcd.stat().st_size==offset+12*count
        dense_revision=None
        if (output/'global_grid_map.npz').exists():
            with np.load(output/'global_grid_map.npz',allow_pickle=False) as data:
                if 'map_revision' in data:dense_revision=int(data['map_revision'])
        report=dict(output=str(output),map_revision=metadata['revision'],pcd_points=count,
                    pcd_size_verified=True,pcd_bytes=pcd.stat().st_size,
                    dense_npz_revision=dense_revision,dense_npz_is_latest=dense_revision==metadata['revision'],
                    authoritative_elevation='global_grid_map.sqlite3 and _internal/grid_map_snapshot/map_index.yaml',
                    export_wall_sec=time.monotonic()-start)
        existing.write_text(json.dumps(report,indent=2));exports.append(report);print('EXPORTED',case['name'],flush=True)
    source=suite/args.saved_case
    if source.parent!=suite or not (source/'benchmark_complete.json').exists():raise ValueError('Saved view requires a passed case')
    output=source/'verified_view'
    if not output.exists():
        metadata=json.loads((source/'map_metadata.json').read_text());profile=metadata['profile']
        size,resolution=profile['tile_cells'],profile['map_resolution']
        database=source/metadata['elevation_database']
        with sqlite3.connect('file:'+str(database)+'?mode=ro',uri=True) as db:
            db.execute('PRAGMA query_only=ON');db.execute('PRAGMA cache_size=-2048')
            x0,x1,y0,y1=db.execute('SELECT MIN(x),MAX(x),MIN(y),MAX(y) FROM tiles').fetchone()
            nx,ny=(x1-x0+1)*size,(y1-y0+1)*size
            if nx*ny>2000000:raise ValueError('Saved display exceeds the explicit 2 million cell cap')
            elevation=np.full((ny,nx),np.nan,np.float32)
            for x,y,blob in db.execute('SELECT x,y,payload FROM tiles'):
                with np.load(BytesIO(blob),allow_pickle=False) as tile:
                    values=tile['elevation_mean'].astype(np.float32);values[tile['elevation_count']==0]=np.nan
                    ix,iy=(x-x0)*size,(y-y0)*size;elevation[iy:iy+size,ix:ix+size]=values
        output.mkdir()
        np.savez_compressed(output/'global_grid_map.npz',elevation=elevation,resolution=resolution,
                            origin_x=x0*size*resolution,origin_y=y0*size*resolution,map_revision=metadata['revision'])
        for name in ['trajectory_map.tum','lidar_map.pcd']:(output/name).symlink_to(source/name)
        with (output/'pose_processing_time.csv').open('w',newline='') as stream:
            writer=csv.writer(stream);writer.writerow(['stamp_sec','processing_time_sec'])
            for line in (source/'learned_metrics.jsonl').read_text().splitlines():
                row=json.loads(line)
                if 'processing_sec' in row:writer.writerow([row['sensor_stamp_sec'],row['processing_sec']])
        (output/'view_metadata.json').write_text(json.dumps(dict(source=str(source),static_snapshot=True,
            ros_domain_id=60,map_revision=metadata['revision'],full_elevation_cells=nx*ny,
            elevation_array_mib=elevation.nbytes/2**20,displayed_timing='Recorded visual processing time; saved snapshot, not live odometry'),indent=2))
    (suite/'artifact_verification.json').write_text(json.dumps(dict(exports=exports,
        saved_view=json.loads((output/'view_metadata.json').read_text())),indent=2))
    print('ARTIFACTS VERIFIED',flush=True)


if __name__=='__main__':main()
