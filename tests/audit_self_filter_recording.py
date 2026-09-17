"""Read recorded normalized LiDAR only; never starts drivers or ROS publishers."""
import argparse,json,sqlite3,time
from pathlib import Path
import numpy as np,yaml
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import PointCloud2
from t3_lidar_visual_fusion.ros_utils import cloud_arrays
from t3_lidar_visual_fusion.self_filter import SelfFilter


def main():
    p=argparse.ArgumentParser();p.add_argument('--bag',type=Path,required=True)
    p.add_argument('--profile',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();cfg=yaml.safe_load(a.profile.read_text());f=SelfFilter(**cfg['self_filter'])
    t=np.array(cfg['base_from_lidar']);frames=[];extrema=[]
    for part in sorted(a.bag.glob('*.db3')):
        with sqlite3.connect('file:'+str(part)+'?mode=ro',uri=True) as db:
            for blob, in db.execute("SELECT m.data FROM messages m JOIN topics t ON m.topic_id=t.id WHERE t.name='/fusion/lidar' ORDER BY m.timestamp"):
                msg=deserialize_message(blob,PointCloud2);xyz=cloud_arrays(msg);r=np.linalg.norm(xyz,axis=1)
                valid=np.isfinite(xyz).all(axis=1)&(r>=cfg['min_range'])&(r<=cfg['max_range'])
                xyz=xyz[valid];base=xyz@t[:3,:3].T+t[:3,3]
                started=time.perf_counter();keep=f.keep_mask(base);elapsed=time.perf_counter()-started
                removed=base[~keep]
                if len(removed):extrema.extend([removed.min(axis=0),removed.max(axis=0)])
                # Explicit checks against the 104 exclusion contract, separate
                # from the filter helper's implementation and frame conversion.
                external=(base[:,0]<-1.47)|(base[:,0]>.53)|(abs(base[:,1])>1.)|(base[:,2]<.13)|(base[:,2]>1.03)
                assert keep[external].all()
                assert not keep[~external].any()
                frames.append(dict(points=len(base),removed=int((~keep).sum()),filter_sec=elapsed,
                    below_box_retained=int((base[:,2]<.13).sum())))
    assert frames
    report=dict(passed=True,scans=len(frames),range_points=sum(x['points'] for x in frames),
        removed_points=sum(x['removed'] for x in frames),min_removed_per_scan=min(x['removed'] for x in frames),
        max_removed_per_scan=max(x['removed'] for x in frames),
        median_filter_ms=float(np.median([x['filter_sec'] for x in frames]))*1000,
        below_box_points_retained=sum(x['below_box_retained'] for x in frames),
        removed_bounds_m=[np.min(extrema,axis=0).tolist(),np.max(extrema,axis=0).tolist()] if extrema else None,
        profile_self_filter=cfg['self_filter'],all_points_outside_box_retained=True,frames=frames,
        scope='Recorded-point geometry check, no claim that every removed point is a verified body surface')
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k!='frames'}))


if __name__=='__main__':main()
