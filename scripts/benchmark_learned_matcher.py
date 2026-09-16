#!/usr/bin/env python3
"""Deferred static-pair inference check; this does not measure full odometry or accuracy."""
import argparse,json,sys,time
from pathlib import Path
import cv2
import numpy as np
import yaml
ROOT=Path(__file__).resolve().parents[1]


def main():
    a=argparse.ArgumentParser(description=__doc__)
    a.add_argument("--left",required=True);a.add_argument("--right",required=True)
    a.add_argument("--backend",default="xfeat_lighterglue",
        choices=["xfeat_lighterglue","xfeat_mnn","superpoint_lightglue","aliked_lightglue"])
    a.add_argument("--device",choices=["cpu","cuda"],default="cuda")
    a.add_argument("--repetitions",type=int,default=20)
    a.add_argument("--output",required=True)
    args=a.parse_args()
    if not 1<=args.repetitions<=200:raise ValueError("Repetitions must be 1..200")
    from t3_lidar_visual_fusion.learned_matching import LearnedMatcher
    cfg=yaml.safe_load((ROOT/"config/simulation_xfeat.yaml").read_text())["learned_visual"]
    cfg.update(backend=args.backend,device=args.device)
    images=[]
    for name in [args.left,args.right]:
        image=cv2.imread(name,cv2.IMREAD_GRAYSCALE)
        if image is None:raise ValueError("Cannot read image: "+name)
        scale=min(1.,cfg["max_image_side"]/max(image.shape))
        image=cv2.resize(image,(max(64,round(image.shape[1]*scale)),max(64,round(image.shape[0]*scale))))
        images.append(image)
    model=LearnedMatcher(ROOT,cfg)
    rows=[]
    for i in range(args.repetitions+2):
        start=time.perf_counter()
        first=model.extract(images[0]);second=model.extract(images[1])
        extracted=time.perf_counter()
        matches=model.match(first,second)
        finished=time.perf_counter()
        if i>=2:rows.append(dict(extract_sec=extracted-start,match_sec=finished-extracted,
            total_sec=finished-start,matches=len(matches),left_features=len(first["pixels"]),
            right_features=len(second["pixels"])))
    result=dict(backend=args.backend,device=args.device,repetitions=args.repetitions,
        scope="Repeated static pair only; not full odometry, live throughput or robustness",
        timings={key:dict(zip(["p50","p95","p99"],map(float,np.percentile([r[key] for r in rows],[50,95,99]))))
            for key in ["extract_sec","match_sec","total_sec"]},
        samples=rows,resources=model.resource_snapshot(),image_shapes=[list(i.shape) for i in images])
    path=Path(args.output);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(result,indent=2))
    print(json.dumps(result["timings"],indent=2))


if __name__=="__main__":main()
