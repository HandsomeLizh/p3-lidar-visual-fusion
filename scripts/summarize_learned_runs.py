#!/usr/bin/env python3
"""Summarize stored run snapshots without launching ROS, importing models or reading maps."""
import argparse,json
from pathlib import Path


def main():
    a=argparse.ArgumentParser();a.add_argument("runs",nargs="+");a.add_argument("--output",required=True)
    args=a.parse_args();rows=[]
    for name in args.runs:
        root=Path(name)
        path=root/"learned_metrics.json"
        if not path.exists():
            rows.append(dict(run=str(root),status="no learned metrics; inspect LiDAR-only fallback and logs"))
            continue
        data=json.loads(path.read_text())
        fusion_file=root/"fusion_status.json"
        fusion=json.loads(fusion_file.read_text()) if fusion_file.exists() else {}
        resource_file=root/"resources_latest.json"
        resources=json.loads(resource_file.read_text()) if resource_file.exists() else {}
        rows.append(dict(run=str(root),backend=data.get("resources",{}).get("backend"),
            synchronized=data.get("synchronized",0),started=data.get("started",0),
            tracked=data.get("tracked",0),anchors=data.get("anchors",0),
            rejected=data.get("rejected",0),replaced_pending=data.get("replaced_pending",0),
            recent_timing=data.get("recent_timing",{}),timing_samples=data.get("timing_samples",0),
            model_resources=data.get("resources",{}),latest_tree_rss_mib=resources.get("tree_rss_mib"),
            peak_tree_rss_mib=resources.get("tree_rss_peak_mib"),
            fusion_mode=fusion.get("operating_mode"),fusion_visual_accepted=fusion.get("visual_accepted"),
            fusion_output_age_p95_sec=fusion.get("filtered_output_age_p95_sec"),
            note="Timing quantiles use the last <=256 processed samples; static model bytes are not total runtime memory"))
    output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(rows,indent=2))
    print(json.dumps(rows,indent=2))


if __name__=="__main__":main()
