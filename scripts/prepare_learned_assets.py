#!/usr/bin/env python3
"""Prepare pinned source, private wheels and local checkpoints; never run inference."""
from pathlib import Path
import argparse, hashlib, json, subprocess, sys, urllib.request
ROOT=Path(__file__).resolve().parents[1]
SOURCES={
 "XFeat":("https://github.com/verlab/accelerated_features.git","e92685f57f8318b18725c5c8c0bd28c7fe188d9a"),
 "LightGlue":("https://github.com/cvg/LightGlue.git","eb42fee2d71449efb0aa5c10549752b5d75384d8")}
WEIGHTS={
 "superpoint_v1.pth":"https://github.com/cvg/LightGlue/releases/download/v0.1_arxiv/superpoint_v1.pth",
 "superpoint_lightglue_v0-1_arxiv.pth":"https://github.com/cvg/LightGlue/releases/download/v0.1_arxiv/superpoint_lightglue.pth",
 "aliked_lightglue_v0-1_arxiv.pth":"https://github.com/cvg/LightGlue/releases/download/v0.1_arxiv/aliked_lightglue.pth",
 "aliked-n16.pth":"https://raw.githubusercontent.com/Shiaoming/ALIKED/683d7c65197395c0b3f01ebe76e1084a27e73a65/models/aliked-n16.pth"}


def sha(path):
    digest=hashlib.sha256()
    with path.open("rb") as f:
        for data in iter(lambda:f.read(1024*1024),b""):digest.update(data)
    return digest.hexdigest()


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--check",action="store_true")
    args=parser.parse_args()
    manifest_path=ROOT/"docs/learned_assets.json"
    if args.check:
        manifest=json.loads(manifest_path.read_text())
        for rel,record in manifest["files"].items():
            path=ROOT/rel
            if not path.is_file() or sha(path)!=record["sha256"]:
                raise RuntimeError("Missing or changed learned asset: "+rel)
        print("ASSETS VERIFIED (file checks only; no model loading):",len(manifest["files"]))
        return
    previous=json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    for name,(url,commit) in SOURCES.items():
        dest=ROOT/"vendor"/name
        if not dest.exists():
            subprocess.run(["git","clone","--no-checkout",url,str(dest)],check=True)
            subprocess.run(["git","-C",str(dest),"checkout",commit],check=True)
        head=subprocess.check_output(["git","-C",str(dest),"rev-parse","HEAD"],text=True).strip()
        if head!=commit:raise RuntimeError("Unexpected vendor version: "+name)
    # Device selection is explicit; do not allocate a CUDA model for a CPU profile.
    path=ROOT/"vendor/XFeat/modules/xfeat.py";source=path.read_text()
    edits=[
      ("top_k = 4096, detection_threshold=0.05):","top_k = 4096, detection_threshold=0.05, device=None):"),
      ("torch.device('cuda' if torch.cuda.is_available() else 'cpu')",
       "torch.device(device if device is not None else ('cuda' if torch.cuda.is_available() else 'cpu'))"),
      ("torch.load(weights, map_location=self.dev)","torch.load(weights, map_location=self.dev, weights_only=True)")]
    for old,new in edits:
        if new in source:continue
        if source.count(old)!=1:raise RuntimeError("Upstream XFeat device patch needs review")
        source=source.replace(old,new,1)
    empty_patch={'old': '\t\tmkpts = self.NMS(K1h, threshold=detection_threshold, kernel_size=5)\n', 'replacement': "\t\tmkpts = self.NMS(K1h, threshold=detection_threshold, kernel_size=5)\n\t\tif mkpts.shape[1] == 0:\n\t\t\treturn [{'keypoints': M1.new_empty((0, 2)),\n\t\t\t\t\t 'descriptors': M1.new_empty((0, 64)),\n\t\t\t\t\t 'scores': M1.new_empty((0,))} for _ in range(B)]\n"}
    if empty_patch["replacement"] not in source:
        if source.count(empty_patch["old"])!=1:raise RuntimeError("Upstream empty-feature patch needs review")
        source=source.replace(empty_patch["old"],empty_patch["replacement"],1)
    path.write_text(source)
    wheel_dir=ROOT/"deps/learned-wheels";wheel_dir.mkdir(parents=True,exist_ok=True)
    subprocess.run([sys.executable,"-m","pip","download","--only-binary=:all:","--no-deps",
        "--dest",str(wheel_dir),"kornia==0.7.2","kornia_rs==0.1.9"],check=True)
    target=ROOT/"deps/learned-python"
    if not (target/"kornia").exists() or not (target/"kornia_rs").exists():
        subprocess.run([sys.executable,"-m","pip","install","--no-deps","--no-compile","--target",str(target),
            *map(str,sorted(wheel_dir.glob("*.whl")))],check=True)
    cache=ROOT/"deps/learned-checkpoints";cache.mkdir(exist_ok=True)
    records={}
    for name,url in WEIGHTS.items():
        path=cache/name
        if not path.exists():
            temp=path.with_suffix(".download")
            print("Downloading",name,flush=True)
            subprocess.run(["curl","--fail","--location","--silent","--show-error",
                "--retry","2","--retry-delay","1","--connect-timeout","15","--max-time","120",
                "--output",str(temp),url+"?download=1"],check=True)
            temp.replace(path)
        rel=str(path.relative_to(ROOT))
        digest=sha(path)
        if rel in previous.get("files",{}) and previous["files"][rel]["sha256"]!=digest:
            raise RuntimeError("Existing checkpoint changed: "+rel)
        records[rel]=dict(sha256=digest,bytes=path.stat().st_size,source=url)
    for path in list((ROOT/"vendor/XFeat/weights").glob("*.pt"))+list(wheel_dir.glob("*.whl")):
        records[str(path.relative_to(ROOT))]=dict(sha256=sha(path),bytes=path.stat().st_size)
    manifest=dict(sources={name:dict(url=v[0],commit=v[1]) for name,v in SOURCES.items()},
        files=records,runtime_models_loaded=False,inference_executed=False,
        existing_torch_requirement="Reuse system/user torch; never pip-upgrade torch or numpy here")
    manifest_path.write_text(json.dumps(manifest,indent=2))
    patch=subprocess.check_output(["git","-C",str(ROOT/"vendor/XFeat"),"diff"],text=True)
    (ROOT/"docs/XFeat.patch").write_text(patch)
    print("Prepared",len(records),"assets; no inference executed",flush=True)


if __name__=="__main__":main()
