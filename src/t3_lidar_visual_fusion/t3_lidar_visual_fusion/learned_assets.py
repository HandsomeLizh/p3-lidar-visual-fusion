"""Offline asset integrity checks. This module does not import torch or ROS."""
from pathlib import Path
import hashlib,json

BACKENDS=("xfeat_lighterglue","xfeat_mnn","superpoint_lightglue","aliked_lightglue")


def required_weights(backend):
    if backend not in BACKENDS:raise ValueError("Unknown learned backend: "+backend)
    if backend.startswith("xfeat"):
        paths=["vendor/XFeat/weights/xfeat.pt"]
        if backend=="xfeat_lighterglue":paths+=["vendor/XFeat/weights/xfeat-lighterglue.pt"]
        return paths
    name=backend.split("_")[0]
    return ["deps/learned-checkpoints/"+("superpoint_v1.pth" if name=="superpoint" else "aliked-n16.pth"),
            "deps/learned-checkpoints/"+name+"_lightglue_v0-1_arxiv.pth"]


def verify_assets(root,backend):
    root=Path(root)
    manifest=json.loads((root/"docs/learned_assets.json").read_text())
    for rel in required_weights(backend):
        path=root/rel
        record=manifest["files"].get(rel)
        if record is None or not path.is_file():raise RuntimeError("Missing prepared model asset: "+rel)
        digest=hashlib.sha256()
        with path.open("rb") as f:
            for block in iter(lambda:f.read(1024*1024),b""):digest.update(block)
        if digest.hexdigest()!=record["sha256"]:raise RuntimeError("Model asset checksum mismatch: "+rel)
    return manifest
