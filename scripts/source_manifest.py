#!/usr/bin/env python3
"""Verify both edited sources and the code ROS will actually load."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fingerprint():
    digest=hashlib.sha256()
    for folder in ["src/t3_lidar_visual_fusion","src/t3_voxelmap","src/t3_pv_lio","src/grid_map_msgs","src/t3_interfaces",
                   "vendor/VINS-Fusion","vendor/XFeat","vendor/LightGlue","visual"]:
        for p in sorted((ROOT/folder).rglob("*")):
            if not p.is_file() or any(s in p.parts for s in [".git","__pycache__","build","install"]):
                continue
            if p.suffix not in [".py",".cpp",".h",".hpp",".c",".xml",".txt",".msg",".srv",".yaml",".cfg"]:
                continue
            digest.update(str(p.relative_to(ROOT)).encode());digest.update(p.read_bytes())
    return digest.hexdigest()


def python_parity(source, runtime):
    expected = {str(p.relative_to(source)): sha256(p) for p in source.rglob("*.py")}
    actual = {str(p.relative_to(runtime)): sha256(p) for p in runtime.rglob("*.py")}
    changed = sorted(k for k in expected.keys() | actual.keys() if expected.get(k) != actual.get(k))
    if changed:
        raise RuntimeError("Installed Python differs from source: " + ", ".join(changed) + ". Run ./build.sh.")
    return actual


def runtime_fingerprint():
    spec = importlib.util.find_spec("t3_lidar_visual_fusion")
    if spec is None or not spec.origin:
        raise RuntimeError("Python package unavailable. Source scripts/env.sh and run ./build.sh.")
    runtime = Path(spec.origin).resolve().parent
    if not runtime.is_relative_to(ROOT):
        raise RuntimeError("Python package resolves outside this workspace: " + str(runtime))
    python_files = python_parity(ROOT / "src/t3_lidar_visual_fusion/t3_lidar_visual_fusion", runtime)
    binaries = ["install/t3_voxelmap/lib/t3_voxelmap/voxelmap_node", "install/vins/lib/vins/vins_node"]
    return {"python_root": str(runtime), "python_sha256": python_files,
            "binary_sha256": {rel: sha256(ROOT / rel) for rel in binaries}}


def check():
    path = ROOT / "build_manifest.json"
    if not path.exists():
        raise RuntimeError("Build manifest missing. Run ./build.sh.")
    manifest = json.loads(path.read_text())
    if manifest.get("source_sha256") != fingerprint():
        raise RuntimeError("Source differs from the last build. Run ./build.sh.")
    if manifest.get("runtime") != runtime_fingerprint():
        raise RuntimeError("Runtime differs from the verified build. Run ./build.sh.")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--record", action="store_true", help="Used by build.sh after successful colcon build")
    args = parser.parse_args()
    if args.record:
        record = {"source_sha256": fingerprint(), "runtime": runtime_fingerprint()}
        target = ROOT / "build_manifest.json"
        temporary = target.with_suffix(".tmp")
        temporary.write_text(json.dumps(record, indent=2))
        temporary.replace(target)
    else:
        check()
