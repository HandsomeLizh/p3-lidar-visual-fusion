from pathlib import Path
import json,time,subprocess
r=Path("/home/yanfa/P3/lidar_visual_fusion")
baseline=r/"results/xfeat_only_long_80m_20260915"
deadline=time.monotonic()+500
while not (baseline/"benchmark_complete.json").exists():
 if (baseline/"benchmark_error.json").exists():raise RuntimeError("Raw visual baseline failed to finish")
 if time.monotonic()>deadline:raise RuntimeError("Baseline wait timeout")
 time.sleep(2)
assert json.loads((r/"results/stereo_transport_equivalence_20260915.json").read_text())["passed"]
with (r/"logs/compact_visual_comparison_20260915.log").open("w") as f:
 p=subprocess.Popen(["/usr/bin/python3",str(r/"scripts/run_compact_visual_comparison.py")],stdout=f,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
(r/"logs/compact_visual_supervisor.json").write_text(json.dumps({"pid":p.pid}))
print("Compact visual comparison started",p.pid,flush=True)
