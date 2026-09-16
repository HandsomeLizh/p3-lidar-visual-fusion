"""Small /proc samples; no historical in-memory telemetry buffer."""
import shutil
from pathlib import Path


def memory_sample(directory):
    status = {}
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith(("VmRSS:", "VmHWM:")):
            key, value, _ = line.split()
            status[key[:-1].lower() + "_mib"] = int(value) / 1024.
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            status["system_available_mib"] = int(line.split()[1]) / 1024.
            break
    status["disk_free_gib"] = shutil.disk_usage(directory).free / 2**30
    return status
