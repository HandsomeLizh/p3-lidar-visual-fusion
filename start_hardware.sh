#!/usr/bin/env bash
# Sensor drivers run in domain 19. Keep the car separate from simulation domain 57.
set -eo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# The observed mapping/window peak is about 2.7 GiB, excluding sensor drivers.
# Leave startup headroom instead of allocating CUDA while the Jetson is swapping.
available_kib="$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)"
if [[ ! "$available_kib" =~ ^[0-9]+$ ]] || (( available_kib < 4 * 1024 * 1024 )); then
  echo "Mapping not started: less than 4 GiB available RAM. Existing processes were left running." >&2
  echo "Check free -h and docs/HARDWARE104_CN.md before starting the CUDA frontend." >&2
  exit 1
fi
export ROS_DOMAIN_ID="${P3_HARDWARE_DOMAIN:-59}"
export ROS_LOCALHOST_ONLY=0
if [[ -f /home/yanfa/program/cyclonedds.xml ]]; then
  export CYCLONEDDS_URI="file:///home/yanfa/program/cyclonedds.xml"
fi
source "$ROOT/scripts/env.sh"
exec python3 "$ROOT/scripts/control.py" start --profile "$ROOT/config/hardware104.yaml" --network "$@"
