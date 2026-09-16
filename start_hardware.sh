#!/usr/bin/env bash
# Sensor drivers run in domain 19. Keep the car separate from simulation domain 57.
set -eo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export ROS_DOMAIN_ID="${P3_HARDWARE_DOMAIN:-59}"
export ROS_LOCALHOST_ONLY=0
if [[ -f /home/yanfa/program/cyclonedds.xml ]]; then
  export CYCLONEDDS_URI="file:///home/yanfa/program/cyclonedds.xml"
fi
source "$ROOT/scripts/env.sh"
exec python3 "$ROOT/scripts/control.py" start --profile "$ROOT/config/hardware104.yaml" --network "$@"
