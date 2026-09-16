#!/usr/bin/env bash
set -eo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export ROS_DOMAIN_ID="${P3_SENSOR_DOMAIN:-19}"
export ROS_LOCALHOST_ONLY=0
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export CYCLONEDDS_URI="file:///home/yanfa/program/cyclonedds.xml"
source "$ROOT/scripts/env.sh"
exec python3 "$ROOT/scripts/hardware_sensors.py" "${@:-start}"
