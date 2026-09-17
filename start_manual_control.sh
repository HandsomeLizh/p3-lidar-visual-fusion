#!/usr/bin/env bash
# Real 104 chassis + original manual GUI; independent of capture, P3 and P4.
set -eo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source /opt/ros/humble/setup.bash
source /home/yanfa/program/Mars_Car2/install/setup.bash
source /home/yanfa/program/UI/install/setup.bash
export ROS_DOMAIN_ID=19 ROS_LOCALHOST_ONLY=0 RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export CYCLONEDDS_URI=file:///home/yanfa/program/cyclonedds.xml
exec /usr/bin/python3 "$ROOT/scripts/manual_control104.py" "$@"
