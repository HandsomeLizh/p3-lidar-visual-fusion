#!/usr/bin/env bash
# Uses the existing /sensors_trigger (5 Hz on rover 104); never starts vehicle control.
set -eo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source /opt/ros/humble/setup.bash
source "$ROOT/install_hardware/setup.bash"
export ROS_DOMAIN_ID="${P3_SENSOR_DOMAIN:-19}"
export ROS_LOCALHOST_ONLY=0
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export CYCLONEDDS_URI="file:///home/yanfa/program/cyclonedds.xml"
export LD_LIBRARY_PATH="/home/yanfa/SDK/Galaxy_camera/lib/armv8:${LD_LIBRARY_PATH:-}"
exec ros2 launch galaxy2 galaxy2_dual.launch.py trigger_mode:=software \
  enable_save:=false exposure_time:=3000.0 publish_raw:=false publish_color:=false \
  publish_mapping:=true mapping_width:=640 mapping_height:=536 \
  publish_color_compressed:=false publish_light:=false "$@"
