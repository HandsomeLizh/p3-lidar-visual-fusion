#!/usr/bin/env bash
# Sensor capture only. Vehicle motion stays with the existing simulator/planner.
set -eo pipefail
T3_CAPTURE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source "$T3_CAPTURE_ROOT/scripts/env.sh"
source /home/yanfa/Env_X/InterFace/install/setup.bash
exec "$T3_CAPTURE_ROOT/vendor/sensor_input/sensor_capture_node" --ros-args \
 -r __node:=fusion_sensor_capture \
 -p tcp_host:="${1:-192.168.10.22}" -p tcp_port:=6665 \
 -p timestamp_mode:=meta_relative -p auto_start:=true \
 -p batch_count:=20000 -p capture_freq:=1.0 -p timeout:=60.0 \
 -p save_data:=false -p publish_thumbnail:=false \
 -p save_dir:="$2/_internal/capture" \
 -p tof_depth_scale:=0.01 -p tof_max_range:=9.5 -p lidar_scale:=0.01 \
 -p big_endian:=false -p normalize_save:=false -p exposure_offset_ms:=600.0
