#!/usr/bin/env bash
# Prefer the verified private capture build; keep the existing binary as a fallback.
set -eo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
RUNTIME="${P3_RUNTIME_ROOT:-$ROOT/../roma_t3_algorithm_bundle_20260825/envx_runtime}"
source /opt/ros/humble/setup.bash
source "$RUNTIME/install/setup.bash"
set -u
UE_HOST=192.168.10.22
UE_PORT=6665
CAPTURE_HZ="${P3_CAPTURE_HZ:-2.0}"
BATCH_GAP="${P3_CAPTURE_BATCH_GAP:-0.0}"
while [[ $# -gt 0 ]]; do
 case "$1" in
  --host) UE_HOST="$2"; shift 2;;
  --port) UE_PORT="$2"; shift 2;;
  --capture-freq) CAPTURE_HZ="$2"; shift 2;;
  --batch-interval) BATCH_GAP="$2"; shift 2;;
  *) echo "Unsupported argument: $1" >&2; exit 2;;
 esac
done
CAPTURE=(ros2 run sensor_capture_ros2 sensor_capture_node)
PRIVATE_BINARY="$ROOT/install/capture_transport/lib/sensor_capture_ros2/sensor_capture_node"
case "${P3_CAPTURE_OPTIMIZED:-auto}" in
 0) ;;
 1|auto)
   if [[ -x "$PRIVATE_BINARY" ]]; then
     python3 "$ROOT/scripts/prepare_capture_transport.py" --check
     CAPTURE=("$PRIVATE_BINARY")
   elif [[ "${P3_CAPTURE_OPTIMIZED:-auto}" == 1 ]]; then
     echo 'Run scripts/build_capture_transport.sh before requesting the optimized capture.' >&2
     exit 2
   fi;;
 *) echo 'P3_CAPTURE_OPTIMIZED must be auto, 0, or 1.' >&2; exit 2;;
esac
printf 'Capture executable: %s\n' "${CAPTURE[*]}"
exec "${CAPTURE[@]}" --ros-args \
 -p tcp_host:="$UE_HOST" -p tcp_port:="$UE_PORT" \
 -p capture_freq:="$CAPTURE_HZ" -p batch_interval:="$BATCH_GAP" -p batch_count:=20000 \
 -p timestamp_mode:="${SENSOR_TIMESTAMP_MODE:-meta_relative}" \
 -p save_dir:="$ROOT/results/capture_scratch" -p save_data:=false \
 -p auto_start:=true -p big_endian:=false -p publish_thumbnail:=false \
 -p 'output_topics:=[rgb_0:=/Car/T5/Cam_Left/image_raw/color,rgb_1:=/Car/T5/Cam_Right/image_raw/color,cpulidar_0:=/Car/T5/OS1/points,tof_0_depth:=/Car/T5/TOF_Left/image_raw/depth,tof_1_depth:=/Car/T5/TOF_Right/image_raw/depth]'
