#!/usr/bin/env bash
set -eo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
P3="/home/yanfa/P3/roma_t3_algorithm_bundle_20260825"
set +u
if [[ -f "$P3/workspace/install_native/setup.bash" ]]; then
  source "$P3/workspace/install_native/setup.bash"
fi
source "$ROOT/scripts/env.sh"
export PYTHONDONTWRITEBYTECODE=1
if [[ -d "$P3/integration_demo/rviz_plugins/opt/ros/humble" ]]; then
  export AMENT_PREFIX_PATH="$P3/integration_demo/rviz_plugins/opt/ros/humble:${AMENT_PREFIX_PATH:-}"
  export LD_LIBRARY_PATH="$P3/integration_demo/rviz_plugins/opt/ros/humble/lib:${LD_LIBRARY_PATH:-}"
fi
mode="$1"
shift
if [[ "$mode" == "monitor" ]]; then
  exec /usr/bin/python3 "$ROOT/scripts/p3_visual_monitor.py" --ros-args \
    -r /T3/mapping/lidar_status:=/fusion/map_status \
    -r /Car/T5/Cam_Left/image_raw/color:=/fusion/left \
    -r /Car/T5/Cam_Right/image_raw/color:=/fusion/right \
    -r /Car/T3/metrics/frame_timing:=/fusion/learned_status "$@"
elif [[ "$mode" == "window" ]]; then
  WINDOW="$ROOT/build_visual/t3_visual_window"
  if [[ ! -x "$WINDOW" ]]; then WINDOW="$P3/integration_demo/rviz_window/build/t3_visual_window"; fi
  exec "$WINDOW" \
    "$ROOT/config/p3_visual_window.rviz" "$@"
else
  exit 2
fi
