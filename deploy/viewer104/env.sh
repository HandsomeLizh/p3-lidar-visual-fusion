#!/usr/bin/env bash
P3_VIEWER_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# Only source dependencies. No process in the 237 robot project is launched.
source /home/yanfa/P3/lidar_visual_fusion/scripts/env.sh
# Pin the display context AFTER all overlays, even from a robot-237 terminal.
export ROS_DOMAIN_ID=59 ROS_LOCALHOST_ONLY=0 RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export CYCLONEDDS_URI="file://$P3_VIEWER_ROOT/config/cyclonedds.xml"
unset ROS_NAMESPACE ROS_STATIC_PEERS ROS_AUTOMATIC_DISCOVERY_RANGE
unset T3_VISUAL_RUNTIME T3_RENDER_AUDIT
export PYTHONPATH="$P3_VIEWER_ROOT/python:${PYTHONPATH:-}"
export PYTHONDONTWRITEBYTECODE=1
export QT_QPA_PLATFORM=xcb
export DISPLAY="${DISPLAY:-:0}"
export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}"
if [[ -z "${XAUTHORITY:-}" && -f "$XDG_RUNTIME_DIR/gdm/Xauthority" ]]; then
 export XAUTHORITY="$XDG_RUNTIME_DIR/gdm/Xauthority"
fi
