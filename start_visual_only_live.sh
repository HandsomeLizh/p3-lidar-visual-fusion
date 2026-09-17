#!/usr/bin/env bash
set -eo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$ROOT/scripts/env.sh"
export ROS_DOMAIN_ID=57 ROS_LOCALHOST_ONLY=0
export DISPLAY="${DISPLAY:-:0}"
export XAUTHORITY="${XAUTHORITY:-/run/user/1000/gdm/Xauthority}"
exec /usr/bin/python3 "$ROOT/scripts/online_visual_only.py" "$@"
