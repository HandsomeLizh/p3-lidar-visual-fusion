#!/usr/bin/env bash
set -eo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export ROS_DOMAIN_ID=59 ROS_LOCALHOST_ONLY=0
export CYCLONEDDS_URI="file://$ROOT/config/cyclonedds_mapping104.xml"
source "$ROOT/scripts/env.sh"
exec /usr/bin/python3 "$ROOT/scripts/preview_hybrid_map.py" "$@"
