#!/usr/bin/env bash
set -eo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$ROOT/scripts/env.sh"
export ROS_DOMAIN_ID="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["domain"])' "$ROOT/run_state.json")"
export ROS_LOCALHOST_ONLY="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get("localhost_only","1"))' "$ROOT/run_state.json")"
export CYCLONEDDS_URI="$(python3 -c 'import json,sys,os; print(json.load(open(sys.argv[1])).get("cyclonedds_uri") or os.environ.get("CYCLONEDDS_URI",""))' "$ROOT/run_state.json")"
exec ros2 service call /Car/T3/mapping/save_grid_map std_srvs/srv/Trigger "{}"
