#!/usr/bin/env bash
# Mapping + passive sensor relay + RViz. Vehicle control stays with the operator.
set -eo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$ROOT/scripts/env.sh"
export ROS_DOMAIN_ID=57 ROS_LOCALHOST_ONLY=0
out="$ROOT/results/live_mapping_$(date +%Y%m%d_%H%M%S)_$$"
profile="$ROOT/config/simulation_live.yaml"
record_args=()
verify_args=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --record-input) record_args=(--record-input); shift;;
    --verify) verify_args+=(--verify); shift;;
    --verify-seconds) verify_args+=(--verify-seconds "$2"); shift 2;;
    --profile) profile="$2"; shift 2;;
    -h|--help) echo 'Usage: ./start_live.sh [--record-input] [--verify --verify-seconds N] [--profile FILE]'; exit 0;;
    *) echo "Unsupported argument: $1" >&2; exit 2;;
  esac
done
relay_pid=''
cleanup() {
  trap - EXIT INT TERM
  if [[ -d "$out" ]]; then
    touch "$out/live.stop"
    if [[ -n "$relay_pid" ]]; then wait "$relay_pid" || true; fi
    if /usr/bin/python3 - "$ROOT/run_state.json" "$out" <<'PY'
import json, sys
raise SystemExit(0 if json.load(open(sys.argv[1])).get('output') == sys.argv[2] else 1)
PY
    then
      sleep 2
      timeout 30 "$ROOT/save_map.sh" > "$out/live_cleanup.log" 2>&1 || true
      touch "$out/verifier.stop"
      "$ROOT/stop.sh" >> "$out/live_cleanup.log" 2>&1 || true
    fi
  fi
  printf 'Mapping stopped; capture and vehicle control are unchanged. Results: %s\n' "$out"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
"$ROOT/start.sh" --profile "$profile" --domain 57 --network --rviz \
  --output "$out" "${verify_args[@]}"
cp "$ROOT/build_manifest.json" "$out/build_manifest.json"
echo 'Manual live mapping: drive using your own controller; Ctrl+C saves and stops mapping only.'
setsid /usr/bin/python3 "$ROOT/scripts/live_sensor_relay.py" --output "$out" \
  "${record_args[@]}" > "$out/live_relay.log" 2>&1 &
relay_pid=$!
set +e
wait "$relay_pid"
result=$?
set -e
exit "$result"
