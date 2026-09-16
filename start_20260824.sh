#!/usr/bin/env bash
set -eo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
mode="${1:-fusion}"
if [[ "$#" -gt 0 ]]; then shift; fi
profile="$ROOT/config/bag_20260824_xfeat.yaml"
if [[ "$mode" == "lidar" ]]; then profile="$ROOT/config/bag_20260824_lidar.yaml";
elif [[ "$mode" != "fusion" ]]; then echo "Usage: $0 [fusion|lidar] [start options]"; exit 2; fi
exec "$ROOT/start.sh" --profile "$profile" \
  --bag /home/yanfa/Env_X/InterFace/bags/20260824_235238 \
  --frame-index "$ROOT/test_data/20260824_235238/first_5min_index.json" \
  --header-time-scale 1 --rate 1 --domain 57 --rviz "$@"
