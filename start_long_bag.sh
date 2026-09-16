#!/usr/bin/env bash
set -eo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "$ROOT/start.sh" --profile "$ROOT/config/long_bag_voxelmap.yaml" \
 --bag /home/yanfa/Env_X/InterFace/bags/20260824_235238 \
 --frame-index "$ROOT/test_data/20260824_235238/first_10min_index.json" \
 --rate 1 --header-time-scale 1 --domain 57 --rviz --verify --verify-seconds 680 "$@"
