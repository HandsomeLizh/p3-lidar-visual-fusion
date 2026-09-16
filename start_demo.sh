#!/usr/bin/env bash
set -eo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "$ROOT/start_bag.sh" /home/yanfa/Env_X/InterFace/bags/20260824_235238 --header-time-scale 1000000 --frames 80 --rviz "$@"
