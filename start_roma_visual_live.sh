#!/usr/bin/env bash
# Existing P3 RoMa frontend, current LiDAR-only mapper and original UI.
set -eo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$ROOT/start_visual_only_live.sh" --backend roma "$@"
