#!/usr/bin/env bash
set -eo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/env.sh"
cmake -S "$T3_FUSION_ROOT/visual/rviz_window" -B "$T3_FUSION_ROOT/build_visual" -DCMAKE_BUILD_TYPE=Release
cmake --build "$T3_FUSION_ROOT/build_visual" -j2
