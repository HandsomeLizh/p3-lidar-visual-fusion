#!/usr/bin/env bash
set -eo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/env.sh"
cd "$T3_FUSION_ROOT"
colcon build --base-paths drivers/galaxy2 --build-base build_hardware \
  --install-base install_hardware --executor sequential --packages-select galaxy2 \
  --cmake-args -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF
