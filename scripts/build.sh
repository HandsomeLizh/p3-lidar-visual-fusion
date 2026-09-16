#!/usr/bin/env bash
set -eo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/env.sh"
cd "$T3_FUSION_ROOT"
export MAKEFLAGS="-j3 -l6"
export CMAKE_BUILD_PARALLEL_LEVEL=3
colcon build --symlink-install --executor sequential --base-paths src \
 --packages-select grid_map_msgs t3_interfaces camera_models vins t3_lidar_visual_fusion t3_voxelmap \
 --cmake-args -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF \
 -DCeres_DIR="$T3_FUSION_ROOT/config/cmake/Ceres" \
 -DPython3_EXECUTABLE=/usr/bin/python3 -DPYTHON_EXECUTABLE=/usr/bin/python3

/usr/bin/python3 "$T3_FUSION_ROOT/scripts/source_manifest.py" --record
