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

# The first build creates this overlay. Reload it before checking what ROS
# actually imports; the environment sourced above may predate the install.
source "$T3_FUSION_ROOT/scripts/env.sh"
/usr/bin/python3 "$T3_FUSION_ROOT/scripts/source_manifest.py" --record
