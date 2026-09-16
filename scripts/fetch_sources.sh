#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
mkdir -p vendor
if [[ ! -d vendor/VoxelMap ]]; then
 git clone https://github.com/hku-mars/VoxelMap.git vendor/VoxelMap
 git -C vendor/VoxelMap checkout --detach d787ee8ccfb0e509a36adb2c52bd5da97b29c39a
fi
if [[ ! -d vendor/VINS-Fusion ]]; then
 git clone https://github.com/leohaijunli/VINS-Fusion-ROS2-humble-arm.git vendor/VINS-Fusion
 git -C vendor/VINS-Fusion checkout --detach ff167b7933a7389ee0b57659c21475599bdc92a5
fi
if [[ "$(git -C vendor/VINS-Fusion rev-parse HEAD)" != ff167b7933a7389ee0b57659c21475599bdc92a5 ]]; then
 echo "Unexpected VINS-Fusion version; existing checkout was left intact." >&2
 exit 2
fi
if git -C vendor/VINS-Fusion apply --reverse --check "$ROOT/docs/VINS-Fusion.patch" 2>/dev/null; then
 echo "VINS-Fusion patch already applied."
else
 git -C vendor/VINS-Fusion apply --check "$ROOT/docs/VINS-Fusion.patch"
 git -C vendor/VINS-Fusion apply "$ROOT/docs/VINS-Fusion.patch"
fi
mkdir -p src
for package in camera_models vins; do
 target="../vendor/VINS-Fusion/$package"
 if [[ ! -e "src/$package" && ! -L "src/$package" ]]; then
  ln -s "$target" "src/$package"
 elif [[ "$(readlink -f "src/$package")" != "$ROOT/vendor/VINS-Fusion/$package" ]]; then
  echo "Unexpected src/$package; existing path was left intact." >&2
  exit 2
 fi
done
echo "Pinned source dependencies and VINS-Fusion build patch are ready."
