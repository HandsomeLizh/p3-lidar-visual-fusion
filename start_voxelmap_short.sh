#!/usr/bin/env bash
set -eo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
mode="${1:-fusion}"
if [[ $# -gt 0 ]]; then shift; fi
case "$mode" in
 fusion) profile="$ROOT/config/selected_bag_voxelmap.yaml" ;;
 lidar) profile="$ROOT/config/selected_bag_voxelmap_lidar.yaml" ;;
 *) echo "Usage: $0 [fusion|lidar] [start options]" >&2; exit 2 ;;
esac
echo "VoxelMap: pure LiDAR; short_20260915_190151; native sensor timestamps; no IMU."
echo "The lidar diagnostic mode does not certify localization in degenerate geometry."
exec "$ROOT/start.sh" --profile "$profile" \
 --bag /home/yanfa/P3/roma_t3_algorithm_bundle_20260825/recordings/short_20260915_190151/bag \
 --rate 1 --header-time-scale 1 --domain 57 --rviz --verify "$@"
