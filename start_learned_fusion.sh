#!/usr/bin/env bash
# Select exactly one learned model. All modes reuse the same LiDAR/maps/interfaces.
set -eo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BACKEND=xfeat_lighterglue
if [[ $# -gt 0 && "$1" != --* ]]; then
 BACKEND="$1"
 shift
fi
case "$BACKEND" in
 xfeat_lighterglue) PROFILE=simulation_xfeat.yaml ;;
 xfeat_mnn) PROFILE=simulation_xfeat_mnn.yaml ;;
 superpoint_lightglue) PROFILE=simulation_superpoint.yaml ;;
 aliked_lightglue) PROFILE=simulation_aliked.yaml ;;
 *) echo "Backend must be xfeat_lighterglue, xfeat_mnn, superpoint_lightglue or aliked_lightglue" >&2
    exit 2 ;;
esac
exec "$ROOT/start.sh" --profile "$ROOT/config/$PROFILE" "$@"
