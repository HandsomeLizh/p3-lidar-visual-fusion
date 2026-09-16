#!/usr/bin/env bash
# Fixed multi-trajectory suite or one 174-frame case; verify before stopping owned nodes.
set -eo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$ROOT/scripts/env.sh"
mode="${1:-clean}"
if [[ $# -gt 0 ]]; then shift; fi
case "$mode" in
  suite) exec /usr/bin/python3 "$ROOT/scripts/run_trajectory_suite.py" \
    --definition "$ROOT/test_data/fusion_multitrajectory_suite.json" \
    --output "$ROOT/results/fusion_suite_$(date +%Y%m%d_%H%M%S)" --rviz "$@" ;;
  clean) extra=() ;;
  perturbed) extra=(--perturbations "$ROOT/config/long_image_perturbations.json") ;;
  *) echo "Usage: $0 [suite|clean|perturbed] [benchmark options]" >&2; exit 2 ;;
esac
exec /usr/bin/python3 "$ROOT/scripts/run_fusion_repair_benchmark.py" \
  --output "$ROOT/results/fusion_${mode}_$(date +%Y%m%d_%H%M%S)" \
  --rviz "${extra[@]}" "$@"
