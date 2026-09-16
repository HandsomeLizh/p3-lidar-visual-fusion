#!/usr/bin/env bash
set -eo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$ROOT/scripts/env.sh"
OUTPUT="$ROOT/results/xfeat_only_$(date +%Y%m%d_%H%M%S)"
exec /usr/bin/python3 "$ROOT/scripts/run_compact_visual_comparison.py" --output "$OUTPUT" "$@"
