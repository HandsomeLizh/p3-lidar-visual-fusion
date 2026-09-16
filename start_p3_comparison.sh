#!/usr/bin/env bash
set -eo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$ROOT/scripts/env.sh"
OUTPUT="$ROOT/results/p3_roma_comparison_$(date +%Y%m%d_%H%M%S)"
exec /usr/bin/python3 "$ROOT/scripts/run_p3_comparison.py" --output "$OUTPUT" "$@"
