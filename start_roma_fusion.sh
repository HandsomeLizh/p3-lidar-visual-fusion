#!/usr/bin/env bash
# Uses an externally launched RoMa pose source on /fusion/roma_raw.
set -eo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "$ROOT/start.sh" --profile "$ROOT/config/simulation_roma.yaml" "$@"
