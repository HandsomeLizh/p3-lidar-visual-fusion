#!/usr/bin/env bash
# Start/reuse the real vehicle mapper; render on the separate 237 viewer.
set -eo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$ROOT/scripts/env.sh"
exec /usr/bin/python3 "$ROOT/scripts/ensure_hardware104.py" "$@"
