#!/usr/bin/env bash
set -eo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/scripts/env.sh"
exec /usr/bin/python3 "$T3_FUSION_ROOT/scripts/control.py" start "$@"
