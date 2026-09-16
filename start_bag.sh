#!/usr/bin/env bash
set -eo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ $# -lt 1 ]]; then echo "Usage: $0 BAG_DIRECTORY [--header-time-scale 1000000] [--rviz]"; exit 2; fi
BAG="$1"; shift
exec "$ROOT/start.sh" --bag "$BAG" "$@"
