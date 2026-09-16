#!/usr/bin/env bash
# Compatibility entry: manual live mapping only. No automatic vehicle motion.
set -eo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "$ROOT/start_live.sh" "$@"
