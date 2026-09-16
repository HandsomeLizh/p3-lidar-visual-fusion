#!/usr/bin/env bash
# Execute only after the shared remote has been released for our tests.
set -eo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/scripts/env.sh"
cd "$ROOT"
python3 -c 'import sys; sys.path.insert(0,"scripts"); from source_manifest import check; check()'
python3 -m unittest discover -s tests -p 'test_*.py' -v
ROS_DOMAIN_ID=58 python3 tests/ros_fault_test.py
