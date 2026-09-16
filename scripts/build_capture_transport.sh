#!/usr/bin/env bash
# Compile only a private copy; the shared P3 capture source/install is untouched.
set -eo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
RUNTIME="${P3_RUNTIME_ROOT:-$ROOT/../roma_t3_algorithm_bundle_20260825/envx_runtime}"
source /opt/ros/humble/setup.bash
source "$RUNTIME/install/setup.bash"
set -u
python3 "$ROOT/scripts/prepare_capture_transport.py" --runtime "$RUNTIME"
cmake -S "$ROOT/vendor/capture_overlap_source" -B "$ROOT/build/capture_transport" \
  -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF \
  -DCMAKE_INSTALL_PREFIX="$ROOT/install/capture_transport" \
  -DPython3_EXECUTABLE=/usr/bin/python3 -DPYTHON_EXECUTABLE=/usr/bin/python3
cmake --build "$ROOT/build/capture_transport" -j2
cmake --install "$ROOT/build/capture_transport"
python3 "$ROOT/scripts/prepare_capture_transport.py" --record
