#!/usr/bin/env bash
# Source in this shell. All overlay paths are private to this task directory.
T3_FUSION_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
set +u
source /opt/ros/humble/setup.bash
T3_DEPS="$T3_FUSION_ROOT/deps/root"
export AMENT_PREFIX_PATH="$T3_DEPS/opt/ros/humble:${AMENT_PREFIX_PATH:-}"
export CMAKE_PREFIX_PATH="$T3_DEPS/opt/ros/humble:$T3_DEPS/usr:${CMAKE_PREFIX_PATH:-}"
export LD_LIBRARY_PATH="$T3_DEPS/opt/ros/humble/lib:$T3_DEPS/usr/lib:$T3_DEPS/usr/lib/aarch64-linux-gnu:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$T3_DEPS/opt/ros/humble/local/lib/python3.10/dist-packages:$T3_DEPS/opt/ros/humble/lib/python3.10/site-packages:${PYTHONPATH:-}"
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-57}"
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export CYCLONEDDS_URI="${CYCLONEDDS_URI:-file://$T3_FUSION_ROOT/config/cyclonedds.xml}"
export ROS_LOCALHOST_ONLY="${ROS_LOCALHOST_ONLY:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1
if [[ -f "$T3_FUSION_ROOT/install/setup.bash" ]]; then
 source "$T3_FUSION_ROOT/install/setup.bash"
fi
export T3_FUSION_ROOT
