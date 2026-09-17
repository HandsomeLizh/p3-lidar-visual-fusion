#!/usr/bin/env bash
# Private parameters: keep the installed sensor calibration, disable raw PCD saving.
set -eo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source /opt/ros/humble/setup.bash
source /home/yanfa/program/Lidar2/install/setup.bash
export ROS_DOMAIN_ID="${P3_SENSOR_DOMAIN:-19}"
export ROS_LOCALHOST_ONLY=0
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export CYCLONEDDS_URI="file:///home/yanfa/program/cyclonedds.xml"
PARAMS="$ROOT/results/hardware_lidar_runtime/params.yaml"
python3 - "$PARAMS" <<'PY'
from pathlib import Path
import sys
import yaml
from ament_index_python.packages import get_package_share_directory

source = Path(get_package_share_directory('ouster_ros'))/'config/community_driver_config.yaml'
config = yaml.safe_load(source.read_text())
parameters = config['Car']['T5']['OS1']['ouster_driver']['ros__parameters']
parameters['save_enable'] = False
target = Path(sys.argv[1])
target.parent.mkdir(parents=True, exist_ok=True)
temporary = target.with_suffix('.tmp')
temporary.write_text(yaml.safe_dump(config, sort_keys=False))
temporary.replace(target)
print('Ouster parameters:', target, '; raw point-cloud saving disabled', flush=True)
PY
# driver_launch.py hard-codes its YAML and ignores save_enable launch arguments.
exec ros2 launch ouster_ros driver.launch.py params_file:="$PARAMS" \
  ouster_ns:=Car/T5/OS1 os_driver_name:=ouster_driver viz:=False
