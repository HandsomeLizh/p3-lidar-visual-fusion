cmake_minimum_required(VERSION 3.8)
project(grid_map_msgs)

find_package(ament_cmake REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)

# These are the unmodified public interfaces from ANYbotics/grid_map 2.4.0.
# Keeping the standard package and type names makes this small vendored copy
# wire-compatible with systems that install grid_map_msgs from binary packages.
rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/GridMapInfo.msg"
  "msg/GridMap.msg"
  "srv/SetGridMap.srv"
  "srv/GetGridMap.srv"
  "srv/GetGridMapInfo.srv"
  "srv/ProcessFile.srv"
  DEPENDENCIES geometry_msgs std_msgs
)

ament_export_dependencies(rosidl_default_runtime)
ament_package()
