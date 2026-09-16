cmake_minimum_required(VERSION 3.10)
project(galaxy2)

# C++17
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "GNU")
  add_compile_options(-Wall -Wextra -Wno-unused-parameter -Wno-unused-but-set-variable)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(Threads REQUIRED)
find_package(OpenCV REQUIRED)

# ---- Galaxy SDK (大恒图像 Galaxy SDK) ----
# Headers + lib live under the SDK install dir; a system copy may also exist.
set(GALAXY_SDK_ROOT "/home/yanfa/SDK/Galaxy_camera" CACHE PATH "Galaxy SDK root")
set(GALAXY_SDK_INC  "${GALAXY_SDK_ROOT}/inc")
set(GALAXY_SDK_LIB  "${GALAXY_SDK_ROOT}/lib/armv8/libgxiapi.so")

if(NOT EXISTS "${GALAXY_SDK_LIB}")
  message(FATAL_ERROR "Galaxy SDK library not found: ${GALAXY_SDK_LIB}")
endif()

add_executable(galaxy2_node
  src/galaxy_camera.cpp
  src/galaxy_node.cpp
)

target_include_directories(galaxy2_node PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${GALAXY_SDK_INC}
)

# Link the Galaxy SDK shared object by absolute path; pthread for the
# acquisition thread.
target_link_libraries(galaxy2_node
  ${GALAXY_SDK_LIB}
  ${OpenCV_LIBS}
  Threads::Threads
)

# Keep the SDK lib discoverable at runtime (libgxiapi.so + liblog4cplus_gx.so).
set_target_properties(galaxy2_node PROPERTIES
  BUILD_RPATH "${GALAXY_SDK_ROOT}/lib/armv8"
  INSTALL_RPATH "${GALAXY_SDK_ROOT}/lib/armv8"
)

ament_target_dependencies(galaxy2_node
  rclcpp
  sensor_msgs
  std_msgs
)

install(TARGETS galaxy2_node
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY launch config
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
