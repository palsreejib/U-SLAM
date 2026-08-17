cmake_minimum_required(VERSION 3.8)
project(mps_sim)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# ── Dependencies ──────────────────────────────────────────────────────
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(trajectory_msgs REQUIRED)
find_package(builtin_interfaces REQUIRED)

# ── Custom messages ───────────────────────────────────────────────────
rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/MpsSector.msg"
  "msg/DvlReading.msg"
  "msg/SyncedSample.msg"
  DEPENDENCIES std_msgs sensor_msgs nav_msgs geometry_msgs builtin_interfaces
)

# ── Python nodes ──────────────────────────────────────────────────────

install(PROGRAMS
  mps_driver/mps_driver_node.py
  mps_driver/ground_truth_node.py
  mps_driver/dvl_node.py
  mps_driver/pressure_node.py
  mps_driver/sync_node.py
  DESTINATION lib/${PROJECT_NAME}
)

# ── URDF, worlds, launch, config ───────────────────────────────────────
install(DIRECTORY
  urdf
  worlds
  launch
  config
  DESTINATION share/${PROJECT_NAME}
)

ament_export_dependencies(rosidl_default_runtime)
ament_package()
