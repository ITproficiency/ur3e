cmake_minimum_required(VERSION 3.8)
project(ur3_moveit_control)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_action REQUIRED)
find_package(moveit_ros_planning_interface REQUIRED)
find_package(moveit_core REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(action_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "action/UR3Control.action"
  DEPENDENCIES geometry_msgs sensor_msgs action_msgs
)

include_directories(include)

add_library(ur3_motion_interface
  src/ur3_motion_interface.cpp
)

ament_target_dependencies(ur3_motion_interface
  rclcpp
  moveit_ros_planning_interface
  moveit_core
  geometry_msgs
  sensor_msgs
  std_msgs
)

add_executable(ur3_control_node
  src/ur3_control_node.cpp
)

target_link_libraries(ur3_control_node
  ur3_motion_interface
)

rosidl_get_typesupport_target(cpp_typesupport_target ${PROJECT_NAME} "rosidl_typesupport_cpp")
target_link_libraries(ur3_control_node "${cpp_typesupport_target}")

ament_target_dependencies(ur3_control_node
  rclcpp
  rclcpp_action
  moveit_ros_planning_interface
  geometry_msgs
  sensor_msgs
  visualization_msgs
  std_msgs
)

install(
  TARGETS
    ur3_motion_interface
    ur3_control_node
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION lib/${PROJECT_NAME}
)

install(
  PROGRAMS
    src/send_goal_client.py
    src/random_pick_test.py
    scripts/save_scene_to_warehouse.py
  DESTINATION lib/${PROJECT_NAME}
)

# Shared by both installed Python executables. Keeping the grasp profile in one
# file prevents the fixed and random test paths from drifting apart.
install(
  FILES src/grasp_profile.py
  DESTINATION lib/${PROJECT_NAME}
)

install(
  DIRECTORY include/
  DESTINATION include
)

install(
  # The semantic robot description belongs to ur_moveit_config and is loaded
  # from that package by the launch files. ur3_moveit_control has no srdf/
  # directory, so listing it here makes the install phase fail.
  DIRECTORY config launch urdf rviz models
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
