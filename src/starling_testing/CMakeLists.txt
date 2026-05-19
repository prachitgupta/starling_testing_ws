cmake_minimum_required(VERSION 3.8)
project(starling_testing)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(px4_msgs REQUIRED)
find_package(px4_ros2_cpp REQUIRED)

add_executable(voxl_px4_ros2_goto_waypoint_cpp src/04_px4_ros2_goto_waypoint.cpp)
target_compile_features(voxl_px4_ros2_goto_waypoint_cpp PUBLIC cxx_std_17)
ament_target_dependencies(voxl_px4_ros2_goto_waypoint_cpp rclcpp px4_msgs)
target_link_libraries(voxl_px4_ros2_goto_waypoint_cpp px4_ros2_cpp::px4_ros2_cpp)

add_executable(voxl_px4_ros2_goto_mission_cpp src/05_px4_ros2_goto_mission.cpp)
target_compile_features(voxl_px4_ros2_goto_mission_cpp PUBLIC cxx_std_17)
ament_target_dependencies(voxl_px4_ros2_goto_mission_cpp rclcpp px4_msgs)
target_link_libraries(voxl_px4_ros2_goto_mission_cpp px4_ros2_cpp::px4_ros2_cpp)

install(TARGETS
  voxl_px4_ros2_goto_waypoint_cpp
  voxl_px4_ros2_goto_mission_cpp
  DESTINATION lib/${PROJECT_NAME}
)

install(PROGRAMS
  scripts/01_takeoff_land.py
  scripts/02_fmu_trajectory_waypoint.py
  scripts/03_px4_ros2_goto_waypoint.py
  scripts/run_voxl_sequence.sh
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
