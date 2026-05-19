cmake_minimum_required(VERSION 3.8)
project(llm_vision_planner)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(px4_msgs REQUIRED)
find_package(px4_ros2_cpp REQUIRED)
find_package(jsoncpp CONFIG REQUIRED)

add_executable(trajectory_follower_cpp src/trajectory_follower.cpp)
target_compile_features(trajectory_follower_cpp PUBLIC cxx_std_17)
ament_target_dependencies(trajectory_follower_cpp rclcpp std_msgs px4_msgs)
target_link_libraries(trajectory_follower_cpp px4_ros2_cpp::px4_ros2_cpp JsonCpp::JsonCpp)

install(TARGETS
  trajectory_follower_cpp
  DESTINATION lib/${PROJECT_NAME}
)

install(PROGRAMS
  scripts/perception.py
  scripts/perception_detection.py
  scripts/prompt_generator.py
  scripts/llm_planner.py
  scripts/refinment.py
  scripts/verifier.py
  scripts/visualize.py
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY
  launch
  config
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
