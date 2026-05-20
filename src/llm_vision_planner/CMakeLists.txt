cmake_minimum_required(VERSION 3.8)
project(llm_vision_planner)

find_package(ament_cmake REQUIRED)

install(PROGRAMS
  scripts/perception.py
  scripts/perception_detection.py
  scripts/prompt_generator.py
  scripts/llm_planner.py
  scripts/refinment.py
  scripts/verifier.py
  scripts/visualize.py
  scripts/trajectory_follower.py
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY
  launch
  config
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
