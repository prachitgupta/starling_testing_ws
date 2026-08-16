cmake_minimum_required(VERSION 3.8)
project(llm_vision_planner)

find_package(ament_cmake REQUIRED)

install(PROGRAMS
  scripts/perception.py
  scripts/perception_detection.py
  scripts/debug_perception.py
  scripts/ros_wifi_dds.sh
  scripts/prompt_generator.py
  scripts/interactive_mission_gateway.py
  scripts/interactive_web_ui.py
  scripts/dataset_scene_publisher.py
  scripts/llm_planner.py
  scripts/refinment.py
  scripts/verifier.py
  scripts/visualize.py
  scripts/verify_contraction.py
  scripts/mission_takeoff.py
  scripts/trajectory_follower.py
  scripts/trajectory_follower_continuous.py
  scripts/control_law_executer.py
  fine_tuning/scripts/vision_error_dataset_generattor.py
  DESTINATION lib/${PROJECT_NAME}
)

install(FILES
  fine_tuning/scripts/min_control_qp.py
  fine_tuning/scripts/lqr.py
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY
  launch
  config
  web
  fine_tuning
  DESTINATION share/${PROJECT_NAME}
  PATTERN "__pycache__" EXCLUDE
  PATTERN "*.pyc" EXCLUDE
)

if(BUILD_TESTING)
  find_package(Python3 REQUIRED COMPONENTS Interpreter)
  add_test(
    NAME interactive_web_ui
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test/test_interactive_web_ui.py
  )
  add_test(
    NAME conformal_obstacle_safety
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test/test_conformal_obstacle_safety.py
  )
endif()

ament_package()
