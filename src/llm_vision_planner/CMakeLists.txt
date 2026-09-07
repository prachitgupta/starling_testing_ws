cmake_minimum_required(VERSION 3.8)
project(llm_vision_planner)

find_package(ament_cmake REQUIRED)

install(PROGRAMS
  scripts/prompt_generator.py
  scripts/refinment.py
  scripts/verifier.py
  DESTINATION lib/${PROJECT_NAME}
)

install(FILES
  fine_tuning/scripts/min_control_qp.py
  fine_tuning/scripts/lqr.py
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY
  config
  fine_tuning
  DESTINATION share/${PROJECT_NAME}
  PATTERN "__pycache__" EXCLUDE
  PATTERN "*.pyc" EXCLUDE
)

if(BUILD_TESTING)
  find_package(Python3 REQUIRED COMPONENTS Interpreter)
  add_test(
    NAME expert_calibration_pipeline
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test/test_expert_calibration_pipeline.py
  )
endif()

ament_package()
