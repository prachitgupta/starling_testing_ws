cmake_minimum_required(VERSION 3.8)
project(starling_testing)

find_package(ament_cmake REQUIRED)

install(PROGRAMS
  scripts/01_takeoff_land.py
  scripts/02_offboard_waypoint.py
  scripts/03_bezier_offboard_waypoint.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
