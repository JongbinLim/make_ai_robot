cmake_minimum_required(VERSION 3.8)
project(localization)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# --- 1. Find Dependencies ---
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED) # 추가됨
find_package(tf2_eigen REQUIRED)
find_package(pcl_conversions REQUIRED)

# Find System Libraries
find_package(PCL REQUIRED COMPONENTS common io registration)
find_package(Eigen3 REQUIRED)
find_package(OpenMP REQUIRED)

# --- 2. Build Particle Filter Library ---
add_library(particle_filter
  localization/particle_filter.cpp
)

target_include_directories(particle_filter PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/localization>
  $<INSTALL_INTERFACE:include>
)

# OpenMP Link
if(OpenMP_CXX_FOUND)
  target_link_libraries(particle_filter PUBLIC OpenMP::OpenMP_CXX)
endif()

# --- 3. Build Odom Localizer Node (UKF + PCL) ---
add_executable(odom_localizer_node
  localization/odom_localizer_node.cpp
)

target_include_directories(odom_localizer_node PUBLIC
  ${PCL_INCLUDE_DIRS}
  ${Eigen3_INCLUDE_DIRS}
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/localization>
)

target_link_libraries(odom_localizer_node
  ${PCL_LIBRARIES}
  Eigen3::Eigen
  particle_filter
)

ament_target_dependencies(odom_localizer_node
  rclcpp
  geometry_msgs
  sensor_msgs
  nav_msgs
  tf2
  tf2_ros
  tf2_eigen
  pcl_conversions
)

# --- 4. Build Global Localizer Node (MCL) ---
add_executable(global_localizer_node
  localization/global_localizer_node.cpp
)

# particle_filter 헤더를 찾을 수 있도록 포함 경로 추가
target_include_directories(global_localizer_node PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/localization>
)

# Particle Filter 라이브러리와 링크
target_link_libraries(global_localizer_node
  particle_filter
)

ament_target_dependencies(global_localizer_node
  rclcpp
  geometry_msgs
  nav_msgs
  sensor_msgs
  tf2
  tf2_ros
  tf2_geometry_msgs
)


# --- 5. Install ---
install(
  TARGETS
    odom_localizer_node
    global_localizer_node
    particle_filter
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION lib/${PROJECT_NAME}
)

# 헤더 파일 설치
install(
  DIRECTORY localization/
  DESTINATION include/${PROJECT_NAME}
  FILES_MATCHING PATTERN "*.hpp"
)

# Launch 파일 설치
install(
  DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}
  OPTIONAL
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
