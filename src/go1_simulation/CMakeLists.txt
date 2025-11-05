# CMake Version Requirement
# Specifies minimum CMake version needed for this package
# 3.8+ required for modern CMake features used by ROS2
cmake_minimum_required(VERSION 3.8)

# Project Declaration
# Defines the package name - must match package.xml <name> tag
project(go1_simulation)

# =============================================================================
# COMPILER CONFIGURATION
# =============================================================================

# Enable additional compiler warnings for better code quality
# These flags help catch potential issues during compilation
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# =============================================================================
# DEPENDENCY DISCOVERY
# =============================================================================
# Find and configure all required ROS2 packages and external dependencies
# Each find_package() call locates libraries and sets up CMake variables

# Core ROS2 Build System Dependencies
find_package(ament_cmake REQUIRED)           # Main ROS2 CMake macros and functions
find_package(ament_cmake_python REQUIRED)    # Python integration support

# Language-Specific ROS2 Client Libraries
find_package(rclcpp REQUIRED)                # ROS2 C++ client library
find_package(rclpy REQUIRED)                 # ROS2 Python client library

# ROS2 Message and Interface Packages
find_package(sensor_msgs REQUIRED)           # Standard sensor message definitions
find_package(geometry_msgs REQUIRED)         # Standard geometry message definitions
find_package(std_msgs REQUIRED)              # Standard message definitions
find_package(nav_msgs REQUIRED)              # Navigation message definitions
find_package(rosgraph_msgs REQUIRED)         # ROS graph message definitions (Clock)
find_package(tf2 REQUIRED)                   # Transform library
find_package(tf2_ros REQUIRED)               # Transform ROS interface

# Computer Vision and Sensor Processing
find_package(cv_bridge REQUIRED)             # OpenCV bridge for ROS
find_package(message_filters REQUIRED)       # Message synchronization utilities

# SLAM and Navigation
find_package(slam_toolbox REQUIRED)          # SLAM toolbox for mapping

# URDF Processing
find_package(xacro REQUIRED)                 # Xacro for URDF processing

# =============================================================================
# RESOURCE FILE INSTALLATION
# =============================================================================
# Install package resources (launch files, config files, etc.) to share directory
# These files are accessible to other packages and launch systems

install(
  DIRECTORY
    launch                    # Launch files for starting nodes and systems
    urdf                      # Robot description files (URDF/Xacro)
    meshes                    # 3D model files for visualization
    rviz                      # RViz configuration files
    config                    # Configuration files for controllers and simulation
    maps                      # 2D Occupancy Grid Map files
    go1_simulation            # Python package directory
  DESTINATION share/${PROJECT_NAME}  # Install to: install/share/go1_simulation/
)

# =============================================================================
# PYTHON PACKAGE INTEGRATION
# =============================================================================
# Configure Python components of the mixed-language package

# Install Python package using standard Python packaging
# This makes the go1_simulation Python module importable
ament_python_install_package(${PROJECT_NAME})

# Install Python executables (scripts that can be run with ros2 run)
install(
  PROGRAMS
    go1_simulation/go1_gt_pose_publisher.py    # Python node for ground truth pose publisher
    go1_simulation/publish_pointcloud.py        # Python script for publishing the pointcloud
    go1_simulation/move_go1_with_keyboard.py    # Python script for moving the robot with the keyboard
    go1_simulation/save_map.py                # Python script for saving the 2D Occupancy Grid Map
  DESTINATION lib/${PROJECT_NAME}       # Install to: install/lib/go1_simulation/
)

# =============================================================================
# TESTING AND CODE QUALITY
# =============================================================================
# Configure testing framework and code quality checks

if(BUILD_TESTING)
  # Find testing framework
  find_package(ament_lint_auto REQUIRED)
  
  # Skip copyright checking (commented lines explain when to enable)
  # Uncomment next line when copyright headers are added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  
  # Skip C++ linting (commented lines explain when to enable)  
  # Uncomment next line when package is in git repo with proper copyright headers
  set(ament_cmake_cpplint_FOUND TRUE)
  
  # Run automatic linting and testing
  # This finds and runs all available linters for the package
  ament_lint_auto_find_test_dependencies()
endif()

# =============================================================================
# PACKAGE FINALIZATION
# =============================================================================
# Complete the package configuration and generate necessary files

# Generate package configuration files
# This creates the files needed for other packages to find and use this package
ament_package()