cmake_minimum_required(VERSION 3.8)
project(rl_with_cpp)

# Compiler settings
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(px4_msgs REQUIRED)
find_package(Torch REQUIRED)

# Include directories
include_directories(
  include
  ${TORCH_INCLUDE_DIRS}
)

# PX4 Node library
add_library(px4_node src/px4_node.cpp)
ament_target_dependencies(px4_node rclcpp px4_msgs)
target_link_libraries(px4_node ${TORCH_LIBRARIES})

# PX4 Accel Environment library
add_library(px4_accel_env src/px4_accel_env.cpp)
ament_target_dependencies(px4_accel_env rclcpp px4_msgs)
target_link_libraries(px4_accel_env px4_node ${TORCH_LIBRARIES})

# Policy network library
add_library(policy_network src/policy_network.cpp)
target_link_libraries(policy_network ${TORCH_LIBRARIES})

# Training executable
add_executable(train_rl src/train_rl.cpp)
ament_target_dependencies(train_rl rclcpp px4_msgs)
target_link_libraries(train_rl 
  px4_node 
  px4_accel_env 
  policy_network 
  ${TORCH_LIBRARIES}
)

# Install targets
install(TARGETS
  train_rl
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY
  launch
  config
  DESTINATION share/${PROJECT_NAME}/
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
