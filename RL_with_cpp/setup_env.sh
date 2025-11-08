#!/bin/bash
# Source this file before building or running the C++ RL trainer

# Source ROS2
source /opt/ros/jazzy/setup.bash

# Set LibTorch path
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
export CMAKE_PREFIX_PATH="$SCRIPT_DIR/libtorch:$CMAKE_PREFIX_PATH"
export LD_LIBRARY_PATH="$SCRIPT_DIR/libtorch/lib:$LD_LIBRARY_PATH"

# Local ephemeral build workspace location (created by install_and_setup.sh)
export RL_CPP_LOCAL_WS="$SCRIPT_DIR/../ros2_local_ws"
if [ -f "$RL_CPP_LOCAL_WS/install/setup.bash" ]; then
    source "$RL_CPP_LOCAL_WS/install/setup.bash"
fi
if [ -f "$SCRIPT_DIR/px4_msgs_ws/install/setup.bash" ]; then
    source "$SCRIPT_DIR/px4_msgs_ws/install/setup.bash"
fi

echo "Environment configured for C++ RL training"
echo "LibTorch path: $SCRIPT_DIR/libtorch"
