#!/bin/bash
# Workaround script for Fast DDS payload size issues

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
WORKSPACE_ROOT="$REPO_ROOT/ros2_local_ws"

# Force Fast DDS to use larger preallocated sizes via environment
export FASTRTPS_DEFAULT_PROFILES_FILE="$SCRIPT_DIR/fastdds_profile.xml"
export FASTDDS_DEFAULT_PROFILES_FILE="$SCRIPT_DIR/fastdds_profile.xml"
export RMW_IMPLEMENTATION="rmw_fastrtps_cpp"

# Additional Fast DDS workarounds
export RMW_FASTRTPS_PUBLICATION_MODE=ASYNCHRONOUS
export FASTRTPS_BUILTIN_TRANSPORTS=UDPv4

# Source ROS2
source /opt/ros/jazzy/setup.bash

# Source px4_msgs
if [ -f "$SCRIPT_DIR/px4_msgs_ws/install/setup.bash" ]; then
    source "$SCRIPT_DIR/px4_msgs_ws/install/setup.bash"
fi

# Source workspace
cd "$WORKSPACE_ROOT"
if [ -f "$WORKSPACE_ROOT/install/setup.bash" ]; then
    source "$WORKSPACE_ROOT/install/setup.bash"
fi

# Source LibTorch environment
if [ -f "$SCRIPT_DIR/setup_env.sh" ]; then
    source "$SCRIPT_DIR/setup_env.sh"
fi

echo "=== Fast DDS Workaround Configuration ==="
echo "RMW_IMPLEMENTATION=$RMW_IMPLEMENTATION"
echo "FASTDDS_DEFAULT_PROFILES_FILE=$FASTDDS_DEFAULT_PROFILES_FILE"
echo "RMW_FASTRTPS_PUBLICATION_MODE=$RMW_FASTRTPS_PUBLICATION_MODE"
echo ""
echo "Starting C++ RL Training..."
echo "Make sure PX4 SITL is running!"
echo ""

ros2 run rl_with_cpp train_rl
