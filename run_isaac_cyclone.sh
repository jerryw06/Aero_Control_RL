#!/bin/bash
# Isaac Sim startup script configured for CycloneDDS (to match C++ trainer)
# Uses Isaac Sim's INTERNAL ROS2 to avoid Python version conflicts

set -e

ISAAC_SIM_ROOT="$HOME/isaacsim_5_1_pegasus"
ISAAC_ROS2_PATH="$ISAAC_SIM_ROOT/exts/isaacsim.ros2.bridge/jazzy"

echo "[Isaac Sim] Starting with CycloneDDS..."

if [ ! -d "$ISAAC_SIM_ROOT" ]; then
    echo "❌ ERROR: Isaac Sim not found"
    exit 1
fi

cd "$(dirname "$0")"

env -i \
    HOME="$HOME" \
    USER="$USER" \
    PATH="$PATH" \
    DISPLAY="$DISPLAY" \
    TERM="$TERM" \
    ROS_DISTRO="jazzy" \
    ROS_DOMAIN_ID="0" \
    RMW_IMPLEMENTATION="rmw_cyclonedds_cpp" \
    LD_LIBRARY_PATH="$ISAAC_ROS2_PATH/lib" \
    "$ISAAC_SIM_ROOT/python.sh" 1_px4_single_vehicle_copy.py
