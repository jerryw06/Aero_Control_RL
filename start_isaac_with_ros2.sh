#!/bin/bash
# Wrapper script to run Isaac Sim with ROS2 properly configured
# This script ISOLATES Isaac Sim's Python 3.11 + internal ROS2 Jazzy environment

set -e  # Exit on error

# Unset all ROS2 environment variables from system installation
unset ROS_VERSION ROS_PYTHON_VERSION AMENT_PREFIX_PATH CMAKE_PREFIX_PATH COLCON_PREFIX_PATH PYTHONPATH

# Clean LD_LIBRARY_PATH of system ROS2 paths
if [ -n "$LD_LIBRARY_PATH" ]; then
    LD_LIBRARY_PATH=$(echo "$LD_LIBRARY_PATH" | tr ':' '\n' | grep -v -E '/(ros|jazzy|humble|iron)/' | tr '\n' ':' | sed 's/:$//')
fi

# Isaac Sim paths
ISAAC_SIM_ROOT="$HOME/isaacsim_5_1_pegasus"
ISAAC_ROS2_PATH="$ISAAC_SIM_ROOT/exts/isaacsim.ros2.bridge/jazzy"

# Verify Isaac Sim installation
if [ ! -d "$ISAAC_SIM_ROOT" ]; then
    echo "❌ ERROR: Isaac Sim not found at $ISAAC_SIM_ROOT"
    exit 1
fi

if [ ! -d "$ISAAC_ROS2_PATH" ]; then
    echo "❌ ERROR: Isaac Sim's internal ROS2 not found"
    exit 1
fi

echo "[Isaac Sim] Starting with internal ROS2 (CycloneDDS)..."

# Change to script directory
cd "$(dirname "$0")"

# Run using Isaac Sim's Python wrapper (Python 3.11)
# Use env -i to start with a completely clean environment, then add only what we need
env -i \
    HOME="$HOME" \
    USER="$USER" \
    PATH="$PATH" \
    DISPLAY="$DISPLAY" \
    TERM="$TERM" \
    ROS_DISTRO="jazzy" \
    RMW_IMPLEMENTATION="rmw_cyclonedds_cpp" \
    LD_LIBRARY_PATH="$ISAAC_ROS2_PATH/lib" \
    "$ISAAC_SIM_ROOT/python.sh" 1_px4_single_vehicle_copy.py
