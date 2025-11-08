#!/bin/bash
# Convenience script to run C++ RL training

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Resolve repo root and local workspace
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
WORKSPACE_ROOT="$REPO_ROOT/ros2_local_ws"

# Select RMW implementation with fallback:
# - Prefer CycloneDDS if available (avoids Fast DDS history payload issues)
# - Else fallback to Fast DDS with a profile that allows realloc
choose_rmw() {
    # If user already set RMW, honor it but verify availability for CycloneDDS
    if [ -n "$RMW_IMPLEMENTATION" ]; then
        if [ "$RMW_IMPLEMENTATION" = "rmw_cyclonedds_cpp" ]; then
            if ! ldconfig -p 2>/dev/null | grep -q librmw_cyclonedds_cpp && [ ! -f "/opt/ros/jazzy/lib/librmw_cyclonedds_cpp.so" ]; then
                echo "[INFO] CycloneDDS requested but not installed. Falling back to Fast DDS with realloc profile."
                export RMW_IMPLEMENTATION="rmw_fastrtps_cpp"
                export FASTDDS_DEFAULT_PROFILES_FILE="$SCRIPT_DIR/fastdds_profile.xml"
            fi
        fi
        return
    fi

    # Auto-select when RMW not set
    if ldconfig -p 2>/dev/null | grep -q librmw_cyclonedds_cpp || [ -f "/opt/ros/jazzy/lib/librmw_cyclonedds_cpp.so" ]; then
        export RMW_IMPLEMENTATION="rmw_cyclonedds_cpp"
    else
        export RMW_IMPLEMENTATION="rmw_fastrtps_cpp"
        export FASTDDS_DEFAULT_PROFILES_FILE="$SCRIPT_DIR/fastdds_profile.xml"
    fi
}

choose_rmw

# Source ROS2
source /opt/ros/jazzy/setup.bash

# Source px4_msgs if available (user or local)
if [ -f "$HOME/ros2_ws/install/setup.bash" ]; then
    source "$HOME/ros2_ws/install/setup.bash"
fi
if [ -f "$HOME/px4_ws/install/setup.bash" ]; then
    source "$HOME/px4_ws/install/setup.bash"
fi
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

echo "Starting C++ RL Training..."
echo "Make sure PX4 SITL and MicroXRCEAgent are running!"
echo ""
echo "RMW_IMPLEMENTATION=$RMW_IMPLEMENTATION"
if [ -n "$FASTDDS_DEFAULT_PROFILES_FILE" ]; then
    echo "FASTDDS_DEFAULT_PROFILES_FILE=$FASTDDS_DEFAULT_PROFILES_FILE"
fi

# Quick sanity check that PX4 topics are present
if ! ros2 topic list 2>/dev/null | grep -q "/fmu/vehicle_local_position/out"; then
    echo "[WARN] PX4 topic /fmu/vehicle_local_position/out not detected."
    echo "       Ensure PX4 SITL and micro XRCE-DDS agent are running, then retry."
    # Proceed anyway, trainer will wait, but this helps diagnose earlier.
fi

ros2 run rl_with_cpp train_rl
