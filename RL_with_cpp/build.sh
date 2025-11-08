#!/bin/bash

# Quick build script for C++ RL Training

set -e

echo "Building C++ RL Training Package..."

# Get the workspace root (assuming we're in RL_with_cpp folder)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
WORKSPACE_ROOT="$SCRIPT_DIR/../ros2_local_ws"
mkdir -p "$WORKSPACE_ROOT/src"
if [ ! -d "$WORKSPACE_ROOT/src/rl_with_cpp" ]; then
    ln -s "$SCRIPT_DIR" "$WORKSPACE_ROOT/src/rl_with_cpp"
fi

# Source environment
if [ -f "$SCRIPT_DIR/setup_env.sh" ]; then
    source "$SCRIPT_DIR/setup_env.sh"
else
        echo "setup_env.sh missing – running setup.sh to create it"; bash "$SCRIPT_DIR/setup.sh"; source "$SCRIPT_DIR/setup_env.sh"
fi

cd "$WORKSPACE_ROOT"
echo "Building in workspace: $(pwd)"
colcon build --packages-select rl_with_cpp --cmake-args -DCMAKE_BUILD_TYPE=Release

echo ""
echo "Build complete!"
echo ""
echo "To run the training:"
echo "  source install/setup.bash"
echo "  ros2 run rl_with_cpp train_rl"
