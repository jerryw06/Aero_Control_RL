#!/bin/bash

# Complete installation and first-run guide for C++ RL Training

echo "=============================================="
echo "C++ RL Training - Complete Setup Guide"
echo "=============================================="
echo ""

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}This script will guide you through the complete setup process.${NC}"
echo ""

# Step 1: Check prerequisites
echo -e "${YELLOW}Step 1: Checking prerequisites...${NC}"
echo ""

# Check ROS2
if [ -f "/opt/ros/jazzy/setup.bash" ]; then
    echo -e "${GREEN}✓ ROS2 Jazzy found${NC}"
else
    echo -e "${RED}✗ ROS2 Jazzy not found${NC}"
    echo "Please install ROS2 Jazzy first: https://docs.ros.org/en/jazzy/Installation.html"
    exit 1
fi

# Check if we're in the right directory
if [ ! -f "CMakeLists.txt" ] || [ ! -f "package.xml" ]; then
    echo -e "${RED}✗ Please run this script from the RL_with_cpp directory${NC}"
    exit 1
fi

echo -e "${GREEN}✓ In correct directory${NC}"
echo ""

# Step 2: Run setup
echo -e "${YELLOW}Step 2: Running automated setup...${NC}"
echo ""
if [ -f "setup.sh" ]; then
    chmod +x setup.sh
    ./setup.sh
else
    echo -e "${RED}✗ setup.sh not found${NC}"
    exit 1
fi

echo ""
echo -e "${YELLOW}Step 3: px4_msgs verification (already handled by setup.sh)${NC}"
if ros2 pkg list | grep -q "px4_msgs"; then
  echo -e "${GREEN}✓ px4_msgs present${NC}"
else
  echo -e "${YELLOW}Warning: px4_msgs still missing; continuing (training will fail until present).${NC}"
fi

# Step 4: Build the C++ package
echo ""
echo -e "${YELLOW}Step 4: Building C++ RL package...${NC}"
echo ""

# Go back to script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Source (may be absent if px4_msgs auto-install failed)
if [ -f "$SCRIPT_DIR/setup_env.sh" ]; then
    source "$SCRIPT_DIR/setup_env.sh"
else
    echo -e "${YELLOW}setup_env.sh missing – regenerating via setup.sh...${NC}"
    bash "$SCRIPT_DIR/setup.sh"
    source "$SCRIPT_DIR/setup_env.sh"
fi

# Create / refresh local workspace under repository root
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
WORKSPACE_ROOT="$REPO_ROOT/ros2_local_ws"
mkdir -p "$WORKSPACE_ROOT/src"
if [ ! -e "$WORKSPACE_ROOT/src/rl_with_cpp" ]; then
    ln -s "$SCRIPT_DIR" "$WORKSPACE_ROOT/src/rl_with_cpp"
fi
cd "$WORKSPACE_ROOT"
echo "Building in local workspace: $WORKSPACE_ROOT"

# Try selective build; if package not discoverable, build all then verify
if ! colcon list 2>/dev/null | grep -q '^rl_with_cpp'; then
    echo -e "${YELLOW}Package rl_with_cpp not discoverable via colcon list – attempting full build then retry...${NC}"
    colcon build --cmake-args -DCMAKE_BUILD_TYPE=Release || { echo -e "${RED}Full build failed${NC}"; exit 1; }
else
    colcon build --packages-select rl_with_cpp --cmake-args -DCMAKE_BUILD_TYPE=Release || { echo -e "${RED}Selective build failed${NC}"; exit 1; }
fi

if ! colcon list 2>/dev/null | grep -q '^rl_with_cpp'; then
    echo -e "${RED}rl_with_cpp still not found after build. Workspace structure:"${NC}
    find "$WORKSPACE_ROOT/src" -maxdepth 2 -type f -name package.xml -print
    echo -e "${RED}Aborting. Please ensure package.xml exists at $SCRIPT_DIR/package.xml${NC}"
    exit 1
fi

if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}✓ Build successful!${NC}"
else
    echo ""
    echo -e "${RED}✗ Build failed${NC}"
    echo "Please check the error messages above."
    exit 1
fi

# Step 5: Create convenience scripts
echo ""
echo -e "${YELLOW}Step 5: Creating convenience scripts...${NC}"

cd "$SCRIPT_DIR"

# Create run script
cat > run_training.sh << 'EOF'
#!/bin/bash
# Convenience script to run C++ RL training

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
WORKSPACE_ROOT="$REPO_ROOT/ros2_local_ws"

# Source ROS2
source /opt/ros/jazzy/setup.bash

# Source px4_msgs if available
if [ -f "$HOME/ros2_ws/install/setup.bash" ]; then
    source "$HOME/ros2_ws/install/setup.bash"
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

ros2 run rl_with_cpp train_rl
EOF

chmod +x run_training.sh
echo -e "${GREEN}✓ Created run_training.sh${NC}"

# Final instructions
echo ""
echo -e "${GREEN}=============================================="
echo "Setup Complete!"
echo "==============================================${NC}"
echo ""
echo -e "${BLUE}Before running training, make sure:${NC}"
echo "  1. PX4 SITL is running"
echo "  2. MicroXRCEAgent is running"
echo "  3. The drone is spawned in the simulation"
echo ""
echo -e "${BLUE}To run training:${NC}"
echo "  ${YELLOW}cd $SCRIPT_DIR${NC}"
echo "  ${YELLOW}./run_training.sh${NC}"
echo ""
echo -e "${BLUE}Or manually:${NC}"
echo "  ${YELLOW}source $SCRIPT_DIR/setup_env.sh${NC}"
echo "  ${YELLOW}cd $WORKSPACE_ROOT${NC}"
echo "  ${YELLOW}source install/setup.bash${NC}"
echo "  ${YELLOW}ros2 run rl_with_cpp train_rl${NC}"
echo ""
echo -e "${GREEN}Good luck with your training!${NC}"
