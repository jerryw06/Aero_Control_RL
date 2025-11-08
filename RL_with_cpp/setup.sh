#!/bin/bash

# Setup script for C++ RL Training with LibTorch and ROS2 Jazzy
# This script will download LibTorch and install necessary dependencies

set -e

echo "=========================================="
echo "C++ RL Training Setup Script"
echo "=========================================="

# Color codes for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check if running on Linux
if [[ "$OSTYPE" != "linux-gnu"* ]]; then
    echo -e "${RED}Error: This script is designed for Linux only${NC}"
    exit 1
fi

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Step 1: Check ROS2 installation
echo -e "\n${YELLOW}Step 1: Checking ROS2 Jazzy installation...${NC}"
if [ -f "/opt/ros/jazzy/setup.bash" ]; then
    echo -e "${GREEN}✓ ROS2 Jazzy found${NC}"
    source /opt/ros/jazzy/setup.bash
else
    echo -e "${RED}✗ ROS2 Jazzy not found. Please install ROS2 Jazzy first.${NC}"
    echo "Visit: https://docs.ros.org/en/jazzy/Installation.html"
    exit 1
fi

# Step 2: Install system dependencies
echo -e "\n${YELLOW}Step 2: Installing system dependencies...${NC}"
sudo apt-get update
sudo apt-get install -y \
    build-essential \
    cmake \
    git \
    wget \
    unzip \
    python3-pip \
    libeigen3-dev

echo -e "${GREEN}✓ System dependencies installed${NC}"

# Step 3: Download LibTorch
echo -e "\n${YELLOW}Step 3: Downloading LibTorch (C++ PyTorch)...${NC}"
LIBTORCH_DIR="$SCRIPT_DIR/libtorch"

if [ -d "$LIBTORCH_DIR" ]; then
    echo -e "${YELLOW}LibTorch directory already exists. Skipping download.${NC}"
else
    # Download LibTorch (CPU version, pre-cxx11 ABI for ROS2 compatibility)
    LIBTORCH_URL="https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.1.0%2Bcpu.zip"
    
    echo "Downloading from: $LIBTORCH_URL"
    echo "This is ~200MB and may take a few minutes..."
    
    wget -O libtorch.zip "$LIBTORCH_URL"
    
    echo "Extracting LibTorch..."
    unzip -q libtorch.zip
    rm libtorch.zip
    
    echo -e "${GREEN}✓ LibTorch downloaded and extracted${NC}"
fi

# Step 4: Install nlohmann/json for JSON handling
echo -e "\n${YELLOW}Step 4: Installing nlohmann/json library...${NC}"
JSON_INCLUDE_DIR="/usr/local/include/nlohmann"
if [ -d "$JSON_INCLUDE_DIR" ]; then
    echo -e "${YELLOW}nlohmann/json already installed. Skipping.${NC}"
else
    cd /tmp
    git clone https://github.com/nlohmann/json.git
    cd json
    mkdir -p build
    cd build
    cmake ..
    sudo make install
    cd "$SCRIPT_DIR"
    echo -e "${GREEN}✓ nlohmann/json installed${NC}"
fi

# Step 5: Ensure px4_msgs (auto-install if missing)
echo -e "\n${YELLOW}Step 5: Ensuring px4_msgs package...${NC}"
if ros2 pkg list | grep -q "px4_msgs"; then
    echo -e "${GREEN}✓ px4_msgs package found${NC}"
else
    echo -e "${YELLOW}px4_msgs not found – attempting automatic install...${NC}"
    PX4_WS="$SCRIPT_DIR/px4_msgs_ws"
    mkdir -p "$PX4_WS/src"
    if [ ! -d "$PX4_WS/src/px4_msgs" ]; then
        git clone https://github.com/PX4/px4_msgs.git "$PX4_WS/src/px4_msgs"
    fi
    cd "$PX4_WS"
    colcon build --packages-select px4_msgs || {
        echo -e "${RED}Failed to build px4_msgs automatically. Please install manually.${NC}"
    }
    # Source if built
    if [ -f "$PX4_WS/install/setup.bash" ]; then
        source "$PX4_WS/install/setup.bash"
        echo -e "${GREEN}✓ px4_msgs installed locally at $PX4_WS${NC}"
    fi
    cd "$SCRIPT_DIR"
fi

# Step 6: Create / refresh environment setup script (always create even if px4_msgs missing)
echo -e "\n${YELLOW}Step 6: Creating environment setup script...${NC}"
cat > setup_env.sh << 'EOF'
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
EOF

chmod +x setup_env.sh

echo -e "${GREEN}✓ Environment setup script created${NC}"

# Step 7: Instructions for building
echo -e "\n${GREEN}=========================================="
echo "Setup Complete!"
echo "==========================================${NC}"
echo ""
echo "Next steps:"
echo ""
echo "1. Build px4_msgs if you haven't already:"
echo "   ${YELLOW}cd ~/ros2_ws/src${NC}"
echo "   ${YELLOW}git clone https://github.com/PX4/px4_msgs.git${NC}"
echo "   ${YELLOW}cd ..${NC}"
echo "   ${YELLOW}colcon build --packages-select px4_msgs${NC}"
echo ""
echo "2. Source the environment setup:"
echo "   ${YELLOW}source $SCRIPT_DIR/setup_env.sh${NC}"
echo ""
echo "3. Build this package:"
echo "   ${YELLOW}cd $SCRIPT_DIR/../..${NC}"
echo "   ${YELLOW}colcon build --packages-select rl_with_cpp${NC}"
echo ""
echo "4. Run the training:"
echo "   ${YELLOW}source install/setup.bash${NC}"
echo "   ${YELLOW}ros2 run rl_with_cpp train_rl${NC}"
echo ""
echo -e "${GREEN}Happy training!${NC}"
