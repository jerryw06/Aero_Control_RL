#!/bin/bash

# Verification script to check that all files are present and correct

echo "╔══════════════════════════════════════════════════════════╗"
echo "║     C++ RL Training - File Verification Script          ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Color codes
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

ERRORS=0
WARNINGS=0

echo "Checking files..."
echo ""

# Core implementation files
FILES=(
    "include/px4_node.hpp"
    "include/px4_accel_env.hpp"
    "include/policy_network.hpp"
    "src/px4_node.cpp"
    "src/px4_accel_env.cpp"
    "src/policy_network.cpp"
    "src/train_rl.cpp"
    "CMakeLists.txt"
    "package.xml"
)

# Documentation files
DOCS=(
    "README.md"
    "QUICKSTART.md"
    "IMPLEMENTATION_SUMMARY.md"
    "GETTING_STARTED.txt"
)

# Script files
SCRIPTS=(
    "setup.sh"
    "build.sh"
    "install_and_setup.sh"
)

echo "Core Implementation Files:"
for file in "${FILES[@]}"; do
    if [ -f "$file" ]; then
        SIZE=$(wc -l < "$file")
        echo -e "  ${GREEN}✓${NC} $file ($SIZE lines)"
    else
        echo -e "  ${RED}✗${NC} $file (MISSING)"
        ((ERRORS++))
    fi
done

echo ""
echo "Documentation Files:"
for file in "${DOCS[@]}"; do
    if [ -f "$file" ]; then
        SIZE=$(wc -l < "$file")
        echo -e "  ${GREEN}✓${NC} $file ($SIZE lines)"
    else
        echo -e "  ${RED}✗${NC} $file (MISSING)"
        ((ERRORS++))
    fi
done

echo ""
echo "Setup Scripts:"
for file in "${SCRIPTS[@]}"; do
    if [ -f "$file" ]; then
        if [ -x "$file" ]; then
            echo -e "  ${GREEN}✓${NC} $file (executable)"
        else
            echo -e "  ${YELLOW}⚠${NC} $file (exists but not executable)"
            ((WARNINGS++))
        fi
    else
        echo -e "  ${RED}✗${NC} $file (MISSING)"
        ((ERRORS++))
    fi
done

echo ""
echo "─────────────────────────────────────────────────────────"
echo "File Statistics:"
echo "─────────────────────────────────────────────────────────"

TOTAL_FILES=$(find . -type f ! -path './.git/*' | wc -l)
TOTAL_LINES=$(find . -name "*.cpp" -o -name "*.hpp" | xargs wc -l | tail -1 | awk '{print $1}')
HEADER_FILES=$(find include -name "*.hpp" | wc -l)
SOURCE_FILES=$(find src -name "*.cpp" | wc -l)

echo "  Total files: $TOTAL_FILES"
echo "  Total C++ lines of code: $TOTAL_LINES"
echo "  Header files: $HEADER_FILES"
echo "  Source files: $SOURCE_FILES"
echo ""

echo "─────────────────────────────────────────────────────────"
echo "Prerequisites Check:"
echo "─────────────────────────────────────────────────────────"

# Check ROS2
if [ -f "/opt/ros/jazzy/setup.bash" ]; then
    echo -e "  ${GREEN}✓${NC} ROS2 Jazzy installed"
else
    echo -e "  ${RED}✗${NC} ROS2 Jazzy not found"
    ((ERRORS++))
fi

# Check for CMake
if command -v cmake &> /dev/null; then
    CMAKE_VERSION=$(cmake --version | head -1)
    echo -e "  ${GREEN}✓${NC} CMake available ($CMAKE_VERSION)"
else
    echo -e "  ${RED}✗${NC} CMake not found"
    ((ERRORS++))
fi

# Check for g++
if command -v g++ &> /dev/null; then
    GCC_VERSION=$(g++ --version | head -1)
    echo -e "  ${GREEN}✓${NC} G++ available ($GCC_VERSION)"
else
    echo -e "  ${RED}✗${NC} G++ not found"
    ((ERRORS++))
fi

echo ""
echo "─────────────────────────────────────────────────────────"
echo "Summary:"
echo "─────────────────────────────────────────────────────────"

if [ $ERRORS -eq 0 ] && [ $WARNINGS -eq 0 ]; then
    echo -e "${GREEN}✓ All checks passed!${NC}"
    echo ""
    echo "You're ready to proceed with installation:"
    echo "  $ ./install_and_setup.sh"
    echo ""
elif [ $ERRORS -eq 0 ]; then
    echo -e "${YELLOW}⚠ $WARNINGS warning(s) found${NC}"
    echo ""
    echo "Fix warnings by running:"
    echo "  $ chmod +x setup.sh build.sh install_and_setup.sh"
    echo ""
else
    echo -e "${RED}✗ $ERRORS error(s) found${NC}"
    if [ $WARNINGS -gt 0 ]; then
        echo -e "${YELLOW}⚠ $WARNINGS warning(s) found${NC}"
    fi
    echo ""
    echo "Please ensure all files are present before proceeding."
    exit 1
fi

echo "╔══════════════════════════════════════════════════════════╗"
echo "║              Verification Complete                       ║"
echo "╚══════════════════════════════════════════════════════════╝"
