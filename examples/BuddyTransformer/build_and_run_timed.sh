#!/bin/bash
# ===- build_and_run_timed.sh ----------------------------------------------
#
# Script to build and run the timed transformer executable
#
# ===---------------------------------------------------------------------------

set -e  # Exit on error

# Get the script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
BUDDY_ROOT="$SCRIPT_DIR/../.."
BUILD_DIR="$BUDDY_ROOT/build"

echo "========================================="
echo "  Building Timed Transformer Executable"
echo "========================================="
echo ""
echo "Buddy-MLIR Root: $BUDDY_ROOT"
echo "Build Directory: $BUILD_DIR"
echo ""

# Step 1: Reconfigure CMake to pick up new targets
echo "Reconfiguring CMake..."
cd "$BUILD_DIR"
cmake .. -G Ninja \
  -DBUDDY_TRANSFORMER_EXAMPLES=ON \
  -DCMAKE_BUILD_TYPE=Release
echo "✓ CMake reconfigured"
echo ""

# Step 2: Build the project
echo "Building the project..."
ninja buddy-transformer-timed-executable
echo "✓ Build completed"
echo ""

# Step 3: Run the timed executable
echo "Running timed transformer executable..."
echo "========================================="
echo ""
"$BUILD_DIR/bin/transformer-runner-timed"
echo ""
echo "✓ Generated profile artifacts during execution"
echo "  Full heatmap: $BUILD_DIR/examples/BuddyTransformer/subgraph0_profile.svg"
echo "  Module drill-down: $BUILD_DIR/examples/BuddyTransformer/subgraph0_module_hierarchy.svg"
