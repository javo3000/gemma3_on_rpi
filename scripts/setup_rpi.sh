#!/usr/bin/env bash
# ==============================================================================
# setup_rpi.sh: Automated Provisioning for llama.cpp & Gemma 3 on Raspberry Pi
# ==============================================================================

set -euo pipefail

echo "==> [1/5] Updating system packages and installing build dependencies..."
sudo apt-get update
sudo apt-get install -y \
    build-essential \
    cmake \
    git \
    curl \
    python3 \
    python3-pip \
    python3-venv \
    libgomp1

echo "==> [2/5] Configuring Swap Space (Ensures 2GB headroom against OOM peaks)..."
if grep -q "CONF_SWAPSIZE" /etc/dphys-swapfile 2>/dev/null; then
    sudo sed -i 's/^CONF_SWAPSIZE=.*/CONF_SWAPSIZE=2048/' /etc/dphys-swapfile
    sudo systemctl restart dphys-swapfile || true
    echo "Swap configured to 2048 MB."
fi

echo "==> [3/5] Cloning and Compiling llama.cpp with ARM NEON SIMD optimizations..."
LLAMA_DIR="$HOME/llama.cpp"
if [ ! -d "$LLAMA_DIR" ]; then
    git clone https://github.com/ggerganov/llama.cpp "$LLAMA_DIR"
fi

cd "$LLAMA_DIR"
cmake -B build -DGGML_NATIVE=ON
cmake --build build --config Release -j4 --target llama-mtmd-cli llama-cli

echo "==> [4/5] Pruning intermediate CMake build objects to save SD card space..."
rm -rf build/CMakeFiles/
rm -rf build/ggml/src/CMakeFiles/

echo "==> [5/5] Creating dedicated project directories..."
mkdir -p "$HOME/models"
mkdir -p "$HOME/datasets"
mkdir -p "$HOME/predictions"

echo "======================================================================="
echo "Setup Complete! llama-mtmd-cli binary is ready at: $LLAMA_DIR/build/bin/llama-mtmd-cli"
echo "Next step: Run 'curl' to download Gemma 3 4B Q4_K_M and mmproj models."
echo "======================================================================="
