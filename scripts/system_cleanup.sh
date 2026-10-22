#!/usr/bin/env bash
# ==============================================================================
# system_cleanup.sh: Storage Scrubber for Low-Capacity MicroSD Disks
# Recovers ~6-8 GB of space by clearing caches, build objects, and logs
# ==============================================================================

set -euo pipefail

echo "==> Initial Disk Status:"
df -h /

echo ""
echo "==> [1/4] Purging pip package download cache..."
rm -rf ~/.cache/pip
rm -rf ~/.cache/huggingface/hub/tmp* 2>/dev/null || true

echo "==> [2/4] Pruning CMake intermediate build object files in llama.cpp..."
if [ -d "$HOME/llama.cpp/build" ]; then
    rm -rf "$HOME/llama.cpp/build/CMakeFiles/"
    rm -rf "$HOME/llama.cpp/build/ggml/src/CMakeFiles/"
fi

echo "==> [3/4] Cleaning APT local repository cache..."
sudo apt-get clean
sudo apt-get autoremove -y

echo "==> [4/4] Vacuuming systemd journal logs to max 100 MB..."
sudo journalctl --vacuum-size=100M

echo ""
echo "==> Post-Cleanup Disk Status:"
df -h /
echo "======================================================================="
echo "Storage recovery complete. Verify python temp directory works:"
python3 -c "import tempfile; print('Tempdir working:', tempfile.gettempdir())"
echo "======================================================================="
