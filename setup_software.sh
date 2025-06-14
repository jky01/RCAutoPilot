#!/bin/bash
# Install system packages (may require sudo)
set -e

apt-get update
apt-get install -y swig cmake ffmpeg

# Ensure required repositories are owned by the invoking user so editable
# installs succeed even if previous steps were run with sudo
OWNER="${SUDO_USER:-$USER}"
if [ "$(id -u)" -eq 0 ]; then
    chown -R "$OWNER":"$OWNER" rl-baselines3-zoo CLRNet
    # Clean up any previous build artifacts that might cause install issues
    find CLRNet -name '*.egg-info' -exec rm -rf {} + || true
    find CLRNet -name 'build' -type d -exec rm -rf {} + || true
    find CLRNet -name 'dist' -type d -exec rm -rf {} + || true
    find . -maxdepth 1 -iname 'clrnet*.egg-info' -exec rm -rf {} + || true
else
    sudo chown -R "$OWNER":"$OWNER" rl-baselines3-zoo CLRNet
    sudo find CLRNet -name '*.egg-info' -exec rm -rf {} + || true
    sudo find CLRNet -name 'build' -type d -exec rm -rf {} + || true
    sudo find CLRNet -name 'dist' -type d -exec rm -rf {} + || true
    sudo find . -maxdepth 1 -iname 'clrnet*.egg-info' -exec rm -rf {} + || true
fi
