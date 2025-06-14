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
else
    sudo chown -R "$OWNER":"$OWNER" rl-baselines3-zoo CLRNet
fi
