#!/bin/bash
# Install system packages (may require sudo)
set -e

apt-get update
apt-get install -y swig cmake ffmpeg

# Ensure the rl-baselines3-zoo directory is owned by the invoking user
OWNER="${SUDO_USER:-$USER}"
if [ "$(id -u)" -eq 0 ]; then
    chown -R "$OWNER":"$OWNER" rl-baselines3-zoo
else
    sudo chown -R "$OWNER":"$OWNER" rl-baselines3-zoo
fi
