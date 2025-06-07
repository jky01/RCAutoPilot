#!/bin/bash
# Create Python virtual environment and install dependencies.
# Source this script to keep the environment active.
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}" )" && pwd)"
cd "$SCRIPT_DIR"

if [ ! -d "venv" ]; then
    python3 -m venv venv
fi

# Activate the environment
# shellcheck disable=SC1091
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install system packages (may require sudo)
apt-get update
apt-get install -y swig cmake ffmpeg

# Install Python dependencies
pip install -r rl-baselines3-zoo/requirements.txt
pip install -e rl-baselines3-zoo
pip install -e gym-donkeycar

echo "Environment ready and activated. Use 'deactivate' to exit."
