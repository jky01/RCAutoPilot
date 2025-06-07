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

sudo ./setup_software.sh

echo "Environment ready and activated. Use 'deactivate' to exit."
