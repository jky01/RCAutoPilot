#!/bin/bash
# Start RL Baselines3 Zoo training for DonkeyCar
# Usage: ./start_training.sh [additional train.py arguments]

set -e

# Move to the rl-baselines3-zoo directory
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/rl-baselines3-zoo"

python train.py --algo sac --env donkey-generated-track-v0 --gym-packages gym_donkeycar \
       --eval-freq 10000 --eval-episodes 10 --n-eval-envs 1 "$@"
