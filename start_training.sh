#!/bin/bash
# Start RL Baselines3 Zoo training for DonkeyCar
# Usage: ./start_training.sh [additional train.py arguments]

set -e

# Move to the rl-baselines3-zoo directory
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/rl-baselines3-zoo"


# Ensure the local gym-donkeycar package is on the Python path so
# its environments get registered even if it has not been installed.
export PYTHONPATH="$SCRIPT_DIR/gym-donkeycar${PYTHONPATH:+:$PYTHONPATH}"

# Check that gymnasium is installed. The RL Zoo and gym-donkeycar
# environments rely on it and wrappers are optional.
if ! python - <<'EOF' >/dev/null 2>&1
import gymnasium
EOF
then
    echo "Error: gymnasium is not installed. Please run 'source ./setup_env.sh'" >&2
    exit 1
fi

python train.py --algo sac --env donkey-generated-track-v0 --gym-packages gym_donkeycar \
       --eval-freq 10000 --eval-episodes 10 --n-eval-envs 1 "$@"
