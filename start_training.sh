#!/bin/bash
# Start RL Baselines3 Zoo training for DonkeyCar
# Usage: ./start_training.sh [additional train.py arguments]

set -e

# Move to the rl-baselines3-zoo directory
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

cd "$SCRIPT_DIR"

source venv/bin/activate

cd "$SCRIPT_DIR/rl-baselines3-zoo"
# Launch DonkeyCar simulator if available
SIM_PATH="$HOME/DonkeySimLinux/donkey_sim.x86_64"
if [ -x "$SIM_PATH" ]; then
    "$SIM_PATH" &
    SIM_PID=$!
    trap '[ -n "$SIM_PID" ] && kill $SIM_PID' EXIT
    sleep 2
fi



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

EXTRA_ARGS=""
if [ -n "$LANE_CFG" ] && [ -n "$LANE_CKPT" ]; then
    EXTRA_ARGS="--env-wrapper rl_zoo3.lane_detection_wrapper.CLRLaneDetectionWrapper --env-kwargs config_path=$LANE_CFG checkpoint_path=$LANE_CKPT"
fi

python train.py --algo sac --env donkey-generated-track-v0 --gym-packages gym_donkeycar \
       --eval-freq 10000 --eval-episodes 10 --n-eval-envs 1 $EXTRA_ARGS "$@"
