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

LANE_CFG="${LANE_CFG:-$SCRIPT_DIR/CLRNet/configs/clrnet/clr_dla34_culane.py}"
LANE_CKPT="${LANE_CKPT:-$SCRIPT_DIR/culane_dla34.pth}"
# Pass wrapper path as a string so the hyperparameters parser does not attempt
# to evaluate it as Python code.
EXTRA_ARGS="--hyperparams env_wrapper:'\''rl_zoo3.lane_detection_wrapper.CLRLaneDetectionWrapper'\'' --env-kwargs config_path:'$LANE_CFG' checkpoint_path:'$LANE_CKPT'"

python train.py --algo sac --env donkey-generated-track-v0 --gym-packages gym_donkeycar \
       --eval-freq 10000 --eval-episodes 10 --n-eval-envs 1 $EXTRA_ARGS "$@"
