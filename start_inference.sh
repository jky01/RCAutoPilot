#!/bin/bash
# Run a trained agent with lane detection enabled using CLRNet
# Usage: ./start_inference.sh [enjoy.py arguments]

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

source venv/bin/activate
cd "$SCRIPT_DIR/rl-baselines3-zoo"

# Ensure gym-donkeycar is available
export PYTHONPATH="$SCRIPT_DIR/gym-donkeycar${PYTHONPATH:+:$PYTHONPATH}"

LANE_CFG="${LANE_CFG:-$SCRIPT_DIR/CLRNet/configs/clrnet/clr_dla34_culane.py}"
LANE_CKPT="${LANE_CKPT:-$SCRIPT_DIR/culane_dla34.pth}"
LANE_CAPTURE_DIR="${LANE_CAPTURE_DIR:-lane_captures}"
export LANE_CFG
export LANE_CKPT
export LANE_CAPTURE_DIR
mkdir -p "$LANE_CAPTURE_DIR"

EXTRA_ARGS=(
  --hyperparams
  env_wrapper:"'rl_zoo3.lane_detection_wrapper.CLRLaneDetectionWrapper'"
)

python enjoy.py --algo sac --env donkey-generated-track-v0 --gym-packages gym_donkeycar "${EXTRA_ARGS[@]}" "$@"
