#!/bin/bash

set -e

# Use the model generated during training located in the logs folder
FOLDER="rl-baselines3-zoo/logs"

if [ ! -d "$FOLDER" ]; then
    echo "Error: log directory '$FOLDER' does not exist. Please run start_training.sh first." >&2
    exit 1
fi

source venv/bin/activate
python -m rl_zoo3.record_video --algo sac \
    --env donkey-generated-track-v0 \
    --folder "$FOLDER" \
    --env-kwargs lane_detection:True \
    --n-timesteps 1000 \
    --output-folder videos
