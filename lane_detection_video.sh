#!/bin/bash

set -e

MODEL_PATH="rl-baselines3-zoo/rl-trained-agents/sac/donkey-generated-track-v0.zip"

if [ ! -f "$MODEL_PATH" ]; then
    echo "Pre-trained agents not found at $MODEL_PATH." >&2
    if [ ! -d "rl-baselines3-zoo/rl-trained-agents" ]; then
        echo "Cloning rl-trained-agents repository..." >&2
        git clone https://github.com/DLR-RM/rl-trained-agents rl-baselines3-zoo/rl-trained-agents
    fi
    ln -sf rl-baselines3-zoo/rl-trained-agents
fi

source venv/bin/activate
python -m rl_zoo3.record_video --algo sac \
    --env donkey-generated-track-v0 \
    --env-kwargs lane_detection:True \
    --n-timesteps 1000 \
    --output-folder videos
