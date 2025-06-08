source venv/bin/activate
python -m rl_zoo3.record_video --algo sac \
    --env donkey-generated-track-v0 \
    --env-kwargs lane_detection:True \
    --n-timesteps 1000 \
    --output-folder videos
