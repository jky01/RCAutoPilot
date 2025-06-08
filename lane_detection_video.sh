source ./setup_env.sh  # 啟動虛擬環境
python -m rl_zoo3.record_video --algo sac \
    --env donkey-generated-track-v0 \
    --gym-packages gym_donkeycar \
    --env-kwargs lane_detection:True \
    --n-timesteps 1000 \
    --output-folder videos
