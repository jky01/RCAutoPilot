# RCAutoPilot

This repository bundles [RL Baselines3 Zoo](https://github.com/DLR-RM/rl-baselines3-zoo), [Stable Baselines3](https://github.com/DLR-RM/stable-baselines3) and [gym-donkeycar](https://github.com/tawnkramer/gym-donkeycar) to train autonomous driving agents.

## Quick Start

1. **Install dependencies** (recommended inside a virtual environment):

   ```bash
   apt-get install swig cmake ffmpeg
   pip install -r rl-baselines3-zoo/requirements.txt
   pip install -e rl-baselines3-zoo
   pip install -e gym-donkeycar
   ```

   Optionally install extra tools for plotting and tests:

   ```bash
   pip install -e rl-baselines3-zoo[plots,tests]
   ```

2. **Launch the DonkeyCar simulator**. If you have the simulator installed, start it manually or via script before training.

3. **Run training** using the helper script:

   ```bash
   ./start_training.sh
   ```

   The script runs RL Baselines3 Zoo with the SAC algorithm on the `donkey-generated-track-v0` environment. Additional arguments will be forwarded to `train.py`.

