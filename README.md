# RCAutoPilot

This repository bundles [RL Baselines3 Zoo](https://github.com/DLR-RM/rl-baselines3-zoo), [Stable Baselines3](https://github.com/DLR-RM/stable-baselines3) and [gym-donkeycar](https://github.com/tawnkramer/gym-donkeycar) to train autonomous driving agents.

## Quick Start

1. **Create a Python environment and install dependencies**:

   Run the setup script which creates a virtual environment in `venv`, activates
   it and installs all required packages.

   ```bash
   source ./setup_env.sh
   ```

   Optionally install extra tools for plotting and tests after the environment
   is activated:

   ```bash
   pip install -e rl-baselines3-zoo[plots,tests]
   ```

2. **Launch the DonkeyCar simulator**. If you have the simulator installed, start it manually or via script before training.

3. **(Optional) Download the pre-trained agents** used by the example scripts.
   The RL Zoo references them as a Git submodule, but this repository includes
   the RL Zoo code directly without its Git metadata. To obtain the agents,
   clone the repository manually into the expected location:

   ```bash
   git clone https://github.com/DLR-RM/rl-trained-agents \
       rl-baselines3-zoo/rl-trained-agents
   ln -sf rl-baselines3-zoo/rl-trained-agents 
   ```

   This step is required if you wish to run scripts such as
   `lane_detection_video.sh` without having trained an agent yourself.

4. **Run training** using the helper script:

   ```bash
   ./start_training.sh
   ```

   The script runs RL Baselines3 Zoo with the SAC algorithm on the `donkey-generated-track-v0` environment. Additional arguments will be forwarded to `train.py`.

   To run the HuggingFace Transformer example, pass the `--transformer` flag (the setup script installs the required `transformers` package):

   ```bash
   ./start_training.sh --transformer
   ```

  By default, the example trains continuously until interrupted. Pass
  `--epochs <n>` after `--transformer` to train for a fixed number of epochs.
  You can also save and resume training with `--checkpoint <file>` which stores
  a checkpoint every epoch (change the interval with `--checkpoint-freq`). If an
  existing file contains only model weights, the script will still load them but
  the optimizer state will start from scratch. The model weights are periodically saved to
  `rl-baselines3-zoo/donkey_transformer.pt` (relative to the repository root)
  by default. Use `--model-path` to change the location and `--save-freq` to
  control the saving interval.
  When training starts, the script automatically reloads weights from the last
  saved file if present. Press `Ctrl+C` at any time to stop training; if
  `--checkpoint` is specified, a final checkpoint is written before exiting.

   If you encounter an error about `donkey-generated-track-v0` not being found,
   ensure the `gym-donkeycar` package is available. The provided scripts add it
   to the `PYTHONPATH` and the transformer example imports it automatically so
   environments are registered.

5. **Optional screenshot capture**

   The DonkeyCar environment supports saving raw camera images every few seconds
   during training. Set `screenshot_interval` (in seconds) and `screenshot_dir`
   in the environment configuration to enable this feature. Captured images are
   stored in the specified directory for later inspection.

