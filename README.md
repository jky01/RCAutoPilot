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

3. **Run training** using the helper script:

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
   You can also save and
   resume training with `--checkpoint <file>` which stores a checkpoint every
   epoch (change the interval with `--checkpoint-freq`).

   If you encounter an error about `donkey-generated-track-v0` not being found,
   ensure the `gym-donkeycar` package is available. The provided scripts add it
   to the `PYTHONPATH` and the transformer example imports it automatically so
   environments are registered.

