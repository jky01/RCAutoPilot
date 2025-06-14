# RCAutoPilot

This repository bundles [RL Baselines3 Zoo](https://github.com/DLR-RM/rl-baselines3-zoo), [Stable Baselines3](https://github.com/DLR-RM/stable-baselines3) and [gym-donkeycar](https://github.com/tawnkramer/gym-donkeycar) to train autonomous driving agents.

## Quick Start

1. **Create a Python environment and install dependencies**:

   Run the setup script which creates a virtual environment in `venv`, activates
   it and installs all required packages.

   ```bash
   source ./setup_env.sh
   ```

   This installs `gym-donkeycar` and then installs CLRNet with its dependencies
   (`torch`, `torchvision`, `opencv-python`, `mmcv`). The setup script uses
   `pip install --no-build-isolation -e CLRNet` so the already-installed
   PyTorch is reused during the build and lane detection works out of the box.

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

  Lane-line preprocessing using [CLRNet](https://github.com/Turoad/CLRNet) is enabled by default. The training
  script reads the `LANE_CFG` and `LANE_CKPT` environment variables to locate the CLRNet configuration and
  checkpoint. The pre-trained CULane model can be downloaded from the
  [CLRNet releases](https://github.com/Turoad/CLRNet/releases) page (file `culane_dla34.pth`).
  Place it in the repository root or set `LANE_CKPT` to its location. If the CLRNet
  dependencies are missing, the wrapper falls back to using the raw camera frames instead of lane masks.
  When lane detection is active, a lane-mask image is saved every five seconds in the `lane_captures/` directory
  (configurable via `LANE_CAPTURE_DIR` and `LANE_CAPTURE_INTERVAL`). If this directory stays empty, make sure the
  CLRNet dependencies are installed and the configuration and checkpoint paths are correct.

   If you encounter an error about `donkey-generated-track-v0` not being found,
   make sure the `gym-donkeycar` package is installed (the setup script installs
   it automatically) or run training via `./start_training.sh` which adds the
   package to the `PYTHONPATH`.

