# Upgrade pip
pip install --upgrade pip

# Install Python dependencies
pip install -r rl-baselines3-zoo/requirements.txt
pip install -e rl-baselines3-zoo
pip install -e gym-donkeycar

# Install CLRNet and its runtime dependencies for lane detection
pip install opencv-python torch torchvision mmcv
pip install -e CLRNet
