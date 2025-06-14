# Upgrade pip
pip install --upgrade pip

# Install Python dependencies
pip install -r rl-baselines3-zoo/requirements.txt
pip install -e rl-baselines3-zoo
pip install -e gym-donkeycar

# Install CLRNet and its runtime dependencies for lane detection
pip install torch torchvision opencv-python mmcv
# Install CLRNet using the already installed PyTorch
pip install --no-build-isolation -e CLRNet
