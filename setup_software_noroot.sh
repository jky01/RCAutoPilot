# Upgrade pip
pip install --upgrade pip

# Install Python dependencies
pip install -r rl-baselines3-zoo/requirements.txt
pip install -e rl-baselines3-zoo
pip install -e gym-donkeycar

# Install CLRNet and its runtime dependencies for lane detection
pip install torch torchvision opencv-python mmcv
# CLRNet pins very old versions of PyTorch and MMCV that are not available on
# newer Python versions. Remove those pins so the currently installed packages
# are used instead.
sed -i \
  -e '/^torch==/d' \
  -e '/^torchvision==/d' \
  -e '/^mmcv==/d' \
  -e 's/^sklearn$/scikit-learn/' \
  CLRNet/requirements.txt
# Install CLRNet using the already installed PyTorch
pip install --no-build-isolation -e CLRNet
