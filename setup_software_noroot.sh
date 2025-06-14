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
  -e 's/^Shapely==.*/Shapely/' \
  CLRNet/requirements.txt
# Remove any stale build or egg-info directories from previous installs
find CLRNet -name '*.egg-info' -exec rm -rf {} +
find CLRNet -name 'build' -type d -exec rm -rf {} +
# Skip building CLRNet's CUDA extension which fails with modern toolchains
sed -i \
  -e 's/from torch.utils.cpp_extension import CUDAExtension, BuildExtension/from torch.utils.cpp_extension import BuildExtension/' \
  -e '/def get_extensions/,/return extensions/c\
def get_extensions():\n    return []\n' \
  -e 's/ext_modules=get_extensions()/ext_modules=[]/' \
  -e 's/cmdclass={"build_ext": BuildExtension}/cmdclass={}/' \
  CLRNet/setup.py
# Install CLRNet using the already installed PyTorch
pip install --no-build-isolation -e CLRNet
