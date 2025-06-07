
# Upgrade pip
pip install --upgrade pip

# Install system packages (may require sudo)
apt-get update
apt-get install -y swig cmake ffmpeg

# Install Python dependencies
pip install -r rl-baselines3-zoo/requirements.txt
pip install -e rl-baselines3-zoo
pip install -e gym-donkeycar
