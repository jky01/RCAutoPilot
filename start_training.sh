#!/bin/bash
# Start RL Baselines3 Zoo training for DonkeyCar or train the HuggingFace
# transformer example.
# Usage: ./start_training.sh [--transformer [args...]] [additional train.py arguments]

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

cd "$SCRIPT_DIR"

source venv/bin/activate


# Ensure the local gym-donkeycar package is on the Python path so
# its environments get registered even if it has not been installed.
export PYTHONPATH="$SCRIPT_DIR/gym-donkeycar${PYTHONPATH:+:$PYTHONPATH}"

# Check that gymnasium is installed. The RL Zoo and gym-donkeycar
# environments rely on it and wrappers are optional.
if ! python - <<'EOF' >/dev/null 2>&1
import gymnasium
EOF
then
    echo "Error: gymnasium is not installed. Please run 'source ./setup_env.sh'" >&2
    exit 1
fi

if [[ "$1" == "--transformer" ]]; then
    shift
    # Ensure transformers is installed when running the example
    if ! python - <<'EOF' >/dev/null 2>&1
import transformers
EOF
    then
        echo "Error: transformers is not installed. Run 'pip install transformers' inside the virtualenv." >&2
        exit 1
    fi
    # Run the HuggingFace transformer example located in gym-donkeycar
    # Additional arguments are forwarded to the example script
    python "$SCRIPT_DIR/gym-donkeycar/examples/huggingface_transformer/train_transformer.py" "$@"
else
    cd "$SCRIPT_DIR/rl-baselines3-zoo"
    python train.py --algo sac --env donkey-generated-track-v0 --gym-packages gym_donkeycar \
        --eval-freq 10000 --eval-episodes 10 --n-eval-envs 1 "$@"
fi
