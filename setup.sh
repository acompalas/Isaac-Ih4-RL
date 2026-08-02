#!/bin/bash
set -e

echo "=== Isaac-Ih4-RL pod setup ==="

# 1. Get a genuinely separate Python (Isaac Sim's bundled one breaks venv isolation)
apt-get update && apt-get install -y python3.10 python3.10-venv python3.10-dev

# 2. Build the venv from the real system Python, not Isaac Sim's
rm -rf /workspace/isaaclab/rl_env
/usr/bin/python3.10 -m venv /workspace/isaaclab/rl_env
source /workspace/isaaclab/rl_env/bin/activate

# 3. Clone the physics solver repo (skip if already present)
if [ ! -d /workspace/isaaclab/i4h-tutorials ]; then
  git clone https://github.com/isaac-for-healthcare/i4h-tutorials.git /workspace/isaaclab/i4h-tutorials
fi

# 4. Install everything into the isolated venv
pip install --upgrade pip
pip install -e /workspace/isaaclab/i4h-tutorials/catheter-vasculature-solver
pip install --no-deps -e .
pip install torch torchrl tyro tensorboard wandb "numpy<2" h5py tensordict gymnasium mani_skill==3.0.0b22

echo "=== Setup complete. Run: source /workspace/isaaclab/rl_env/bin/activate ==="