#!/usr/bin/env bash
set -euo pipefail

cd /home/jvwei/robot_lab
source /home/jvwei/env_isaaclab/bin/activate

exec python scripts/reinforcement_learning/rsl_rl/train.py \
  --task RobotLab-Isaac-AMP-Flat-Deeprobotics-DR02-Pro-v0 \
  --num_envs 64 \
  --max_iterations 10 \
  --seed 45 \
  --headless \
  --run_name dr02_pro_amp_smoke_stable_10it_20260823_seed45
