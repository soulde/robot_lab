#!/usr/bin/env bash
set -euo pipefail

cd /home/jvwei/robot_lab
source /home/jvwei/env_isaaclab/bin/activate

exec python scripts/reinforcement_learning/rsl_rl/train.py \
  --task RobotLab-Isaac-AMP-Flat-Deeprobotics-DR02-Pro-v0 \
  --num_envs 2048 \
  --max_iterations 30000 \
  --seed 42 \
  --headless \
  --run_name dr02_pro_amp_all_20260823_seed42
