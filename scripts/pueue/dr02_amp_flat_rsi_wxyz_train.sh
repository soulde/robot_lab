#!/usr/bin/env bash
# DR02 Pro flat AMP training with the corrected WXYZ motion-data loaders.
set -euo pipefail

cd /home/jvwei/robot_lab
source /home/jvwei/env_isaaclab_ea/bin/activate
export RLL_MOTION_DATA_DIR=/home/jvwei/datasets/dr02_kit_locomotion
export OMNI_KIT_ACCEPT_EULA=YES
PHYSICS_BACKEND="${DR02_PHYSICS_BACKEND:-isaacsim_physx}"
case "$PHYSICS_BACKEND" in
  isaacsim_physx|newton_mjwarp) ;;
  *) echo "Unsupported DR02 physics backend: $PHYSICS_BACKEND" >&2; exit 2 ;;
esac

exec uv run --active --no-project --offline python scripts/reinforcement_learning/rsl_rl/train.py \
  --task RobotLab-Isaac-AMP-Flat-Deeprobotics-DR02-Pro-v0 \
  --viz none \
  --device cuda:0 \
  --num_envs 4096 \
  --seed 42 \
  --max_iterations 30000 \
  --logger tensorboard \
  --run_name ea_amp_flat_rsi_wxyz_20260919_s42 \
  "physics=$PHYSICS_BACKEND"
