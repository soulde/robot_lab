#!/usr/bin/env bash
# Full DR02 Pro AMP rough-terrain training on the IsaacLab v3.0.0-EA host env.
# Motion data: config (bodies.json) + npz in RLL_MOTION_DATA_DIR.
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
  --task RobotLab-Isaac-AMP-Rough-Deeprobotics-DR02-Pro-v0 \
  --viz none \
  --logger tensorboard \
  --run_name ea_amp_chocolate_sync \
  "physics=$PHYSICS_BACKEND"
