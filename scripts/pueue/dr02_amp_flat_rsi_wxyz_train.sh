#!/usr/bin/env bash
# DR02 Pro flat AMP training with the corrected WXYZ motion-data loaders.
set -euo pipefail

cd /home/jvwei/robot_lab
source /home/jvwei/env_isaaclab_ea/bin/activate
export RLL_MOTION_DATA_DIR=/home/jvwei/datasets/dr02_kit_locomotion_with_toe
export OMNI_KIT_ACCEPT_EULA=YES
PHYSICS_BACKEND="${DR02_PHYSICS_BACKEND:-isaacsim_physx}"
MAX_ITERATIONS="${DR02_MAX_ITERATIONS:-30000}"
NUM_ENVS="${DR02_NUM_ENVS:-4096}"
RUN_NAME="${DR02_RUN_NAME:-ea_amp_flat_rsi_wxyz_20260919_s42}"
case "$PHYSICS_BACKEND" in
  isaacsim_physx|newton_mjwarp) ;;
  *) echo "Unsupported DR02 physics backend: $PHYSICS_BACKEND" >&2; exit 2 ;;
esac

TRAIN_CMD=(uv run --active --no-project --offline python scripts/reinforcement_learning/rsl_rl/train.py \
  --task RobotLab-Isaac-AMP-Flat-Deeprobotics-DR02-Pro-v0 \
  --viz none \
  --device cuda:0 \
  --num_envs "$NUM_ENVS" \
  --seed 42 \
  --max_iterations "$MAX_ITERATIONS" \
  --logger tensorboard \
  --run_name "$RUN_NAME")
if [[ -n "${DR02_RESUME_RUN:-}" ]]; then
  TRAIN_CMD+=(--resume --load_run "$DR02_RESUME_RUN" --checkpoint "${DR02_RESUME_CHECKPOINT:-model_1000.pt}")
fi
exec "${TRAIN_CMD[@]}" "physics=$PHYSICS_BACKEND"
