#!/usr/bin/env bash
# Retry failed MJLab PPO verification tasks after the main batch finishes.
# Per-task extra CLI args (e.g. fewer envs for heavy models).
set -u
ROOT=/home/soulde/robot_lab
LOGROOT="$ROOT/logs/mjlab_ppo"
STATUS="$LOGROOT/status.tsv"

# Wait until the main sequential batch is done.
while tmux has-session -t mjlab_ppo_verify 2>/dev/null; do sleep 60; done
[ -f "$LOGROOT/STOP" ] && exit 0

declare -A OVERRIDE=(
  [RobotLab-MJLab-Velocity-Flat-DDTRobot-Tita]="--env.scene.num-envs 1024"
  [RobotLab-MJLab-Velocity-Flat-Deeprobotics-M20]="--env.scene.num-envs 1024"
  [RobotLab-MJLab-Velocity-Flat-MagicLab-Dog-W]="--env.scene.num-envs 1024"
  [RobotLab-MJLab-Velocity-Flat-Openloong-Loong]="--env.scene.num-envs 1024"
  [RobotLab-MJLab-Velocity-Flat-RoboParty-ATOM01]="--env.scene.num-envs 1024"
  [RobotLab-MJLab-Velocity-Flat-RobotEra-Xbot]="--env.scene.num-envs 1024"
  [RobotLab-MJLab-Velocity-Flat-Unitree-B2W]="--env.scene.num-envs 1024"
  [RobotLab-MJLab-Velocity-Flat-Unitree-Go2W]="--env.scene.num-envs 1024"
  [RobotLab-MJLab-Velocity-Flat-Unitree-H1]="--env.scene.num-envs 1024"
  [RobotLab-MJLab-Velocity-Flat-Zsibot-ZSL1W]="--env.scene.num-envs 1024"
  [RobotLab-MJLab-Velocity-Rough-DDTRobot-Tita]="--env.scene.num-envs 1024"
  [RobotLab-MJLab-Velocity-Rough-Deeprobotics-M20]="--env.scene.num-envs 1024"
)

awk -F'\t' '$2 == "FAIL" {print $3}' "$STATUS" | sort -u | while IFS= read -r task; do
  [ -f "$LOGROOT/STOP" ] && break
  [ -f "$LOGROOT/${task}.done" ] && continue
  extra="${OVERRIDE[$task]:-}"
  echo "$(date +%FT%T)	RETRY	$task" >> "$STATUS"
  if WANDB_DISABLED=true timeout 21600 /home/soulde/mjlab/.venv/bin/train "$task" \
      --agent.logger tensorboard --env.scene.num-envs 4096 $extra \
      >> "$LOGROOT/${task}.retry.log" 2>&1; then
    echo "$(date +%FT%T)	PASS	$task" >> "$STATUS"
    touch "$LOGROOT/${task}.done"
  else
    echo "$(date +%FT%T)	FAIL	$task" >> "$STATUS"
  fi
done
