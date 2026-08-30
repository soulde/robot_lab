#!/usr/bin/env bash
# Sequentially train every RobotLab MJLab velocity task with the built-in PPO.
# AMP tasks are excluded: they need GMR motion data and are not built-in PPO.
set -u
VENV_BIN=/home/soulde/mjlab/.venv/bin
ROOT=/home/soulde/robot_lab
LOGROOT="$ROOT/logs/mjlab_ppo"
STATUS="$LOGROOT/status.tsv"
TASKS="$LOGROOT/tasks.txt"
mkdir -p "$LOGROOT"

# Discover the task list once.
if [ ! -s "$TASKS" ]; then
  "$VENV_BIN/train" 2>&1 | grep -o 'RobotLab-MJLab-[A-Za-z0-9-]*' | sort -u > "$TASKS"
fi

# Per-task extra CLI args: cap marathon iteration budgets and shrink
# env counts for models that exceed GPU memory at 4096 envs.
declare -A OVERRIDE=(
  [RobotLab-MJLab-Velocity-Flat-Unitree-G1]="--agent.max-iterations 3000"
  [RobotLab-MJLab-Velocity-Rough-Zsibot-ZSL1]="--agent.max-iterations 3000"
  [RobotLab-MJLab-Velocity-Rough-Zsibot-ZSL1W]="--agent.max-iterations 3000"
  [RobotLab-MJLab-Velocity-Flat-DDTRobot-Tita]="--env.scene.num-envs 1024"
  [RobotLab-MJLab-Velocity-Flat-Deeprobotics-M20]="--env.scene.num-envs 1024"
  [RobotLab-MJLab-Velocity-Flat-MagicLab-Dog-W]="--env.scene.num-envs 1024"
  [RobotLab-MJLab-Velocity-Flat-Openloong-Loong]="--env.scene.num-envs 1024"
  [RobotLab-MJLab-Velocity-Flat-RoboParty-ATOM01]="--env.scene.num-envs 1024"
  [RobotLab-MJLab-Velocity-Flat-RobotEra-Xbot]="--env.scene.num-envs 1024"
  [RobotLab-MJLab-Velocity-Flat-Unitree-B2W]="--env.scene.num-envs 1024"
)
# Rough terrain needs far more GPU memory than flat; run all rough tasks at 2048.
for _t in $(grep "Velocity-Rough" "$TASKS"); do OVERRIDE[$_t]="--env.scene.num-envs 2048"; done

while IFS= read -r task; do
  [ -f "$LOGROOT/STOP" ] && { echo "$(date +%FT%T)	STOPPED" >> "$STATUS"; break; }
  case "$task" in
    RobotLab-MJLab-AMP-*) continue ;;
  esac
  [ -f "$LOGROOT/${task}.done" ] && continue
  log="$LOGROOT/${task}.log"
  extra="${OVERRIDE[$task]:-}"
  echo "$(date +%FT%T)	START	$task" >> "$STATUS"
  if WANDB_DISABLED=true timeout 21600 "$VENV_BIN/train" "$task" \
      --agent.logger tensorboard --env.scene.num-envs 4096 $extra \
      > "$log" 2>&1; then
    echo "$(date +%FT%T)	PASS	$task" >> "$STATUS"
    touch "$LOGROOT/${task}.done"
  else
    echo "$(date +%FT%T)	FAIL	$task" >> "$STATUS"
  fi
done < "$TASKS"
echo "$(date +%FT%T)	ALL_DONE" >> "$STATUS"
