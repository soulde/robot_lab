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

while IFS= read -r task; do
  [ -f "$LOGROOT/STOP" ] && { echo "$(date +%FT%T)	STOPPED" >> "$STATUS"; break; }
  case "$task" in
    RobotLab-MJLab-AMP-*) continue ;;
  esac
  [ -f "$LOGROOT/${task}.done" ] && continue
  log="$LOGROOT/${task}.log"
  echo "$(date +%FT%T)	START	$task" >> "$STATUS"
  if WANDB_DISABLED=true timeout 21600 "$VENV_BIN/train" "$task" \
      --agent.logger tensorboard --env.scene.num-envs 4096 \
      > "$log" 2>&1; then
    echo "$(date +%FT%T)	PASS	$task" >> "$STATUS"
    touch "$LOGROOT/${task}.done"
  else
    echo "$(date +%FT%T)	FAIL	$task" >> "$STATUS"
  fi
done < "$TASKS"
echo "$(date +%FT%T)	ALL_DONE" >> "$STATUS"
