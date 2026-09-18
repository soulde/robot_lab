#!/usr/bin/env bash
# Manage the DR02 Isaac Lab 3.0.0-rc1 training container.
#
# Usage: ./docker/dr02.sh <command> [args...]
#
# Commands:
#   pull              Pull the 3.0.0-rc1 image (run once, ~20 GB)
#   start             Create and start the container (idempotent)
#   stop              Stop and remove the container
#   restart           stop + start
#   status            Container state
#   shell             Interactive bash inside the container (workspace mounted)
#   exec <cmd...>     Run a command inside the container
#   train [args...]   Run the DR02 train script (defaults to AMP rough)
#   play  [args...]   Run the DR02 play script
#   doctor            Check host prerequisites (docker, GPU, image, mounts)
#
# Files created by the container are owned by the calling host user
# (--user $(id -u):$(id -g)); HOME is redirected to a writable temp dir.

set -euo pipefail

IMAGE="nvcr.io/nvidia/isaac-lab:3.0.0-rc1"
NAME="dr02-isaac"
ROBOT_LAB="${ROBOT_LAB:-$HOME/robot_lab}"
MOTION_DATA_HOST="${MOTION_DATA_HOST:-$HOME/GMR-private}"

mounts=(
  -v "$ROBOT_LAB:/workspace/robot_lab"
)
if [ -d "$MOTION_DATA_HOST" ]; then
  mounts+=(-v "$MOTION_DATA_HOST:/workspace/motion-data")
fi
# rsl_rl custom fork lives outside robot_lab on hosts that have it
if [ -d "$HOME/humanoid_amp" ]; then
  mounts+=(-v "$HOME/humanoid_amp:/workspace/humanoid_amp")
fi

docker_run_flags=(
  --name "$NAME"
  --init
  --gpus all
  --network host
  --ipc host
  --shm-size 16g
  --user "$(id -u):$(id -g)"
  -e HOME=/tmp/home
  -e OMNI_KIT_ACCEPT_EULA=YES
  -e RLL_MOTION_DATA_ROOT=/workspace/motion-data
  -e TERM="${TERM:-xterm}"
  -w /workspace/robot_lab
  "${mounts[@]}"
)

container_exists() { docker ps -a --format '{{.Names}}' | grep -qx "$NAME"; }
container_running() { docker ps --format '{{.Names}}' | grep -qx "$NAME"; }

cmd_start() {
  if container_running; then echo "[OK] $NAME already running"; return 0; fi
  if container_exists; then docker rm "$NAME" >/dev/null; fi
  mkdir -p /tmp/home
  echo "[INFO] Starting $NAME from $IMAGE"
  docker run -d "${docker_run_flags[@]}" "$IMAGE" sleep infinity
  echo "[OK] Container up. Use './docker/dr02.sh shell' to enter."
}

cmd_stop() {
  if ! container_exists; then echo "[OK] $NAME not present"; return 0; fi
  docker rm -f "$NAME" >/dev/null
  echo "[OK] $NAME stopped and removed"
}

cmd_status() {
  if ! container_exists; then echo "$NAME: not created"; exit 1; fi
  docker ps -a --filter "name=$NAME" --format "table {{.Names}}\t{{.Status}}\t{{.Image}}"
  docker exec "$NAME" nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader 2>/dev/null || true
}

cmd_shell() {
  container_running || cmd_start
  docker exec -it "$NAME" bash
}

cmd_exec() {
  container_running || cmd_start
  docker exec -i "$NAME" "$@"
}

cmd_train() {
  cmd_exec python scripts/reinforcement_learning/rsl_rl/train.py --task RobotLab-Isaac-AMP-Rough-Deeprobotics-DR02-Pro-v0 "$@"
}

cmd_play() {
  cmd_exec bash -lc 'python scripts/reinforcement_learning/rsl_rl/play.py --task RobotLab-Isaac-AMP-Rough-Deeprobotics-DR02-Pro-v0 "$@"' _ "$@"
}

cmd_doctor() {
  echo "docker:   $(docker --version 2>/dev/null || echo MISSING)"
  echo "gpu:      $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo 'nvidia-smi failed')"
  echo "uid/gid:  $(id -u)/$(id -g)"
  echo "image:    $(docker image inspect "$IMAGE" --format 'present ({{.Size}})' 2>/dev/null || echo 'NOT PULLED — run ./docker/dr02.sh pull')"
  echo "mounts:"
  echo "  robot_lab:     $([ -d "$ROBOT_LAB" ] && echo "$ROBOT_LAB" || echo "MISSING: $ROBOT_LAB")"
  echo "  motion-data:   $([ -d "$MOTION_DATA_HOST" ] && echo "$MOTION_DATA_HOST" || echo "(absent, AMP tasks unavailable)")"
  echo "  humanoid_amp:  $([ -d "$HOME/humanoid_amp" ] && echo "$HOME/humanoid_amp" || echo "(absent, rsl_rl fork not mounted)")"
}

cmd_pull() {
  docker pull "$IMAGE"
}

case "${1:-help}" in
  pull)   shift; cmd_pull "$@" ;;
  start)  shift; cmd_start "$@" ;;
  stop)   shift; cmd_stop "$@" ;;
  restart)shift; cmd_stop; cmd_start "$@" ;;
  status) shift; cmd_status "$@" ;;
  shell)  shift; cmd_shell "$@" ;;
  exec)   shift; cmd_exec "$@" ;;
  train)  shift; cmd_train "$@" ;;
  play)   shift; cmd_play "$@" ;;
  doctor) shift; cmd_doctor "$@" ;;
  help|*) sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//' ;;
esac
