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
#   tb                Serve TensorBoard for logs/ on localhost:6006
#   doctor            Check host prerequisites (docker, GPU, image, mounts)
#
# Files created by the container are owned by the calling host user
# (--user $(id -u):$(id -g)); HOME is redirected to a writable temp dir.

set -euo pipefail

IMAGE="nvcr.io/nvidia/isaac-lab:3.0.0-rc1"
NAME="dr02-isaac"
ROBOT_LAB="${ROBOT_LAB:-$HOME/robot_lab}"
MOTION_DATA_HOST="${MOTION_DATA_HOST:-$HOME/datasets}"

mounts=(
  -v "$ROBOT_LAB:/workspace/robot_lab"
)
# training logs live on /media via the workspace `logs` symlink; docker
# resolves the symlink when setting up the bind mount, so the real dir is
# mounted at the default path (subpath mounts override the repo mount)
if [ -L "$ROBOT_LAB/logs" ]; then
  mounts+=(-v "$ROBOT_LAB/logs:/workspace/robot_lab/logs")
fi
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
  --gpus '"device=0"'
  --network host
  --ipc host
  --shm-size 16g
  --cap-add SYS_PTRACE
  --user "$(id -u):$(id -g)"
  -e HOME=/tmp/home
  -e OMNI_KIT_ACCEPT_EULA=YES
  -e ACCEPT_EULA=Y
  -e RLL_MOTION_DATA_DIR=${RLL_MOTION_DATA_DIR:-/workspace/motion-data}
  -e TERM="${TERM:-xterm}"
  -w /workspace/robot_lab
  "${mounts[@]}"
)

container_exists() { docker ps -a --format '{{.Names}}' | grep -qx "$NAME"; }
container_running() { docker ps --format '{{.Names}}' | grep -qx "$NAME"; }

PY=/workspace/isaaclab/_isaac_sim/python.sh

ensure_passwd() {
  # uid/gid mapped from the host have no passwd entry in the image; torch
  # inductor (getpass.getuser) and friends crash without one
  docker exec -u root "$NAME" groupadd -g "$(id -g)" rll 2>/dev/null || true
  docker exec -u root "$NAME" useradd -u "$(id -u)" -g "$(id -g)" -M -d /tmp/home -s /bin/bash rll 2>/dev/null || true
}

install_pkgs() {
  # remove the stock rsl_rl the image ships in site-packages: it shadows the
  # editable fork install (path entries win over .pth). Only the stock copy
  # (5.4.1); the fork's editable dist-info must survive. The pip installs
  # themselves run in the container bootstrap command (see cmd_start).
  docker exec -u root "$NAME" bash -c \
    'sp=/isaac-sim/kit/python/lib/python3.12/site-packages; rm -rf "$sp/rsl_rl" "$sp"/rsl_rl_lib-5.4.*.dist-info' \
    || true
}

# Runs as the container command (as the mapped user): editable installs land
# in the ephemeral HOME; the marker file signals completion to cmd_start.
bootstrap_cmd=(
  bash -c '
    set -e
    PY=/workspace/isaaclab/_isaac_sim/python.sh
    "$PY" -m pip install -q --no-deps \
      -e /workspace/robot_lab/source/robot_learning_lab_tasks \
      -e /workspace/robot_lab/source/robot_learning_lab_zoo \
      -e /workspace/robot_lab/source/rll_rl \
      -e /workspace/robot_lab/source/rsl_rl
    "$PY" -m pip install -q "moviepy>=1.0.3,<2" || true
    exec sleep infinity
  '
)

cmd_start() {
  if container_running; then echo "[OK] $NAME already running"; return 0; fi
  if container_exists; then
    # stopped container: reuse it, editable installs are already in place
    echo "[INFO] Starting existing $NAME"
    docker start "$NAME" >/dev/null
  else
    mkdir -p /tmp/home
    echo "[INFO] Creating $NAME from $IMAGE (packages install in the background)"
    docker run -d --entrypoint "" "${docker_run_flags[@]}" "${IMAGE}" "${bootstrap_cmd[@]}" >/dev/null
    ensure_passwd
    install_pkgs
  fi
  echo "[OK] Container started. Use './docker/dr02.sh shell' to enter."
}

cmd_stop() {
  if ! container_exists; then echo "[OK] $NAME not present"; return 0; fi
  docker stop "$NAME" >/dev/null
  echo "[OK] $NAME stopped (kept; use './docker/dr02.sh rm' to delete)"
}

cmd_rm() {
  if ! container_exists; then echo "[OK] $NAME not present"; return 0; fi
  docker rm -f "$NAME" >/dev/null
  echo "[OK] $NAME removed"
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
  cmd_exec "$PY" scripts/reinforcement_learning/rsl_rl/train.py --task RobotLab-Isaac-AMP-Rough-Deeprobotics-DR02-Pro-v0 "$@"
}

cmd_play() {
  cmd_exec bash -lc 'python scripts/reinforcement_learning/rsl_rl/play.py --task RobotLab-Isaac-AMP-Rough-Deeprobotics-DR02-Pro-v0 "$@"' _ "$@"
}

TB_PORT="${TB_PORT:-6006}"
TB_LOGDIR="${TB_LOGDIR:-/workspace/robot_lab/logs}"

cmd_tb() {
  container_running || cmd_start
  # container runs with --network host: the port is directly reachable on the
  # host, no -p mapping needed
  if docker exec "$NAME" pgrep -f "tensorboard.*$TB_LOGDIR" >/dev/null 2>&1; then
    echo "[OK] TensorBoard already running on http://localhost:$TB_PORT"
    return 0
  fi
  docker exec -d "$NAME" "$PY" -m tensorboard.main \
    --logdir "$TB_LOGDIR" --host 0.0.0.0 --port "$TB_PORT"
  sleep 2
  docker exec "$NAME" pgrep -f "tensorboard.*$TB_LOGDIR" >/dev/null \
    && echo "[OK] TensorBoard at http://localhost:$TB_PORT (logdir $TB_LOGDIR)" \
    || echo "[ERROR] TensorBoard failed to start"
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
  rm)     shift; cmd_rm "$@" ;;
  restart)shift; cmd_stop; cmd_start "$@" ;;
  status) shift; cmd_status "$@" ;;
  shell)  shift; cmd_shell "$@" ;;
  exec)   shift; cmd_exec "$@" ;;
  train)  shift; cmd_train "$@" ;;
  play)   shift; cmd_play "$@" ;;
  tb)     shift; cmd_tb "$@" ;;
  doctor) shift; cmd_doctor "$@" ;;
  help|*) sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//' ;;
esac
