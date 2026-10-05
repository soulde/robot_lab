#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
COMPOSE_FILE="$ROOT/docker/docker-compose.yaml"
DOCKERFILE="$ROOT/docker/Dockerfile"
IMAGE="${ROBOT_LAB_IMAGE:-robot-lab:isaac-lab-3.0.0-rc1}"
DEFAULT_TASK="RobotLab-Isaac-AMP-Rough-Deeprobotics-DR02-Pro-v0"
DEFAULT_RSL_RL_COMMIT="8eef047f1d9b5e07d54d743387da2f0fb42791fa"

usage() {
  cat <<'EOF'
Usage:
  scripts/docker_train.sh doctor
  scripts/docker_train.sh build
  ACCEPT_EULA=Y scripts/docker_train.sh smoke --motion-dir PATH --output-dir PATH [--task TASK]
  ACCEPT_EULA=Y scripts/docker_train.sh train --motion-dir PATH --output-dir PATH [--task TASK] [--uid UID] [--gid GID] -- [train args...]

Commands:
  doctor  Check Docker, Compose, Buildx, and host GPU visibility.
  build   Build the Isaac Lab 3.0.0-rc1 image using a pinned local RSL-RL checkout.
  smoke   Run one headless environment for one iteration.
  train   Run headless RSL-RL training; arguments after -- go to Isaac Lab.

Runtime options:
  --motion-dir PATH  Host directory containing the DR02 AMP config and motions.
  --output-dir PATH  Host directory for logs and checkpoints.
  --task TASK        Isaac Lab task (default: DR02 rough AMP).
  --uid UID          Container UID (default: current host UID).
  --gid GID          Container GID (default: current host GID).

Build options:
  ROBOT_LAB_IMAGE    Override the local image tag.
  RSL_RL_DIR         Local soulde/rsl_rl checkout (default: ~/rsl_rl).
  RSL_RL_COMMIT      Commit to export from that checkout.
EOF
}

fail() {
  printf '[ERROR] %s\n' "$*" >&2
  exit 2
}

require_docker() {
  command -v docker >/dev/null 2>&1 || fail "Docker is not installed or not on PATH."
}

cmd_doctor() {
  require_docker
  printf 'Docker:  '
  docker --version
  printf 'Compose: '
  docker compose version
  printf 'Buildx:  '
  docker buildx version
  printf 'GPU:     '
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1 || true
  else
    printf 'nvidia-smi not found (check NVIDIA Container Toolkit and host drivers)\n'
  fi
}

cmd_build() {
  (($# == 0)) || fail "build takes no positional arguments; configure RSL_RL_COMMIT with an environment variable."
  require_docker
  docker buildx version >/dev/null 2>&1 || fail "Docker Buildx is required."

  local rsl_rl_commit="${RSL_RL_COMMIT:-$DEFAULT_RSL_RL_COMMIT}"
  [[ "$rsl_rl_commit" =~ ^[0-9a-fA-F]{40}$ ]] || fail "RSL_RL_COMMIT must be a full 40-character commit hash."

  local rsl_rl_dir="${RSL_RL_DIR:-$HOME/rsl_rl}"
  [[ -d "$rsl_rl_dir/.git" || -f "$rsl_rl_dir/.git" ]] \
    || fail "RSL_RL_DIR must point to a Git checkout: $rsl_rl_dir"
  git -C "$rsl_rl_dir" cat-file -e "${rsl_rl_commit}^{commit}" 2>/dev/null \
    || fail "Pinned commit $rsl_rl_commit is not present in $rsl_rl_dir"

  local rsl_rl_context
  rsl_rl_context="$(mktemp -d "${TMPDIR:-/tmp}/robot-lab-rsl-rl.XXXXXX")"
  trap 'rm -rf -- "$rsl_rl_context"' RETURN
  git -C "$rsl_rl_dir" archive "$rsl_rl_commit" | tar -x -C "$rsl_rl_context"

  docker buildx build \
    --load \
    --build-arg "RSL_RL_COMMIT=$rsl_rl_commit" \
    --build-context "rsl_rl=$rsl_rl_context" \
    --file "$DOCKERFILE" \
    --tag "$IMAGE" \
    "$ROOT"
}

canonical_dir() {
  local path="$1"
  [[ -d "$path" ]] || return 1
  (cd -- "$path" && pwd -P)
}

cmd_runtime() {
  local mode="$1"
  shift

  local motion_dir="" output_dir="" task="$DEFAULT_TASK"
  local host_uid="$(id -u)" host_gid="$(id -g)"
  local -a train_args=()

  while (($#)); do
    case "$1" in
      --motion-dir)
        (($# >= 2)) || fail "--motion-dir requires a path."
        motion_dir="$2"
        shift 2
        ;;
      --output-dir)
        (($# >= 2)) || fail "--output-dir requires a path."
        output_dir="$2"
        shift 2
        ;;
      --task)
        (($# >= 2)) || fail "--task requires a task ID."
        task="$2"
        shift 2
        ;;
      --uid)
        (($# >= 2)) || fail "--uid requires an integer."
        host_uid="$2"
        shift 2
        ;;
      --gid)
        (($# >= 2)) || fail "--gid requires an integer."
        host_gid="$2"
        shift 2
        ;;
      --)
        shift
        train_args=("$@")
        break
        ;;
      *)
        fail "Unknown runtime option: $1 (put Isaac Lab arguments after --)."
        ;;
    esac
  done

  [[ -n "$motion_dir" ]] || fail "--motion-dir is required."
  [[ -n "$output_dir" ]] || fail "--output-dir is required."
  [[ "$host_uid" =~ ^[0-9]+$ ]] || fail "UID must be a non-negative integer."
  [[ "$host_gid" =~ ^[0-9]+$ ]] || fail "GID must be a non-negative integer."
  [[ "${ACCEPT_EULA:-}" == "Y" ]] \
    || fail "Set ACCEPT_EULA=Y to accept the NVIDIA software license before running a container."

  motion_dir="$(canonical_dir "$motion_dir")" \
    || fail "Motion directory does not exist or cannot be accessed: $motion_dir"
  mkdir -p -- "$output_dir"
  output_dir="$(cd -- "$output_dir" && pwd -P)"

  require_docker
  docker image inspect "$IMAGE" >/dev/null 2>&1 \
    || fail "Image $IMAGE is not present locally; run '$0 build' first."

  export ACCEPT_EULA HOST_UID="$host_uid" HOST_GID="$host_gid"
  export MOTION_DATA_HOST="$motion_dir" OUTPUT_DIR_HOST="$output_dir"
  export ROBOT_LAB_IMAGE="$IMAGE"

  local -a command=(
    /workspace/isaaclab/isaaclab.sh
    -p /workspace/robot_lab/scripts/reinforcement_learning/rsl_rl/train.py
    --task "$task"
    --headless
  )
  if [[ "$mode" == "smoke" ]]; then
    local run_name="docker_smoke_$(date -u +%Y%m%dT%H%M%SZ)"
    command+=(--num_envs 1 --max_iterations 1 --run_name "$run_name")
  else
    command+=("${train_args[@]}")
  fi

  docker compose -f "$COMPOSE_FILE" run \
    --rm \
    --no-deps \
    --pull never \
    trainer \
    "${command[@]}"
}

command_name="${1:-help}"
if (($#)); then shift; fi

case "$command_name" in
  build) cmd_build "$@" ;;
  smoke) cmd_runtime smoke "$@" ;;
  train) cmd_runtime train "$@" ;;
  doctor) cmd_doctor "$@" ;;
  help|-h|--help) usage ;;
  *) usage >&2; fail "Unknown command: $command_name" ;;
esac
