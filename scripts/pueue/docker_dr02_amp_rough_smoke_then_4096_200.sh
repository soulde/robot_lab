#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)"
MOTION_DIR="${MOTION_DIR:-${HOME}/datasets/dr02_kit_locomotion_with_toe}"
RUN_NAME="dr02_amp_rough_docker_4096x200_$(date -u +%Y%m%dT%H%M%SZ)"
RUN_ROOT="${RUN_ROOT:-$ROOT/logs/$RUN_NAME}"

cd "$ROOT"
export RSL_RL_DIR="${RSL_RL_DIR:-$HOME/rsl_rl}"
[[ -f "$MOTION_DIR/bodies.json" ]] || { echo "AMP bodies.json missing: $MOTION_DIR" >&2; exit 2; }
[[ ! -e "$RUN_ROOT" ]] || { echo "Refusing to reuse existing experiment output: $RUN_ROOT" >&2; exit 2; }
mkdir -p "$RUN_ROOT"
printf 'Run root: %s\nMotion data: %s\n' "$RUN_ROOT" "$MOTION_DIR"

./scripts/docker_train.sh build
ACCEPT_EULA=Y ./scripts/docker_train.sh smoke \
  --motion-dir "$MOTION_DIR" \
  --output-dir "$RUN_ROOT/smoke"
ACCEPT_EULA=Y ./scripts/docker_train.sh train \
  --motion-dir "$MOTION_DIR" \
  --output-dir "$RUN_ROOT/train" \
  -- \
  --num_envs 4096 \
  --max_iterations 200 \
  --seed 42 \
  --run_name "$RUN_NAME"
