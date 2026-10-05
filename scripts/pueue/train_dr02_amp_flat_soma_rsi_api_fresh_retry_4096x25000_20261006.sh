#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/jvwei/robot_lab"
DATASET="/home/jvwei/datasets/soma_retargeted/locomotion_minimal"
RUN_ROOT="$ROOT/logs/dr02_amp_flat_soma_rsi_api_fresh_retry_4096x25000_20261006"
MOTION_DIR="$RUN_ROOT/motion_data"
RSL_RL_DIR="/home/jvwei/rsl_rl"
RSL_RL_COMMIT="8eef047f1d9b5e07d54d743387da2f0fb42791fa"
RSL_RL_SNAPSHOT="$RUN_ROOT/source_snapshot/rsl_rl"
IMAGE="sha256:5f6f461a739232b822771a42893b0f159946e5f9ae5a3a21bba2d9cb7a56ef09"
TASK="RobotLab-Isaac-AMP-Flat-Deeprobotics-DR02-Pro-v0"
TASK_PACKAGE="$ROOT/source/robot_learning_lab_tasks"
ENV_CFG="$TASK_PACKAGE/robot_learning_lab_tasks/tasks/isaaclab/manager_based/amp/config/dr02/flat_env_cfg.py"
AGENT_CFG="$TASK_PACKAGE/robot_learning_lab_tasks/tasks/isaaclab/manager_based/amp/config/dr02/agents/rsl_rl_amp_cfg.py"
CONTAINER_NAME="dr02-soma-rsi-api-fresh-retry-20261006"
ACTIVE_CONTAINER=""
export ACCEPT_EULA=Y

cleanup() {
  if [[ -n "$ACTIVE_CONTAINER" ]] && docker inspect "$ACTIVE_CONTAINER" >/dev/null 2>&1; then
    docker stop --timeout 30 "$ACTIVE_CONTAINER" >/dev/null || true
  fi
}
on_signal() {
  exit 143
}
trap cleanup EXIT
trap on_signal TERM INT HUP

cd "$ROOT"
[[ -d "$DATASET" ]]
[[ -f "$ENV_CFG" && -f "$AGENT_CFG" ]]
[[ ! -e "$RUN_ROOT" ]] || { echo "Refusing to reuse output: $RUN_ROOT" >&2; exit 2; }
[[ "$(git -C "$RSL_RL_DIR" cat-file -t "$RSL_RL_COMMIT")" == commit ]]
docker image inspect "$IMAGE" >/dev/null
if ss -ltn 2>/dev/null | rg -q ':8080\b'; then
  echo "Port 8080 is already in use; refusing to launch Viser." >&2
  exit 2
fi

mkdir -p "$RUN_ROOT/train" "$RUN_ROOT/source_snapshot" "$MOTION_DIR" "$RSL_RL_SNAPSHOT"
git -C "$RSL_RL_DIR" archive "$RSL_RL_COMMIT" | tar -x -C "$RSL_RL_SNAPSHOT"
RSL_RL_PATCH="$RUN_ROOT/source_snapshot/rsl_rl_worktree.patch"
git -C "$RSL_RL_DIR" diff --binary "$RSL_RL_COMMIT" > "$RSL_RL_PATCH"
if [[ -s "$RSL_RL_PATCH" ]]; then
  patch --batch --forward -p1 -d "$RSL_RL_SNAPSHOT" < "$RSL_RL_PATCH"
fi
grep -q 'rsi_support_body_boxes = cfg\["algorithm"\].pop' \
  "$RSL_RL_SNAPSHOT/rsl_rl/algorithms/amp.py"
grep -q 'def configure_rsi_support_height_adjustment' \
  "$RSL_RL_SNAPSHOT/rsl_rl/datasets/base_motion_dataset.py"

source /home/jvwei/env_isaaclab_ea/bin/activate
python - "$DATASET" "$MOTION_DIR" "$ENV_CFG" <<'PY'
from __future__ import annotations

import ast
import errno
import json
import os
import re
import shutil
import sys
from pathlib import Path

import numpy as np

source_root, staged_root, config_path = map(Path, sys.argv[1:])
source_files = sorted(source_root.glob("*.npz"))
if not source_files:
    raise SystemExit(f"No NPZ motions found in {source_root}")
tree = ast.parse(config_path.read_text(encoding="utf-8"))
exclude_assignment = next(
    node for node in tree.body
    if isinstance(node, ast.Assign)
    and any(isinstance(target, ast.Name) and target.id == "DR02_AMP_MOTION_EXCLUDES" for target in node.targets)
)
exclude_patterns = [re.compile(pattern) for pattern in ast.literal_eval(exclude_assignment.value)]
files = [path for path in source_files if not any(pattern.fullmatch(path.name) for pattern in exclude_patterns)]
excluded_names = sorted(path.name for path in source_files if path not in files)
stair_files = sorted(path.name for path in source_files if path.name.lower().startswith("stairs"))
if any(name not in excluded_names for name in stair_files):
    raise SystemExit(f"Stair motions must be excluded from flat AMP and RSI: {stair_files}")
if not files:
    raise SystemExit("DR02 AMP motion exclusions removed every motion")

required = {
    "fps", "joint_pos", "joint_vel", "body_pos_w", "body_quat_w",
    "body_lin_vel_w", "body_ang_vel_w", "joint_names", "body_names",
}
reference_joint_names = None
reference_body_names = None
for path in files:
    with np.load(path, allow_pickle=False) as data:
        missing = required - set(data.files)
        if missing:
            raise SystemExit(f"{path.name}: missing fields {sorted(missing)}")
        if float(data["fps"]) != 50.0:
            raise SystemExit(f"{path.name}: expected 50 Hz, got {float(data['fps'])}")
        joint_names = list(map(str, data["joint_names"].tolist()))
        body_names = list(map(str, data["body_names"].tolist()))
        if reference_joint_names is None:
            reference_joint_names, reference_body_names = joint_names, body_names
        if joint_names != reference_joint_names or body_names != reference_body_names:
            raise SystemExit(f"{path.name}: joint/body name order differs from dataset contract")
        frames = len(data["joint_pos"])
        expected_shapes = {
            "joint_pos": (frames, 29), "joint_vel": (frames, 29),
            "body_pos_w": (frames, len(body_names), 3),
            "body_quat_w": (frames, len(body_names), 4),
            "body_lin_vel_w": (frames, len(body_names), 3),
            "body_ang_vel_w": (frames, len(body_names), 3),
        }
        for key, shape in expected_shapes.items():
            if data[key].shape != shape:
                raise SystemExit(f"{path.name}: {key} shape {data[key].shape}, expected {shape}")
            if not np.isfinite(data[key]).all():
                raise SystemExit(f"{path.name}: non-finite values in {key}")

        destination = staged_root / path.name
        try:
            os.link(path, destination)
        except OSError as error:
            if error.errno != errno.EXDEV:
                raise
            shutil.copy2(path, destination)

if reference_body_names is None or not {"base_link", "left_toe_link", "right_toe_link"}.issubset(reference_body_names):
    raise SystemExit("Dataset does not contain required DR02 AMP root/key bodies")

(staged_root / "bodies.json").write_text(
    json.dumps({"format_version": 1, "robot": "dr02", "body_names": reference_body_names}, indent=2) + "\n",
    encoding="utf-8",
)
(staged_root / "dataset_manifest.json").write_text(
    json.dumps(
        {
            "retargeted_npz_source": str(source_root),
            "source_motion_count": len(source_files),
            "motion_count": len(files),
            "fps": 50.0,
            "selection": "DR02_AMP_MOTION_EXCLUDES from flat_env_cfg.py",
            "excluded_motion_names": excluded_names,
            "body_names_source": files[0].name,
        },
        indent=2,
    ) + "\n",
    encoding="utf-8",
)
print(f"Prepared {len(files)} of {len(source_files)} motions at 50 Hz; 29 joints, {len(reference_body_names)} bodies")
print(f"Excluded {len(excluded_names)} motions, including all {len(stair_files)} stair motions: {stair_files}")
PY

cp "$ENV_CFG" "$AGENT_CFG" "$RUN_ROOT/source_snapshot/"
cp "$TASK_PACKAGE/robot_learning_lab_tasks/tasks/isaaclab/manager_based/amp/mdp/commands.py" \
   "$TASK_PACKAGE/robot_learning_lab_tasks/tasks/isaaclab/manager_based/amp/mdp/rewards.py" \
   "$TASK_PACKAGE/robot_learning_lab_tasks/tasks/isaaclab/manager_based/amp/mdp/events.py" \
   "$RUN_ROOT/source_snapshot/"
cp "$ROOT/scripts/reinforcement_learning/rsl_rl/viser_controls.py" \
   "$ROOT/scripts/reinforcement_learning/rsl_rl/reward_diagnostics.py" \
   "$ROOT/scripts/reinforcement_learning/rsl_rl/train.py" \
   "$RUN_ROOT/source_snapshot/"
git -C "$TASK_PACKAGE" diff --binary > "$RUN_ROOT/source_snapshot/task_worktree.patch"
git -C "$ROOT" diff --binary > "$RUN_ROOT/source_snapshot/robot_lab_worktree.patch"
printf '%s\n' "$RSL_RL_COMMIT" > "$RUN_ROOT/source_snapshot/rsl_rl_commit.txt"
git -C "$TASK_PACKAGE" status --short --branch > "$RUN_ROOT/source_snapshot/task_status.txt"
git -C "$ROOT" status --short --branch > "$RUN_ROOT/source_snapshot/robot_lab_status.txt"

printf 'Task: %s\nRetargeted NPZ: %s\nStaged motions: %s\nRSL-RL base commit: %s\nDataset contract: BeyondMimic WXYZ NPZ; RSL-RL converts to XYZW\nRSI sampler: RSL-RL root/joint state plus precomputed support height and vertical-velocity offsets\nResume: false\nSeed: 42\nTraining environments: 4096\nVisible environments: 512\nIterations: 25000\nSmoke tests: skipped by user request\n' \
  "$TASK" "$DATASET" "$MOTION_DIR" "$RSL_RL_COMMIT" | tee "$RUN_ROOT/launch.txt"

phase_root="$RUN_ROOT/train"
mkdir -p "$phase_root"
ACTIVE_CONTAINER="$CONTAINER_NAME"
docker run --rm --pull never --name "$CONTAINER_NAME" \
  --gpus all --shm-size 8g -p 8080:8080 --user "$(id -u):$(id -g)" \
  -e ACCEPT_EULA=Y -e OMNI_KIT_ACCEPT_EULA=YES -e HOME=/tmp -e XDG_CACHE_HOME=/tmp/.cache \
  -e PYTHONUNBUFFERED=1 -e RLL_MOTION_DATA_DIR=/workspace/motion-data \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  -v "$ROOT/source":/workspace/robot_lab/source:ro \
  -v "$ROOT/scripts":/workspace/robot_lab/scripts:ro \
  -v "$RSL_RL_SNAPSHOT":/opt/robot_lab_rsl_rl:ro \
  -v "$MOTION_DIR":/workspace/motion-data:ro \
  -v "$phase_root":/workspace/robot_lab/logs \
  --workdir /workspace/robot_lab --entrypoint /bin/bash "$IMAGE" \
  /workspace/isaaclab/isaaclab.sh -p /workspace/robot_lab/scripts/reinforcement_learning/rsl_rl/train.py \
  --task "$TASK" --visualizer viser --viser_start_paused --max_visible_envs 512 \
  --seed 42 --num_envs 4096 --max_iterations 25000 \
  --run_name dr02_amp_flat_soma_rsi_api_fresh_retry_4096x25000_20261006 \
  2>&1 | tee "$phase_root/docker.log"
ACTIVE_CONTAINER=""
