#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/jvwei/robot_lab"
RSL_RL_DIR="/home/jvwei/rsl_rl"
RUN_ROOT="$ROOT/logs/dr02_amp_flat_bounds_jointscale_torso_viser_fresh_4096x5000_20261004"
MOTION_DIR="/home/jvwei/datasets/dr02_kit_locomotion_with_toe"
IMAGE="sha256:5f6f461a739232b822771a42893b0f159946e5f9ae5a3a21bba2d9cb7a56ef09"
TASK="RobotLab-Isaac-AMP-Flat-Deeprobotics-DR02-Pro-v0"
ENV_CFG="$ROOT/source/robot_learning_lab_tasks/robot_learning_lab_tasks/tasks/isaaclab/manager_based/amp/config/dr02/flat_env_cfg.py"
AGENT_CFG="$ROOT/source/robot_learning_lab_tasks/robot_learning_lab_tasks/tasks/isaaclab/manager_based/amp/config/dr02/agents/rsl_rl_amp_cfg.py"
export ACCEPT_EULA=Y

cd "$ROOT"
[[ -f "$MOTION_DIR/bodies.json" ]]
[[ "$(git -C "$RSL_RL_DIR" branch --show-current)" == "debug/dr02-amp-training" ]]
[[ ! -e "$RUN_ROOT" ]] || { echo "Refusing to reuse output: $RUN_ROOT" >&2; exit 2; }
docker image inspect "$IMAGE" >/dev/null

mkdir -p "$RUN_ROOT/smoke" "$RUN_ROOT/train" "$RUN_ROOT/source_snapshot"
cp "$ENV_CFG" "$AGENT_CFG" "$RUN_ROOT/source_snapshot/"
cp "$ROOT/source/robot_learning_lab_tasks/robot_learning_lab_tasks/tasks/isaaclab/manager_based/amp/mdp/commands.py" "$ROOT/source/robot_learning_lab_tasks/robot_learning_lab_tasks/tasks/isaaclab/manager_based/amp/mdp/rewards.py" "$RUN_ROOT/source_snapshot/"
cp "$ROOT/scripts/reinforcement_learning/rsl_rl/viser_controls.py" "$RUN_ROOT/source_snapshot/"
cp "$ROOT/scripts/reinforcement_learning/rsl_rl/reward_diagnostics.py" "$ROOT/scripts/reinforcement_learning/rsl_rl/train.py" "$RUN_ROOT/source_snapshot/"
git -C "$ROOT/source/robot_learning_lab_tasks" diff --binary > "$RUN_ROOT/source_snapshot/task_worktree.patch"
git -C "$ROOT" diff --binary > "$RUN_ROOT/source_snapshot/robot_lab_worktree.patch"
git -C "$RSL_RL_DIR" rev-parse HEAD > "$RUN_ROOT/source_snapshot/rsl_rl_commit.txt"
git -C "$RSL_RL_DIR" diff --binary > "$RUN_ROOT/source_snapshot/rsl_rl_worktree.patch"
git -C "$RSL_RL_DIR" status --short --branch > "$RUN_ROOT/source_snapshot/rsl_rl_status.txt"
printf 'Task: %s\nImage: %s\nMotion directory: %s\nOutput: %s\nRSL-RL branch: %s\nRSL-RL base: %s\nResume: false\nSeed: 42\nEnvironments: 4096\nIterations: 5000\nRMS weights: linear=-0.2 angular=-0.1\nAMP updates: shared Adam, one per PPO minibatch\nGradient clipping: actor and critic at 1.0; discriminator unclipped\n' \
  "$TASK" "$IMAGE" "$MOTION_DIR" "$RUN_ROOT" "$(git -C "$RSL_RL_DIR" branch --show-current)" \
  "$(git -C "$RSL_RL_DIR" rev-parse HEAD)" | tee "$RUN_ROOT/launch.txt"

run_training() {
  local phase="$1"
  shift
  local container_name="dr02-flat-torso-viser-fresh-${phase}-20261004"
  docker run --rm --pull never --name "$container_name" \
    --gpus all --shm-size 8g -p 8080:8080 --user "$(id -u):$(id -g)" \
    -e ACCEPT_EULA=Y -e OMNI_KIT_ACCEPT_EULA=YES -e HOME=/tmp -e XDG_CACHE_HOME=/tmp/.cache \
    -e PYTHONUNBUFFERED=1 -e RLL_MOTION_DATA_DIR=/workspace/motion-data \
    -e NVIDIA_DRIVER_CAPABILITIES=all \
    -v "$ROOT/source":/workspace/robot_lab/source:ro \
    -v "$ROOT/scripts":/workspace/robot_lab/scripts:ro \
    -v "$RSL_RL_DIR":/opt/robot_lab_rsl_rl:ro \
    -v "$MOTION_DIR":/workspace/motion-data:ro \
    -v "$RUN_ROOT/$phase":/workspace/robot_lab/logs \
    --workdir /workspace/robot_lab --entrypoint /bin/bash "$IMAGE" \
    /workspace/isaaclab/isaaclab.sh -p /workspace/robot_lab/scripts/reinforcement_learning/rsl_rl/train.py \
    --task "$TASK" --visualizer viser --viser_start_paused --max_visible_envs 512 --reward_diagnostics --seed 42 "$@"
}

run_training smoke --num_envs 1 --max_iterations 3 \
  --run_name dr02_amp_flat_bounds_jointscale_torso_viser_fresh_smoke_20261004

source /home/jvwei/env_isaaclab_ea/bin/activate
python - "$RUN_ROOT/smoke" <<'PY'
from pathlib import Path
import sys

import torch
import yaml

root = Path(sys.argv[1])
agent_path = next(root.rglob("agent.yaml"))
env_path = agent_path.with_name("env.yaml")
agent = yaml.load(agent_path.read_text(), Loader=yaml.BaseLoader)
env = yaml.load(env_path.read_text(), Loader=yaml.BaseLoader)
assert env["rewards"]["track_lin_vel_xy_rms"]["weight"] == "-0.2"
assert env["rewards"]["track_ang_vel_z_rms"]["weight"] == "-0.1"
assert agent["algorithm"]["max_grad_norm"] == "1.0"
assert float(agent["clip_actions"]) == 100.0
assert "track_body_lin_vel_xy_yaw_frame_exp" in env["rewards"]["track_lin_vel_xy_exp"]["func"]
assert "track_body_lin_vel_xy_yaw_frame_rms" in env["rewards"]["track_lin_vel_xy_rms"]["func"]
assert "HorizontalVelocityCommand" in env["commands"]["base_velocity"]["class_type"]
assert env["commands"]["base_velocity"]["velocity_body_name"] == "body"
for name in ("track_lin_vel_xy_exp", "track_lin_vel_xy_rms", "track_ang_vel_z_exp", "track_ang_vel_z_rms"):
    assert env["rewards"][name]["params"]["body_name"] == "body"
for group in ("policy", "critic"):
    terms = [v for v in env["observations"][group].values() if isinstance(v, dict) and "func" in v]
    assert terms
    for term in terms:
        assert [float(x) for x in term["clip"]] == [-100.0, 100.0]
assert float(env["terminations"]["torso_to_toe_height"]["params"]["height_ratio"]) == 0.35
for name, base in {"joint_acc_l2": -2.5e-7, "joint_torques_l2": -1e-4, "action_rate_l2": -0.1, "joint_pos_limits": -2.0}.items():
    assert abs(float(env["rewards"][name]["weight"]) - base * 23 / 29) < abs(base) * 1e-8
assert float(env["rewards"]["undesired_contacts"]["weight"]) == -10.0
import json
rows = [json.loads(line) for line in next(root.rglob("rollout.jsonl")).read_text().splitlines()]
assert {r["kind"] for r in rows} == {"reward", "targets", "update"}
assert agent["resume"] == "false"
checkpoint = next(root.rglob("model_2.pt"))
state = torch.load(checkpoint, map_location="cpu", weights_only=False)

def check_finite(value):
    if isinstance(value, torch.Tensor) and value.is_floating_point():
        assert torch.isfinite(value).all(), "non-finite tensor in smoke checkpoint"
    elif isinstance(value, dict):
        for item in value.values():
            check_finite(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            check_finite(item)

check_finite(state)
print("SMOKE_PASSED: bounds, clearance, 23/29 penalties, diagnostics, finite model_2", flush=True)
PY

echo 'Smoke passed; starting the full 4096-environment fresh run.'
run_training train --num_envs 4096 --max_iterations 5000 \
  --seed 42 \
  --run_name dr02_amp_flat_bounds_jointscale_torso_viser_fresh_4096x5000_20261004
