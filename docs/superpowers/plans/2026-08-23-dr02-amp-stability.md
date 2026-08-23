# DR02 AMP Stability and Velocity Task Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans (recommended) to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stabilize DR02 Pro AMP and combine motion style reward with velocity-tracking RL rewards.

**Architecture:** Keep AMP in the local RSL-RL fork. The motion dataset owns expert contract validation and root-frame conversion; the AMP algorithm owns valid online transition filtering; the Isaac task config owns DR02 gains, commands, observations, and reward weights.

**Tech Stack:** Python, PyTorch, RSL-RL, Isaac Lab, pytest, Pueue.

**Spec:** `docs/superpowers/specs/2026-08-23-dr02-amp-stability-design.md`

## Global Constraints

- Preserve the 29-DoF order in `~/GMR-private/retarget_data/dr02/joints.json`.
- Use the existing `dr02_pos.xml` control groups as the gain source.
- Do not delete existing logs or checkpoints.
- Use one GPU task at a time and run smoke tests before formal training.
- Do not add explicit gait timing/contact shaping rewards.

---

### Task 1: Add failing dataset-contract tests

**Files:**
- Modify: `/home/jvwei/rsl_rl/tests/algorithms/test_amp.py`
- Test: `/home/jvwei/rsl_rl/tests/algorithms/test_amp.py`

- [ ] **Step 1: Add tests** for rejecting an invalid joint contract, root-frame body conversion under a rotated root, and sampling no terminal motion frame.
- [ ] **Step 2: Run the focused tests** with `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q tests/algorithms/test_amp.py` and confirm the new assertions fail for the current loader.

### Task 2: Fix the motion dataset contract and frame conversion

**Files:**
- Modify: `/home/jvwei/rsl_rl/rsl_rl/datasets/motion_dataset.py`
- Test: `/home/jvwei/rsl_rl/tests/algorithms/test_amp.py`

- [ ] **Step 1: Implement explicit joint-name validation and root-frame conversion.** Use the configured external contract, compute key-body indices by names, and apply the inverse root quaternion to body offsets.
- [ ] **Step 2: Exclude the final frame of each motion from transition start indices.** Raise a clear error for motions shorter than two frames.
- [ ] **Step 3: Run the focused tests** and confirm all pass.

### Task 3: Filter invalid online AMP transitions

**Files:**
- Modify: `/home/jvwei/rsl_rl/rsl_rl/algorithms/amp.py`
- Test: `/home/jvwei/rsl_rl/tests/algorithms/test_amp.py`

- [ ] **Step 1: Add a test** proving terminated environments are not sent to the discriminator transition batch.
- [ ] **Step 2: Implement filtering** in `process_env_step`, preserving PPO termination rewards while excluding reset-state AMP transitions; skip discriminator update only when no valid transition exists.
- [ ] **Step 3: Run all local RSL-RL AMP tests** and confirm they pass.

### Task 4: Add DR02 velocity task settings and validated gains

**Files:**
- Modify: `/home/jvwei/robot_lab/source/robot_learning_lab_zoo/robot_learning_lab_zoo/assets/isaaclab/deeprobotics.py`
- Modify: `/home/jvwei/robot_lab/source/robot_learning_lab_tasks/robot_learning_lab_tasks/tasks/isaaclab/manager_based/amp/tracking_env_cfg.py`
- Modify: `/home/jvwei/robot_lab/source/robot_learning_lab_tasks/robot_learning_lab_tasks/tasks/isaaclab/manager_based/amp/config/dr02/flat_env_cfg.py`
- Modify: `/home/jvwei/robot_lab/source/robot_learning_lab_tasks/robot_learning_lab_tasks/tasks/isaaclab/manager_based/amp/config/dr02/agents/rsl_rl_amp_cfg.py`
- Modify: `/home/jvwei/robot_lab/tests/test_dr02_finetune_assets.py`

- [ ] **Step 1: Add static tests** for all actuator gain groups, command observation presence, enabled velocity rewards, disabled gait terms, and task/style scales.
- [ ] **Step 2: Run tests to confirm the new configuration assertions fail.**
- [ ] **Step 3: Implement the gain groups, command observation terms, velocity rewards, non-gait RL rewards, and `task_reward_scale=1.0`.**
- [ ] **Step 4: Run static tests and Python compilation.**

### Task 5: Smoke test and formal training submission

**Files:**
- Modify: `/home/jvwei/robot_lab/scripts/pueue_train_dr02_pro_amp_all.sh` only if smoke/full settings need an explicit configuration flag.

- [ ] **Step 1: Run focused unit tests and DR02 static tests.**
- [ ] **Step 2: Run Isaac Lab AMP smoke test with a small environment count and one or two iterations.** Verify no exception, AMP discriminator update, and episode length greater than one.
- [ ] **Step 3: Check `nvidia-smi`, `free -h`, and `pueue status`.**
- [ ] **Step 4: Submit the full all-data run with a new unique output directory through the existing wrapper.**
- [ ] **Step 5: Verify initial logs, selected file count, total frames, first iteration, and checkpoint creation.
