# Unitree G1 Dex3 Backpack Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add independently selectable G1 Dex3 variants carrying a fixed, visible, collidable 1 kg cuboid backpack to Isaac Lab and MJLab, with six distinct task registrations.

**Architecture:** Derive checked-in URDF and MJCF files from the existing complete Dex3 assets, appending only one fixed backpack link/body. Construct new backend configurations by deep-copying the Dex3 configurations and changing only the source model path, then add thin task subclasses/factories that select the backpack configuration while preserving the 43-joint action and body-only AMP reference contracts.

**Tech Stack:** Python 3.10+, uv, pytest, XML, URDF, MuJoCo MJCF, Isaac Lab, MJLab, Gymnasium

**Spec:** `docs/superpowers/specs/2026-08-20-unitree-g1-dex3-backpack-design.md`

## Global Constraints

- Keep all existing G1 and G1 Dex3 public symbols, source files, task IDs, and behavior unchanged.
- Backpack dimensions are exactly `(0.25, 0.20, 0.30)` m in `(X, Y, Z)`.
- Backpack pose relative to `torso_link` is exactly `xyz=(-0.12, 0.0, 0.05)` m with identity orientation.
- Backpack mass is exactly `1.0` kg with diagonal inertia `(0.0108333333, 0.0133333333, 0.0085416667)` kg m^2 and zero products of inertia.
- The backpack is fixed and adds no actuator; the policy action space remains 43 joints.
- AMP reference features remain the existing 29 body joints and existing link list.
- Build, install, and Python verification commands use uv; do not use pip directly.
- Do not launch training or modify experiment data, checkpoints, containers, Pueue tasks, or tmux sessions.

---

### Task 1: Define and build the derived backpack assets

**Files:**
- Create: `source/robot_learning_lab_zoo/robots/unitree/g1_description/urdf/g1_29dof_with_hand_backpack_1kg.urdf`
- Create: `source/robot_learning_lab_zoo/robots/unitree/g1_description/xmls/g1_29dof_with_hand_backpack_1kg.xml`
- Modify: `source/robot_learning_lab_zoo/test/test_unitree_g1_variants.py`

**Interfaces:**
- Consumes: the existing complete Dex3 URDF/MJCF.
- Produces: two derived models with a fixed `backpack_link`, unchanged 43-joint robot mechanics, and exactly 1 kg additional mass.

- [ ] **Step 1: Write failing URDF behavior assertions**

  Parse the derived URDF using `ElementTree`. Assert `backpack_joint` is fixed from `torso_link` to `backpack_link`; origin is the literal `-0.12 0 0.05`; visual/collision boxes are `0.25 0.20 0.30`; mass is `1.0`; inertia literals match the global constraints; and the actuated joint list is identical to ordinary Dex3's 43 names.

- [ ] **Step 2: Write failing MJCF compile assertions**

  Compile ordinary Dex3 and backpack specs. Assert the backpack model contains one named body and its visual/collision geoms, has the same 44 MuJoCo joints including free root, and satisfies:

  ```python
  assert backpack_model.body_mass.sum() == pytest.approx(dex3_model.body_mass.sum() + 1.0)
  ```

- [ ] **Step 3: Run focused tests and verify RED**

  ```bash
  PYTHONPATH=source/robot_learning_lab_zoo MPLCONFIGDIR=/tmp/matplotlib \
    UV_CACHE_DIR=/tmp/robot-lab-uv-cache \
    uv run --python /home/jvwei/mjlab/.venv/bin/python --no-project \
    python -m pytest source/robot_learning_lab_zoo/test/test_unitree_g1_variants.py -v
  ```

  Expected: failures because both derived files and public spec are absent.

- [ ] **Step 4: Create the derived URDF with only the backpack addition**

  Copy the existing Dex3 URDF mechanically, change its robot name, and append a `backpack` material plus `backpack_link` inertial/visual/collision and fixed `backpack_joint` immediately before `</robot>`. Use a box, not a mesh, and do not alter any existing body or hand element.

- [ ] **Step 5: Create the derived MJCF with only the backpack addition**

  Copy the existing Dex3 MJCF mechanically, change its model name, locate `torso_link`, and append a fixed child body at the specified pose. Give it one mass/inertia declaration, one non-colliding visual box geom, and one colliding box geom with half-size `0.125 0.10 0.15`; preserve all existing mesh paths and robot content.

- [ ] **Step 6: Re-run focused tests and verify the asset portion is GREEN**

  Confirm the two files differ from ordinary Dex3 only by model name and backpack elements, then commit:

  ```bash
  git add robots/unitree/g1_description/urdf/g1_29dof_with_hand_backpack_1kg.urdf robots/unitree/g1_description/xmls/g1_29dof_with_hand_backpack_1kg.xml test/test_unitree_g1_variants.py
  git commit -m "feat: add G1 Dex3 1kg backpack models"
  ```

---

### Task 2: Expose backpack configurations in both asset backends

**Files:**
- Modify: `source/robot_learning_lab_zoo/robot_learning_lab_zoo/assets/isaaclab/unitree.py`
- Modify: `source/robot_learning_lab_zoo/robot_learning_lab_zoo/assets/mjlab/unitree.py`
- Modify: `source/robot_learning_lab_zoo/robot_learning_lab_zoo/assets/mjlab/__init__.py`
- Modify: `source/robot_learning_lab_zoo/test/test_unitree_g1_variants.py`
- Modify: `source/robot_learning_lab_zoo/test/test_mjlab_asset_references.py`

**Interfaces:**
- Consumes: the two derived files and existing Dex3 configurations/scales.
- Produces: `UNITREE_G1_29DOF_DEX3_BACKPACK_CFG` in both backends, `UNITREE_G1_29DOF_DEX3_BACKPACK_ACTION_SCALE`, and `G1_DEX3_BACKPACK_ACTION_SCALE`.

- [ ] **Step 1: Add failing public-configuration assertions**

  Assert each backpack config is a distinct object pointing at the derived file, while its actuator names and action scale equal the ordinary Dex3 value. Assert no actuator/action expression contains `backpack`.

- [ ] **Step 2: Run the MJLab and available Isaac tests and verify RED**

  Use the Task 1 command for MJLab. For Isaac Lab, use its explicitly activated environment and disable third-party pytest plugin auto-loading; if `pxr` remains unavailable, record the dependency failure and retain backend-independent source/URDF checks.

- [ ] **Step 3: Implement minimal copied configurations**

  Deep-copy the corresponding Dex3 config and change only its spawn/spec source. Copy action-scale dictionaries rather than aliasing mutable mappings. Add `G1_DEX3_BACKPACK_XML` and `get_g1_dex3_backpack_spec()` through the existing shared G1 spec normalizer, then export the MJLab entity.

- [ ] **Step 4: Run tests and compile both MJLab variants**

  Verify ordinary Dex3 total mass is unchanged, backpack mass delta is 1 kg, both have 43 actuated joints, and all exported unique specs compile. Commit:

  ```bash
  git add robot_learning_lab_zoo/assets test
  git commit -m "feat: expose G1 Dex3 backpack assets"
  ```

---

### Task 3: Add six separately registered backpack tasks

**Files:**
- Modify: `source/robot_learning_lab_tasks/robot_learning_lab_tasks/tasks/isaaclab/manager_based/locomotion/velocity/config/humanoid/unitree_g1/rough_env_cfg.py`
- Modify: `source/robot_learning_lab_tasks/robot_learning_lab_tasks/tasks/isaaclab/manager_based/locomotion/velocity/config/humanoid/unitree_g1/flat_env_cfg.py`
- Modify: `source/robot_learning_lab_tasks/robot_learning_lab_tasks/tasks/isaaclab/manager_based/locomotion/velocity/config/humanoid/unitree_g1/__init__.py`
- Modify: `source/robot_learning_lab_tasks/robot_learning_lab_tasks/tasks/isaaclab/manager_based/amp/config/g1/flat_env_cfg.py`
- Modify: `source/robot_learning_lab_tasks/robot_learning_lab_tasks/tasks/isaaclab/manager_based/amp/config/g1/__init__.py`
- Modify: `source/robot_learning_lab_tasks/robot_learning_lab_tasks/tasks/mjlab/velocity/unitree_g1/env_cfgs.py`
- Modify: `source/robot_learning_lab_tasks/robot_learning_lab_tasks/tasks/mjlab/velocity/unitree_g1/rl_cfg.py`
- Modify: `source/robot_learning_lab_tasks/robot_learning_lab_tasks/tasks/mjlab/velocity/unitree_g1/__init__.py`
- Modify: `source/robot_learning_lab_tasks/robot_learning_lab_tasks/tasks/mjlab/amp/unitree_g1/env_cfgs.py`
- Modify: `source/robot_learning_lab_tasks/robot_learning_lab_tasks/tasks/mjlab/amp/unitree_g1/rl_cfg.py`
- Modify: `source/robot_learning_lab_tasks/robot_learning_lab_tasks/tasks/mjlab/amp/unitree_g1/__init__.py`
- Modify: `source/robot_learning_lab_tasks/test/test_unitree_g1_dex3_registration.py`
- Modify: `source/robot_learning_lab_tasks/test/test_mjlab_registration.py`

**Interfaces:**
- Consumes: backpack asset configurations and unchanged Dex3 action scales.
- Produces: six task IDs from the spec, thin backpack env classes/factories, and backpack-specific runner experiment names.

- [ ] **Step 1: Add failing registration and selection tests**

  Assert all six literal IDs register separately. Load MJLab old, Dex3, and backpack velocity configs and compare spec functions. Load the backpack AMP config and assert it uses the derived URDF, has 29 AMP reference joints, and its action scale contains hand joints but no backpack entry.

- [ ] **Step 2: Run the task tests and verify RED**

  Use uv with the MJLab interpreter and local package paths. Expected: only the six backpack IDs and factories are absent.

- [ ] **Step 3: Add thin backpack velocity variants**

  Derive from the existing Dex3 classes/factories, replace only the robot entity with the backpack entity, retain the Dex3 action scale, and register rough/flat IDs. Add runner config copies whose only semantic change is a backpack-specific experiment name.

- [ ] **Step 4: Add thin backpack AMP variants**

  Derive from existing Dex3 AMP configurations, select the backpack asset/URDF, retain `G1_JOINT_NAMES` and `G1_AMP_LINK_NAMES`, and register the two IDs. Give each backend a distinct backpack experiment name.

- [ ] **Step 5: Run focused and complete registration tests and verify GREEN**

  Update the exact expected task set by six and ensure no duplicate warnings. Run Python compilation for both backend trees, then commit:

  ```bash
  git add robot_learning_lab_tasks test
  git commit -m "feat: add G1 Dex3 backpack tasks"
  ```

---

### Task 4: Document and verify uv wheel installation

**Files:**
- Modify: `source/robot_learning_lab_zoo/README.md`
- Modify: `source/robot_learning_lab_tasks/README.md`

**Interfaces:**
- Consumes: backpack symbols, files, and task IDs.
- Produces: selection documentation plus install/build evidence independent of the source checkout.

- [ ] **Step 1: Document the backpack variant**

  Add the new symbol beside ordinary Dex3 and list all six task IDs. State the physical dimensions, 1 kg fixed payload, 43-joint action count, and body-only AMP references.

- [ ] **Step 2: Build both wheels with uv**

  ```bash
  UV_CACHE_DIR=/tmp/robot-lab-uv-cache uv build source/robot_learning_lab_zoo
  UV_CACHE_DIR=/tmp/robot-lab-uv-cache uv build source/robot_learning_lab_tasks
  ```

- [ ] **Step 3: Inspect and install the wheels with uv**

  Confirm both derived asset paths appear in the zoo wheel. Install the zoo/tasks wheels with `uv pip install --python /home/jvwei/mjlab/.venv/bin/python --force-reinstall`, ensure local `rll_rl` is installed through uv, then import the installed packages, compile the backpack MJCF, and enumerate all three MJLab backpack task IDs without source `PYTHONPATH`.

- [ ] **Step 4: Run final verification**

  Run the complete zoo/tasks test suites with `--import-mode=importlib`, Ruff on every changed Python file, `git diff --check`, and clean-status checks in both subrepositories and the top-level repository. Before any Isaac Sim App smoke test, inspect `nvidia-smi`, `free -h`, and `pueue status`; skip and report it if runtime or resources are unavailable.

- [ ] **Step 5: Commit documentation**

  ```bash
  git add README.md
  git commit -m "docs: describe G1 Dex3 backpack workflows"
  ```

