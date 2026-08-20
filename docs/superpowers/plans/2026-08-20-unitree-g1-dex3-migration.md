# Unitree G1 Dex3 Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add separately named G1 Dex3 assets and 43DoF task variants to Isaac Lab and MJLab without changing the existing G1 API or task behavior.

**Architecture:** Keep `UNITREE_G1_29DOF_CFG` as the existing model and build a parallel `DEX3` configuration from the checked-in hand URDF. Share body constants and task customization helpers, add hand-specific actuator/action mappings, and register new task IDs instead of changing old registrations. AMP policies may control all 43 joints, but AMP reference features remain body-only because current motion data has no hand trajectories.

**Tech Stack:** Python 3.10+, uv, pytest, Isaac Lab, MJLab, MuJoCo MJCF/URDF tooling, Gymnasium task registration

**Spec:** `docs/superpowers/specs/2026-08-20-unitree-g1-dex3-migration-design.md`

## Global Constraints

- `UNITREE_G1_29DOF_CFG`, existing action-scale symbols, existing task IDs, and existing configuration classes retain their current meanings.
- Only new three-finger-hand symbols and task IDs use the `DEX3`/`Dex3` suffix; do not introduce a `RUBBER_HAND` public name.
- Use `source/robot_learning_lab_zoo/robots/unitree/g1_description/urdf/g1_29dof_with_hand_rev_1_0.urdf` as the Dex3 source model.
- The Dex3 model has 29 body joints plus 14 independently actuated hand joints.
- Do not fabricate tactile sensors or AMP hand-reference trajectories.
- Use uv for builds, installs, and Python test execution. Use `/home/jvwei/mjlab/.venv` and `/home/jvwei/env_isaaclab` only through uv's selected interpreter or their explicitly activated environment.
- Do not launch training or disturb existing GPU, Pueue, tmux, container, checkpoint, log, dataset, or experiment state.

---

### Task 1: Lock down the public model distinction with source-level tests

**Files:**
- Create: `source/robot_learning_lab_zoo/test/test_unitree_g1_variants.py`
- Modify: `source/robot_learning_lab_zoo/test/test_mjlab_asset_references.py`

**Interfaces:**
- Consumes: G1 URDF/XML paths and public symbols in the Isaac Lab and MJLab Unitree modules.
- Produces: regression tests proving the old model stays unchanged and the Dex3 model has fourteen named hand joints.

- [ ] **Step 1: Write a backend-independent failing asset test**

  Parse both URDF files with `xml.etree.ElementTree`; assert the old file has 29 revolute/continuous joints and no joint matching `.*_hand_.*`, while the Dex3 file has 43 and exactly these suffixes on both sides:

  ```python
  DEX3_JOINT_SUFFIXES = {
      "thumb_0_joint", "thumb_1_joint", "thumb_2_joint",
      "index_0_joint", "index_1_joint",
      "middle_0_joint", "middle_1_joint",
  }

  assert {name.removeprefix("left_hand_") for name in left_hand_joints} == DEX3_JOINT_SUFFIXES
  assert {name.removeprefix("right_hand_") for name in right_hand_joints} == DEX3_JOINT_SUFFIXES
  ```

- [ ] **Step 2: Add failing source-contract assertions**

  Import each backend only when available and assert `UNITREE_G1_29DOF_DEX3_CFG` exists, is distinct from `UNITREE_G1_29DOF_CFG`, and its source path contains `with_hand`. Assert the old path remains the current non-Dex3 source. Extend the MJLab compile test expected count from 26 to 27 only after the distinct spec factory exists.

- [ ] **Step 3: Run the tests and verify RED**

  Run:

  ```bash
  uv run --python /home/jvwei/mjlab/.venv/bin/python --with pytest pytest source/robot_learning_lab_zoo/test/test_unitree_g1_variants.py source/robot_learning_lab_zoo/test/test_mjlab_asset_references.py -v
  ```

  Expected: backend-independent URDF assertions pass; public Dex3 symbol and compiled-spec assertions fail because the migration is not implemented.

- [ ] **Step 4: Commit the failing contract tests**

  ```bash
  git add source/robot_learning_lab_zoo/test/test_unitree_g1_variants.py source/robot_learning_lab_zoo/test/test_mjlab_asset_references.py
  git commit -m "test: define G1 Dex3 asset contract"
  ```

---

### Task 2: Add the Isaac Lab Dex3 articulation configuration

**Files:**
- Modify: `source/robot_learning_lab_zoo/robot_learning_lab_zoo/assets/isaaclab/unitree.py`
- Modify: `source/robot_learning_lab_zoo/test/test_unitree_g1_variants.py`

**Interfaces:**
- Consumes: existing G1 body actuator constants/configuration and the Dex3 URDF joint limits.
- Produces: `UNITREE_G1_29DOF_DEX3_CFG` and `UNITREE_G1_29DOF_DEX3_ACTION_SCALE`.

- [ ] **Step 1: Add failing hand-action assertions**

  Assert the old scale has no key containing `_hand_`, the Dex3 scale resolves all fourteen hand joint expressions, and the Dex3 configuration has a dedicated actuator named `hands` whose joint expression is `.*_hand_.*_joint`.

- [ ] **Step 2: Run the focused test and verify RED**

  Run the test through the Isaac Lab interpreter with uv:

  ```bash
  uv run --python /home/jvwei/env_isaaclab/bin/python --with pytest pytest source/robot_learning_lab_zoo/test/test_unitree_g1_variants.py -v
  ```

  Expected: failure because the new configuration/action mapping is absent.

- [ ] **Step 3: Extract a private G1 configuration factory and add Dex3 hand control**

  Keep the public old object unchanged in meaning. Build both objects through a private helper accepting `asset_path` and `include_dex3`; copy the existing initial state and actuator dictionaries, then add:

  ```python
  DEX3_HAND_JOINT_EXPR = ".*_hand_.*_joint"
  DEX3_HAND_STIFFNESS = 20.0
  DEX3_HAND_DAMPING = 0.5

  "hands": ImplicitActuatorCfg(
      joint_names_expr=[DEX3_HAND_JOINT_EXPR],
      effort_limit_sim={".*_hand_thumb_0_joint": 2.45, DEX3_HAND_JOINT_EXPR: 1.4},
      velocity_limit_sim={".*_hand_thumb_0_joint": 3.14, DEX3_HAND_JOINT_EXPR: 12.0},
      stiffness=DEX3_HAND_STIFFNESS,
      damping=DEX3_HAND_DAMPING,
  )
  ```

  Define a deterministic open/natural initial hand pose for all seven joints per side. Derive the new action mapping with the same `0.25 * effort / stiffness` rule without mutating the old mapping.

- [ ] **Step 4: Run Isaac Lab import/config tests and verify GREEN**

  Use the command from Step 2, followed by the existing zoo tests that do not start simulation. Confirm the old and Dex3 objects reference different URDF paths.

- [ ] **Step 5: Commit the Isaac Lab asset**

  ```bash
  git add source/robot_learning_lab_zoo/robot_learning_lab_zoo/assets/isaaclab/unitree.py source/robot_learning_lab_zoo/test/test_unitree_g1_variants.py
  git commit -m "feat: add Isaac Lab G1 Dex3 asset"
  ```

---

### Task 3: Add and compile the MJLab Dex3 model

**Files:**
- Create: `source/robot_learning_lab_zoo/robots/unitree/g1_description/xmls/g1_29dof_with_dex3_rev_1_0.xml`
- Modify: `source/robot_learning_lab_zoo/robot_learning_lab_zoo/assets/mjlab/unitree.py`
- Modify: `source/robot_learning_lab_zoo/robot_learning_lab_zoo/assets/mjlab/__init__.py`
- Modify: `source/robot_learning_lab_zoo/test/test_unitree_g1_variants.py`
- Modify: `source/robot_learning_lab_zoo/test/test_mjlab_asset_references.py`

**Interfaces:**
- Consumes: `g1_29dof_with_hand_rev_1_0.urdf`, existing G1 MuJoCo spec normalization, and existing body actuator groups.
- Produces: a self-contained Dex3 MJCF, `get_g1_dex3_spec()`, `G1_DEX3_ARTICULATION`, `UNITREE_G1_29DOF_DEX3_CFG`, and `G1_DEX3_ACTION_SCALE`.

- [ ] **Step 1: Extend the failing MJLab test**

  Assert the Dex3 compiled model reports 43 hinge joints, contains all fourteen hand joint names, contains actuators resolving those joints, has both foot sites and the IMU sensors, and loads meshes only from package-relative paths.

- [ ] **Step 2: Run focused MJLab tests and verify RED**

  Use the Task 1 MJLab command. Expected: failure because the Dex3 XML and MJLab config do not exist.

- [ ] **Step 3: Generate the MJCF with the installed MuJoCo toolchain**

  Use a temporary output for the first conversion and validate it before adding it:

  ```bash
  uv run --python /home/jvwei/mjlab/.venv/bin/python python -m mujoco.mjcf --help
  ```

  If that installed version has no supported CLI conversion entry point, use its Python `MjSpec.from_file(URDF).to_xml()` API from a short checked-in-free command. Normalize mesh paths relative to `g1_description`, retain all inertial/collision/joint-limit data, and add the resulting XML with `apply_patch`; do not hand-author robot geometry.

- [ ] **Step 4: Add the MJLab Dex3 config minimally**

  Introduce a shared private spec-normalization helper used by `get_spec()` and `get_g1_dex3_spec()`. Add one hand actuator group with the same gains and limits as the Isaac Lab config, append it only to `G1_DEX3_ARTICULATION`, and construct a distinct entity configuration and scale map. Export the new entity from `assets/mjlab/__init__.py`.

- [ ] **Step 5: Compile both MJLab variants and verify GREEN**

  Run both focused tests. Check `nq`, named joints, foot collision naming, foot/IMU sites, actuator resolution, and the full package asset-reference suite.

- [ ] **Step 6: Commit the MJLab asset**

  ```bash
  git add source/robot_learning_lab_zoo/robots/unitree/g1_description/xmls/g1_29dof_with_dex3_rev_1_0.xml source/robot_learning_lab_zoo/robot_learning_lab_zoo/assets/mjlab source/robot_learning_lab_zoo/test
  git commit -m "feat: add MJLab G1 Dex3 asset"
  ```

---

### Task 4: Add Dex3 velocity task variants for both backends

**Files:**
- Modify: `source/robot_learning_lab_tasks/robot_learning_lab_tasks/tasks/isaaclab/manager_based/locomotion/velocity/config/humanoid/unitree_g1/rough_env_cfg.py`
- Modify: `source/robot_learning_lab_tasks/robot_learning_lab_tasks/tasks/isaaclab/manager_based/locomotion/velocity/config/humanoid/unitree_g1/flat_env_cfg.py`
- Modify: `source/robot_learning_lab_tasks/robot_learning_lab_tasks/tasks/isaaclab/manager_based/locomotion/velocity/config/humanoid/unitree_g1/__init__.py`
- Modify: `source/robot_learning_lab_tasks/robot_learning_lab_tasks/tasks/mjlab/velocity/unitree_g1/env_cfgs.py`
- Modify: `source/robot_learning_lab_tasks/robot_learning_lab_tasks/tasks/mjlab/velocity/unitree_g1/__init__.py`
- Create: `source/robot_learning_lab_tasks/test/test_unitree_g1_dex3_registration.py`

**Interfaces:**
- Consumes: both backends' Dex3 asset configuration and action-scale mapping.
- Produces: `UnitreeG1Dex3RoughEnvCfg`, `UnitreeG1Dex3FlatEnvCfg`, `unitree_g1_dex3_rough_env_cfg()`, `unitree_g1_dex3_flat_env_cfg()`, and four new velocity task registrations.

- [ ] **Step 1: Write failing registration and action-space tests**

  Assert these IDs exist independently of their old counterparts:

  ```text
  RobotLab-Isaac-Velocity-Rough-Unitree-G1-Dex3-v0
  RobotLab-Isaac-Velocity-Flat-Unitree-G1-Dex3-v0
  RobotLab-MJLab-Velocity-Rough-Unitree-G1-Dex3
  RobotLab-MJLab-Velocity-Flat-Unitree-G1-Dex3
  ```

  Load each configuration and assert it uses the Dex3 entity and a joint-position scale that covers all 43 joints. Also assert all existing G1 IDs still load their old entity.

- [ ] **Step 2: Run backend-specific registration tests and verify RED**

  Run the new test once with each backend interpreter using `uv run --python ... --with pytest`. Expected: only the four new IDs/configurations fail.

- [ ] **Step 3: Refactor task customization into reusable private helpers**

  Preserve existing public factories/classes. Parameterize only robot configuration and action-scale inputs so old and Dex3 paths execute the same terrain, observation, reward, event, and play-mode customization. Add thin Dex3 subclass/factory wrappers that pass the new inputs.

- [ ] **Step 4: Register the four Dex3 velocity IDs**

  Reuse the existing runner configurations; give MJLab Dex3 runs a distinct `experiment_name="g1_dex3_velocity"` copy so outputs cannot collide with old G1 runs. Do not change existing runner objects or experiment names.

- [ ] **Step 5: Run focused and existing registration suites and verify GREEN**

  Confirm old task count expectations are updated only by the newly registered tasks, old IDs retain old paths, and no duplicate registration warnings appear.

- [ ] **Step 6: Commit velocity variants**

  ```bash
  git add source/robot_learning_lab_tasks/robot_learning_lab_tasks/tasks/isaaclab/manager_based/locomotion/velocity/config/humanoid/unitree_g1 source/robot_learning_lab_tasks/robot_learning_lab_tasks/tasks/mjlab/velocity/unitree_g1 source/robot_learning_lab_tasks/test/test_unitree_g1_dex3_registration.py
  git commit -m "feat: add G1 Dex3 velocity tasks"
  ```

---

### Task 5: Add body-referenced, whole-policy-action Dex3 AMP variants

**Files:**
- Modify: `source/robot_learning_lab_tasks/robot_learning_lab_tasks/tasks/isaaclab/manager_based/amp/config/g1/flat_env_cfg.py`
- Modify: `source/robot_learning_lab_tasks/robot_learning_lab_tasks/tasks/isaaclab/manager_based/amp/config/g1/__init__.py`
- Modify: `source/robot_learning_lab_tasks/robot_learning_lab_tasks/tasks/mjlab/amp/unitree_g1/env_cfgs.py`
- Modify: `source/robot_learning_lab_tasks/robot_learning_lab_tasks/tasks/mjlab/amp/unitree_g1/__init__.py`
- Modify: `source/robot_learning_lab_tasks/test/test_unitree_g1_dex3_registration.py`

**Interfaces:**
- Consumes: the Dex3 assets, 43-joint policy action scales, and existing 29-entry `G1_JOINT_NAMES`/link reference lists.
- Produces: `UnitreeG1Dex3AMPFlatEnvCfg`, `unitree_g1_dex3_amp_flat_env_cfg()`, and two new AMP registrations while retaining body-only AMP feature dimensions.

- [ ] **Step 1: Add failing AMP invariants**

  Assert these IDs register:

  ```text
  RobotLab-Isaac-AMP-Flat-Unitree-G1-Dex3-v0
  RobotLab-MJLab-AMP-Flat-Unitree-G1-Dex3
  ```

  For each loaded config assert: simulated/action-controlled joints total 43; action scale includes fourteen hand joints; AMP `joint_names` contains exactly the existing 29 body joints; AMP link names are unchanged; and the MJLab AMP URDF path ends in `g1_29dof_with_hand_rev_1_0.urdf`.

- [ ] **Step 2: Run the AMP tests and verify RED**

  Run the new focused test in both backend environments. Expected: new registrations/configs are missing while all old AMP invariants pass.

- [ ] **Step 3: Add thin Dex3 AMP configurations**

  Share the existing body-only constants. Parameterize robot and action scale, and for MJLab parameterize the kinematic URDF path. Do not append hand joints to `G1_JOINT_NAMES` and do not alter motion files or discriminator observation terms.

- [ ] **Step 4: Register and verify both AMP variants**

  Register the two IDs, reuse existing runner hyperparameters, and ensure any experiment/output name for Dex3 is distinct. Run focused tests plus existing AMP loader/runner tests.

- [ ] **Step 5: Commit AMP variants**

  ```bash
  git add source/robot_learning_lab_tasks/robot_learning_lab_tasks/tasks/isaaclab/manager_based/amp/config/g1 source/robot_learning_lab_tasks/robot_learning_lab_tasks/tasks/mjlab/amp/unitree_g1 source/robot_learning_lab_tasks/test/test_unitree_g1_dex3_registration.py
  git commit -m "feat: add G1 Dex3 AMP tasks"
  ```

---

### Task 6: Document, install, and smoke-test the migration with uv

**Files:**
- Modify: `source/robot_learning_lab_zoo/README.md`
- Modify: `source/robot_learning_lab_tasks/README.md`
- Modify only if required by a failing build: `source/robot_learning_lab_zoo/pyproject.toml`
- Modify only if required by a failing build: `source/robot_learning_lab_tasks/pyproject.toml`

**Interfaces:**
- Consumes: all new asset symbols and six new task IDs.
- Produces: user-facing selection examples and evidence that both packages install and import through uv.

- [ ] **Step 1: Add documentation contract checks**

  Extend the focused test to assert both READMEs mention `UNITREE_G1_29DOF_CFG` as the default/current model and `UNITREE_G1_29DOF_DEX3_CFG` as the explicit three-finger option, plus list all six Dex3 task IDs.

- [ ] **Step 2: Run the documentation assertions and verify RED**

  Expected: failure because the new names and IDs are undocumented.

- [ ] **Step 3: Update README examples**

  Show side-by-side imports without introducing a rubber-hand alias. Explain that Dex3 provides joint/contact simulation only, and that AMP hand actions have no demonstration supervision.

- [ ] **Step 4: Build and install both packages using uv**

  Run:

  ```bash
  uv build source/robot_learning_lab_zoo
  uv build source/robot_learning_lab_tasks
  uv pip install --python /home/jvwei/mjlab/.venv/bin/python -e 'source/robot_learning_lab_zoo[mjlab]' -e 'source/robot_learning_lab_tasks[mjlab]'
  uv pip install --python /home/jvwei/env_isaaclab/bin/python -e 'source/robot_learning_lab_zoo[isaaclab]' -e 'source/robot_learning_lab_tasks[isaaclab]'
  ```

  Do not substitute `pip install` commands.

- [ ] **Step 5: Run final backend verification**

  MJLab: run all zoo asset tests, task registration tests, compile both G1 specs, and instantiate (but do not step/train) flat Dex3 velocity and AMP configs. Isaac Lab: import both assets, enumerate old/new Gymnasium registrations, and perform the smallest headless configuration/URDF conversion smoke test allowed by current GPU/process state. Before any simulator App launch, inspect `nvidia-smi`, `free -h`, and `pueue status`; skip and report the runtime smoke test if resources are occupied.

- [ ] **Step 6: Run quality and repository checks**

  ```bash
  uvx ruff check source/robot_learning_lab_zoo source/robot_learning_lab_tasks
  git diff --check
  git status --short --branch
  ```

  Confirm no generated caches, build distributions, simulator outputs, or experiment data are staged.

- [ ] **Step 7: Commit documentation and any verified packaging fix**

  ```bash
  git add source/robot_learning_lab_zoo/README.md source/robot_learning_lab_tasks/README.md source/robot_learning_lab_zoo/pyproject.toml source/robot_learning_lab_tasks/pyproject.toml
  git commit -m "docs: distinguish G1 Dex3 workflows"
  ```

