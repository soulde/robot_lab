# Task Migration Completion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Finish the interrupted task-package migration so clean backend installs import successfully, all migrated MJLab tasks register, referenced robot assets exist, and the top-level repository records every migrated submodule.

**Architecture:** Keep simulator-specific registration isolated under `robot_learning_lab_tasks.tasks.{isaaclab,mjlab}`. Treat `robot_learning_lab_zoo`, `robot_learning_lab_datasets`, and `rll_rl` as explicit package dependencies rather than relying on the monorepo source tree being on `PYTHONPATH`. Validate migration boundaries with lightweight package and registry tests before any simulator smoke test.

**Tech Stack:** Python 3.13, uv, PEP 517/621, setuptools, pytest, MJLab task registry, Git submodules

**Spec:** User request on 2026-08-20 to inspect and complete the interrupted task migration; repository state at `robot_learning_lab_tasks@5690147` (`not finish yet`).

## Global Constraints

- Use uv for package builds, dependency resolution, and installation; use `/home/jvwei/mjlab/.venv` for MJLab checks and `/home/jvwei/env_isaaclab` for Isaac Lab checks.
- Do not start training; simulator validation is limited to import, registry, and minimal smoke checks.
- Preserve all pre-existing edits across the dirty submodules and never overwrite experiment data or assets.
- Do not commit or push without an explicit user request; keep changes reviewable in their owning repositories.

---

### Task 1: Make the MJLab package dependency boundary self-contained

**Files:**
- Create: `source/robot_learning_lab_tasks/pyproject.toml`
- Modify: `source/robot_learning_lab_tasks/setup.py`
- Modify: `source/robot_learning_lab_tasks/test/test_package_structure.py`

**Interfaces:**
- Consumes: `robot_learning_lab_tasks.tasks.mjlab.amp.unitree_g1` imports `AMPRunner` and `AMPRunnerCfg` from `rll_rl`.
- Produces: the `mjlab` optional dependency set includes the package needed by every eager registration import.

- [x] Add a test that runs `uv build` and checks the built wheel declares `mjlab`, `robot_learning_lab_zoo[mjlab]`, `rll_rl[amp]`, and the MJLab entry point.
- [x] Run uv against the legacy package and confirm it fails without PEP 517 metadata and again when the sdist omits `config/extension.toml`.
- [x] Add complete PEP 621 metadata and reduce `setup.py` to a compatibility shim.
- [x] Run package tests, `uv build`, and a full local-package `uv pip install --dry-run` successfully.

### Task 2: Lock down migrated MJLab registration coverage

**Files:**
- Create: `source/robot_learning_lab_tasks/test/test_mjlab_registration.py`
- Modify only the specific registration/config modules exposed by failing assertions.

**Interfaces:**
- Consumes: `mjlab.tasks.registry.list_tasks`, `load_env_cfg`, `load_rl_cfg`, and `load_runner_cls`.
- Produces: a test that imports the backend and verifies all migrated robot task IDs plus train/play and runner configurations.

- [x] Add a subprocess-based test with explicit local dependency paths that imports the MJLab backend and verifies the expected 54 RobotLab task IDs.
- [x] Run it before production edits and capture any missing or duplicate registrations.
- [x] Correct only confirmed registration/config defects (none found).
- [x] Re-run the focused and complete tasks test suites.

### Task 3: Validate migrated asset references

**Files:**
- Create: `source/robot_learning_lab_zoo/test/test_mjlab_asset_references.py`
- Modify only asset modules or model files proven invalid by the test.

**Interfaces:**
- Consumes: exported MJLab robot configuration objects and their MJCF/URDF mesh paths.
- Produces: a lightweight invariant that every source model and referenced mesh exists inside the zoo package.

- [x] Add tests that enumerate exported migrated MJLab robot configs and compile every unique model spec without starting an environment.
- [x] Run the tests and record exact missing paths or malformed references (all 26 unique specs compiled).
- [x] Repair confirmed path/export defects while preserving existing model edits (none found); update the stale manufacturer inventory for `anybotics`.
- [x] Re-run zoo and tasks tests.

### Task 4: Verify both backend boundaries and repository integration

**Files:**
- Modify: `.gitmodules` only if the existing sim-infer addition is valid and complete.
- Modify: top-level dependency/install documentation only where verification shows stale commands.

**Interfaces:**
- Consumes: the completed tasks, zoo, datasets, rll_rl, and sim-infer repositories.
- Produces: reproducible import/registry commands and valid top-level submodule metadata.

- [x] Run the MJLab import/list command in `/home/jvwei/mjlab/.venv` with local packages available.
- [ ] Run Isaac Lab package structure/import checks in `/home/jvwei/env_isaaclab` without launching training (structure passed; runtime App enumeration deferred because an existing process occupies 17.1 GiB GPU memory at 93% utilization).
- [x] Run `pytest` for tasks, zoo, datasets, and rll_rl tests relevant to migration.
- [x] Verify `.gitmodules`, gitlinks, and nested repository remotes agree; publish sim-infer and register gitlink `611086d` in the top-level repository.
