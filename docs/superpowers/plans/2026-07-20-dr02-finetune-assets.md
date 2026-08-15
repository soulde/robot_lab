# DR02 Finetune Assets Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Synchronize robot_lab DR02 Standard and Pro assets and actuator parameters with their finetuned robot-zoo models.

**Architecture:** Keep robot assets vendored under robot_lab's existing data tree. Point the unchanged public articulation configs at those assets, enforce user-specific conversion caches, and verify the model/config contract with static tests.

**Tech Stack:** Python, Isaac Lab ArticulationCfg, URDF/XML, pytest

## Global Constraints

- Preserve `DEEPROBOTICS_DR02_STANDARD_CFG` and `DEEPROBOTICS_DR02_PRO_CFG`.
- Do not add a runtime dependency on robot_lab_zoo.
- Require `TMPDIR` and use only its per-robot IsaacLab cache directories.

---

### Task 1: Synchronize DR02 descriptions and configuration

**Files:**
- Modify: `source/robot_lab/data/Robots/deeprobotics/dr02_standard_description/urdf/dr02_std.urdf`
- Modify: `source/robot_lab/data/Robots/deeprobotics/dr02_pro_description/urdf/dr02_pro.urdf`
- Modify: `source/robot_lab/robot_lab/assets/deeprobotics.py`
- Create: `tests/test_dr02_finetune_assets.py`

**Interfaces:**
- Consumes: `TMPDIR`, local URDF and mesh trees, Isaac Lab `DCMotorCfg`.
- Produces: the existing Standard and Pro `ArticulationCfg` constants.

- [ ] **Step 1: Add static tests for joint counts, mesh resolution, cache policy, and gain groups**
- [ ] **Step 2: Run the tests and confirm they fail against the old configuration**
- [ ] **Step 3: Replace both URDFs and update actuator groups and cache validation**
- [ ] **Step 4: Run static tests, Ruff, and Python compilation**
- [ ] **Step 5: Run one-environment Isaac Lab articulation smoke tests for both models**
- [ ] **Step 6: Commit the synchronized assets and configuration**
