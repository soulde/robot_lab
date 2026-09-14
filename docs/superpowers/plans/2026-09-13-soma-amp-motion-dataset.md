# SOMA AMP Motion Dataset Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a format-specific SOMA AMP motion loader behind a shared dataset base class while preserving the existing BeyondMimic loader.

**Architecture:** Extract format-neutral NPZ selection, tensor conversion, joint-contract handling, AMP feature construction, and transition sampling into `BaseMotionDataset`. Keep `MotionDataset` as the compatibility-preserving BeyondMimic subclass and add `SomaMotionDataset` for strict embedded SOMA metadata validation. The AMP factory selects the subclass from `motion_dataset_format`, defaulting to BeyondMimic.

**Tech Stack:** Python, NumPy, PyTorch, pytest, local RSL-RL fork.

**Spec:** `docs/superpowers/specs/2026-09-08-soma-amp-motion-dataset-design.md`

## Global Constraints

- Do not resample, interpolate, decimate, or control motion frame rate in the loader.
- Do not change the existing BeyondMimic `MotionDataset` behavior.
- Require SOMA `joint_names` and `body_names` metadata.
- Continue passing `joint_names` and complete configured `body_names` explicitly from the task configuration.
- Preserve the AMP feature order: root height, root orientation 6D, root linear velocity, root angular velocity, joint position, joint velocity, key-body positions.
- Preserve the current local RSL-RL transition preloading changes.
- Do not modify generated datasets, checkpoints, logs, or start GPU training.

## File Map

- Create `/home/jvwei/rsl_rl/rsl_rl/datasets/base_motion_dataset.py`: shared NPZ selection, loading pipeline, joint handling, feature construction, and transition sampling.
- Modify `/home/jvwei/rsl_rl/rsl_rl/datasets/motion_dataset.py`: make `MotionDataset` the BeyondMimic subclass without SOMA-specific branches.
- Create `/home/jvwei/rsl_rl/rsl_rl/datasets/soma_motion_dataset.py`: strict SOMA metadata and body-contract validation.
- Modify `/home/jvwei/rsl_rl/rsl_rl/datasets/__init__.py`: export the base class and both concrete loaders.
- Modify `/home/jvwei/rsl_rl/rsl_rl/algorithms/amp.py`: select the loader using `motion_dataset_format`, defaulting to `beyondmimic`.
- Modify `/home/jvwei/rsl_rl/tests/algorithms/test_amp.py`: unit tests for hierarchy, contracts, selection, ordering, and adjacent-frame sampling.

### Task 1: Extract the shared motion dataset base

**Files:**
- Create: `/home/jvwei/rsl_rl/rsl_rl/datasets/base_motion_dataset.py`
- Modify: `/home/jvwei/rsl_rl/rsl_rl/datasets/motion_dataset.py`
- Test: `/home/jvwei/rsl_rl/tests/algorithms/test_amp.py`

**Interfaces:**
- `BaseMotionDataset(..., motion_dir: str, device: str = "cpu", amp_observation_dim: int = 190, time_between_frames: float = 0.02, key_body_names: list[str] | None = None, body_names: list[str] | None = None, joint_names: list[str] | None = None, motion_file_pattern: str | None = None, motion_files: list[str] | None = None)`
- `BaseMotionDataset.sample_amp_observations(batch_size: int) -> torch.Tensor`
- Concrete subclasses provide `_load_motion_file(path: str) -> dict` and may validate format metadata before the base pipeline consumes the record.

- [x] **Step 1: Add a failing hierarchy test.**

```python
def test_beyondmimic_motion_dataset_uses_shared_base():
    from rsl_rl.datasets.base_motion_dataset import BaseMotionDataset
    from rsl_rl.datasets.motion_dataset import MotionDataset

    assert issubclass(MotionDataset, BaseMotionDataset)
```

- [x] **Step 2: Run the focused test and verify it fails because `BaseMotionDataset` is missing or `MotionDataset` is not a subclass.**

Run from `/home/jvwei/rsl_rl`:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q tests/algorithms/test_amp.py::test_beyondmimic_motion_dataset_uses_shared_base
```

- [x] **Step 3: Move format-neutral logic into `BaseMotionDataset`.** Keep the existing selection semantics, tensor field names, configured joint reordering, root-frame body feature construction, final-frame exclusion, transition preloading, and sampling behavior unchanged. Make the base class call a concrete `_load_motion_file` hook.
- [x] **Step 4: Implement `MotionDataset._load_motion_file` with the existing BeyondMimic compatibility behavior, including external body names and optional embedded joint/body metadata. Do not add SOMA validation to this subclass.**
- [x] **Step 5: Run the full existing AMP dataset tests and verify the extraction preserves their behavior.**

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q tests/algorithms/test_amp.py
```

- [x] **Step 6: Commit the base extraction in the `/home/jvwei/rsl_rl` repository.**

```bash
git add rsl_rl/datasets/base_motion_dataset.py rsl_rl/datasets/motion_dataset.py tests/algorithms/test_amp.py
git commit -m "refactor: extract shared AMP motion dataset base"
```

### Task 2: Add strict SOMA metadata validation

**Files:**
- Create: `/home/jvwei/rsl_rl/rsl_rl/datasets/soma_motion_dataset.py`
- Modify: `/home/jvwei/rsl_rl/rsl_rl/datasets/base_motion_dataset.py` only if a protected hook is needed
- Test: `/home/jvwei/rsl_rl/tests/algorithms/test_amp.py`

**Interfaces:**
- `SomaMotionDataset` has the same constructor and sampling method as `BaseMotionDataset`.
- SOMA records must contain `fps`, all six motion arrays, `joint_names`, and `body_names`.

- [x] **Step 1: Add failing tests using temporary NPZ files for required metadata, duplicate names, body-order mismatch, and accepted 34-body metadata.** Use a small 29-joint synthetic payload with the new body sequence beginning `base_link, waist_z_link, waist_x_link, body` and containing `neck_link`, `head_link`, `left_toe_link`, and `right_toe_link`.
- [x] **Step 2: Run only the new tests and verify they fail because `SomaMotionDataset` is unavailable.**

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q tests/algorithms/test_amp.py -k soma
```

- [x] **Step 3: Implement `SomaMotionDataset` as a `BaseMotionDataset` subclass.** Require all SOMA fields, convert `fps` with `np.asarray(...).reshape(-1)[0]` without changing it, require one-dimensional unique name arrays, validate the configured body sequence exactly, and pass the arrays through the shared joint/feature pipeline. Do not resample or alter adjacent-frame transitions.
- [x] **Step 4: Add a test proving the configured joint order is used for both `joint_pos` and `joint_vel`, while the configured body order is retained for key-body indexing.**
- [x] **Step 5: Run all AMP dataset tests and verify the SOMA tests pass.**

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q tests/algorithms/test_amp.py
```

- [x] **Step 6: Commit the SOMA loader and tests in `/home/jvwei/rsl_rl`.**

```bash
git add rsl_rl/datasets/soma_motion_dataset.py tests/algorithms/test_amp.py
git commit -m "feat: add SOMA AMP motion dataset loader"
```

### Task 3: Add explicit AMP dataset-format selection

**Files:**
- Modify: `/home/jvwei/rsl_rl/rsl_rl/datasets/__init__.py`
- Modify: `/home/jvwei/rsl_rl/rsl_rl/algorithms/amp.py`
- Test: `/home/jvwei/rsl_rl/tests/algorithms/test_amp.py`

**Interfaces:**
- `cfg["algorithm"].get("motion_dataset_format", "beyondmimic")` selects the loader.
- Mapping: `"beyondmimic" -> MotionDataset`, `"soma" -> SomaMotionDataset`.
- Unknown values raise `ValueError("Unknown AMP motion dataset format: ...")`.

- [x] **Step 1: Add failing tests for the default format, explicit SOMA format, and unknown format error.** Test the mapping helper separately if introduced; otherwise isolate the factory selection before constructing a full simulator environment.
- [x] **Step 2: Run the focused factory tests and verify they fail because the format setting is not consumed.**
- [x] **Step 3: Export `BaseMotionDataset`, `MotionDataset`, and `SomaMotionDataset` from `rsl_rl.datasets`, add the closed mapping in `AMP.construct_algorithm`, pop the format key before passing remaining algorithm parameters, and instantiate the selected class with the existing arguments.**
- [x] **Step 4: Run the focused factory tests and all AMP tests.**

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q tests/algorithms/test_amp.py
```

- [x] **Step 5: Commit the format selector and tests in `/home/jvwei/rsl_rl`.**

```bash
git add rsl_rl/datasets/__init__.py rsl_rl/algorithms/amp.py tests/algorithms/test_amp.py
git commit -m "feat: select AMP motion dataset format"
```

### Task 4: Final verification and handoff

**Files:**
- No production file changes expected.

- [x] **Step 1: Run the complete local RSL-RL test suite relevant to AMP.**

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q tests/algorithms/test_amp.py tests/algorithms/test_amp_components.py tests/algorithms/test_amp_runner.py
```

- [x] **Step 2: Compile the changed modules.**

```bash
python -m py_compile rsl_rl/datasets/base_motion_dataset.py rsl_rl/datasets/motion_dataset.py rsl_rl/datasets/soma_motion_dataset.py rsl_rl/algorithms/amp.py tests/algorithms/test_amp.py
```

- [x] **Step 3: Run a read-only real-data contract check against `/home/jvwei/datasets/soma_uniform/filtered/retargeted/npz_20260907_164055/dr02` using the new 34-body list and current 29-joint list. Confirm selected files load and report sample shape; do not launch Isaac Lab or training.**
- [x] **Step 4: Inspect `/home/jvwei/rsl_rl` git diff and the top-level repository status to confirm unrelated user changes remain untouched.**
- [x] **Step 5: Report test counts, real-data contract result, and the fact that no frame-rate conversion or training was performed.**
