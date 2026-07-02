# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**robot_lab** is an RL extension library for robots built on top of [Isaac Lab](https://isaac-sim.github.io/IsaacLab) (NVIDIA's robot learning framework on Isaac Sim). It provides locomotion and manipulation training environments for quadrupeds, wheeled robots, and humanoids, supporting RSL-RL, CusRL, and skrl RL frameworks.

## Commands

### Installation
```bash
# Isaac Lab must be installed first (separate repo). Then:
python -m pip install -e source/robot_lab
```

### List available environments
```bash
python scripts/tools/list_envs.py
```

### Training (RSL-RL, primary framework)
```bash
python scripts/reinforcement_learning/rsl_rl/train.py --task=<TASK_NAME> --headless
```
Common flags: `--num_envs 32`, `--video --video_length 200`, `--resume --load_run <folder>`, `--distributed` (multi-GPU).

### Playing (evaluating trained policy)
```bash
python scripts/reinforcement_learning/rsl_rl/play.py --task=<TASK_NAME> [--keyboard] [--num_envs 32]
```

### Training with alternative RL frameworks
```bash
# CusRL
python scripts/reinforcement_learning/cusrl/train.py --task=<TASK_NAME> --headless
# skrl (AMP Dance)
python scripts/reinforcement_learning/skrl/train.py --task=<TASK_NAME> --algorithm AMP --headless
```

### Multi-GPU distributed training
```bash
python -m torch.distributed.run --nnodes=1 --nproc_per_node=2 scripts/reinforcement_learning/rsl_rl/train.py --task=<TASK_NAME> --headless --distributed
```

### Testing with dummy agents (no learning, validates env setup)
```bash
python scripts/tools/zero_agent.py --task=<TASK_NAME>
python scripts/tools/random_agent.py --task=<TASK_NAME>
```

### URDF/MJCF conversion
```bash
python scripts/tools/convert_urdf.py /path/to/robot.urdf
python scripts/tools/convert_mjcf.py /path/to/robot.xml
```

### Tensorboard
```bash
tensorboard --logdir=logs
```

### Docker
```bash
cd docker && docker compose --env-file .env.base --file docker-compose.yaml build robot-lab
docker compose --env-file .env.base --file docker-compose.yaml up
```

### Code quality
```bash
pre-commit run --all-files    # ruff linter + formatter, codespell, license headers
```

### USD cache cleanup (temporary files can consume significant disk)
```bash
rm -rf /tmp/IsaacLab/usd_*
```

## Architecture

### Extension structure

This project follows the Isaac Lab extension pattern. It **must be installed separately from** the core Isaac Lab repository. The entry point is `source/robot_lab/robot_lab/__init__.py`, which imports all task config packages to auto-register Gym environments.

### Key layers

**1. Assets** (`source/robot_lab/robot_lab/assets/`):
Define robot configurations as `ArticulationCfg` objects. Each manufacturer gets its own file (unitree.py, deeprobotics.py, fftai.py, etc.). Configs point to URDF files in `source/robot_lab/data/Robots/`.

**2. Tasks** (`source/robot_lab/robot_lab/tasks/`):
Three task paradigms:
- **Manager-based locomotion** (`manager_based/locomotion/velocity/`): The main paradigm. Uses Isaac Lab's `ManagerBasedRLEnv` with separate manager terms (rewards, observations, commands, terminations, events, curriculum). The base config `velocity_env_cfg.py` defines the scene (`MySceneCfg`), then per-robot configs subclass it.
- **Manager-based beyondmimic** (`manager_based/beyondmimic/`): Motion tracking / imitation learning for humanoids (G1, DR02). Uses motion reference data for full-body tracking.
- **Direct** (`direct/g1_amp/`): Direct-step RL environment for AMP (Adversarial Motion Priors) dance training on G1.

**3. MDP modules** (`mdp/` subdirectories in each task):
Environment logic split into files by concern: `rewards.py`, `observations.py`, `commands.py`, `events.py`, `terminations.py`, `curriculums.py`. These are imported as `import robot_lab.tasks...mdp as mdp` and referenced by string name in config classes.

### Config-based registration pattern

Each robot+terrain combination follows this structure under `tasks/.../velocity/config/<category>/<robot_name>/`:

```
config/<category>/<robot_name>/
├── __init__.py          # gym.register() calls — environment IDs registered here
├── flat_env_cfg.py      # @configclass subclassing RoughEnvCfg, overrides terrain to "plane"
├── rough_env_cfg.py     # @configclass with terrain, sensor, and robot cfg assembled
└── agents/
    ├── __init__.py
    ├── rsl_rl_ppo_cfg.py   # PPO hyperparameters for RSL-RL
    └── cusrl_ppo_cfg.py    # PPO hyperparameters for CusRL
```

Environments are registered via `gym.register()` with three entry points:
- `env_cfg_entry_point`: The `@configclass` defining scene, MDP terms, and robot
- `rsl_rl_cfg_entry_point`: PPO runner config (RSL-RL)
- `cusrl_cfg_entry_point`: Trainer config (CusRL)

The Gym ID follows the convention: `RobotLab-Isaac-<Task>-<Terrain>-<Robot>-v0`.

### Adding a new robot

1. Add the URDF/model files to `source/robot_lab/data/Robots/<manufacturer>/`
2. Define its `ArticulationCfg` in `source/robot_lab/robot_lab/assets/<manufacturer>.py`
3. Create config directory: `tasks/.../velocity/config/<category>/<robot_name>/` with `__init__.py`, `flat_env_cfg.py`, `rough_env_cfg.py`, and `agents/` subdirectory
4. The config class inherits from the appropriate base and sets `self.scene.robot = <ROBOT_CFG>`

### Scripts (`scripts/`)

- `reinforcement_learning/rsl_rl/` — Primary RL training/inference (RSL-RL with PPO)
- `reinforcement_learning/cusrl/` — Alternative training (CusRL)
- `reinforcement_learning/skrl/` — AMP-based training (skrl)
- `tools/` — Utilities: env listing, URDF/MJCF conversion, dummy agents, BeyondMimic data processing

### Dependencies

Isaac Lab extensions are listed in `source/robot_lab/config/extension.toml` under `[dependencies]`: `isaaclab`, `isaaclab_assets`, `isaaclab_mimic`, `isaaclab_rl`, `isaaclab_tasks`. All are required at runtime via Isaac Sim.

### Version compatibility

Version 2.3.2 (current) requires Isaac Lab v2.3.2 and Isaac Sim 4.5/5.0/5.1. Stored in `VERSION` and `extension.toml`.

### Configuration quirks

- Flat environments inherit from Rough and override: terrain type → "plane", disable height scanner, disable terrain curriculum
- The `isort` config in `pyproject.toml` defines custom sections for Isaac Lab extensions (`isaaclab`, `isaaclab-rl`, `isaaclab-assets`, etc.) — do not reorder these import groups
- USD caches accumulate in `/tmp/IsaacLab/usd_*` — disk usage concern for long-running experiments
- `_BLACKLIST_PKGS = ["utils"]` in task `__init__.py` prevents importing config sub-packages from utility directories
