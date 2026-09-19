import robot_learning_lab_tasks.tasks.isaaclab  # noqa: F401

from isaaclab_tasks.utils.hydra import resolve_task_config


TASK = "RobotLab-Isaac-AMP-Flat-Deeprobotics-DR02-Pro-v0"


def test_dr02_amp_flat_exposes_newton_mjwarp_preset():
    cfg, _ = resolve_task_config(TASK, "rsl_rl_cfg_entry_point", overrides=["physics=newton_mjwarp"])
    assert type(cfg.sim.physics).__name__ == "NewtonCfg"


def test_dr02_amp_flat_defaults_to_physx():
    cfg, _ = resolve_task_config(TASK, "rsl_rl_cfg_entry_point")
    assert type(cfg.sim.physics).__name__ == "PhysxCfg"
