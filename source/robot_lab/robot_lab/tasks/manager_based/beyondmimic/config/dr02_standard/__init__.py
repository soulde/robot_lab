# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

import gymnasium as gym

from . import agents, flat_env_cfg

##
# Register Gym environments.
##

_RSL_RL_CFG_ENTRY_POINT = (
    f"{agents.__name__}.rsl_rl_ppo_cfg:DeeproboticsDR02StandardBeyondMimicFlatPPORunnerCfg"
)

gym.register(
    id="RobotLab-Isaac-BeyondMimic-Flat-Deeprobotics-DR02-Standard-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:DeeproboticsDR02StandardBeyondMimicFlatEnvCfg",
        "rsl_rl_cfg_entry_point": _RSL_RL_CFG_ENTRY_POINT,
    },
)
