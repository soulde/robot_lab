# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from isaaclab.utils import configclass

from ...deeprobotics_dr02_standard.agents.rsl_rl_ppo_cfg import DeeproboticsDR02StandardRoughPPORunnerCfg


@configclass
class DeeproboticsDR02ProRoughPPORunnerCfg(DeeproboticsDR02StandardRoughPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.experiment_name = "deeprobotics_dr02_pro_rough"
