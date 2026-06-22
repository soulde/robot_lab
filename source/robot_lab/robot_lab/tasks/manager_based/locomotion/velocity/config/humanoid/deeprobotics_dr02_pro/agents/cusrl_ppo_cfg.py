# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

from ...deeprobotics_dr02_standard.agents.cusrl_ppo_cfg import DeeproboticsDR02StandardRoughTrainerCfg


@dataclass
class DeeproboticsDR02ProRoughTrainerCfg(DeeproboticsDR02StandardRoughTrainerCfg):
    experiment_name = "deeprobotics_dr02_pro_rough"
