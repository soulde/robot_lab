# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from isaaclab.utils import configclass

from robot_lab.assets.deeprobotics import DEEPROBOTICS_DR02_PRO_CFG

from ..deeprobotics_dr02_standard.rough_env_cfg import DeeproboticsDR02StandardRoughEnvCfg


@configclass
class DeeproboticsDR02ProRoughEnvCfg(DeeproboticsDR02StandardRoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # ------------------------------Sence------------------------------
        self.scene.robot = DEEPROBOTICS_DR02_PRO_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # ------------------------------Rewards------------------------------
        self.rewards.create_joint_deviation_l1_rewterm(
            "joint_deviation_torso_l1", -0.1, ["waist_z_joint", "waist_x_joint", "waist_y_joint"]
        )
        self.rewards.create_joint_deviation_l1_rewterm(
            "joint_deviation_head_l1", -0.1, ["neck_z_joint", "neck_y_joint"]
        )

        # If the weight of rewards is 0, set rewards to None
        if self.__class__.__name__ == "DeeproboticsDR02ProRoughEnvCfg":
            self.disable_zero_weight_rewards()
