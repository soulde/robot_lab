# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from isaaclab.utils import configclass

from robot_lab.assets.deeprobotics import DEEPROBOTICS_DR02_PRO_CFG
from robot_lab.tasks.manager_based.beyondmimic.config.dr02_standard.flat_env_cfg import (
    DeeproboticsDR02StandardBeyondMimicFlatEnvCfg,
)


@configclass
class DeeproboticsDR02ProBeyondMimicFlatEnvCfg(DeeproboticsDR02StandardBeyondMimicFlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = DEEPROBOTICS_DR02_PRO_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.commands.motion.body_names = [
            "base_link",
            "body",
            "left_hip_x_link",
            "left_knee_link",
            "left_ankle_x_link",
            "right_hip_x_link",
            "right_knee_link",
            "right_ankle_x_link",
            "left_shoulder_x_link",
            "left_elbow_link",
            "left_wrist_x_link",
            "right_shoulder_x_link",
            "right_elbow_link",
            "right_wrist_x_link",
            "head_link",
        ]
        self.rewards.undesired_contacts.params["sensor_cfg"].body_names = [
            r"^(?!left_ankle_x_link$)(?!right_ankle_x_link$)(?!left_wrist_x_link$)(?!right_wrist_x_link$).+$"
        ]
        self.terminations.ee_body_pos.params["body_names"] = [
            "left_ankle_x_link",
            "right_ankle_x_link",
            "left_wrist_x_link",
            "right_wrist_x_link",
        ]
