# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from isaaclab.utils import configclass

from robot_lab.assets.deeprobotics import DEEPROBOTICS_DR02_STANDARD_CFG
from robot_lab.tasks.manager_based.beyondmimic.tracking_env_cfg import BeyondMimicEnvCfg


@configclass
class DeeproboticsDR02StandardBeyondMimicFlatEnvCfg(BeyondMimicEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = DEEPROBOTICS_DR02_STANDARD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.actions.joint_pos.scale = 0.25
        self.commands.motion.motion_file = ""
        self.commands.motion.anchor_body_name = "base_link"
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
            "right_shoulder_x_link",
            "right_elbow_link",
        ]

        self.events.randomize_com_positions.params["asset_cfg"].body_names = "base_link"
        self.rewards.undesired_contacts.params["sensor_cfg"].body_names = [
            r"^(?!left_ankle_x_link$)(?!right_ankle_x_link$).+$"
        ]
        self.terminations.ee_body_pos.params["body_names"] = [
            "left_ankle_x_link",
            "right_ankle_x_link",
            "left_elbow_link",
            "right_elbow_link",
        ]

        self.observations.policy.motion_anchor_pos_b = None
        self.observations.policy.base_lin_vel = None

        self.episode_length_s = 30.0
