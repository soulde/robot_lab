"""Verify DR02 AMP RSI resets and expert/robot discriminator observation parity."""
import json

from isaaclab.app import AppLauncher

app = AppLauncher(headless=True).app

import gymnasium as gym
import torch

import robot_learning_lab_tasks.tasks.isaaclab  # noqa
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab.utils.math import quat_apply_inverse

from robot_learning_lab_tasks.tasks.isaaclab.manager_based.amp.config.dr02.flat_env_cfg import (
    DR02_AMP_KEY_BODY_NAMES,
    DR02_JOINT_NAMES,
    dr02_amp_body_names_path,
    dr02_amp_motion_dir,
    dr02_amp_motion_files,
)
from robot_learning_lab_tasks.tasks.isaaclab.manager_based.amp.mdp import motion_loader as ml

TASK = "RobotLab-Isaac-AMP-Flat-Deeprobotics-DR02-Pro-v0"
env_cfg = parse_env_cfg(TASK)
env_cfg.scene.num_envs = 8

recorded = {}
orig = ml.AmpMotionLibrary.sample_reset_states


def patched(self, count, pool_probs=None):
    root_states, joint_pos, joint_vel = orig(self, count, pool_probs)
    recorded["states"] = (root_states.clone(), joint_pos.clone(), joint_vel.clone())
    return root_states, joint_pos, joint_vel


ml.AmpMotionLibrary.sample_reset_states = patched

env = gym.make(TASK, cfg=env_cfg)
obs_dict, _ = env.reset()
robot = env.unwrapped.scene["robot"]
lib_root, lib_jpos, lib_jvel = recorded["states"]
origins = env.unwrapped.scene.env_origins

robot_joints = list(robot.joint_names)
print("robot joint order == DR02_JOINT_NAMES:", robot_joints == list(DR02_JOINT_NAMES))
reorder = [list(DR02_JOINT_NAMES).index(n) for n in robot_joints]  # sim order -> NPZ col

print("\n=== RSI: sim state vs reference frame (after joint reorder) ===")
print(f"root z  err {(robot.data.root_pos_w[:,2] - (lib_root[:,2]+origins[:,2])).abs().max():.2e}")
print(f"root xy err vs origin {(robot.data.root_pos_w[:,:2] - origins[:,:2]).abs().max():.2e}")
print(f"quat err {(robot.data.root_quat_w - lib_root[:,3:7]).abs().max():.2e}")
print(f"lin vel err {(robot.data.root_lin_vel_w - lib_root[:,7:10]).abs().max():.2e}")
print(f"ang vel err {(robot.data.root_ang_vel_w - lib_root[:,10:13]).abs().max():.2e}")
print(f"joint pos err {(robot.data.joint_pos - lib_jpos[:, reorder]).abs().max():.2e}")
print(f"joint vel err {(robot.data.joint_vel - lib_jvel[:, reorder]).abs().max():.2e}")

# --- expert dataset (same construction as construct_algorithm) ---------------
from rsl_rl.datasets import MotionDataset

body_names = json.loads(open(dr02_amp_body_names_path()).read())["body_names"]
ds = MotionDataset(
    motion_dir=dr02_amp_motion_dir(),
    device="cpu",
    key_body_names=list(DR02_AMP_KEY_BODY_NAMES),
    body_names=body_names,
    joint_names=list(DR02_JOINT_NAMES),
    motion_files=dr02_amp_motion_files(),
)
print(f"\nexpert dataset: {len(ds.motions)} motions, {len(ds.transitions)} transitions, J={ds.num_joints}")

# --- robot amp obs vs expert layout -------------------------------------------
amp_obs = obs_dict["amp"]
J, K = len(DR02_JOINT_NAMES), len(DR02_AMP_KEY_BODY_NAMES)
key_idx_bodies = [list(robot.body_names).index(n) for n in DR02_AMP_KEY_BODY_NAMES]

m0 = ds.motions[0]
expert0 = ds._all_observations(m0)
print(f"robot amp obs {tuple(amp_obs.shape)}; expert single-obs dim {expert0.shape[-1]} (want equal)")

r_jp = amp_obs[:, 13 : 13 + J]
print(f"amp obs joint_pos vs reference frame joint_pos: {(r_jp - lib_jpos).abs().max():.2e}")

r_h = amp_obs[:, 0]
print(f"amp obs root_height vs reference root z (flat plane): {(r_h - lib_root[:,2]).abs().max():.2e}")

r_links = amp_obs[:, 13 + 2 * J : 13 + 2 * J + 3 * K]
body_pos = robot.data.body_pos_w.torch
pos_b = quat_apply_inverse(
    robot.data.root_quat_w.torch.repeat_interleave(len(key_idx_bodies), dim=0),
    (body_pos[:, key_idx_bodies] - robot.data.root_pos_w.torch[:, None, :]).reshape(-1, 3),
).reshape(-1, 3 * K)
print(f"link obs convention check (env obs vs quat_apply_inverse): {(r_links - pos_b).abs().max():.2e}")

print("\n=== per-slice ranges (robot 8 envs vs expert subsample) ===")
slices = {
    "root_height": (0, 1),
    "root_orientation": (1, 7),
    "root_lin_vel_w": (7, 10),
    "root_ang_vel_w": (10, 13),
    "joint_pos": (13, 13 + J),
    "joint_vel": (13 + J, 13 + 2 * J),
    "link_pos_rel": (13 + 2 * J, 13 + 2 * J + 3 * K),
}
e = expert0[:: max(1, len(expert0) // 8)][:8]
for name, (a, b) in slices.items():
    r = amp_obs[:, a:b]
    print(f"{name:18s} robot[{r.min():+.2f},{r.max():+.2f}] expert[{e[:, a:b].min():+.2f},{e[:, a:b].max():+.2f}]")

fps_set = {float(m["fps"]) for m in ds.motions}
print(f"\nmotion fps {fps_set}, env step dt {env.unwrapped.step_dt}, transitions dt 0.02")
env.close()
app.close()
print("DONE")
