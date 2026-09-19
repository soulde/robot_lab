"""Full parity check: robot AMP observations vs expert dataset observations at identical RSI frames."""
import json

from isaaclab.app import AppLauncher

app = AppLauncher(headless=True).app

import gymnasium as gym
import torch

import robot_learning_lab_tasks.tasks.isaaclab  # noqa
from isaaclab_tasks.utils import parse_env_cfg

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
env_cfg.scene.num_envs = 16

recorded = {}
orig_sample = ml.AmpMotionLibrary._sample_indices


def patched(self, count, pool_probs=None):
    idx = orig_sample(self, count, pool_probs)
    recorded["idx"] = idx.clone()
    return idx


ml.AmpMotionLibrary._sample_indices = patched

env = gym.make(TASK, cfg=env_cfg)
obs_dict, _ = env.reset()
lib = ml.get_amp_motion_library(dr02_amp_motion_dir(), None, dr02_amp_motion_files(), "cpu:0")
robot = env.unwrapped.scene["robot"]

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

g = recorded["idx"].cpu()
lengths = [m["joint_pos"].shape[0] for m in ds.motions]  # frame counts per motion
starts = [0]
for n in lengths[:-1]:
    starts.append(starts[-1] + n)
# global concat index -> (motion, frame); both loaders read the same file list
motion_of = torch.searchsorted(torch.tensor(starts[1:], dtype=torch.long), g, right=True)
frame_of = g - torch.tensor(starts, dtype=torch.long)[motion_of]

robot_joints = list(robot.joint_names)
reorder = [list(DR02_JOINT_NAMES).index(n) for n in robot_joints]
amp_obs = obs_dict["amp"]
J, K = len(DR02_JOINT_NAMES), len(DR02_AMP_KEY_BODY_NAMES)

print("=== per-env robot amp obs vs expert obs at the SAME motion frame ===")
names = [
    ("root_height", 0, 1),
    ("root_orientation", 1, 7),
    ("root_lin_vel_w", 7, 10),
    ("root_ang_vel_w", 10, 13),
    ("joint_pos", 13, 13 + J),
    ("joint_vel", 13 + J, 13 + 2 * J),
    ("link_pos_rel", 13 + 2 * J, 13 + 2 * J + 3 * K),
]
for e in range(len(g)):
    m, f = int(motion_of[e]), int(frame_of[e])
    expert_obs = ds._all_observations(ds.motions[m])[f]
    r = amp_obs[e]
    exp = expert_obs.to(r.device)
    # robot amp obs joint slices are already in DR02_JOINT_NAMES order
    # (SceneEntityCfg resolves joint_names in that order); no remap needed
    r_aligned = r
    if e == 0:
        for name, a, b in names:
            print(f"env0 {name:18s} max|diff| = {(r_aligned[a:b] - exp[a:b]).abs().max():.2e}")
    else:
        d = (r_aligned - exp).abs().max()
        if e < 4:
            print(f"env{e} overall max|diff| = {d:.2e}")

env.close()
app.close()
print("DONE")
