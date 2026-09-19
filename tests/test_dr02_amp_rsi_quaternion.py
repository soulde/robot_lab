"""RSI motions store MuJoCo quaternions; Isaac Lab EA consumes xyzw."""

import importlib.util
from pathlib import Path

import numpy as np
import torch


def test_amp_motion_library_converts_root_quaternion_to_xyzw(tmp_path):
    module_path = (
        Path(__file__).resolve().parents[1]
        / "source/robot_learning_lab_tasks/robot_learning_lab_tasks/tasks/isaaclab/manager_based/amp/mdp/motion_loader.py"
    )
    spec = importlib.util.spec_from_file_location("amp_motion_loader_under_test", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    np.savez(
        tmp_path / "motion.npz",
        joint_pos=np.zeros((2, 1), dtype=np.float32),
        joint_vel=np.zeros((2, 1), dtype=np.float32),
        body_pos_w=np.zeros((2, 1, 3), dtype=np.float32),
        body_quat_w=np.array([[[1.0, 0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0, 1.0]]], dtype=np.float32),
        body_lin_vel_w=np.zeros((2, 1, 3), dtype=np.float32),
        body_ang_vel_w=np.zeros((2, 1, 3), dtype=np.float32),
    )

    library = module.AmpMotionLibrary(str(tmp_path), device="cpu")
    torch.testing.assert_close(
        library.root_states[:, 3:7],
        torch.tensor([[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 1.0, 0.0]]),
    )
