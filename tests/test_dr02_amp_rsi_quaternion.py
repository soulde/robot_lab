import numpy as np
import torch

from rsl_rl.datasets import MotionDataset


def test_beyondmimic_dataset_exposes_xyzw_reference_quaternions(tmp_path):
    source_quaternions_wxyz = np.array(
        [[[1.0, 0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0, 1.0]]], dtype=np.float32
    )
    np.savez(
        tmp_path / "motion.npz",
        fps=np.array(50.0, dtype=np.float32),
        joint_pos=np.zeros((2, 1), dtype=np.float32),
        joint_vel=np.zeros((2, 1), dtype=np.float32),
        body_pos_w=np.zeros((2, 1, 3), dtype=np.float32),
        body_quat_w=source_quaternions_wxyz,
        body_lin_vel_w=np.zeros((2, 1, 3), dtype=np.float32),
        body_ang_vel_w=np.zeros((2, 1, 3), dtype=np.float32),
        joint_names=np.array(["joint"]),
        body_names=np.array(["torso"]),
    )

    dataset = MotionDataset(
        motion_dir=str(tmp_path),
        device="cpu",
        key_body_names=["torso"],
        body_names=["torso"],
        joint_names=["joint"],
    )

    expected_xyzw = torch.tensor([[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 1.0, 0.0]])
    torch.testing.assert_close(dataset.motions[0]["body_quat_xyzw"][:, 0], expected_xyzw)
    sampled_xyzw = dataset.sample_reference_states(16)["root_quat_xyzw"]
    nearest_frame = torch.cdist(sampled_xyzw, expected_xyzw).min(dim=1).values
    torch.testing.assert_close(nearest_frame, torch.zeros_like(nearest_frame))
