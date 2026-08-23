import json
from pathlib import Path
import xml.etree.ElementTree as ET


ROOT = Path(__file__).parents[1]
ZOO = ROOT / "source/robot_learning_lab_zoo"
DATA = ZOO / "robots/deeprobotics"
CONFIG = ZOO / "robot_learning_lab_zoo/assets/isaaclab/deeprobotics.py"
PRO_ENV_CONFIG = (
    ROOT
    / "source/robot_learning_lab_tasks/robot_learning_lab_tasks/tasks/isaaclab/manager_based/locomotion/velocity"
    / "config/humanoid/deeprobotics_dr02_pro/rough_env_cfg.py"
)
AMP_ENV_CONFIG = (
    ROOT
    / "source/robot_learning_lab_tasks/robot_learning_lab_tasks/tasks/isaaclab/manager_based/amp/config/dr02/flat_env_cfg.py"
)
AMP_AGENT_CONFIG = (
    ROOT
    / "source/robot_learning_lab_tasks/robot_learning_lab_tasks/tasks/isaaclab/manager_based/amp/config/dr02/agents/rsl_rl_amp_cfg.py"
)


def _assert_urdf_contract(relative_path: str, expected_joints: int) -> None:
    urdf_path = DATA / relative_path
    root = ET.parse(urdf_path).getroot()
    movable = [joint for joint in root.findall("joint") if joint.attrib["type"] != "fixed"]
    assert len(movable) == expected_joints

    for mesh in root.findall(".//mesh"):
        assert (urdf_path.parent / mesh.attrib["filename"]).resolve().is_file(), mesh.attrib["filename"]


def test_standard_finetune_urdf_contract() -> None:
    _assert_urdf_contract("dr02_standard_description/urdf/dr02_std.urdf", 21)


def test_pro_finetune_urdf_contract() -> None:
    _assert_urdf_contract("dr02_pro_description/urdf/dr02_pro.urdf", 29)


def test_dr02_configs_define_tmpdir_fallback_and_finetune_gains() -> None:
    source = CONFIG.read_text()
    assert 'os.environ.get("TMPDIR")' in source
    assert 'tmp_dir = "/tmp/IsaacLab"' in source
    for cache_name in ("dr02_standard", "dr02_pro"):
        assert f'"IsaacLab", "{cache_name}"' in source

    for parameter in (
        '"left_shoulder_z_joint": 0.765',
        '"right_shoulder_z_joint": -0.765',
        '".*_elbow_joint": 1.25',
        '"knees": DCMotorCfg(',
        "stiffness=625.0",
        "damping=0.25",
    ):
        assert parameter in source


def test_pro_env_does_not_reward_fixed_neck_joints() -> None:
    urdf = ET.parse(DATA / "dr02_pro_description/urdf/dr02_pro.urdf").getroot()
    joint_types = {joint.attrib["name"]: joint.attrib["type"] for joint in urdf.findall("joint")}

    assert joint_types["neck_z_joint"] == "fixed"
    assert joint_types["neck_y_joint"] == "fixed"
    assert "joint_deviation_head_l1" not in PRO_ENV_CONFIG.read_text(encoding="utf-8")


def test_dr02_amp_contract_matches_external_body_order() -> None:
    bodies = json.loads(
        (Path.home() / "GMR-private/retarget_data/dr02/bodies.json").read_text(encoding="utf-8")
    )
    source = AMP_ENV_CONFIG.read_text(encoding="utf-8")
    agent_source = AMP_AGENT_CONFIG.read_text(encoding="utf-8")

    assert tuple(bodies["body_names"]) == tuple(
        name.strip().strip('"')
        for name in source.split("DR02_AMP_BODY_NAMES = (", 1)[1].split(")", 1)[0].split(",")
        if name.strip()
    )
    assert "DR02_AMP_KEY_BODY_NAMES" in source
    assert 'motion_dir' in agent_source
    assert 'body_names = body_names' in agent_source
