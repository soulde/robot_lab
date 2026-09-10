import json
from pathlib import Path
import ast
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
AMP_TRACKING_CONFIG = (
    ROOT
    / "source/robot_learning_lab_tasks/robot_learning_lab_tasks/tasks/isaaclab/manager_based/amp"
    / "tracking_env_cfg.py"
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


def test_dr02_amp_velocity_command_visualization_is_enabled_in_play() -> None:
    tracking_source = (
        ROOT
        / "source/robot_learning_lab_tasks/robot_learning_lab_tasks/tasks/isaaclab/manager_based/amp/tracking_env_cfg.py"
    ).read_text(encoding="utf-8")
    play_source = (ROOT / "scripts/reinforcement_learning/rsl_rl/play.py").read_text(encoding="utf-8")

    assert "debug_vis=True" in tracking_source
    assert "env_cfg.commands.base_velocity.debug_vis = False" not in play_source


def test_amp_viewer_is_not_bound_to_robot_root() -> None:
    tracking_source = (
        ROOT
        / "source/robot_learning_lab_tasks/robot_learning_lab_tasks/tasks/isaaclab/manager_based/amp/tracking_env_cfg.py"
    ).read_text(encoding="utf-8")

    assert 'self.viewer.origin_type = "world"' in tracking_source
    assert 'self.viewer.asset_name = "robot"' not in tracking_source


def test_amp_observation_entities_preserve_configured_feature_order() -> None:
    tree = ast.parse(AMP_TRACKING_CONFIG.read_text(encoding="utf-8"))
    ordered_amp_entities = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
            continue
        if node.func.id != "SceneEntityCfg":
            continue
        keywords = {keyword.arg: keyword.value for keyword in node.keywords}
        selected_names = keywords.get("joint_names", keywords.get("body_names"))
        if not isinstance(selected_names, ast.List) or selected_names.elts:
            continue
        preserve_order = keywords.get("preserve_order")
        ordered_amp_entities.append(
            isinstance(preserve_order, ast.Constant) and preserve_order.value is True
        )

    assert ordered_amp_entities == [True, True, True]


def test_amp_randomizes_existing_arm_motor_armatures_at_startup() -> None:
    source = AMP_TRACKING_CONFIG.read_text(encoding="utf-8")
    tree = ast.parse(source)
    assignments = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "randomize_motor_armature"
            for target in node.targets
        )
    ]

    assert len(assignments) == 1
    event_source = ast.get_source_segment(source, assignments[0].value)
    assert event_source is not None
    assert "func=base_mdp_events.randomize_joint_parameters" in event_source
    assert 'mode="startup"' in event_source
    assert '"armature_distribution_params": (0.9, 1.1)' in event_source
    assert '"operation": "scale"' in event_source
    for joint_pattern in (
        ".*_shoulder_[xyz]_joint",
        ".*_elbow_joint",
        ".*_wrist_[xyz]_joint",
    ):
        assert f'"{joint_pattern}"' in event_source


def test_dr02_amp_uses_reduced_style_reward_scale() -> None:
    source = AMP_AGENT_CONFIG.read_text(encoding="utf-8")

    assert "self.algorithm.task_reward_scale = 1.0" in source
    assert "self.algorithm.style_reward_scale = 0.1" in source
