from pathlib import Path
import xml.etree.ElementTree as ET


ROOT = Path(__file__).parents[1]
ZOO = ROOT / "source/soulde_robot_zoo"
DATA = ZOO / "robots/deeprobotics"
CONFIG = ZOO / "soulde_robot_zoo/assets/deeprobotics.py"


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
