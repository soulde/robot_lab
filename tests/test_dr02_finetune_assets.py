from pathlib import Path
import xml.etree.ElementTree as ET


ROOT = Path(__file__).parents[1]
DATA = ROOT / "source/robot_lab/data/Robots/deeprobotics"
CONFIG = ROOT / "source/robot_lab/robot_lab/assets/deeprobotics.py"


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


def test_dr02_configs_require_tmpdir_and_contain_finetune_gains() -> None:
    source = CONFIG.read_text()
    assert 'os.environ.get("TMPDIR")' in source
    assert "TMPDIR must be set" in source
    for cache_name in ("dr02_standard", "dr02_pro"):
        assert f'"IsaacLab", "{cache_name}"' in source

    for gain in (
        "stiffness=80.0",
        "stiffness=90.0",
        "stiffness=100.0",
        "stiffness=120.0",
        "stiffness=150.0",
        "stiffness=200.0",
        "stiffness=300.0",
        "stiffness=2300.0",
        "stiffness=2800.0",
    ):
        assert gain in source
