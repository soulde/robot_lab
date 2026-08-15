# DR02 Finetune Assets Design

## Scope

Update both DR02 Standard and DR02 Pro in robot_lab to use repository-local,
finetuned robot descriptions and actuator parameters from robot_lab_zoo.
Keep all public configuration names and task registrations unchanged.

## Asset Layout

Retain the existing `source/robot_lab/data/Robots/deeprobotics` layout and its
meshes. Replace the Standard URDF with the robot-zoo version and the Pro URDF
with the equivalent 29-DoF description already validated by humanoid_amp and
GMR. No runtime dependency on another repository is introduced.

## Configuration

Require `TMPDIR` when importing the DR02 assets and place generated USD files
under `$TMPDIR/IsaacLab/dr02_standard` and `$TMPDIR/IsaacLab/dr02_pro`.
Standard uses the robot-zoo position gains grouped by joint role and damping 8.
Pro uses the robot-zoo explicit position `kp` and `kv` values grouped by role.
Effort and velocity limits continue to reflect each motor class.

## Verification

Static tests parse both URDF files, verify expected movable-joint counts, ensure
all mesh references resolve, and assert that actuator expressions cover every
movable joint exactly once. Isaac Lab smoke tests instantiate each articulation
when a GPU runtime is available.
