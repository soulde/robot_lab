# Unitree G1 Dex3 Backpack Design

## Goal

Add a separately named Unitree G1 Dex3 robot variant carrying a visible,
collidable, rigidly mounted 1 kg cuboid backpack, together with distinct Isaac
Lab and MJLab velocity and AMP task registrations.

## Naming and compatibility

- Existing G1 and G1 Dex3 asset symbols, files, action scales, task IDs, and
  behavior remain unchanged.
- The new public asset symbol is `UNITREE_G1_29DOF_DEX3_BACKPACK_CFG` in both
  backends.
- Isaac Lab exports `UNITREE_G1_29DOF_DEX3_BACKPACK_ACTION_SCALE`; MJLab exports
  `G1_DEX3_BACKPACK_ACTION_SCALE`.
- New task IDs use the suffix `Unitree-G1-Dex3-Backpack`; Isaac Lab retains its
  final `-v0` suffix.

## Backpack physics and geometry

The backpack is a fixed child of `torso_link` named `backpack_link`, connected
by `backpack_joint`. It adds no controllable degree of freedom, so the complete
robot remains a 43-actuated-joint system.

The backpack properties are:

- dimensions: 0.25 m in X, 0.20 m in Y, and 0.30 m in Z;
- center relative to `torso_link`: `(-0.12, 0.0, 0.05)` m;
- orientation relative to `torso_link`: identity;
- mass: exactly 1.0 kg;
- center of mass: the cuboid center;
- diagonal inertia for a uniform cuboid: `ixx=0.0108333333`,
  `iyy=0.0133333333`, and `izz=0.0085416667` kg m^2;
- off-diagonal inertia: zero;
- visual geometry: a clearly distinguishable backpack-colored cuboid;
- collision geometry: the same cuboid dimensions.

The backpack is structural payload, not an independently simulated loose
object. Its fixed attachment must not create an actuator, action, observation,
or command entry.

## Derived asset files

Create a Dex3-backpack URDF derived from
`g1_29dof_with_hand_rev_1_0.urdf`, preserving the entire G1 and Dex3 definition
and appending only the backpack material, link, inertial, visual, collision, and
fixed joint elements.

Create a matching Dex3-backpack MJCF derived from
`g1_29dof_with_hand_rev_1_0.xml`. The MJCF attaches a body named
`backpack_link` to `torso_link` with the same pose, mass, diagonal inertia,
visual geom, and collision geom. The derived MJCF continues to use package
relative mesh paths and must compile from an installed wheel.

Isaac Lab and MJLab receive separate backpack configurations that reuse the
Dex3 initial state, actuator groups, and action scales without mutating the
existing Dex3 configuration objects. Runtime model mass must increase by
exactly 1.0 kg relative to the ordinary Dex3 model.

## Task variants

Each backend gains three backpack variants:

- rough-terrain velocity;
- flat-terrain velocity;
- flat-terrain AMP.

Isaac Lab task IDs:

- `RobotLab-Isaac-Velocity-Rough-Unitree-G1-Dex3-Backpack-v0`;
- `RobotLab-Isaac-Velocity-Flat-Unitree-G1-Dex3-Backpack-v0`;
- `RobotLab-Isaac-AMP-Flat-Unitree-G1-Dex3-Backpack-v0`.

MJLab task IDs:

- `RobotLab-MJLab-Velocity-Rough-Unitree-G1-Dex3-Backpack`;
- `RobotLab-MJLab-Velocity-Flat-Unitree-G1-Dex3-Backpack`;
- `RobotLab-MJLab-AMP-Flat-Unitree-G1-Dex3-Backpack`.

The backpack variants use the same 43-joint policy action space as ordinary
Dex3. AMP reference observations remain restricted to the existing 29 body
joints and existing body-link list because current motion data contains no hand
or backpack trajectories. Backpack runner experiment names must be distinct
from both ordinary G1 and ordinary Dex3 runs.

## Packaging and validation

All package builds and installations use uv. Both derived files must be present
in the zoo wheel, and an installed-wheel MJLab test must compile the backpack
model without relying on the source checkout.

Automated tests must verify:

- existing G1 and Dex3 files/configurations are unchanged;
- the backpack URDF has the expected fixed link, pose, dimensions, mass, and
  inertia;
- the backpack MJCF compiles with 43 actuated joints and one `backpack_link`;
- total compiled mass is exactly 1.0 kg above ordinary Dex3 within numerical
  tolerance;
- no backpack joint appears in actuators or action-scale maps;
- all six backpack task IDs register separately from existing tasks;
- velocity and AMP backpack variants select the backpack configuration;
- AMP reference joint names remain the existing 29 body joints;
- zoo and tasks wheels build and install with uv and contain the new assets.

No training is launched as part of validation. Isaac Sim runtime smoke testing
is conditional on GPU, memory, Pueue, and Isaac runtime availability.

## Non-goals

- Simulating straps, flexible attachment, backpack contents, sensors, or
  independently moving payloads.
- Adding manipulation rewards or new motion/reference datasets.
- Replacing existing G1 or Dex3 assets and task registrations.
