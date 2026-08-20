# Unitree G1 Dex3 Migration Design

## Goal

Add a clearly distinguished Unitree G1 29DoF body with two Dex3-1 seven-joint
hands to the Isaac Lab and MJLab asset and task stacks, while preserving every
existing G1 symbol and task as the current model without an explicit hand-type
suffix.

## Naming and compatibility

- `UNITREE_G1_29DOF_CFG` continues to mean the existing G1 model. It is not
  renamed and no `RUBBER_HAND` public name is introduced.
- `UNITREE_G1_29DOF_ACTION_SCALE` and MJLab's `G1_ACTION_SCALE` retain their
  existing meanings.
- The new public asset names are `UNITREE_G1_29DOF_DEX3_CFG` and
  `UNITREE_G1_29DOF_DEX3_ACTION_SCALE` for Isaac Lab, plus
  `UNITREE_G1_29DOF_DEX3_CFG` and `G1_DEX3_ACTION_SCALE` for MJLab.
- New task IDs append `-Dex3` immediately after `Unitree-G1`, before Isaac
  Lab's `-v0` suffix. Existing task IDs and Python configuration classes remain
  unchanged.

## Assets

The checked-in
`g1_29dof_with_hand_rev_1_0.urdf` is the source model for the Dex3 variant. It
contains the unchanged 29-joint G1 body and seven joints per hand:
`thumb_0`, `thumb_1`, `thumb_2`, `index_0`, `index_1`, `middle_0`, and
`middle_1`, with a `left_` or `right_hand_` prefix.

Isaac Lab loads this URDF through a separate `ArticulationCfg`. The body
initial state and body actuators are shared structurally with the existing
configuration, while a dedicated hand actuator group controls the fourteen
Dex3 joints. Hand effort and velocity limits come from the URDF: thumb root is
2.45 Nm and 3.14 rad/s; the remaining joints are 1.4 Nm and 12 rad/s. Hand
position-control gains must be explicit constants in the asset module, and the
derived action scale must cover all 43 actuated joints.

MJLab receives a checked-in, self-contained MJCF model generated from the Dex3
URDF and normalized to the conventions already applied by `get_spec`: named
collision geoms, foot sites, and an IMU site and sensors. A separate spec
factory loads this model. Its body articulation reuses the current six body
actuator groups and adds one Dex3 hand group. The existing G1 XML and spec
factory are unchanged.

No tactile sensors are invented in this migration. The source URDF contains
hand mechanics but not the physical Dex3-1 tactile array, so the public model
supports joint control and contact geometry only.

## Task variants

Existing velocity and AMP registrations remain byte-for-byte compatible in
their public IDs and continue to instantiate `UNITREE_G1_29DOF_CFG`.

Each backend gains Dex3 variants of the currently registered G1 tasks:

- velocity rough terrain;
- velocity flat terrain;
- AMP flat terrain.

The Dex3 velocity configurations expose all 43 actuated joints through the
existing joint-position action term and use the Dex3 action-scale mapping.
Their initial hand pose is explicitly defined so reset behavior does not depend
on converter defaults.

The Dex3 AMP configurations also expose all 43 joints to the policy, but AMP
reference observations and discriminator inputs remain restricted to the
existing 29 body joints and existing link list. Existing motion files contain
no hand trajectories; the fourteen hand joints therefore receive no fabricated
demonstration targets. The AMP URDF path for the Dex3 variant points at the
Dex3 URDF so kinematic loading matches the simulated body, while its configured
AMP joint list stays body-only.

## Package surface

The MJLab asset package exports both `UNITREE_G1_29DOF_CFG` and
`UNITREE_G1_29DOF_DEX3_CFG`. Direct imports from the Isaac Lab Unitree module
remain supported and gain the Dex3 symbols. README examples document the two
choices without renaming the old model.

All package installation and import checks use uv. No pip-only installation
path is accepted as verification.

## Validation

Automated tests must establish the distinction rather than merely import both
names:

- the original asset path still targets `g1_29dof_rev_1_0.urdf` or its current
  MJCF and contains 29 actuated joints;
- the Dex3 asset path targets `g1_29dof_with_hand_rev_1_0.urdf` or its distinct
  MJCF and contains 43 actuated joints;
- the Dex3 action-scale maps include all fourteen hand joints while the old
  maps do not;
- old task IDs still register and new `Dex3` IDs register separately;
- Dex3 velocity actions cover all 43 joints;
- Dex3 AMP reference joint lists remain the 29 body joints;
- both project packages install and import under uv-managed environments.

MJLab validation compiles both specs and checks joint and actuator resolution.
Isaac Lab validation imports both configurations and performs the smallest
available URDF conversion or environment construction smoke test. No training
job is part of this migration.

## Non-goals

- Adding tactile sensor simulation or tactile observations.
- Creating manipulation rewards, objects, demonstrations, or grasping tasks.
- Replacing or renaming existing G1 configurations, tasks, checkpoints, or
  datasets.
- Claiming AMP supervision for hand motion when no hand reference data exists.
