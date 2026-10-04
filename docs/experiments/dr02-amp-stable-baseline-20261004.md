# DR02 AMP stable baseline — 2026-10-04

## Source

- robot_lab branch: `debug/dr02-amp-training`.
- Task submodule commit: `ae57128`.
- External `/home/jvwei/rsl_rl` branch: `debug/dr02-amp-training`, commit `4c734de`.
- Launch wrapper: `scripts/pueue/train_dr02_amp_flat_torso_viser_fresh_20261004.sh`.
- Pueue task: 182. Fresh policy, seed 42, 4096 environments, 5000 iterations.
- Output: `logs/dr02_amp_flat_bounds_jointscale_torso_viser_fresh_4096x5000_20261004`.
- Viser starts with rendering paused; the service remains available.

## Changes

AMP discriminator shares the PPO Adam optimizer and adaptive learning rate.
Actor and critic gradients are clipped separately; discriminator gradients are not clipped.
Actions and policy/critic observations are clipped to ±100.
Joint acceleration history is initialized from the RSI reset velocity.
Joint acceleration, torque, action rate and position limit penalties are scaled by 23/29.
Linear RMS weight is -0.2 and angular RMS weight is -0.1 (10× Chocolate).
Collision weight remains -10.0 with the expanded upper-body contact selection.
Torso-to-highest-toe clearance termination uses 0.35×initial root height.
Velocity tracking and arrows use the DR02 torso body. Linear tracking uses the
horizontal torso yaw frame; angular tracking uses torso-local Z, as Chocolate does.
Exponential tracking calls official Isaac Lab kernels through a body-state adapter.
Reward diagnostics record per-term extremes and stop on nonfinite values.

## Observed training evidence

The 3-iteration smoke completed with a finite checkpoint. At iteration 4446,
mean return was 124.34, mean episode length 1934.18 steps, value loss 0.0687,
and discriminator loss 0.4206. Timeouts accounted for 94.18% of terminations;
low torso clearance accounted for 5.82%. No diagnostic snapshots were triggered.
This run crossed the previous failure region near iteration 3000, but the
combined changes do not isolate which change removed the failure.

## Next work

Continue tuning reward proportions. Keep this stable baseline for comparison.
Prioritize balancing velocity tracking, torque, action rate and joint acceleration
against AMP style reward; use separate experiment directories and record each
weight change and the observed gait, tracking error and stability.
Do not treat reward scale changes alone as proof of better locomotion.
