# DR02 AMP Stability and Velocity Task Design

## Goal

Make the DR02 Pro Isaac Lab AMP task survive beyond the first rollout steps and provide a useful velocity-tracking RL objective alongside AMP style reward.

## Root Causes

- The current task gives AMP no task reward because `task_reward_scale` is zero.
- The Isaac actuator groups use one `625/0.25` gain pair instead of the validated DR02 position-control groups from `dr02_pos.xml`.
- Expert key-body positions are translated relative to the root but are not rotated into the root frame, while online AMP observations are root-frame positions.
- The motion loader wraps the last frame to frame zero, creating an invalid cross-clip transition.
- Isaac Lab returns observations after automatic reset, so a terminated environment can produce a fall-to-reset transition unless AMP excludes it.

## Design

Keep the existing RSL-RL AMP architecture and make the smallest contract corrections:

1. Add explicit DR02 joint-contract validation and preserve the verified 29-DoF order.
2. Convert expert body positions into the same root-local frame used by Isaac observations.
3. Sample only valid consecutive motion frames and exclude terminated online transitions from discriminator training.
4. Use DR02 role-specific position gains: waist `(200,10)`, `(2800,15)`, `(2300,20)`; hip/knee `(300,10)`; ankle pitch `(80,3)`; ankle roll `(30,1)`; arms `(100,5)`; wrists `(90,2)`.
5. Add velocity commands to policy and critic observations and enable linear/angular velocity tracking rewards.
6. Keep termination, posture, torque, acceleration, action-rate, joint-limit, and non-foot contact terms. Disable explicit foot timing/gait/contact shaping terms.
7. Mix task and style rewards at scale `1.0` each and reduce initial exploration noise to avoid immediately saturating the position targets.

## Verification

- Unit tests cover motion-frame sampling, root-frame conversion, terminal filtering, and joint contract validation.
- Existing AMP and DR02 asset tests remain green.
- A fresh Isaac Lab smoke test uses a small environment count and one or two iterations, checking that AMP updates complete and mean episode length is greater than one.
- The full run is submitted through Pueue only after the smoke test passes.
