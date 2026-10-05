"""Optional DR02 rollout diagnostics captured before automatic resets."""

import json
import math
from pathlib import Path

import torch


class RewardDiagnostics:
    def __init__(self, env, log_dir):
        self.env = env
        self.root = Path(log_dir) / "diagnostics"
        self.root.mkdir(parents=True, exist_ok=False)
        self.stream = (self.root / "rollout.jsonl").open("w", buffering=1)
        self.step = 0
        self.snapshots = 0
        self.algorithm = None
        self.raw_actions = None
        manager = env.reward_manager
        self.names = manager.active_terms
        self.compute = manager.compute
        manager.compute = self.compute_reward

    def write(self, kind, **values):
        self.stream.write(json.dumps({"kind": kind, "step": self.step, **values}) + "\n")

    def snapshot(self, ids, reason, fatal=False):
        if self.snapshots >= 8 and not fatal:
            return
        ids = ids[:16]
        robot = self.env.scene["robot"]
        state = {
            "reason": reason, "step": self.step, "env_ids": ids.cpu(),
            "reward_names": self.names, "joint_names": list(robot.joint_names),
            "actions": self.env.action_manager.action[ids].detach().cpu(),
            "previous_actions": self.env.action_manager.prev_action[ids].detach().cpu(),
            "joint_pos": robot.data.joint_pos.torch[ids].detach().cpu(),
            "joint_vel": robot.data.joint_vel.torch[ids].detach().cpu(),
            "root_state": robot.data.root_state_w.torch[ids].detach().cpu(),
            "reward_rates": self.env.reward_manager._step_reward[ids].detach().cpu(),
        }
        if self.raw_actions is not None:
            state["raw_policy_actions"] = self.raw_actions[ids].detach().cpu()
        if fatal and self.algorithm is not None:
            state["actor_state_dict"] = self.algorithm._raw_actor.state_dict()
            state["critic_state_dict"] = self.algorithm._raw_critic.state_dict()
        torch.save(state, self.root / f"step_{self.step}_{self.snapshots}.pt")
        self.snapshots += 1
        print(f"[REWARD_DIAGNOSTICS] step={self.step} reason={reason} env_ids={ids.tolist()}", flush=True)

    def compute_reward(self, dt):
        reward = self.compute(dt)
        self.step += 1
        terms = self.env.reward_manager._step_reward * dt
        actions = self.env.action_manager.action
        robot = self.env.scene["robot"]
        joint_vel = robot.data.joint_vel.torch
        joint_pos = robot.data.joint_pos.torch
        root_state = robot.data.root_state_w.torch
        minimum, min_ids = terms.min(dim=0)
        maximum, max_ids = terms.max(dim=0)
        self.write("reward", names=self.names, mean=terms.mean(dim=0).tolist(),
                   minimum=minimum.tolist(), minimum_env=min_ids.tolist(),
                   maximum=maximum.tolist(), maximum_env=max_ids.tolist(),
                   action_absmax=actions.abs().max().item(),
                   raw_action_absmax=(self.raw_actions.abs().max().item() if self.raw_actions is not None else None),
                   joint_velocity_absmax=joint_vel.abs().max().item())
        bad = ~torch.isfinite(reward) | ~torch.isfinite(terms).all(dim=1)
        for value in (actions, joint_vel, joint_pos, root_state):
            bad |= ~torch.isfinite(value).all(dim=1)
        if bad.any():
            self.snapshot(bad.nonzero().flatten(), "nonfinite rollout", fatal=True)
            raise FloatingPointError("Nonfinite rollout; inspect diagnostics before restarting")
        outliers = (reward.abs() > 100.0) | (actions.abs().amax(dim=1) > 50.0)
        if outliers.any() and (self.snapshots == 0 or self.step % 2400 == 0):
            self.snapshot(outliers.nonzero().flatten(), "finite reward/action outlier")
        return reward

    def attach_algorithm(self, algorithm):
        self.algorithm = algorithm
        compute_returns = algorithm.compute_returns
        update = algorithm.update
        act = algorithm.act

        def checked_act(obs):
            self.raw_actions = act(obs).detach()
            bad = ~torch.isfinite(self.raw_actions).all(dim=1)
            if bad.any():
                self.snapshot(bad.nonzero().flatten(), "nonfinite policy action", fatal=True)
                raise FloatingPointError("Nonfinite policy action; inspect diagnostics")
            return self.raw_actions

        def checked_returns(obs):
            compute_returns(obs)
            storage = algorithm.storage
            tensors = {name: getattr(storage, name) for name in ("rewards", "values", "returns", "advantages")}
            self.write("targets", **{name + "_absmax": value.abs().max().item() for name, value in tensors.items()})
            if not all(torch.isfinite(value).all() for value in tensors.values()):
                self.snapshot(torch.arange(min(self.env.num_envs, 16), device=self.env.device),
                              "nonfinite PPO targets", fatal=True)
                raise FloatingPointError("Nonfinite PPO targets; inspect diagnostics")

        def checked_update():
            try:
                losses = update()
                self.write("update", **losses)
                if not all(math.isfinite(value) for value in losses.values()):
                    raise FloatingPointError("Nonfinite training loss")
                return losses
            except Exception:
                self.snapshot(torch.arange(min(self.env.num_envs, 16), device=self.env.device),
                              "PPO update failed", fatal=True)
                raise

        algorithm.compute_returns = checked_returns
        algorithm.update = checked_update
        algorithm.act = checked_act
