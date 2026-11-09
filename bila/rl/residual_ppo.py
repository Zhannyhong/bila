"""
The learner side of train_residual.py: the policy (torch) and its PPO
update. Kept apart so the worker processes never import torch (~700 MB each
with torch._dynamo; 12 of them don't fit in the laptop's WSL).

  - Actor: an MLP on the actor observation, its last layer zero, so it starts
    as the script (a zero correction); a state-independent log std. The
    workers run a numpy copy of it (actor_weights, train_residual.NumpyActor)
    only to choose actions: log-probabilities and values are computed here.
  - Critic: an MLP on the actor observation and the critic's privileged
    extras (asymmetric: only the actor has to run on the robot).
  - PPO: GAE, the clipped objective, Adam with the learning rate going
    linearly to 0, gradients clipped. An episode cut off by the decision
    limit is bootstrapped from the value of where it stopped.
  - Observations normalised by running mean/std (clipped to +-10), rewards
    scaled by the running std of the discounted return.
"""

from __future__ import annotations

import math
from dataclasses import asdict

import numpy as np
import torch
import torch._dynamo  # noqa: F401  (now, not lazily at Adam's first step: imported after the
#                        wrist camera has rendered through Mesa's software EGL, it segfaults)
import torch.nn as nn

from bila.rl.train_residual import Config, RunningStats


def mlp(n_in: int, n_out: int, hidden: int, last_gain: float) -> nn.Sequential:
    net = nn.Sequential(nn.Linear(n_in, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh(),
                        nn.Linear(hidden, n_out))
    for layer, gain in zip(net[::2], (math.sqrt(2), math.sqrt(2), last_gain)):
        nn.init.orthogonal_(layer.weight, gain)
        nn.init.zeros_(layer.bias)
    return net


class Policy(nn.Module):
    """Gaussian actor on the actor observation (its last layer zero: it starts
    as a zero correction) and a critic on both observations."""

    def __init__(self, n_actor: int, n_critic: int, hidden: int = 256, log_std: float = -1.0):
        super().__init__()
        self.actor = mlp(n_actor, 3, hidden, last_gain=0.0)
        self.critic = mlp(n_actor + n_critic, 1, hidden, last_gain=1.0)
        self.log_std = nn.Parameter(torch.full((3,), float(log_std)))

    def dist(self, actor: torch.Tensor) -> torch.distributions.Normal:
        return torch.distributions.Normal(self.actor(actor), self.log_std.exp())

    def value(self, actor: torch.Tensor, critic: torch.Tensor) -> torch.Tensor:
        return self.critic(torch.cat([actor, critic], dim=-1)).squeeze(-1)

    def actor_weights(self) -> dict:
        """The actor as numpy arrays, for train_residual.NumpyActor."""
        layers = [(l.weight.detach().numpy().copy(), l.bias.detach().numpy().copy()) for l in self.actor[::2]]
        return {"layers": layers, "log_std": self.log_std.detach().numpy().copy()}


def gae(rewards, values, terminated, truncated, last_value, final_values, gamma, lam):
    """Advantages of one worker's rollout (generalised advantage estimation).
    An episode that terminated is worth nothing after; one truncated is worth
    the value of where it stopped (final_values, at those steps); either way
    the next step starts a new episode. last_value: of the observation after
    the rollout's last step."""
    T = len(rewards)
    adv, carry = np.zeros(T), 0.0
    for t in reversed(range(T)):
        if terminated[t]:
            nxt = 0.0
        elif truncated[t]:
            nxt = final_values[t]
        else:
            nxt = last_value if t == T - 1 else values[t + 1]
        carry = carry if not (terminated[t] or truncated[t]) else 0.0
        delta = rewards[t] + gamma * nxt - values[t]
        carry = delta + gamma * lam * carry
        adv[t] = carry
    return adv


class Learner:
    def __init__(self, cfg: Config, sizes):
        torch.manual_seed(cfg.seed)
        self.cfg = cfg
        self.policy = Policy(*sizes, cfg.hidden, cfg.log_std)
        self.opt = torch.optim.Adam(self.policy.parameters(), lr=cfg.lr, eps=1e-5)
        self.actor_stats, self.critic_stats = RunningStats(sizes[0]), RunningStats(sizes[1])
        self.ret_stats = RunningStats()
        self.ret = np.zeros(cfg.workers)            # each worker's running discounted return (reward scaling)
        self.decisions, self.updates, self.best = 0, 0, None

    def message(self, **extra) -> dict:
        """What a worker needs to act as the policy does now."""
        return {"actor": self.policy.actor_weights(), "actor_stats": self.actor_stats.state(),
                "critic_stats": self.critic_stats.state(), **extra}

    def tensors(self, actor, critic):
        return (torch.as_tensor(self.actor_stats.normalize(actor)), torch.as_tensor(self.critic_stats.normalize(critic)))

    def scale_rewards(self, rollouts) -> np.ndarray:
        """Rewards over the running std of the discounted return (as VecNormalize)."""
        R = np.stack([r["reward"] for r in rollouts])
        ends = np.stack([r["terminated"] | r["truncated"] for r in rollouts])
        seen = np.zeros_like(R)
        for t in range(R.shape[1]):
            self.ret = self.ret * self.cfg.gamma + R[:, t]
            seen[:, t] = self.ret
            self.ret[ends[:, t]] = 0.0
        self.ret_stats.update(seen.ravel())
        return R / np.sqrt(self.ret_stats.var + 1e-8)

    def update(self, rollouts) -> dict:
        cfg = self.cfg
        W, T = len(rollouts), cfg.rollout
        actor = np.stack([r["actor"] for r in rollouts])
        critic = np.stack([r["critic"] for r in rollouts])
        rewards = self.scale_rewards(rollouts)
        a_t, c_t = self.tensors(actor.reshape(W * T, -1), critic.reshape(W * T, -1))
        actions = torch.as_tensor(np.stack([r["action"] for r in rollouts]).reshape(W * T, 3))
        with torch.no_grad():
            values = self.policy.value(a_t, c_t).numpy().reshape(W, T)
            old_logp = self.policy.dist(a_t).log_prob(actions).sum(-1)
            adv = np.zeros((W, T))
            for w, r in enumerate(rollouts):
                last = self.policy.value(*[x[None] for x in self.tensors(*r["last"])]).item()
                finals = np.zeros(T)
                for t, (fa, fc) in r["finals"].items():
                    finals[t] = self.policy.value(*[x[None] for x in self.tensors(fa, fc)]).item()
                adv[w] = gae(rewards[w], values[w], r["terminated"], r["truncated"], last, finals, cfg.gamma, cfg.lam)
        returns = torch.as_tensor((adv + values).ravel(), dtype=torch.float32)
        adv_t = torch.as_tensor(adv.ravel(), dtype=torch.float32)
        # The observations are normalised with the statistics the workers acted on; update them after.
        self.actor_stats.update(actor.reshape(W * T, -1))
        self.critic_stats.update(critic.reshape(W * T, -1))

        frac = max(0.0, 1.0 - self.decisions / cfg.decisions)
        for g in self.opt.param_groups:
            g["lr"] = cfg.lr * frac
        stats = {"kl": [], "clip_frac": [], "pg_loss": [], "v_loss": []}
        n = W * T
        for _ in range(cfg.epochs):
            for idx in torch.randperm(n).split(cfg.minibatch):
                d = self.policy.dist(a_t[idx])
                logp = d.log_prob(actions[idx]).sum(-1)
                ratio = (logp - old_logp[idx]).exp()
                A = adv_t[idx]
                A = (A - A.mean()) / (A.std() + 1e-8)
                pg = -torch.min(ratio * A, ratio.clamp(1 - cfg.clip, 1 + cfg.clip) * A).mean()
                v = ((self.policy.value(a_t[idx], c_t[idx]) - returns[idx]) ** 2).mean()
                loss = pg + cfg.vf_coef * v - cfg.ent_coef * d.entropy().sum(-1).mean()
                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), cfg.max_grad_norm)
                self.opt.step()
                with torch.no_grad():
                    stats["kl"].append(((ratio - 1) - (logp - old_logp[idx])).mean().item())
                    stats["clip_frac"].append(((ratio - 1).abs() > cfg.clip).float().mean().item())
                    stats["pg_loss"].append(pg.item())
                    stats["v_loss"].append(v.item())
        var_y = returns.var().item()
        out = {k: float(np.mean(v)) for k, v in stats.items()}
        out.update(explained_variance=1 - float(np.var(returns.numpy() - values.ravel())) / (var_y + 1e-8),
                   std=float(self.policy.log_std.detach().exp().mean()), lr=cfg.lr * frac,
                   reward_mean=float(np.mean([r["reward"].mean() for r in rollouts])))
        self.decisions += n
        self.updates += 1
        return out

    # --- checkpoints -------------------------------------------------------------
    def save(self, path: str, **extra) -> None:
        torch.save({"policy": self.policy.state_dict(), "opt": self.opt.state_dict(),
                    "actor_stats": self.actor_stats.state(), "critic_stats": self.critic_stats.state(),
                    "ret_stats": self.ret_stats.state(), "ret": self.ret, "decisions": self.decisions,
                    "updates": self.updates, "best": self.best, "config": asdict(self.cfg), **extra}, path)

    def load(self, path: str) -> None:
        ck = torch.load(path, weights_only=False)
        self.policy.load_state_dict(ck["policy"])
        self.opt.load_state_dict(ck["opt"])
        self.actor_stats.load(ck["actor_stats"])
        self.critic_stats.load(ck["critic_stats"])
        self.ret_stats.load(ck["ret_stats"])
        self.ret, self.decisions, self.updates, self.best = ck["ret"], ck["decisions"], ck["updates"], ck["best"]
