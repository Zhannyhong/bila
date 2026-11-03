"""Checks the residual's PPO pieces: advantages, normalisation, the policy's
start, the workers' numpy copy of it, the action-scale ramp, one update on
made-up rollouts, and that the worker side never loads torch.

    uv run --group rl pytest tests/test_train_residual.py
"""

import subprocess
import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from residual_ppo import Learner, Policy, gae  # noqa: E402
from train_residual import Config, NumpyActor, RunningStats, ramp  # noqa: E402


def test_gae_by_hand():
    """Three steps, an episode ending (terminated) after the second."""
    r, v = np.array([1.0, 2.0, 3.0]), np.array([0.5, 0.4, 0.3])
    g, lam, last = 0.9, 0.8, 0.7
    term, trunc = np.array([False, True, False]), np.zeros(3, bool)
    adv = gae(r, v, term, trunc, last, np.zeros(3), g, lam)
    d2 = 3.0 + g * last - 0.3                       # the next episode, bootstrapped from last
    d1 = 2.0 - 0.4                                  # terminated: nothing after
    d0 = 1.0 + g * 0.4 - 0.5
    np.testing.assert_allclose(adv, [d0 + g * lam * d1, d1, d2])


def test_gae_bootstraps_a_truncated_episode_from_where_it_stopped():
    r, v = np.array([1.0, 1.0]), np.array([0.0, 0.0])
    trunc, finals = np.array([True, False]), np.array([5.0, 0.0])
    adv = gae(r, v, np.zeros(2, bool), trunc, 0.0, finals, 0.9, 1.0)
    assert adv[0] == pytest.approx(1.0 + 0.9 * 5.0)  # not the next episode's first value, nor its advantage


def test_running_stats_match_numpy():
    x = np.random.default_rng(0).normal(3.0, 2.0, (1000, 4))
    s = RunningStats(4)
    for chunk in np.split(x, 10):
        s.update(chunk)
    np.testing.assert_allclose(s.mean, x.mean(axis=0), atol=1e-6)
    np.testing.assert_allclose(s.var, x.var(axis=0), rtol=1e-4)
    assert np.abs(s.normalize(x)).max() <= 10.0


def test_the_policy_starts_as_the_script():
    p = Policy(7, 5, hidden=16, log_std=-1.0)
    d = p.dist(torch.randn(4, 7))
    assert torch.equal(d.mean, torch.zeros(4, 3))
    torch.testing.assert_close(d.stddev, torch.full((4, 3), float(np.exp(-1.0))))
    assert p.value(torch.randn(4, 7), torch.randn(4, 5)).shape == (4,)


def test_the_action_scale_ramps_up_then_stays():
    cfg = Config(ramp_start=0.25, ramp_decisions=100)
    assert ramp(cfg, 0) == 0.25 and ramp(cfg, 50) == pytest.approx(0.625) and ramp(cfg, 500) == 1.0


def test_an_update_on_made_up_rollouts_moves_the_policy():
    cfg = Config(workers=2, rollout=64, minibatch=32, epochs=2, hidden=16, decisions=10_000)
    rng = np.random.default_rng(0)
    learner = Learner(cfg, (7, 5))
    rollouts = [{"actor": rng.normal(size=(64, 7)).astype(np.float32),
                 "critic": rng.normal(size=(64, 5)).astype(np.float32),
                 "action": rng.normal(size=(64, 3)).astype(np.float32),
                 "reward": rng.normal(size=64).astype(np.float32),
                 "terminated": rng.random(64) < 0.05, "truncated": np.zeros(64, bool),
                 "finals": {}, "last": (rng.normal(size=7), rng.normal(size=5)), "episodes": []} for _ in range(2)]
    before = [p.detach().clone() for p in learner.policy.parameters()]
    stats = learner.update(rollouts)
    assert learner.decisions == 128 and learner.updates == 1
    assert all(np.isfinite(v) for v in stats.values())
    assert any(not torch.equal(a, b) for a, b in zip(before, learner.policy.parameters()))


def test_the_workers_numpy_actor_acts_as_the_policy():
    cfg = Config(workers=1, rollout=8, hidden=16)
    learner = Learner(cfg, (7, 5))
    for layer in learner.policy.actor[::2]:                      # (a trained actor: its last layer isn't zero)
        torch.nn.init.normal_(layer.weight, std=0.5)
        torch.nn.init.normal_(layer.bias, std=0.5)
    learner.actor_stats.update(np.random.default_rng(1).normal(2.0, 3.0, (100, 7)))
    obs = np.random.default_rng(2).normal(size=(4, 7)).astype(np.float32)
    actor = NumpyActor(learner.message(), np.random.default_rng(0))
    with torch.no_grad():
        expected = learner.policy.dist(torch.as_tensor(learner.actor_stats.normalize(obs))).mean.numpy()
    np.testing.assert_allclose(actor.mean(obs), expected, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(actor.std, np.exp(cfg.log_std))


def test_the_worker_side_never_loads_torch():
    """Each worker would pay ~700 MB for it (they import train_residual and gap_env)."""
    code = "import sys, train_residual, gap_env; print('torch' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"
