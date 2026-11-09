"""Checks the residual's PPO pieces: advantages, normalisation, the policy's
start, the workers' numpy copy of it, the action-scale ramp, one update on
made-up rollouts, and that the worker side never loads torch.

    uv run --group rl pytest tests/test_train_residual.py
"""

import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from insert_book import RESIDUAL_END  # noqa: E402
from residual_ppo import Learner, Policy, gae  # noqa: E402
from train_residual import (Config, NumpyActor, RunningStats, episode_record, evaluate, paired_summary, ramp,
                            run_eval, summary, validation_scenes)  # noqa: E402


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
    """No correction, and the end action well below RESIDUAL_END, with its own spread."""
    p = Policy(7, 5, hidden=16, log_std=-1.0, end_log_std=-0.5, end_bias=-1.8)
    d = p.dist(torch.randn(4, 7))
    assert torch.equal(d.mean[:, :5], torch.zeros(4, 5))
    torch.testing.assert_close(d.mean[:, 5], torch.full((4,), -1.8))
    assert (d.mean[:, 5] < RESIDUAL_END).all()
    torch.testing.assert_close(d.stddev[:, :5], torch.full((4, 5), float(np.exp(-1.0))))
    torch.testing.assert_close(d.stddev[:, 5], torch.full((4,), float(np.exp(-0.5))))
    assert p.value(torch.randn(4, 7), torch.randn(4, 5)).shape == (4,)


def test_a_three_part_policy_has_no_end_action():
    """(As for the gap-creation-only residual of before: Policy(n_action=3).)"""
    p = Policy(7, 5, hidden=16, n_action=3)
    assert torch.equal(p.dist(torch.randn(2, 7)).mean, torch.zeros(2, 3))


def test_the_action_scale_ramps_up_then_stays():
    cfg = Config(ramp_start=0.25, ramp_decisions=100)
    assert ramp(cfg, 0) == 0.25 and ramp(cfg, 50) == pytest.approx(0.625) and ramp(cfg, 500) == 1.0


def test_an_update_on_made_up_rollouts_moves_the_policy():
    cfg = Config(workers=2, rollout=64, minibatch=32, epochs=2, hidden=16, decisions=10_000)
    rng = np.random.default_rng(0)
    learner = Learner(cfg, (7, 5))
    rollouts = [{"actor": rng.normal(size=(64, 7)).astype(np.float32),
                 "critic": rng.normal(size=(64, 5)).astype(np.float32),
                 "action": rng.normal(size=(64, 6)).astype(np.float32),
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
    np.testing.assert_allclose(actor.std, np.exp([cfg.log_std] * 5 + [cfg.end_log_std]), rtol=1e-6)


def test_the_worker_side_never_loads_torch():
    """Each worker would pay ~700 MB for it (they import train_residual and gap_env)."""
    code = "import sys, train_residual, gap_env; print('torch' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"


def test_five_action_policy_update_and_numpy_inference():
    cfg = Config(workers=1, rollout=16, minibatch=8, epochs=1, hidden=16, learn_stop=False)
    learner = Learner(cfg, (7, 5, 5))
    rng = np.random.default_rng(9)
    actor = NumpyActor(learner.message(), rng)
    observations = rng.normal(size=(16, 7)).astype(np.float32)
    actions = np.stack([actor({"actor": x}, sample=True) for x in observations])
    assert actions.shape == (16, 5)
    stats = learner.update([{"actor": observations, "critic": rng.normal(size=(16, 5)).astype(np.float32),
                             "action": actions, "reward": rng.normal(size=16).astype(np.float32),
                             "terminated": np.zeros(16, bool), "truncated": np.zeros(16, bool),
                             "finals": {}, "last": (observations[-1], np.zeros(5)), "episodes": []}])
    assert all(np.isfinite(v) for v in stats.values())
    updated = NumpyActor(learner.message(), rng)
    with torch.no_grad():
        expected = learner.policy.dist(torch.as_tensor(learner.actor_stats.normalize(observations))).mean.numpy()
    np.testing.assert_allclose(updated.mean(observations), expected, rtol=1e-5, atol=1e-6)


def test_validation_counts_layouts_and_keeps_failed_variants(monkeypatch):
    import gap_env
    entries = [gap_env.PoolEntry("clear", seed, None, np.zeros(0), contact=variant != 0,
                                phase="slide" if variant else "", variant=variant,
                                perception_seed=seed + variant, grasp_seed=seed + 10 + variant)
               for seed in (20000, 20001) for variant in (0, 1, 2)]
    monkeypatch.setattr(gap_env, "load_pool", lambda _: [gap_env.Packed(e) for e in reversed(entries)])
    cfg = Config(scenarios=("clear",), val_scenes=1)
    cases = validation_scenes(cfg)
    assert [c["geometry_seed"] for c in cases] == [20000] * 3
    assert [c["variant"] for c in cases] == [0, 1, 2]
    cfg.val_variants = 2
    assert validation_scenes(cfg) == cases[:2]
    cfg.phases = ("slide",)
    assert validation_scenes(cfg) == cases[1:]


def test_evaluation_scale_does_not_follow_the_training_ramp():
    calls = []
    cases = [{"scenario": "clear", "geometry_seed": i, "variant": 0, "perception_seed": i + 10,
              "grasp_seed": i + 20} for i in (1, 2)]

    class Workers:
        def ask(self, messages):
            calls.extend(messages)
            return [[{**c, "outcome": "success"} for c in msg["scenes"]] for _, msg in messages]

    learner = SimpleNamespace(decisions=0, message=lambda **kwargs: kwargs)
    cfg = Config(workers=2, ramp_start=0.25, eval_action_scale=0.8)
    first = run_eval(Workers(), learner, cfg, cases, zero=True)
    learner.decisions = cfg.ramp_decisions
    second = run_eval(Workers(), learner, cfg, cases, zero=False)
    assert first == second
    assert all(msg["scale"] == 0.8 for _, msg in calls)


def test_evaluation_keeps_exact_preparation_failures_and_reward_records():
    from gap_env import PreparationFailed
    cases = [{"scenario": "tight", "geometry_seed": 1, "variant": v,
              "perception_seed": v + 10, "grasp_seed": v + 20} for v in (0, 1)]

    class Env:
        action_scale = 1.0
        action_space = SimpleNamespace(shape=(5,))

        def reset(self, options):
            self.case = options
            if options["variant"] == 0:
                raise PreparationFailed(options, "script could not make room")
            return {}, options

        def step(self, action):
            np.testing.assert_array_equal(action, np.zeros(5))
            return {}, 9.9, True, False, {**self.case, "outcome": "success", "decisions": 1,
                                         "reward_totals": {"terminal": 10.0, "time": -0.1}}

    rows = evaluate(Env(), None, cases, zero=True)
    assert len(rows) == 2 and rows[0]["outcome"] == "preparation_failed"
    assert rows[1]["reward_totals"] == {"terminal": 10.0, "time": -0.1}
    stats = summary(rows)
    assert stats["success"] == 0.5 and stats["success_given_control"] == 1.0
    assert stats["reward_terminal"] == 10 and stats["return_success"] == 9.9
    assert episode_record(rows[1], 9.9)["grasp_seed"] == 21
    assert paired_summary(rows, rows) == {"recovered": 0, "lost": 0, "success_delta": 0.0}
    recovered = [{**r, "outcome": "success"} for r in rows]
    assert paired_summary(rows, recovered)["recovered"] == 1
    with pytest.raises(ValueError, match="same unique"):
        paired_summary(rows, recovered[1:])
