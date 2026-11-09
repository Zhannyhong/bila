"""Checks the gap-creation env: the cached views and first plan change
nothing, zero actions leave the script as it is, and the observation,
action and failure tests say what they should.

    uv run --group rl pytest tests/test_gap_env.py      # ~1 min: the scene's first run plans from scratch
"""

import copy

import mujoco
import numpy as np
import pytest

pytest.importorskip("gymnasium")

from bila.rl.gap_env import (PHASES, TOPPLE_DEG, Abandon, GapEnv, GapEpisode, NoisyEyes, ReachedInsert, digest,  # noqa: E402
                     load_scene, make_entry)
from bila.control.insert_book import RESIDUAL_MAX, RESIDUAL_STEP  # noqa: E402
from bila.sim.shelf_scene import ShelfConfig  # noqa: E402

SEED = 10_000              # a tight training scene the script widens


class Zero:
    """A residual that never corrects anything."""

    def decide(self, run, contact):
        return np.zeros(3)


class Stop:
    """A residual that stops the script at its first contact move."""

    def decide(self, run, contact):
        raise Abandon


def run_to(entry, residual):
    """Run the script on a pool entry with `residual`, until it stops; the
    final (qpos, time)."""
    model, data = load_scene(ShelfConfig(), entry)
    ep = GapEpisode(model, data, entry, NoisyEyes(model, entry.row, entry.noise_rng))
    ep.residual = residual
    try:
        ep.reshelve()
    except (Abandon, ReachedInsert):
        pass
    return data.qpos.copy(), data.time


@pytest.fixture(scope="module")
def warmed():
    """A pool entry, and where its first, uncached run got to by its first contact."""
    entry = make_entry(ShelfConfig(), "tight", SEED)
    cold = run_to(entry, Stop())
    return entry, cold


@pytest.fixture
def env(warmed):
    env = GapEnv(pool={"tight": [copy.deepcopy(warmed[0])]})
    yield env
    env.close()


def test_digest_tells_inputs_apart():
    a = [(np.array([1.0, 2.0]), np.eye(3)), frozenset({3, 1}), None, 0.04]
    assert digest(a) == digest(copy.deepcopy(a))
    b = copy.deepcopy(a)
    b[0][0][1] += 1e-12
    assert digest(a) != digest(b)


def test_cached_views_and_plan_give_the_same_episode(warmed):
    entry, (qpos, t) = warmed
    assert entry.views is not None and entry.first_plan is not None
    hot = run_to(entry, Stop())
    np.testing.assert_array_equal(hot[0], qpos)
    assert hot[1] == t


def test_zero_actions_follow_the_script(warmed, env):
    qpos, t = run_to(copy.deepcopy(warmed[0]), Zero())
    env.reset(seed=0)
    n = 0
    while True:
        obs, reward, terminated, truncated, info = env.step(np.zeros(3))
        n += 1
        assert np.isfinite(reward)
        if terminated or truncated:
            break
    assert info["outcome"] in ("success", "narrow") and info["decisions"] == n
    np.testing.assert_array_equal(env.ep.d.qpos, qpos)
    assert env.ep.d.time == t


def test_the_observation_fits_its_spaces(env):
    obs, _ = env.reset(seed=0)
    for _ in range(3):
        assert env.observation_space.contains(obs)
        assert np.isfinite(obs["actor"]).all() and np.isfinite(obs["critic"]).all()
        phase = obs["actor"][-20:-16]                                  # extras: phase, time, target, rot, offset, action
        assert phase.sum() == 1.0 and PHASES[int(np.argmax(phase))] == env.contact.phase
        obs, *_ = env.step(np.zeros(3))


def test_an_action_is_an_increment_of_the_correction(env):
    env.reset(seed=0)
    env.set_action_scale(0.5)
    obs, *_ = env.step(np.array([0.0, 0.0, 1.0]))                      # up: inside every contact's bounds
    assert env.ep.run.offset[2] == pytest.approx(0.5 * RESIDUAL_STEP)   # ramped in by the next decision
    assert obs["actor"][-4] == pytest.approx(0.5 * RESIDUAL_STEP / RESIDUAL_MAX)
    np.testing.assert_array_equal(obs["actor"][-3:], [0.0, 0.0, 1.0])  # the last action


def test_reset_abandons_a_running_episode(env):
    env.reset(seed=0)
    env.step(np.zeros(3))
    thread = env.thread
    env.reset(seed=1)
    assert not thread.is_alive() and env.thread.is_alive()


def test_a_toppled_book_fails_the_episode(env):
    env.reset(seed=0)
    ep = env.ep
    i = env.steady[0]
    adr = ep.m.jnt_qposadr[ep.m.body_jntadr[ep.row_bodies[i]]]
    quat = ep.d.qpos[adr + 3:adr + 7].copy()
    tip, tipped = np.zeros(4), np.zeros(4)
    mujoco.mju_axisAngle2Quat(tip, np.array([0.0, 1.0, 0.0]), np.radians(TOPPLE_DEG + 5))
    mujoco.mju_mulQuat(tipped, tip, quat)
    ep.d.qpos[adr + 3:adr + 7] = tipped
    mujoco.mj_forward(ep.m, ep.d)
    _, _, tilt = env.potential()
    assert tilt > TOPPLE_DEG
    assert "tilted" in env.failure(env.ground_truth.estimate(ep.d), tilt)


def test_scenes_are_picked_by_their_first_contact(warmed):
    widen, shove = copy.deepcopy(warmed[0]), copy.deepcopy(warmed[0])
    widen.phase, shove.phase, shove.seed = "widen", "shove", SEED + 1
    with pytest.raises(ValueError):
        GapEnv(pool={"tight": [shove]}, first_phases=("widen",))
    env = GapEnv(pool={"tight": [shove, widen]}, first_phases=("widen",))
    try:
        assert [p.seed for p in env.pool["tight"]] == [SEED]
        _, info = env.reset(seed=0, options={"index": 0})
        assert info["seed"] == SEED and env.contact.phase == "widen"
    finally:
        env.close()
