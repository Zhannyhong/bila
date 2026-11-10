"""Checks the reshelving env: the cached views and first plan change
nothing, zero actions leave the script as it is (through the insertion, or to
insert only), the grasp error puts the book off where the script believes it
holds it, the slide may be ended only once the book is over the board, and
the observation, action and failure tests say what they should.

    uv run --group rl pytest tests/test_gap_env.py      # ~2 min: the scenes' first runs plan from scratch
"""

import copy
import os

import mujoco
import numpy as np
import pytest

pytest.importorskip("gymnasium")

from gap_env import (PHASES, PROGRESS_REWARD, TOPPLE_DEG, Abandon, GapEnv, GapEpisode, GraspError, NoisyEyes,  # noqa: E402
                     Packed, PreparationFailed, ReachedInsert, axis_rot, digest, learned_masks, load_scene,
                     make_entry, make_eyes_for, pool_jobs, quat_to_mat, shift_in_hand, sight, to_go, warm)
from insert_book import (CLEAR, PAD_Z, RESIDUAL_MAX, RESIDUAL_STEP, RESIDUAL_TURN_MAX, SLIDE_HI, Episode,  # noqa: E402
                         RESIDUAL_TURN_STEP)
from shelf_scene import ShelfConfig  # noqa: E402

SEED = 10_000              # a tight training scene the script widens; as a clear one, it inserts at once
WEIGHTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bila_weights", "segmenter",
                       "segmenter.pt")       # Victor's learned masks (not in git)
END = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0])


class Zero:
    """A residual that never corrects anything."""

    def decide(self, run, contact):
        return np.zeros(6)


class Stop:
    """A residual that stops the script at its first contact move."""

    def decide(self, run, contact):
        raise Abandon


def run_to(entry, residual, through_insert=True, grasp=()):
    """Run the script on a pool entry with `residual`, until it stops; the
    final (qpos, time)."""
    model, data = load_scene(ShelfConfig(), entry)
    ep = GapEpisode(model, data, entry, NoisyEyes(model, entry.row, entry.noise_rng), through_insert, grasp)
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


@pytest.fixture(scope="module")
def clear():
    """A clear row's pool entry: its first contact is the slide."""
    entry = warm(("clear", SEED))
    assert entry.contact and entry.phase == "slide"
    return entry


@pytest.fixture(scope="module")
def scripted_tight(warmed):
    entry = copy.deepcopy(warmed[0])
    return run_to(entry, Zero(), grasp=GraspError().draws(entry.grasp_rng, 1.0))


@pytest.fixture
def env(warmed):
    env = GapEnv(pool={"tight": [copy.deepcopy(warmed[0])]})
    yield env
    env.close()


@pytest.fixture
def clear_env(clear):
    env = GapEnv(scenarios=("clear",), pool={"clear": [copy.deepcopy(clear)]})
    yield env
    env.close()


def episode(env, action):
    """Step `action` until the episode ends; (steps, the last info)."""
    n = 0
    while True:
        obs, reward, terminated, truncated, info = env.step(action)
        n += 1
        assert np.isfinite(reward)
        if terminated or truncated:
            return n, info


def test_digest_tells_inputs_apart():
    a = [(np.array([1.0, 2.0]), np.eye(3)), frozenset({3, 1}), None, 0.04]
    assert digest(a) == digest(copy.deepcopy(a))
    b = copy.deepcopy(a)
    b[0][0][1] += 1e-12
    assert digest(a) != digest(b)


@pytest.mark.parametrize("rule,spine,distance,in_place", [
    ("flush", -0.03, 0.02, False),
    ("flush", -0.01, 0.0, True),
    ("flush", 0.0, 0.0, True),
    ("flush", 0.01, 0.0, True),
    ("flush", 0.03, 0.02, False),
    ("flush", np.nextafter(-0.01, -np.inf), 0.0, False),
    ("flush", np.nextafter(0.01, np.inf), 0.0, False),
    ("inside", 0.26, 0.02, False),
    ("inside", 0.245, 0.005, False),
    ("inside", 0.24, 0.0, True),
    ("inside", 0.23, 0.0, True),
    ("inside", 0.20, 0.0, True),
    ("inside", np.nextafter(0.24, np.inf), 0.0, False),
])
def test_placement_acceptance_boundaries(rule, spine, distance, in_place):
    """Flush accepts a 1 cm band. Inside accepts everything 1 cm past the
    front edge, even before reaching the controller's deeper target."""
    ep = Episode.__new__(Episode)
    ep.D = 0.5

    def seat_target(truth=False):
        assert truth
        return (0.0 if rule == "flush" else 0.20), rule

    ep.seat_target = seat_target
    _, error, actual_rule = ep.placement(spine)
    assert actual_rule == rule
    assert error == pytest.approx(distance, abs=1e-12)
    assert (error == 0.0) == in_place  # exact acceptance, including just outside each boundary


def test_insertion_progress_matches_placement_score(clear_env, monkeypatch):
    """Reward approaching the accepted region, penalise leaving it on
    either side, and give no further depth reward while already in place."""
    env = clear_env
    env.reset(options={"index": 0})
    ep = env.ep
    qa = ep.m.jnt_qposadr[ep.book_joint]

    def move_spine(y):
        spine = ep.shelf.box(ep.m, ep.d, ep.book_geom)[:, 1].max()
        ep.d.qpos[qa:qa + 3] += ep.shelf.R[:, 1] * (y - spine)
        mujoco.mj_forward(ep.m, ep.d)

    # Values are (spine, expected distance). The inside aim is deliberately
    # deeper than the accepted front-edge threshold.
    cases = [
        ("flush", 0.0, [(0.03, 0.02), (0.005, 0.0), (-0.005, 0.0), (-0.03, 0.02), (-0.005, 0.0)]),
        ("inside", ep.D / 2 - 0.04,
         [(ep.D / 2 + 0.02, 0.03), (ep.D / 2 - 0.015, 0.0), (ep.D / 2 - 0.02, 0.0),
          (ep.D / 2 - 0.005, 0.005), (ep.D / 2 - 0.015, 0.0)]),
    ]
    for rule, target, positions in cases:
        monkeypatch.setattr(ep, "seat_target", lambda truth=False, target=target, rule=rule: (target, rule))
        move_spine(positions[0][0])
        env.start_insertion()
        previous = positions[0][1]
        assert to_go(ep) == pytest.approx(previous)
        for spine, expected in positions[1:]:
            move_spine(spine)
            distance = to_go(ep)
            assert distance == pytest.approx(expected)
            score = ep.score(0.0, 0.0, False)
            assert score["rule"] == rule
            assert score["in_place"] == (expected == 0.0)
            terms = env.insertion_rewards()
            assert terms["insertion_progress"] == pytest.approx(PROGRESS_REWARD * (previous - expected))
            obs = env.observe(env.ground_truth.estimate(ep.d))
            assert obs["critic"][env.critic_layout["to_go"]][0] == pytest.approx(expected)
            previous = expected


def test_cached_views_and_plan_give_the_same_episode(warmed):
    entry, (qpos, t) = warmed
    assert entry.views is not None and entry.first_plan is not None
    hot = run_to(entry, Stop())
    np.testing.assert_array_equal(hot[0], qpos)
    assert hot[1] == t


def test_zero_actions_follow_the_script(scripted_tight, env):
    """Through the insertion, with the grasp error the env draws: the script's
    own reshelve, to its score."""
    qpos, t = scripted_tight
    env.reset(seed=0)
    n, info = episode(env, np.zeros(6))
    assert info["outcome"] in ("success", "missed") and info["decisions"] == n
    assert info["wide_at_insert"] and np.any(info["grasp_error"]) and info["ended"] == {}
    np.testing.assert_array_equal(env.ep.d.qpos, qpos)
    assert env.ep.d.time == t


def test_zero_actions_follow_the_script_to_insert(warmed):
    """Gap creation only (insert=False): up to where the script would insert."""
    qpos, t = run_to(copy.deepcopy(warmed[0]), Zero(), through_insert=False)
    env = GapEnv(pool={"tight": [copy.deepcopy(warmed[0])]}, insert=False)
    try:
        env.reset(seed=0)
        n, info = episode(env, np.zeros(6))
        assert info["outcome"] in ("success", "narrow") and info["decisions"] == n
        np.testing.assert_array_equal(env.ep.d.qpos, qpos)
        assert env.ep.d.time == t
    finally:
        env.close()


def test_the_observation_fits_its_spaces(env):
    obs, _ = env.reset(seed=0)
    for _ in range(3):
        assert env.observation_space.contains(obs)
        assert np.isfinite(obs["actor"]).all() and np.isfinite(obs["critic"]).all()
        phase = obs["actor"][env.actor_layout["phase"]]
        assert phase.sum() == 1.0 and PHASES[int(np.argmax(phase))] == env.contact.phase
        obs, *_ = env.step(np.zeros(6))


def test_an_action_is_an_increment_of_the_correction(env):
    env.reset(seed=0)
    env.set_action_scale(0.5)
    action = np.array([0.0, 0.0, 1.0, 1.0, 0.0, -1.0])                 # up and a yaw: inside every contact's bounds
    obs, *_ = env.step(action)
    offset = env.ep.run.offset                                          # ramped in by the next decision
    assert offset[2] == pytest.approx(0.5 * RESIDUAL_STEP) and offset[3] == pytest.approx(0.5 * RESIDUAL_TURN_STEP)
    seen = obs["actor"][env.actor_layout["offset"]]
    assert seen[2] == pytest.approx(0.5 * RESIDUAL_STEP / RESIDUAL_MAX)
    assert seen[3] == pytest.approx(0.5 * RESIDUAL_TURN_STEP / RESIDUAL_TURN_MAX)
    np.testing.assert_array_equal(obs["actor"][env.actor_layout["last_action"]], action)


def test_the_grasp_error_puts_the_book_off_where_the_script_believes(clear_env):
    """At the first decision (the slide's start): the book is turned from
    where the script believes it by the error's pitch (to its own measure of
    the grasp, which is a physics step old). And shift_in_hand moves it by
    exactly what it is given, about the pads, leaving the belief as it was."""
    obs, _ = clear_env.reset(options={"index": 0})
    ep, d = clear_env.ep, clear_env.ep.d
    assert ep.grasp_error.any()
    np.testing.assert_allclose(obs["critic"][clear_env.critic_layout["grasp_error"]], ep.grasp_error, rtol=1e-6)
    qa = ep.m.jnt_qposadr[ep.book_joint]
    book = lambda: (d.qpos[qa:qa + 3].copy(), quat_to_mat(d.qpos[qa + 3:qa + 7]))
    _, believed_R = ep.held_pose()
    _, true_R = book()
    angle = np.arccos(np.clip((np.trace(believed_R.T @ true_R) - 1) / 2, -1.0, 1.0))
    assert angle == pytest.approx(abs(ep.grasp_error[2]), abs=1e-4)

    belief = ep.in_hand_p.copy(), ep.in_hand_R.copy()
    p0, R0 = book()
    pads = d.xpos[ep.ik.hand] + d.xmat[ep.ik.hand].reshape(3, 3) @ [0.0, 0.0, PAD_Z]
    out, up, pitch = 0.001, 0.002, np.radians(0.5)                      # (up: away from the board)
    assert shift_in_hand(ep, (out, up, pitch))
    p, R = book()
    turn = axis_rot(R0[:, 0], pitch)
    np.testing.assert_allclose(R, turn @ R0, atol=1e-12)
    np.testing.assert_allclose(p, pads + turn @ (p0 - pads) + R0 @ [0.0, -out, up], atol=1e-12)
    np.testing.assert_array_equal(ep.in_hand_p, belief[0])
    np.testing.assert_array_equal(ep.in_hand_R, belief[1])


def test_the_slide_may_be_ended_only_over_the_board(clear_env):
    obs, _ = clear_env.reset(options={"index": 0})
    ep = clear_env.ep
    allowed = []
    while clear_env.contact is not None and clear_env.contact.phase == "slide":
        may = bool(obs["actor"][clear_env.actor_layout["may_end"]][0])
        assert may == (ep.held_y() < ep.D / 2 - CLEAR - 0.005)
        allowed.append(may)
        obs, _, terminated, truncated, _ = clear_env.step(END)
        assert not terminated and not truncated
    assert not allowed[0] and allowed[-1]                              # refused at first, taken once it may be
    assert ep.run.ended == ["slide"] and clear_env.contact.phase == "push"


def test_reset_abandons_a_running_episode(env):
    env.reset(seed=0)
    env.step(np.zeros(6))
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


def test_insertion_only_prepares_the_same_row_and_preserves_zero_residual(warmed, scripted_tight):
    """A tight row still gets widened, but only slide/push produce decisions.
    Five actions keep stopping scripted, with identical final physics."""
    env = GapEnv(pool={"tight": [warmed[0]]}, control_phases=("slide", "push"), learn_stop=False)
    total, parts, phases = 0.0, {}, set()
    try:
        obs, case = env.reset(options={"geometry_seed": SEED})
        assert env.contact.phase == "slide" and "widen" in env.ep.info["method"]
        assert env.action_space.shape == (5,) and env.observation_space.contains(obs)
        assert not obs["actor"][env.actor_layout["may_end"]].any()
        assert case["geometry_seed"] == SEED and case["variant"] == 0
        while True:
            phases.add(env.contact.phase)
            obs, reward, terminated, truncated, info = env.step(np.zeros(5))
            assert reward == pytest.approx(sum(info["reward_terms"].values()))
            total += reward
            for k, v in info["reward_terms"].items():
                parts[k] = parts.get(k, 0.0) + v
            if terminated or truncated:
                break
        assert phases == {"slide", "push"} and info["ended"] == {}
        assert parts == pytest.approx(info["reward_totals"])
        assert sum(parts.values()) == pytest.approx(total)
        assert parts["gap"] == 0 and parts["gap_bonus"] == 0  # no credit for scripted preparation
        np.testing.assert_array_equal(env.ep.d.qpos, scripted_tight[0])
        assert env.ep.d.time == scripted_tight[1]
    finally:
        env.close()


def test_error_variants_have_independent_replayable_streams(warmed):
    jobs = pool_jobs("tight", 2, SEED, error_variants=3, error_seed=4)
    assert jobs == pool_jobs("tight", 2, SEED, error_variants=3, error_seed=4)
    assert [j[1] for j in jobs] == [SEED] * 3 + [SEED + 1] * 3
    assert len({j[3] for j in jobs} | {j[4] for j in jobs}) == 12
    changed = pool_jobs("tight", 2, SEED, error_variants=3, error_seed=5)
    assert [j[:3] for j in jobs] == [j[:3] for j in changed]
    assert all(a[3:] != b[3:] for a, b in zip(jobs, changed))
    assert pool_jobs("tight", 1, SEED) == [("tight", SEED)]  # old pools keep their exact streams
    entry = copy.deepcopy(warmed[0])
    original_grasp = entry.grasp_rng.normal(size=10)
    entry.perception_seed = jobs[0][3]
    np.testing.assert_array_equal(entry.grasp_rng.normal(size=10), original_grasp)
    entry.variant, entry.grasp_seed = jobs[0][2], jobs[0][4]
    packed = Packed(entry)
    assert packed.unpack().identity() == entry.identity()
    np.testing.assert_array_equal(packed.unpack().noise_rng.normal(size=10), entry.noise_rng.normal(size=10))


def test_legacy_pool_seed_streams_survive_missing_variant_fields(warmed):
    entry = copy.deepcopy(warmed[0])
    for k in ("variant", "perception_seed", "grasp_seed"):
        entry.__dict__.pop(k, None)
    np.testing.assert_array_equal(entry.noise_rng.normal(size=10), np.random.default_rng([SEED, 1]).normal(size=10))
    np.testing.assert_array_equal(entry.grasp_rng.normal(size=10), np.random.default_rng([SEED, 2]).normal(size=10))
    assert entry.identity()["variant"] == 0


def test_explicit_case_is_not_substituted_or_removed_after_preparation_failure(warmed, monkeypatch):
    a, b = copy.deepcopy(warmed[0]), copy.deepcopy(warmed[0])
    b.variant = 1
    env = GapEnv(pool={"tight": [a, b]})
    attempts = []

    def failed(entry):
        attempts.append(entry.variant)
        return "done", ("error", "no plan")

    monkeypatch.setattr(env, "start", failed)
    try:
        for _ in range(2):
            with pytest.raises(PreparationFailed) as error:
                env.reset(options={"geometry_seed": SEED, "variant": 0})
            assert error.value.record["geometry_seed"] == SEED
            assert error.value.record["variant"] == 0
        assert attempts == [0, 0]
        assert all(p.contact is not False for p in env.pool["tight"])
        with pytest.raises(IndexError):
            env.reset(options={"index": 2})
    finally:
        env.close()


def test_exact_grasp_error_case_replays_without_changing_geometry(clear_env):
    options = {"geometry_seed": SEED, "perception_seed": [SEED, 1], "grasp_seed": 42}
    obs, info = clear_env.reset(options=options)
    state, error = clear_env.ep.d.qpos.copy(), clear_env.ep.grasp_error.copy()
    again, replay = clear_env.reset(options=options)
    assert info == replay and info["grasp_seed"] == 42
    np.testing.assert_array_equal(clear_env.ep.d.qpos, state)
    np.testing.assert_array_equal(clear_env.ep.grasp_error, error)
    for name in obs:
        np.testing.assert_array_equal(again[name], obs[name])


def test_pool_with_only_failed_preparations_can_be_evaluated(warmed, monkeypatch):
    entry = copy.deepcopy(warmed[0])
    entry.contact, entry.phase, entry.why = False, "", "no plan"
    env = GapEnv(pool={"tight": [entry]})
    monkeypatch.setattr(env, "start", lambda _: ("done", ("error", "no plan")))
    try:
        with pytest.raises(PreparationFailed) as error:
            env.reset(options={"geometry_seed": SEED})
        assert error.value.record["outcome"] == "preparation_failed"
        with pytest.raises(PreparationFailed, match="no case reached"):
            env.reset(seed=0)
    finally:
        env.close()


def test_an_insertion_action_is_a_bounded_target(clear_env):
    """The slide (a target contact): an action is where the correction goes,
    inside SLIDE_LO..SLIDE_HI, reached at RESIDUAL_STEP a decision; a steady
    small action holds a small correction; it never lifts the book."""
    clear_env.reset(options={"index": 0})
    clear_env.set_action_scale(1.0)
    assert clear_env.contact.phase == "slide" and clear_env.contact.target
    run = clear_env.ep.run
    full = np.array([1.0, 1.0, 1.0, 0.0, 0.0, -1.0])                         # along, out of the shelf, up
    clear_env.step(full)
    assert run.offset[0] == pytest.approx(RESIDUAL_STEP)                      # rate-limited ...
    clear_env.step(full)
    np.testing.assert_allclose(run.offset[:3], [SLIDE_HI[0], SLIDE_HI[1], 0.0], atol=1e-12)   # ... to the bounds, no lift
    for _ in range(3):
        clear_env.step(np.zeros(6))
    np.testing.assert_allclose(run.offset, 0.0, atol=1e-12)                   # zero: back to the script's pose
    small = np.array([0.1, 0.0, 0.0, 0.0, 0.0, -1.0])
    for _ in range(20):
        clear_env.step(small)
        assert clear_env.contact.phase == "slide"
        assert run.offset[0] == pytest.approx(0.1 * SLIDE_HI[0])              # held, not adding up


@pytest.fixture(scope="module")
def clear_camera():
    """The clear row's pool entry, seen by the rendered camera (the simulation's masks)."""
    entry = warm(("clear", SEED), perception="camera")
    assert entry.contact and entry.phase == "slide" and entry.perception == "camera"
    return entry


def test_camera_zero_actions_follow_the_script(clear_camera):
    """Through the camera, the env changes nothing either: zero corrections
    end where the script alone does, though the env runs it in a thread."""
    entry = copy.deepcopy(clear_camera)
    model, data = load_scene(ShelfConfig(), entry)
    eyes = make_eyes_for(entry, model, "camera")
    try:
        ep = GapEpisode(model, data, entry, eyes, grasp=GraspError().draws(entry.grasp_rng, 1.0))
        ep.residual = Zero()
        alone = ep.reshelve()
    finally:
        eyes.close()
    env = GapEnv(scenarios=("clear",), pool={"clear": [copy.deepcopy(clear_camera)]}, perception="camera",
                 control_phases=("slide", "push"), learn_stop=False)
    try:
        env.reset(seed=0)
        camera = env.ep.eyes.camera
        n, info = episode(env, np.zeros(5))
        assert info["outcome"] == ("success" if alone["success"] else "missed") and info["decisions"] == n
        np.testing.assert_array_equal(env.ep.d.qpos, data.qpos)
        assert env.ep.d.time == data.time
        assert camera.renderer._gl_context is None   # let go of (by the script's thread) as it ended
        env.reset(seed=0)                            # and a new episode makes its own
        assert env.ep.eyes.camera is not camera
    finally:
        env.close()


def test_an_env_takes_only_a_pool_that_saw_its_rows_as_it_does(clear, clear_camera):
    with pytest.raises(ValueError, match="saw its rows as noisy"):
        GapEnv(scenarios=("clear",), pool={"clear": [copy.deepcopy(clear)]}, perception="camera")
    with pytest.raises(ValueError, match="saw its rows as camera"):
        GapEnv(scenarios=("clear",), pool={"clear": [copy.deepcopy(clear_camera)]})
    with pytest.raises(ValueError, match="not the camera"):
        GapEnv(scenarios=("clear",), pool={"clear": [copy.deepcopy(clear_camera)]}, perception="camera",
               noise_scale=2.0)
    with pytest.raises(ValueError, match="segmenter goes with"):
        sight("noisy", "segmenter.pt")
    assert sight("camera", "/a/segmenter.pt") == "camera+segmenter.pt" and sight() == "noisy"
    assert Packed(clear).perception == "noisy"


@pytest.mark.skipif(not os.path.exists(WEIGHTS), reason="the learned masks' weights aren't here")
def test_camera_with_learned_masks_loads_them_once(clear_camera):
    model, _ = load_scene(ShelfConfig(), clear_camera)
    a, b = (make_eyes_for(clear_camera, model, "camera", segmenter=WEIGHTS) for _ in range(2))
    try:
        assert a.segment is b.segment is learned_masks(WEIGHTS) and a.robust_fit
    finally:
        a.close(), b.close()
