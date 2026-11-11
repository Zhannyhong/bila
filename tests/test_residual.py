"""Checks the residual hook in insert_book.Runner: without a residual, or with
one that never corrects, the hand moves exactly as the script says; with one,
it is moved by the correction (turned about the contact's pivot), within the
contact's bounds, and may end the move.

    uv run pytest tests/test_residual.py
"""

import mujoco
import numpy as np
import pytest

from bila.control.insert_book import (DEFAULT_PANDA, FINGERTIP_Z, GRIP_OPEN, PAD_Z, PANDA_HOME, RESIDUAL_EVERY, RESIDUAL_MAX,
                         RESIDUAL_STEP, RESIDUAL_TURN_MAX, RESIDUAL_TURN_STEP, ArmIK, Runner, ShelfFrame, axis_rot,
                         make_episode)
from bila.sim.shelf_scene import ShelfConfig

MM = 0.001


class Constant:
    """A residual that asks for the same increment at every decision, and
    remembers the correction at each."""

    def __init__(self, inc):
        self.inc, self.seen = np.asarray(inc, dtype=float), []

    def decide(self, run, contact):
        self.seen.append(run.offset.copy())
        return self.inc


@pytest.fixture(scope="module")
def scene():
    model, data, row = make_episode(ShelfConfig(), "tight", np.random.default_rng(0), DEFAULT_PANDA)
    return model, data, row


def runner(scene, residual=None):
    """A Runner on a fresh copy of the settled scene, the arm at rest at home."""
    model, data, row = scene
    d = mujoco.MjData(model)
    mujoco.mj_copyData(d, model, data)
    ik = ArmIK(model, PANDA_HOME[:7])
    d.qpos[ik.qadr] = PANDA_HOME[:7]
    d.qvel[:] = 0
    mujoco.mj_forward(model, d)
    return Runner(model, d, ik, PANDA_HOME[:7], GRIP_OPEN, log=lambda *_: None, residual=residual,
                  shelf=ShelfFrame(row))


def push(run, dx=0.03, **bounds):
    """A 0.6 s move 3 cm along the row, marked as a contact. Returns its goal
    and how many steps it took."""
    goal = run.pos + run.shelf.R @ [dx, 0.0, 0.0]
    with run.contact("widen", **bounds) as c:
        run.move("push", goal, speed=0.1)
    return goal, (c.steps if c is not None else None)


def test_no_residual_and_a_zero_residual_move_the_hand_the_same(scene):
    plain, zero = runner(scene), runner(scene, Constant([0, 0, 0]))
    for run in (plain, zero):
        push(run)
        run.move("away", run.pos + run.shelf.R @ [0, 0.02, 0.0])
    np.testing.assert_array_equal(plain.d.qpos, zero.d.qpos)
    np.testing.assert_array_equal(plain.q_cmd, zero.q_cmd)
    np.testing.assert_array_equal(plain.pos, zero.pos)
    n = int(round(max(0.3, 1.875 * 0.03 / 0.1) / scene[0].opt.timestep))   # steps of the push
    assert len(zero.residual.seen) == -(-n // RESIDUAL_EVERY)              # a decision every RESIDUAL_EVERY steps


def test_a_constant_residual_moves_the_hand_by_its_correction(scene):
    inc = np.array([0.001, -0.002, 0.003])
    run = runner(scene, Constant(inc))
    goal, steps = push(run)
    # Each increment is ramped in over the RESIDUAL_EVERY steps after its decision (the last
    # one only partly, if the move ends first), and the total is capped.
    expected = np.clip(steps / RESIDUAL_EVERY * inc, -RESIDUAL_MAX, RESIDUAL_MAX)
    np.testing.assert_allclose(run.offset[:3], expected, atol=1e-12)
    assert not run.offset[3:].any()                                          # (it asked for no turns)
    np.testing.assert_allclose(run.pos, goal, atol=1e-12)                    # the script's pose is untouched
    aimed, _ = run.ik.fk(run.q_cmd)
    np.testing.assert_allclose(aimed, goal + run.shelf.R @ expected, atol=0.5 * MM)   # IK aims at both
    # The next move fades the correction out over its first RESIDUAL_FADE.
    run.move("on", goal + run.shelf.R @ [0.0, 0.02, 0.0])
    assert not run.offset.any() and run.fade is None
    aimed, _ = run.ik.fk(run.q_cmd)
    np.testing.assert_allclose(aimed, run.pos, atol=0.5 * MM)


def test_an_increment_is_capped_per_decision(scene):
    run = runner(scene, Constant([0.05, 0.0, 0.0]))
    push(run)
    steps = np.diff(np.array(run.residual.seen)[:, 0])
    assert steps.max() == pytest.approx(RESIDUAL_STEP)


def test_halt_keeps_the_script_pose_and_the_correction(scene):
    run = runner(scene, Constant([0.002, 0.0, -0.002]))
    goal = run.pos + run.shelf.R @ [0.03, 0.0, 0.0]
    with run.contact("widen"):
        run.move("push", goal, speed=0.1)
        offset = run.offset.copy()
        run.halt()
        np.testing.assert_array_equal(run.offset, offset)
        assert not run.rate.any()
        actual, _ = run.ik.fk(run.d.qpos[run.ik.qadr])
        np.testing.assert_allclose(run.pos + run.shelf.R @ run.offset[:3], actual, atol=1e-12)


def test_the_correction_stays_inside_the_contact_bounds(scene):
    run = runner(scene, Constant([-0.003, -0.003, 0.003]))
    push(run, lo=(-0.010, -0.004, -RESIDUAL_MAX), hi=(RESIDUAL_MAX, RESIDUAL_MAX, 0.006))
    seen = np.array(run.residual.seen + [run.offset])
    assert seen[:, 0].min() == pytest.approx(-0.010) and seen[:, 1].min() == pytest.approx(-0.004)
    assert seen[:, 2].max() == pytest.approx(0.006)


def test_the_correction_never_takes_the_hand_past_the_room_limit(scene):
    """x_limit 5 mm beyond where the push ends: the correction may take the
    hand no further. And it never pulls back a hand the script takes past it."""
    run = runner(scene, Constant([0.003, 0.0, 0.0]))
    end = run.shelf.to_shelf(run.pos)[0] + 0.03
    aimed = []                                                   # x of every hand target IK was given
    target = run.hand_target
    run.hand_target = lambda: (p := target(), aimed.append(run.shelf.to_shelf(p[0])[0]))[0]
    push(run, x_limit=(+1, end + 0.005))
    assert max(aimed) <= end + 0.005 + 1e-12
    assert run.offset[0] == pytest.approx(0.005, abs=0.01 * MM)              # up against it

    behind = runner(scene, Constant([0.003, 0.0, 0.0]))
    push(behind, x_limit=(+1, behind.shelf.to_shelf(behind.pos)[0] + 0.01))   # the script itself goes 2 cm past
    assert behind.offset[0] == 0.0                                            # ... and isn't pulled back


# --- Turns, the end action, the insertion's depth limit --------------------------------
def test_a_turn_keeps_the_pivot_and_uncorrected_undoes_it(scene):
    run = runner(scene)
    pos, rot = run.pos.copy(), run.rot.copy()
    for pivot in (FINGERTIP_Z, PAD_Z):
        run.pivot = pivot
        run.offset = np.array([0.004, -0.002, 0.001, np.radians(2.0), np.radians(-1.5)])
        p, R = run.corrected(pos, rot)
        turn = axis_rot(run.shelf.R[:, 2], run.offset[3]) @ axis_rot(run.shelf.R[:, 0], run.offset[4])
        np.testing.assert_allclose(R, turn @ rot, atol=1e-12)
        # The pivot only moves by the correction's x, y, z.
        np.testing.assert_allclose(p + R @ [0, 0, pivot], pos + rot @ [0, 0, pivot] + run.shelf.R @ run.offset[:3],
                                   atol=1e-12)
        back = run.uncorrected(p, R)
        np.testing.assert_allclose(back[0], pos, atol=1e-12)
        np.testing.assert_allclose(back[1], rot, atol=1e-12)


def test_turns_are_capped_per_decision_and_in_all(scene):
    run = runner(scene, Constant([0.0, 0.0, 0.0, 1.0, -1.0]))             # far more than a step asks
    push(run, dx=0.06)
    yaw = np.array(run.residual.seen)[:, 3]
    assert np.diff(yaw).max() == pytest.approx(RESIDUAL_TURN_STEP)
    assert run.offset[3] == pytest.approx(RESIDUAL_TURN_MAX) and run.offset[4] == pytest.approx(-RESIDUAL_TURN_MAX)


def test_halt_with_a_turn_keeps_the_script_pose_and_the_correction(scene):
    run = runner(scene, Constant([0.001, 0.0, 0.0, RESIDUAL_TURN_STEP, 0.0]))
    with run.contact("clamp", pivot=PAD_Z):
        run.move("push", run.pos + run.shelf.R @ [0.03, 0.0, 0.0], speed=0.1)
        offset = run.offset.copy()
        run.halt()
        np.testing.assert_array_equal(run.offset, offset)
        actual = run.ik.fk(run.d.qpos[run.ik.qadr])
        p, R = run.corrected(run.pos, run.rot)                          # the script's pose, corrected, is the hand's
        np.testing.assert_allclose(p, actual[0], atol=1e-12)
        np.testing.assert_allclose(R, actual[1], atol=1e-12)


class EndAt:
    """A residual that corrects nothing, and asks to end the move at its n-th decision."""

    def __init__(self, n):
        self.n, self.k = n, 0

    def decide(self, run, contact):
        self.k += 1
        return np.array([0, 0, 0, 0, 0, 1.0 if self.k >= self.n else -1.0])


def test_the_end_action_leaves_the_block_where_the_hand_is(scene):
    run = runner(scene, EndAt(5))
    after = []
    with run.contact("widen") as c:
        run.move("push", run.pos + run.shelf.R @ [0.03, 0.0, 0.0], speed=0.1)
        after.append("the rest of the block")                            # not reached
    assert c.ended and run.ended == ["widen"] and not after
    assert c.steps == 4 * RESIDUAL_EVERY                                 # ended at the 5th decision, before stepping on
    np.testing.assert_array_equal(run.q_cmd, run.d.qpos[run.ik.qadr])     # held where it is
    assert run.touching is None
    # The script carries on from there as after the move.
    assert run.move("away", run.pos + run.shelf.R @ [0.0, 0.02, 0.0])


def test_an_end_the_move_may_not_take_is_ignored(scene):
    run = runner(scene, EndAt(1))
    with run.contact("slide", may_end=lambda: False) as c:
        done = run.move("push", run.pos + run.shelf.R @ [0.03, 0.0, 0.0], speed=0.1)
    assert done and not c.ended and not run.ended


def test_no_deeper_holds_the_fingertips_to_the_moves_end(scene):
    """A residual pushing the hand into the shelf (-y) as fast as it may: with
    no_deeper, the fingertips never go past where the script's move ends."""
    def tips(no_deeper):
        run = runner(scene, Constant([0.0, -0.003, 0.0, 0.0, np.radians(0.4)]))
        aimed = []
        target = run.hand_target
        run.hand_target = lambda: (p := target(), aimed.append(run.tip_y(*p)))[0]
        goal = run.pos + run.shelf.R @ [0.0, -0.01, 0.0]
        with run.contact("push", no_deeper=no_deeper):
            run.move("push", goal, speed=0.1)
        return np.array(aimed), run.tip_y(goal, run.rot)

    held, end = tips(True)
    assert held.min() >= end - 1e-9
    free, _ = tips(False)
    assert free.min() < end - 0.005                                     # (without it, the residual takes it deeper)


# --- Target corrections (the insertion) --------------------------------------------------
class Sequence:
    """A residual that gives these corrections in turn (the last one after), and remembers the correction at each."""

    def __init__(self, *outs):
        self.outs, self.seen = [np.asarray(o, dtype=float) for o in outs], []

    def decide(self, run, contact):
        self.seen.append(run.offset.copy())
        return self.outs[min(len(self.seen) - 1, len(self.outs) - 1)]


def test_a_steady_target_holds_instead_of_adding_up(scene):
    """The same small correction every decision: with increments it would add
    up to the bounds; as a target it is reached and held."""
    run = runner(scene, Constant([0.002, 0.0, 0.0, 0.0, 0.0]))
    push(run, target=True, lo=(-0.004, -0.004, -0.004), hi=(0.004, 0.004, 0.004))
    seen = np.array(run.residual.seen)
    assert len(seen) > 10
    np.testing.assert_allclose(seen[1:, 0], 0.002, atol=1e-12)                 # reached by the next decision, held
    np.testing.assert_allclose(run.offset[0], 0.002, atol=1e-12)
    assert not seen[:, 1:].any()


def test_a_target_is_kept_in_bounds_and_approached_at_the_step_rate(scene):
    run = runner(scene, Constant([0.05, -0.05, 0.05, 1.0, -1.0]))                # far outside every bound
    push(run, target=True, lo=(-0.004, -0.008, -0.002, -0.02, -0.02), hi=(0.004, 0.0, 0.0, 0.02, 0.02))
    seen = np.array(run.residual.seen)
    assert seen[1, 0] == pytest.approx(RESIDUAL_STEP) and seen[1, 3] == pytest.approx(RESIDUAL_TURN_STEP)
    np.testing.assert_allclose(seen[-1], [0.004, -0.008, 0.0, 0.02, -0.02], atol=1e-12)


def test_a_zero_target_brings_the_hand_back_to_the_script(scene):
    run = runner(scene, Sequence(*[[0.004, 0.0, 0.0, 0.0, 0.0]] * 4, [0.0] * 5))
    push(run, target=True, lo=(-0.004,) * 3, hi=(0.004,) * 3)
    seen = np.array(run.residual.seen)
    assert seen[4, 0] == pytest.approx(0.004)                                    # held at the target ...
    assert seen[5, 0] == pytest.approx(0.004 - RESIDUAL_STEP)                   # ... then back at the step rate
    np.testing.assert_allclose(seen[6:], 0.0, atol=1e-12)
