"""Checks the residual hook in insert_book.Runner: without a residual, or with
one that never corrects, the hand moves exactly as the script says; with one,
it is moved by the correction, within the contact's bounds.

    uv run pytest tests/test_residual.py
"""

import mujoco
import numpy as np
import pytest

from bila.control.insert_book import (DEFAULT_PANDA, GRIP_OPEN, PANDA_HOME, RESIDUAL_EVERY, RESIDUAL_MAX, RESIDUAL_STEP, ArmIK,
                         Runner, ShelfFrame, make_episode)
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
    np.testing.assert_allclose(run.offset, expected, atol=1e-12)
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
        np.testing.assert_allclose(run.pos + run.shelf.R @ run.offset, actual, atol=1e-12)


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
    run.hand_target = lambda: (p := target(), aimed.append(run.shelf.to_shelf(p)[0]))[0]
    push(run, x_limit=(+1, end + 0.005))
    assert max(aimed) <= end + 0.005 + 1e-12
    assert run.offset[0] == pytest.approx(0.005, abs=0.01 * MM)              # up against it

    behind = runner(scene, Constant([0.003, 0.0, 0.0]))
    push(behind, x_limit=(+1, behind.shelf.to_shelf(behind.pos)[0] + 0.01))   # the script itself goes 2 cm past
    assert behind.offset[0] == 0.0                                            # ... and isn't pulled back
