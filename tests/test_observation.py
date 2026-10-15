"""Checks the policy's observation: the free-space walk, and that every part of
the actor's and critic's vectors says what its name says.

    uv run pytest tests/test_observation.py
"""

import math

import mujoco
import numpy as np
import pytest

from insert_book import GAP_MARGIN
from observation import (BOOK_OFFSETS, GAP_CAP, ObservationBuilder, book_names_by_offset, robot_state)
from perception import (FREE_SPACE_CAP, GroundTruthPerception, NoisyPerception, ShelfEstimate, book_from_spine,
                        free_space)
from shelf_scene import ShelfConfig, make_episode

PANDA = "mujoco_menagerie/franka_emika_panda/panda.xml"


# ---------------------------------------------------------------------------
# Free space, on hand-made rows
# ---------------------------------------------------------------------------
def row_of(edges, slot_after, panels=(-0.5, 0.5)):
    """An estimate with upright 0.2 m books spanning the given (left, right)
    x edges, the slot after the first `slot_after` of them."""
    books = []
    for i, (lo, hi) in enumerate(edges):
        offset = i - slot_after if i < slot_after else i - slot_after + 1
        books.append(book_from_spine([[lo, 0, 0], [hi, 0, 0], [hi, 0, 0.2], [lo, 0, 0.2]], offset, 0.17, 1.0))
    return ShelfEstimate(np.eye(4), 0.0, 0.1, 0.3, 0.25, panels, books, np.array([0.1]), np.array([0.05]), 0.0, "test")


def test_free_space_cases():
    packed = [(-0.15, -0.10), (-0.10, -0.05), (0.0, 0.05), (0.05, 0.10)]
    # 15 mm between the second and third book out on the left
    spaced = [(-0.215, -0.165), (-0.15, -0.10), (-0.10, -0.05), (0.0, 0.05), (0.05, 0.10)]
    s = free_space(row_of(spaced, 3), -1)
    assert (s.books, s.resolved) == (2, True) and s.width == pytest.approx(0.015)
    # packed up to a side panel that is in view: no room at all on that side
    s = free_space(row_of(packed, 2, panels=(-0.15, 0.10)), -1)
    assert (s.books, s.width, s.resolved) == (FREE_SPACE_CAP, 0.0, True)
    # room between the last book and the panel
    s = free_space(row_of(packed, 2, panels=(-0.20, 0.10)), -1)
    assert (s.books, s.resolved) == (2, True) and s.width == pytest.approx(0.05)
    # the panel out of view after the last book seen: unknown
    s = free_space(row_of(packed, 2, panels=(None, None)), +1)
    assert (s.books, s.width, s.resolved) == (FREE_SPACE_CAP, 0.0, False)


# ---------------------------------------------------------------------------
# The vectors, on a real scene
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def scene():
    model, data, row, _ = make_episode(ShelfConfig(), "leaning", np.random.default_rng(0), panda_xml=PANDA)
    est = GroundTruthPerception(model, row).estimate(data)
    target = (row.target_book.thickness, row.target_book.height)
    return model, data, row, est, target


def part(builder, x, name, critic=False):
    return x[(builder.critic_layout if critic else builder.actor_layout)[name]]


def test_actor_vector_is_complete_and_finite(scene):
    model, data, row, est, target = scene
    obs = ObservationBuilder(est, target)
    x = obs.actor(est, robot_state(model, data), now=data.time)
    assert np.all(np.isfinite(x))
    assert sum(s.stop - s.start for s in obs.actor_layout.values()) == len(x)
    # the same size with a noisy estimate that leaves books out
    noisy = NoisyPerception(GroundTruthPerception(model, row), np.random.default_rng(0), scale=1.5).estimate(data)
    assert len(obs.actor(noisy, robot_state(model, data), now=data.time)) == len(x)


def test_books_and_gap_are_in_the_slot_frame(scene):
    model, data, row, est, target = scene
    obs = ObservationBuilder(est, target)
    x = obs.actor(est, robot_state(model, data), now=data.time)
    for k in BOOK_OFFSETS:
        b = est.book(k)
        assert part(obs, x, f"book{k:+d}_seen")[0] == 1.0
        centre, thickness, height, lean = (part(obs, x, f"book{k:+d}")[:3], *part(obs, x, f"book{k:+d}")[3:6])
        np.testing.assert_allclose(centre, b.spine_corners.mean(axis=0) - obs.origin, atol=1e-6)
        assert (thickness, height, lean) == pytest.approx((b.thickness, b.height, b.lean), abs=1e-6)
    # the slot frame's origin sits between the neighbours' bottom corners, on the board, at its front edge
    left, right = est.book(-1).spine_corners[1] - obs.origin, est.book(+1).spine_corners[0] - obs.origin
    assert left[0] == pytest.approx(-right[0], abs=1e-6)
    assert left[2] == pytest.approx(0.0, abs=2e-3) and right[2] == pytest.approx(0.0, abs=2e-3)
    narrowest = est.min_gap_width(target[1])
    assert part(obs, x, "gap_narrowest")[0] == pytest.approx(min(narrowest, GAP_CAP), abs=1e-6)
    assert part(obs, x, "gap_widening_needed")[0] == pytest.approx(max(0, target[0] + GAP_MARGIN - narrowest), abs=1e-6)
    assert np.all(part(obs, x, "gap_widths") <= GAP_CAP + 1e-6)


def test_unseen_books_are_flagged_and_zero(scene):
    model, data, row, est, target = scene
    obs = ObservationBuilder(est, target)
    partial = ShelfEstimate(**{**est.__dict__, "books": [b for b in est.books if abs(b.slot_offset) < 3]})
    x = obs.actor(partial, robot_state(model, data), now=data.time)
    for k in (-3, 3):
        assert part(obs, x, f"book{k:+d}_seen")[0] == 0.0
        assert np.all(part(obs, x, f"book{k:+d}") == 0.0)


def test_robot_part_matches_the_simulation(scene):
    model, data, row, est, target = scene
    obs = ObservationBuilder(est, target)
    robot = robot_state(model, data)
    x = obs.actor(est, robot, now=data.time + 1.5)
    hand = model.body("panda/hand").id
    tip_world = data.xpos[hand] + data.xmat[hand].reshape(3, 3)[:, 2] * 0.1034
    tip_shelf = (tip_world - est.shelf_pose[:3, 3]) @ est.shelf_pose[:3, :3]
    np.testing.assert_allclose(part(obs, x, "tool_pos"), tip_shelf - obs.origin, atol=1e-6)
    assert part(obs, x, "tool_pos")[1] > 0                      # the hand is in front of the shelf
    assert part(obs, x, "estimate_age")[0] == pytest.approx(1.5)
    np.testing.assert_allclose(part(obs, x, "joint_pos"), robot.joint_pos, atol=1e-6)
    rot = part(obs, x, "tool_rot").reshape(2, 3)
    np.testing.assert_allclose(rot @ rot.T, np.eye(2), atol=1e-5)  # two orthonormal columns


def test_critic_has_the_true_physics(scene):
    model, data, row, est, target = scene
    obs = ObservationBuilder(est, target)
    names = book_names_by_offset(row)
    v = obs.critic(model, data, est, names)
    assert np.all(np.isfinite(v))
    for k in BOOK_OFFSETS:
        physics = part(obs, v, f"true_book{k:+d}_physics", critic=True)
        body = model.body(names[k]).id
        assert physics[0] == pytest.approx(model.body_mass[body], rel=1e-5)
        assert physics[1] == pytest.approx(model.geom_friction[model.geom(names[k] + "_geom").id, 0], rel=1e-5)
        assert np.abs(physics[2:]).max() < 0.05                  # settled: (almost) at rest
