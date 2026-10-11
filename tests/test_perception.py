"""Checks that perception.py reports the target row as it really is.

    uv run pytest tests/test_perception.py
"""

import math

import numpy as np
import pytest

from perception import (DEPTH_PRIOR, GroundTruthPerception, NoisyPerception, book_from_spine,
                        gap_profile, slice_x)
from shelf_scene import ShelfConfig, make_episode

MM = 0.001


def spine(t, h, lean=0.0, yaw=0.0, base=(0.0, 0.0, 0.0)):
    """Spine corners of a book `t` thick and `h` tall, leaning about its
    bottom-left corner, then swivelled about vertical."""
    cl, sl, cy, sy = math.cos(lean), math.sin(lean), math.cos(yaw), math.sin(yaw)
    Ry = np.array([[cl, 0, sl], [0, 1, 0], [-sl, 0, cl]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    local = np.array([[0, 0, 0], [t, 0, 0], [t, 0, h], [0, 0, h]])
    return local @ (Rz @ Ry).T + np.array(base)


# ---------------------------------------------------------------------------
# Derived values, on hand-made spines
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("lean, yaw", [(0.0, 0.0), (0.4, 0.0), (-0.3, 0.0), (0.0, 0.2), (0.0, -0.1)])
def test_book_from_spine_recovers_size_and_angles(lean, yaw):
    b = book_from_spine(spine(0.03, 0.25, lean, yaw), slot_offset=-1, depth=0.17, visible=1.0)
    assert b.thickness == pytest.approx(0.03)
    assert b.height == pytest.approx(0.25)
    assert b.lean == pytest.approx(lean)
    assert b.yaw == pytest.approx(yaw)


def test_slice_x_of_a_leaning_spine():
    c = spine(0.02, 0.20, lean=math.radians(30))
    lo, hi = slice_x(c, 0.10)
    shift = 0.10 * math.tan(math.radians(30))              # the left edge has moved this far right
    assert lo == pytest.approx(shift)
    assert hi - lo == pytest.approx(0.02 / math.cos(math.radians(30)))
    assert slice_x(c, 0.25) is None                         # above the book's top


def test_gap_profile_of_a_v_shaped_gap():
    """Left neighbour leans right into a 60 mm gap, pivoting on its bottom-right
    corner at x = 0, until its top touches the right neighbour."""
    t, h = 0.02, 0.20
    lean = math.asin(0.06 / h)
    s, c = math.sin(lean), math.cos(lean)
    up = np.array([h * s, 0, h * c])
    bl, br = np.array([-t * c, 0, t * s]), np.zeros(3)
    left = book_from_spine([bl, br, br + up, bl + up], -1, 0.17, 1.0)
    right = book_from_spine(spine(0.02, 0.25, base=(0.06, 0, 0)), +1, 0.17, 1.0)
    w = gap_profile([left, right], (-0.5, 0.5), 0.0, [0.0, 0.10, 0.30])
    assert w[0] == pytest.approx(0.06)                      # full width at the board
    assert w[1] == pytest.approx(0.06 - 0.10 * math.tan(lean))
    assert w[2] == pytest.approx(1.0)                       # above both books: panel to panel


# ---------------------------------------------------------------------------
# Ground truth, on settled scenes
# ---------------------------------------------------------------------------
def placed_gap(row):
    """Gap between the neighbours' faces as the generator placed them.
    (row.gap_left is set after the left neighbour's spacing, so
    row.gap_right - row.gap_left is up to 3 mm short of this.)"""
    left, right = row.books[row.target_index - 1], row.books[row.target_index]
    return (right.x - right.thickness / 2) - (left.x + left.thickness / 2)


@pytest.fixture(scope="module")
def clear_scene():
    model, data, row, _ = make_episode(ShelfConfig(), "clear", np.random.default_rng(0))
    return model, data, row


@pytest.fixture(scope="module")
def leaning_scene():
    model, data, row, _ = make_episode(ShelfConfig(), "leaning", np.random.default_rng(0))
    return model, data, row


def test_slot_offsets_name_the_neighbours(clear_scene):
    model, data, row = clear_scene
    est = GroundTruthPerception(model, row).estimate(data)
    ti = row.target_index
    assert [b.slot_offset for b in est.books] == [*range(-ti, 0), *range(1, len(row.books) - ti + 1)]
    assert est.book(-1).thickness == pytest.approx(row.books[ti - 1].thickness)
    assert est.book(+1).thickness == pytest.approx(row.books[ti].thickness)
    assert est.book(0) is None


def test_upright_books_match_the_generator(clear_scene):
    model, data, row = clear_scene
    est = GroundTruthPerception(model, row).estimate(data)
    assert est.source == "ground_truth" and est.stamp == data.time
    for b, true in zip(est.books, row.books):
        assert b.thickness == pytest.approx(true.thickness, abs=1e-6)
        assert b.height == pytest.approx(true.height, abs=1e-6)
        assert abs(b.lean) < math.radians(0.5) and abs(b.yaw) < math.radians(0.5)
        assert b.spine_y == pytest.approx(row.shelf_depth / 2 - true.inset, abs=1 * MM)
        assert b.depth == min(DEPTH_PRIOR, row.shelf_depth - 0.03)   # a prior, never the true depth
        assert b.visible == 1.0


def test_clear_gap_matches_the_generator(clear_scene):
    model, data, row = clear_scene
    est = GroundTruthPerception(model, row).estimate(data)
    below_both = est.gap_heights < min(est.book(-1).height, est.book(+1).height)
    np.testing.assert_allclose(est.gap_widths[below_both], placed_gap(row), atol=1 * MM)
    assert est.min_gap_width(0.15) == pytest.approx(placed_gap(row), abs=1 * MM)


def test_leaning_neighbour_closes_the_gap_toward_its_top(leaning_scene):
    model, data, row = leaning_scene
    est = GroundTruthPerception(model, row).estimate(data)
    ti, leaner_index = row.target_index, row.nudges[0]["book"]
    side = -1 if leaner_index == ti - 1 else +1                  # which neighbour leans
    leaner = est.book(side)
    assert -side * leaner.lean > math.radians(10)                # leans into the gap
    # Its gap-side bottom corner stays put, so the gap keeps its width at the
    # board and narrows steadily up to its gap-side top corner (the lower of
    # its two top corners). Above that, the book it rests on may end.
    assert est.gap_widths[0] == pytest.approx(placed_gap(row), abs=3 * MM)
    top = leaner.spine_corners[2:, 2].min() - est.board_top
    w = est.gap_widths[est.gap_heights < top]
    assert np.all(np.diff(w) <= 1e-6) and w[-1] < w[0] - 10 * MM


# ---------------------------------------------------------------------------
# Noise
# ---------------------------------------------------------------------------
def test_noisy_perception_stays_close_and_is_repeatable(leaning_scene):
    model, data, row = leaning_scene
    gt = GroundTruthPerception(model, row)
    a = NoisyPerception(gt, np.random.default_rng(1)).estimate(data)
    b = NoisyPerception(gt, np.random.default_rng(1)).estimate(data)
    exact = gt.estimate(data)
    assert a.source == "noisy"
    np.testing.assert_array_equal(a.gap_widths, b.gap_widths)
    for noisy, true in zip(a.books, exact.books):
        assert noisy.slot_offset == true.slot_offset
        assert np.abs(noisy.spine_corners - true.spine_corners).max() < 10 * MM    # 5 sigma
        assert abs(noisy.lean - true.lean) < math.radians(5)
    assert np.all((a.gap_widths == 0) | (a.gap_widths >= 0.004))  # narrow gaps read as closed
