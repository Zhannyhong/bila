"""Checks that perception.py reports the target row as it really is.

    uv run pytest tests/test_perception.py
"""

import math

import numpy as np
import pytest

from perception import (CAMERA_ERRORS, DEPTH_PRIOR, SYSTEMATIC_PARAMS, VIEW_REACH, GroundTruthPerception,
                        NoisyPerception, apply_shift, book_from_spine, fit_rigid, gap_profile, matrix_from_rotvec,
                        perturb_spine, rigid_shift, rotvec_from_matrix, slice_x, slot_pivot)
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
def test_perturb_spine_adds_the_errors_asked_for():
    c = spine(0.03, 0.25)
    zero = dict(x=0, z=0, spine_y=0, thickness=0, height=0, lean=0, yaw=0)
    np.testing.assert_allclose(perturb_spine(c, zero), c, atol=1e-12)
    b = book_from_spine(perturb_spine(c, {**zero, "thickness": 0.002, "height": -0.003, "lean": 0.1,
                                          "yaw": 0.05, "x": 0.01}), -1, 0.17, 1.0)
    assert b.thickness == pytest.approx(0.032)
    assert b.height == pytest.approx(0.247)
    assert b.lean == pytest.approx(0.1, abs=1e-3)               # angles are projected, so the two mix slightly
    assert b.yaw == pytest.approx(0.05, abs=1e-3)
    assert b.spine_corners.mean(axis=0)[0] == pytest.approx(c.mean(axis=0)[0] + 0.01)


def test_noisy_perception_errs_like_the_camera(leaning_scene):
    model, data, row = leaning_scene
    gt = GroundTruthPerception(model, row)
    exact = gt.estimate(data)
    a = NoisyPerception(gt, np.random.default_rng(1)).estimate(data)
    b = NoisyPerception(gt, np.random.default_rng(1)).estimate(data)
    assert a.source == "noisy"
    np.testing.assert_array_equal(a.gap_widths, b.gap_widths)              # repeatable with a seed
    truth = {t.slot_offset: t for t in exact.books}
    for noisy in a.books:
        true = truth[noisy.slot_offset]
        assert np.abs(noisy.spine_corners - true.spine_corners).max() < 20 * MM
        assert abs(noisy.lean - true.lean) < math.radians(2)
    # Books further along the row than the camera sees are left out.
    x_slot = exact.slot_x()
    for t in exact.books:
        outer = t.spine_corners[0, 0] if t.slot_offset < 0 else t.spine_corners[1, 0]
        assert (a.book(t.slot_offset) is not None) == (abs(outer - x_slot) <= VIEW_REACH)


def test_noisy_perception_without_spread_keeps_only_the_camera_biases(leaning_scene):
    model, data, row = leaning_scene
    gt = GroundTruthPerception(model, row)
    exact = gt.estimate(data)
    est = NoisyPerception(gt, np.random.default_rng(0), scale=0).estimate(data)
    bias = CAMERA_ERRORS[(1, 1)]
    for offset in (-1, +1):
        assert est.book(offset).thickness == pytest.approx(exact.book(offset).thickness + bias["thickness"][0])
        assert est.book(offset).height == pytest.approx(exact.book(offset).height + bias["height"][0])


# ---------------------------------------------------------------------------
# The systematic error: one rigid motion of the whole row per episode
# ---------------------------------------------------------------------------
NO_BOOK_ERRORS = {band: {k: (0.0, 0.0) for k in by_param} for band, by_param in CAMERA_ERRORS.items()}
SHIFTS = {"x": (0.0, 0.003), "y": (0.0, 0.003), "z": (0.0, 0.003),
          "rx": (0.0, 0.01), "ry": (0.0, 0.01), "rz": (0.0, 0.01)}


def test_rotvec_round_trip():
    w = np.array([0.01, -0.02, 0.005])
    R = matrix_from_rotvec(w)
    np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-12)
    np.testing.assert_allclose(rotvec_from_matrix(R), w, atol=1e-12)
    np.testing.assert_allclose(rotvec_from_matrix(np.eye(3)), 0.0)


def test_fit_rigid_recovers_a_known_motion():
    rng = np.random.default_rng(0)
    pts = rng.uniform(-0.3, 0.3, (40, 3))
    pivot = np.array([0.1, 0.15, 0.8])
    R, t = matrix_from_rotvec([0.004, -0.01, 0.007]), np.array([0.002, -0.001, 0.003])
    R_fit, t_fit = fit_rigid(pts, apply_shift(pts, R, t, pivot), pivot)
    np.testing.assert_allclose(R_fit, R, atol=1e-10)
    np.testing.assert_allclose(t_fit, t, atol=1e-10)


def test_noisy_perception_shifts_the_whole_row_once_per_episode(leaning_scene):
    """With no error of each book's own, every corner is the true one moved by
    the one rigid motion drawn for the episode, the same in every estimate."""
    model, data, row = leaning_scene
    gt = GroundTruthPerception(model, row)
    exact = gt.estimate(data)
    noisy = NoisyPerception(gt, np.random.default_rng(3), errors=NO_BOOK_ERRORS, reach=10.0, systematic=SHIFTS)
    first, again = noisy.estimate(data), noisy.estimate(data)
    for a, b in zip(first.books, again.books):
        np.testing.assert_array_equal(a.spine_corners, b.spine_corners)
    R, t = rigid_shift(noisy.shift)
    src = np.concatenate([b.spine_corners for b in exact.books])
    dst = np.concatenate([b.spine_corners for b in first.books])
    np.testing.assert_allclose(dst, apply_shift(src, R, t, slot_pivot(exact)), atol=1e-12)
    R_fit, t_fit = fit_rigid(src, dst, slot_pivot(exact))
    np.testing.assert_allclose(t_fit, t, atol=1e-9)
    assert np.abs(t).max() > 0.5 * MM                                       # it did move them
    other = NoisyPerception(gt, np.random.default_rng(4), errors=NO_BOOK_ERRORS, systematic=SHIFTS)
    assert other.shift != noisy.shift                                        # a new episode, a new camera


def test_noisy_perception_draws_the_books_noise_the_same_with_or_without_the_shift(leaning_scene):
    """The shift comes from its own random stream: taking it out leaves each
    book's error, for a seed, as it was."""
    model, data, row = leaning_scene
    gt = GroundTruthPerception(model, row)
    with_shift = NoisyPerception(gt, np.random.default_rng(5), systematic=SHIFTS)
    a = with_shift.estimate(data)
    b = NoisyPerception(gt, np.random.default_rng(5), systematic=None).estimate(data)
    R, t = rigid_shift(with_shift.shift)
    for x, y in zip(a.books, b.books):
        np.testing.assert_allclose(x.spine_corners, apply_shift(y.spine_corners, R, t, with_shift.pivot), atol=1e-12)


def test_calibration_takes_the_shift_back_out(leaning_scene):
    """eval_perception fits each scene's rigid motion and takes it out: on an
    estimate that is the truth shifted, what is left is the truth."""
    from eval_perception import SceneRecord, systematic_params, systematic_stats, without_systematic
    model, data, row = leaning_scene
    gt = GroundTruthPerception(model, row)
    truth = gt.estimate(data)
    noisy = NoisyPerception(gt, np.random.default_rng(6), errors=NO_BOOK_ERRORS, reach=10.0, systematic=SHIFTS)
    shifted = noisy.estimate(data)
    index = {b.slot_offset: i for i, b in enumerate(truth.books)}
    rec = SceneRecord("leaning", 0, 0.5, (0.03, 0.25), truth, {"realistic": shifted},
                      {"realistic": [index[b.slot_offset] for b in shifted.books]})
    stats, n = systematic_stats([rec])
    assert n == 1
    for p in SYSTEMATIC_PARAMS:
        assert stats[p][0] == pytest.approx(noisy.shift[p], abs=1e-9)
    assert systematic_params(*rigid_shift(noisy.shift)) == pytest.approx(noisy.shift)
    back = without_systematic([rec])[0].camera["realistic"]
    for b in back.books:
        np.testing.assert_allclose(b.spine_corners, truth.books[index[b.slot_offset]].spine_corners, atol=1e-9)


def test_learned_calibration_is_opt_in(leaning_scene):
    """NoisyPerception's defaults stay the camera's (gap_env trains on them);
    NoisyPerception.learned uses the learned pipeline's calibration."""
    from perception import CAMERA_SYSTEMATIC, LEARNED_CALIBRATION, LEARNED_CAMERA_ERRORS
    model, data, row = leaning_scene
    gt = GroundTruthPerception(model, row)
    default = NoisyPerception(gt, np.random.default_rng(7))
    assert default.errors is CAMERA_ERRORS and default.reach == VIEW_REACH
    learned = NoisyPerception.learned(gt, np.random.default_rng(7))
    assert learned.errors is LEARNED_CAMERA_ERRORS
    same = NoisyPerception(gt, np.random.default_rng(7), **LEARNED_CALIBRATION)
    assert learned.shift == same.shift
    for a, b in zip(learned.estimate(data).books, same.estimate(data).books):
        np.testing.assert_array_equal(a.spine_corners, b.spine_corners)
    assert set(CAMERA_SYSTEMATIC) == set(LEARNED_CALIBRATION["systematic"])


def test_failures_counts_missed_and_misnamed_books(leaning_scene):
    """eval_perception.failures: of the true books within reach, the share the
    camera missed and the share it named a place off."""
    from dataclasses import replace
    from eval_perception import SceneRecord, failures
    model, data, row = leaning_scene
    truth = GroundTruthPerception(model, row).estimate(data)
    near = [i for i, t in enumerate(truth.books) if abs(t.slot_offset) == 1]
    gone, renamed = near[0], near[1]
    books = [replace(t, slot_offset=t.slot_offset + 1) if i == renamed else t
             for i, t in enumerate(truth.books) if i != gone]
    index = [i for i in range(len(truth.books)) if i != gone]
    rec = SceneRecord("leaning", 0, 0.5, (0.03, 0.25), truth, {"learned": replace(truth, books=books)},
                      {"learned": index})
    n, missed, misnamed = failures([rec], "learned", reach=10.0)[(1, 1)]
    assert n == len(near) and missed == pytest.approx(1 / n) and misnamed == pytest.approx(1 / n)
