"""Checks that the camera-based ShelfEstimate matches the ground truth.

    MUJOCO_GL=egl uv run pytest tests/test_rgbd_perception.py      # in WSL
"""

import math

import numpy as np
import pytest

from camera import DepthNoise, WristCamera
from inspection import look_from, solve_inspection
from perception import GroundTruthPerception, book_from_spine, gap_profile
from rgbd_perception import RGBDPerception, fit_spine
from shelf_scene import ShelfConfig, make_episode

PANDA = "mujoco_menagerie/franka_emika_panda/panda.xml"
MM = 0.001


def camera_and_truth(scenario, seed, noise=None):
    """(camera estimate, ground truth, ground-truth book for each camera book, row)."""
    model, data, row, _ = make_episode(ShelfConfig(), scenario, np.random.default_rng(seed), panda_xml=PANDA)
    look_from(model, data, solve_inspection(model, data, row)[0])
    try:
        cam = WristCamera(model, noise=noise, rng=np.random.default_rng(seed))
    except Exception as e:                       # no OpenGL here (e.g. headless without EGL)
        pytest.skip(f"can't render: {e}")
    with RGBDPerception(model, row, camera=cam) as perception:
        est = perception.estimate(data)
        labels = perception.last_labels
    truth = GroundTruthPerception(model, row).estimate(data)
    by_geom = {model.geom(b.name + "_geom").id: t for b, t in zip(row.books, truth.books)}
    return est, truth, [by_geom.get(g) for g in labels], row


@pytest.fixture(scope="module", params=[("leaning", 0), ("collapsed", 1), ("tight", 0)])
def perfect(request):
    return camera_and_truth(*request.param)


@pytest.fixture(scope="module")
def noisy():
    return camera_and_truth("leaning", 1, noise=DepthNoise())


def test_fit_spine_recovers_a_leaning_rectangle():
    """Points on a 30 x 250 mm spine leaning 20 deg, facing out of the shelf."""
    lean = math.radians(20)
    across, up = np.array([math.cos(lean), 0, -math.sin(lean)]), np.array([math.sin(lean), 0, math.cos(lean)])
    a, b = np.meshgrid(np.linspace(0, 0.03, 40), np.linspace(0, 0.25, 300))
    p = np.array([0.1, 0.1, 0.5]) + a.reshape(-1, 1) * across + b.reshape(-1, 1) * up
    normals = np.tile([0.0, 1.0, 0.0], (len(p), 1))
    corners = fit_spine(p, normals, pixel_size=0.001, grow_px=0)
    book = book_from_spine(corners, -1, 0.17, 1.0)
    assert book.thickness == pytest.approx(0.03, abs=1e-4)
    assert book.height == pytest.approx(0.25, abs=1e-4)
    assert book.lean == pytest.approx(lean, abs=1e-3)


def test_gap_is_unknown_without_a_neighbour():
    left = book_from_spine([[-0.05, 0, 0], [-0.02, 0, 0], [-0.02, 0, 0.2], [-0.05, 0, 0.2]], -2, 0.17, 1.0)
    right = book_from_spine([[0.02, 0, 0], [0.05, 0, 0], [0.05, 0, 0.2], [0.02, 0, 0.2]], +1, 0.17, 1.0)
    w = gap_profile([left, right], (-0.5, 0.5), 0.0, [0.1])
    assert math.isnan(w[0])                                  # -1 not seen: don't measure to -2


def test_books_are_named_as_in_the_truth(perfect):
    est, _, truth, _ = perfect
    assert all(t is not None for t in truth), "a segment that isn't one of the row's books"
    assert [b.slot_offset for b in est.books] == [t.slot_offset for t in truth]


def test_neighbours_match_the_truth(perfect):
    est, _, truth, _ = perfect
    for b, t in zip(est.books, truth):
        if abs(t.slot_offset) > 3:
            continue
        corner_error = np.linalg.norm(b.spine_corners - t.spine_corners, axis=1).max()
        limit = 3 * MM if abs(t.slot_offset) == 1 else 5 * MM
        assert corner_error < limit, f"book {t.slot_offset:+d}: corners {corner_error / MM:.1f} mm off"
        assert abs(b.lean - t.lean) < math.radians(1)
        assert abs(b.thickness - t.thickness) < (3 if abs(t.slot_offset) == 1 else 5) * MM
        assert b.depth == t.depth                            # the same prior, not a measurement
    assert est.book(-1).visible == est.book(+1).visible == 1.0


def test_gap_matches_the_truth(perfect):
    est, truth, _, row = perfect
    both = ~np.isnan(est.gap_widths) & ~np.isnan(truth.gap_widths)
    assert both[:3].all()                                    # the gap near the board is seen
    np.testing.assert_allclose(est.gap_widths[both], truth.gap_widths[both], atol=4 * MM)
    h = row.target_book.height
    assert est.min_gap_width(h) == pytest.approx(truth.min_gap_width(h), abs=3 * MM)


def test_noisy_depth_stays_close(noisy):
    est, truth, truth_books, row = noisy
    assert [b.slot_offset for b in est.books] == [t.slot_offset for t in truth_books]
    for b, t in zip(est.books, truth_books):
        if abs(t.slot_offset) == 1:
            assert np.linalg.norm(b.spine_corners - t.spine_corners, axis=1).max() < 8 * MM
            assert abs(b.lean - t.lean) < math.radians(1.5)
    h = row.target_book.height
    assert est.min_gap_width(h) == pytest.approx(truth.min_gap_width(h), abs=6 * MM)
