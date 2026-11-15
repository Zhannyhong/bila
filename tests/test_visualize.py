"""Smoke tests for visualize.py: the pictures get made, and the viewer overlay
adds the expected geoms (no window or GL context needed)."""

import matplotlib
import mujoco
import numpy as np
import pytest

matplotlib.use("Agg")

from bila.sim.shelf_scene import ShelfConfig, make_episode
from bila.viz.visualize import book_box, draw_in_viewer, estimate_row, plot_estimate


@pytest.fixture(scope="module")
def leaning_scene():
    model, data, row, _ = make_episode(ShelfConfig(), "leaning", np.random.default_rng(0))
    return model, data, row


def test_book_box_sits_behind_the_spine(leaning_scene):
    model, data, row = leaning_scene
    est, _ = estimate_row(model, data, row)
    for b in est.books:
        centre, R, half = book_box(b)
        np.testing.assert_allclose(R.T @ R, np.eye(3), atol=1e-9)
        spine_centre = centre + R[:, 1] * half[1]                 # front face of the box
        np.testing.assert_allclose(spine_centre, b.spine_corners.mean(axis=0), atol=1e-9)
        assert R[1, 1] > 0.5                                     # out of the spine = toward the robot


def test_plot_estimate_saves_a_picture(leaning_scene, tmp_path):
    model, data, row = leaning_scene
    est, truth = estimate_row(model, data, row, noisy=True, seed=0)
    path = tmp_path / "estimate.png"
    plot_estimate(est, truth, (row.target_book.thickness, row.target_book.height), str(path))
    assert path.stat().st_size > 10_000


def test_draw_in_viewer_adds_books_corners_and_gap_bars(leaning_scene):
    model, data, row = leaning_scene
    est, _ = estimate_row(model, data, row)
    scn = mujoco.MjvScene(model, maxgeom=1000)
    draw_in_viewer(scn, est)
    bars = int(np.sum(est.gap_widths > 0))
    assert scn.ngeom == 5 * len(est.books) + bars                # box + 4 corners per book
    small = mujoco.MjvScene(model, maxgeom=10)
    draw_in_viewer(small, est)                                   # more than fits: stops at capacity
    assert small.ngeom == 10


def test_sticker_verdicts():
    """--learned --cart's colours: each reading right, wrong, unreadable or no
    sticker; stickers in clear view not found are missed (not tiny ones)."""
    pytest.importorskip("torch")
    from bila.vision.label_reader import Reading
    from bila.viz.visualize import sticker_verdicts

    def box(x, w=20, h=50, y=10):
        return np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], float)

    stickers = [{"corners": box(x).tolist(), "text": t, "seen": 1.0, "pixels": px}
                for x, t, px in ((0, "010.0\nAAA", 900), (40, "011.0\nBBB", 900), (80, "012.0\nCCC", 900),
                                 (120, "013.0\nDDD", 900), (160, "014.0\nEEE", 50))]
    read = lambda x, text: Reading(box(x + 1), text, 0.999, text or "")
    verdicts, missed = sticker_verdicts([read(0, "010.0\nAAA"), read(40, "019.0\nZZZ"), read(80, None),
                                         read(300, "015.0\nFFF")], stickers)
    assert [v for _, v, _ in verdicts] == ["right", "wrong", "unreadable", "no sticker"]
    assert [t for _, _, t in verdicts] == [0, 1, 2, None]
    assert missed == [3]                                   # 4 is too small to count as missed
