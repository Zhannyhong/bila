"""Checks the perception choices of insert_book (eyes_for) that eval_reshelve.py
runs the controller with, and how it names the way a row ended.

    uv run pytest tests/test_eval_reshelve.py
"""

import numpy as np
import pytest

from bila.control.eval_reshelve import outcome
from bila.control.insert_book import DEFAULT_PANDA, eyes_for, make_episode
from bila.vision.perception import CAMERA_ERRORS, LEARNED_CAMERA_ERRORS, NoisyEyes
from bila.sim.shelf_scene import ShelfConfig


@pytest.fixture(scope="module")
def scene():
    return make_episode(ShelfConfig(), "tight", np.random.default_rng(0), DEFAULT_PANDA)


def test_each_perception_gets_its_eyes(scene):
    model, data, row = scene
    assert eyes_for("truth", model, row, 0) is None
    noisy, learned = eyes_for("noisy", model, row, 0), eyes_for("noisy-learned", model, row, 0)
    assert isinstance(noisy, NoisyEyes) and isinstance(learned, NoisyEyes)
    assert noisy.perception.errors is CAMERA_ERRORS and learned.perception.errors is LEARNED_CAMERA_ERRORS
    cam = eyes_for("camera", model, row, 0, segmenter=lambda frame: [])          # (any segmenter: robust fit on)
    try:
        assert cam.robust_fit
    finally:
        cam.close()


def test_noise_scale_widens_the_noisy_eyes_only(scene):
    model, data, row = scene
    assert eyes_for("noisy", model, row, 0).perception.scale == 1.0
    assert eyes_for("noisy", model, row, 0, noise_scale=2.0).perception.scale == 2.0
    assert eyes_for("truth", model, row, 0, noise_scale=2.0) is None


def test_noisy_eyes_draw_as_gap_env_does(scene):
    """The same seed gives the same estimate (default_rng([seed, 1]), a scene's noise in gap_env)."""
    model, data, row = scene
    a, b = eyes_for("noisy-learned", model, row, 3), eyes_for("noisy-learned", model, row, 3)
    for e in (a, b):
        e.capture(data)
    ea, eb = a.estimate_from_frames([]), b.estimate_from_frames([])
    np.testing.assert_array_equal(ea.books[0].spine_corners, eb.books[0].spine_corners)


@pytest.mark.parametrize("row, kind", [
    ({"success": True}, "placed"),
    ({"success": False, "skipped": "the gap stayed narrower than the book"}, "gave up: the gap stayed narrower than the book"),
    ({"success": False, "error": "the book jammed at the slot's entrance"}, "stopped: the book jammed at the slot's entrance"),
    ({"success": False, "upright": False, "in_place": False}, "not upright"),
    ({"success": False, "released": False}, "not let go of"),
    ({"success": False, "in_place": False, "rule": "flush", "offset_m": 0.03}, "left proud"),
    ({"success": False, "in_place": False, "rule": "flush", "offset_m": -0.02}, "pushed too deep"),
    ({"success": False, "in_place": False, "rule": "inside", "offset_m": 0.004}, "left proud"),
    ({"success": False, "crash": "ValueError: x"}, "crash"),
])
def test_how_a_row_ended(row, kind):
    assert outcome(row) == kind
