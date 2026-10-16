"""Checks on the packed scenario.

    uv run pytest tests/test_scenes.py
"""

import numpy as np
import pytest

from perception import GroundTruthPerception
from shelf_scene import ShelfConfig, make_episode

MM = 0.001


@pytest.mark.parametrize("seed", [0, 1])
def test_packed_slot_is_closed(seed):
    """The slot's neighbours touch: no gap anywhere up the shorter of them."""
    model, data, row, report = make_episode(ShelfConfig(), "packed", np.random.default_rng(seed))
    assert report["stable"]
    est = GroundTruthPerception(model, row).estimate(data)
    shorter = min(est.book(-1).height, est.book(1).height)
    assert est.min_gap_width(shorter - 0.02) < 1.5 * MM
