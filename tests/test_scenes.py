"""Checks on the packed scenario and the Panda's thin fingers.

    uv run pytest tests/test_scenes.py
"""

import os

import mujoco
import numpy as np
import pytest

import shelf_scene
from perception import GroundTruthPerception
from shelf_scene import FINGER_HALF_W, FINGER_LEN, ShelfConfig, build_scene, make_episode, sample_row

MM = 0.001
PANDA = os.path.join(os.path.dirname(os.path.abspath(shelf_scene.__file__)),
                     "mujoco_menagerie", "franka_emika_panda", "panda.xml")


@pytest.mark.parametrize("seed", [0, 1])
def test_packed_slot_is_closed(seed):
    """The slot's neighbours touch: no gap anywhere up the shorter of them."""
    model, data, row, report = make_episode(ShelfConfig(), "packed", np.random.default_rng(seed))
    assert report["stable"]
    est = GroundTruthPerception(model, row).estimate(data)
    shorter = min(est.book(-1).height, est.book(1).height)
    assert est.min_gap_width(shorter - 0.02) < 1.5 * MM


def test_panda_has_thin_fingers():
    """Each finger collides as a bar and a pad, 10 mm wide across its travel,
    ending where Menagerie's fingers end; nothing is left of Menagerie's."""
    row = sample_row(ShelfConfig(), "clear", np.random.default_rng(0))
    model = mujoco.MjModel.from_xml_string(build_scene(ShelfConfig(), row, PANDA))
    for side in ("left_finger", "right_finger"):
        body = model.body("panda/" + side).id
        geoms = [g for g in range(model.ngeom) if model.geom_bodyid[g] == body and model.geom_contype[g]]
        assert len(geoms) == 2
        assert all(model.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX for g in geoms)
        assert all(model.geom_size[g][0] == pytest.approx(FINGER_HALF_W) for g in geoms)
        assert max(model.geom_pos[g][2] + model.geom_size[g][2] for g in geoms) == pytest.approx(FINGER_LEN)
