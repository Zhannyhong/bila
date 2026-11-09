"""Checks on the packed scenario, the Panda's thin fingers and articulated
books.

    uv run pytest tests/test_scenes.py
"""

import os

import mujoco
import numpy as np
import pytest

from bila.sim import shelf_scene
from bila.vision.perception import GroundTruthPerception
from bila.sim.shelf_scene import (FINGER_HALF_W, FINGER_LEN, ShelfConfig, book_children, build_scene, make_episode,
                         sample_row)

MM = 0.001
from bila.control.insert_book import DEFAULT_PANDA as PANDA  # noqa: E402


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


def test_articulated_book_keeps_its_box():
    """ShelfConfig(book_pages=N) leaves <book>_geom the book's box, unchanged
    and non-colliding, for the controllers and perception that reconstruct a
    book from it; the pages become sections that carry the collision and the
    mass instead."""
    row = sample_row(ShelfConfig(book_pages=4), "clear", np.random.default_rng(0))
    model = mujoco.MjModel.from_xml_string(build_scene(ShelfConfig(book_pages=4), row))
    b = row.books[0]
    env = model.geom(b.name + "_geom")
    box = [b.thickness / 2, b.depth / 2, b.height / 2]
    assert model.geom_size[env.id].tolist() == pytest.approx(box, rel=1e-5)   # the XML keeps 6 digits
    assert model.geom_contype[env.id] == 0 and model.geom_conaffinity[env.id] == 0

    sections = book_children(b)
    assert sections == [p.name for p in b.parts[::-1]]              # back cover first: the welded one
    mass = 0.0
    for name in sections:
        g = model.geom(name + "_geom")
        assert model.geom_contype[g.id] != 0
        assert model.geom_size[g.id].tolist() == pytest.approx([box[0] / 4, box[1], box[2]], rel=1e-5)  # 4 sections
        mass += model.body_mass[model.body(model.geom_bodyid[g.id]).id]
    assert mass == pytest.approx(b.mass, rel=1e-4)                  # the row still weighs what it did


def test_articulated_pages_fan_from_the_spine():
    """Cutting a book into sections adds one hinge per section after the welded
    back one, about the spine's vertical line: closed, the sections make up the
    book's box; opened, their fore edges swing out of the shelf."""
    row = sample_row(ShelfConfig(book_pages=4), "clear", np.random.default_rng(0))
    model = mujoco.MjModel.from_xml_string(build_scene(ShelfConfig(book_pages=4), row))
    data = mujoco.MjData(model)
    b = row.books[0]
    sections, env = book_children(b), model.geom(b.name + "_geom")
    hinges = [model.joint(name + "_joint") for name in sections[1:]]
    assert all(model.jnt_type[h.id] == mujoco.mjtJoint.mjJNT_HINGE for h in hinges)

    c, s = np.cos(row.shelf_yaw), np.sin(row.shelf_yaw)
    along, depth = np.array([c, s, 0.0]), np.array([-s, c, 0.0])     # the shelf's axes in the world

    def offsets():                                                   # section centres, off the book's box
        mujoco.mj_forward(model, data)
        return np.array([data.geom_xpos[model.geom(name + "_geom").id] - data.geom_xpos[env.id]
                         for name in sections])

    mujoco.mj_resetDataKeyframe(model, data, model.key("init").id)   # the keyframe's hinges, all closed
    closed = offsets()
    assert closed @ depth == pytest.approx(0.0, abs=1e-9)            # every one on the spine's plane
    span = closed @ along
    assert span.max() - span.min() == pytest.approx(3 * b.thickness / 4, rel=1e-5)

    for h in hinges:
        data.qpos[model.jnt_qposadr[h.id]] = -0.6
    assert (offsets() @ depth).max() > b.depth / 4                   # fore edges out of the shelf


def test_articulated_books_do_not_shift_the_keyframes_qpos():
    """Every joint of every book is in the 'init' keyframe's qpos: the books
    stand upright on their boards and not at the origin, where a qpos that ran
    out of values (a page hinge left out) would drop them."""
    cfg = ShelfConfig(book_pages=4, static_other_levels=False)       # every book free, so every book joint appears
    row = sample_row(cfg, "clear", np.random.default_rng(0))
    model = mujoco.MjModel.from_xml_string(build_scene(cfg, row))
    qpos = model.key("init").qpos
    books = row.books + [bk for fl in row.filler_levels for bk in fl.books]
    for bk in books:
        adr = model.jnt_qposadr[model.joint(bk.name + "_joint").id]
        assert qpos[adr + 2] > 0.05                                  # above the plinth
        assert shelf_scene.book_tilt_deg(qpos[adr + 3:adr + 7]) == pytest.approx(0.0, abs=1e-6)
