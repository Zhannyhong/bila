"""Checks of the oracle's book masks and of make_vision_dataset's labels.

    MUJOCO_GL=egl uv run pytest tests/test_vision_dataset.py      # in WSL
"""

import json

import cv2
import mujoco
import numpy as np
import pytest

import make_vision_dataset as mvd
from camera import WristCamera, project
from rgbd_perception import OracleSegmenter
from shelf_scene import ShelfConfig, book_assets, build_scene, sample_row

PANDA = "mujoco_menagerie/franka_emika_panda/panda.xml"


def scene(cfg, scenario="tight", seed=0):
    """A model at its keyframe (not settled: these checks are about drawing, not physics)."""
    row = sample_row(cfg, scenario, np.random.default_rng(seed))
    model = mujoco.MjModel.from_xml_string(build_scene(cfg, row, PANDA))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    return model, data, row


def capture(model, data, name):
    try:
        cam = WristCamera(model, name=name)
    except Exception as e:                       # no OpenGL here
        pytest.skip(f"can't render: {e}")
    with cam:
        return cam.capture(data)


def test_oracle_unchanged_on_box_books():
    """Plain boxes: the same masks and labels as the oracle had before (one geom per book)."""
    model, data, _ = scene(ShelfConfig(), "leaning", 3)
    frame = capture(model, data, "scene_cam")
    before = [g for g in range(model.ngeom) if "book" in (model.geom(g).name or "")]
    before = [(int(g), frame.seg == g) for g in np.intersect1d(np.unique(frame.seg), before)]
    after = [(s.label, s.mask) for s in OracleSegmenter(model)(frame)]
    assert len(after) == len(before) > 10
    assert all(a[0] == b[0] and np.array_equal(a[1], b[1]) for a, b in zip(after, before))


@pytest.mark.skipif(not book_assets(), reason="needs dae_to_obj.py's book meshes")
def test_oracle_one_mask_per_book_with_covers_and_pages():
    model, data, row = scene(ShelfConfig(book_meshes=True, book_pages=4), "packed", 2)
    frame = capture(model, data, "scene_cam")
    segs = OracleSegmenter(model)(frame)
    names = [model.geom(s.label).name for s in segs]
    books = {b.name for b in row.books} | {b.name for fl in row.filler_levels for b in fl.books}
    assert len(set(names)) == len(names) > 10
    assert {n[:-len("_geom")] for n in names} <= books
    # Every drawn pixel of a book (cover faces, page sections) is in its book's mask.
    drawn = np.isin(frame.seg, [g for g in range(model.ngeom)
                                if model.body(model.body_rootid[model.geom_bodyid[g]]).name in books])
    assert np.array_equal(drawn, np.any([s.mask for s in segs], axis=0))


@pytest.fixture(scope="module")
def cart_frame():
    import cart_demo
    from inspection import look_from
    cfg, row, tray, _ = cart_demo.sample("tight", 0)
    numbers = mvd.cart_numbers(row, tray, 0)
    model, tray = cart_demo.build(cfg, row, tray, numbers)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    look_from(model, data, cart_demo.SCAN_POSE)
    return model, data, row, capture(model, data, "row_cam")


def test_stickers_are_part_of_their_book(cart_frame):
    model, _, row, frame = cart_frame
    segs = OracleSegmenter(model)(frame)
    assert not any(model.geom(s.label).name.endswith("_label") for s in segs)
    by_label = {s.label: s.mask for s in segs}
    seen = 0
    for b in row.books:
        sticker = frame.seg == model.geom(b.name + "_label").id
        if sticker.any():
            seen += 1
            assert by_label[model.geom(b.name + "_geom").id][sticker].all()
    assert seen > 5


def test_sticker_corners_in_reading_order(cart_frame):
    """Top-left is left of top-right and above bottom-left in the image, and
    the corners outline the sticker's pixels."""
    model, data, row, frame = cart_frame
    checked = 0
    for b in row.books:
        g = model.geom(b.name + "_label").id
        v, u = np.nonzero(frame.seg == g)
        if len(u) < 50:
            continue
        pu, pv, _ = project(frame, mvd.sticker_corners(model, data, g))
        assert pu[0] < pu[1] and pu[3] < pu[2]          # left before right
        assert pv[0] < pv[3] and pv[1] < pv[2]          # top above bottom
        assert pu.min() - 2 <= u.min() and u.max() <= pu.max() + 2
        assert pv.min() - 2 <= v.min() and v.max() <= pv.max() + 2
        checked += 1
    assert checked > 5


def test_make_scene_writes_matching_files(tmp_path, monkeypatch):
    """One shelf scene, one view: the mask's values match the label's book list."""
    import insert_book
    monkeypatch.setattr(insert_book, "SWEEP", (0.0,))
    monkeypatch.setattr(mvd, "SHELF_VIEWS", 0)
    for sub in ("rgb", "depth", "masks", "labels", "scenes"):
        (tmp_path / sub).mkdir()
    seed = next(s for s in range(2000, 2100) if mvd.scene_kind(s) == "shelf")
    try:
        records = mvd.make_scene((str(tmp_path), seed))
    except Exception as e:
        if "EGL" in str(e) or "OpenGL" in str(e):
            pytest.skip(f"can't render: {e}")
        raise
    assert len(records) == 1
    stem = records[0]["image"]
    label = json.loads((tmp_path / "labels" / f"{stem}.json").read_text())
    mask = cv2.imread(str(tmp_path / "masks" / f"{stem}.png"), cv2.IMREAD_UNCHANGED)
    depth = cv2.imread(str(tmp_path / "depth" / f"{stem}.png"), cv2.IMREAD_UNCHANGED)
    rgb = cv2.imread(str(tmp_path / "rgb" / f"{stem}.png"))
    assert mask.dtype == np.uint16 and depth.dtype == np.uint16 and rgb.shape == mask.shape + (3,)
    assert mask.max() == len(label["books"]) > 10
    for k, b in enumerate(label["books"], 1):
        assert (mask == k).sum() == b["pixels"]
    assert any(b["where"] == "target_row" for b in label["books"])
    metres = depth[depth > 0] * label["depth_unit"]
    assert 0.05 < np.median(metres) < 1.5
    assert json.loads((tmp_path / "scenes" / f"{seed:05d}.json").read_text())["images"] == 1
