"""Checks that the camera-based ShelfEstimate matches the ground truth.

    MUJOCO_GL=egl uv run pytest tests/test_rgbd_perception.py      # in WSL
"""

import math

import numpy as np
import pytest

from bila.vision.camera import DepthNoise, WristCamera
from bila.control.inspection import look_from, solve_inspection
from bila.vision.perception import GroundTruthPerception, book_from_spine, gap_profile
from bila.vision.rgbd_perception import RGBDPerception, fit_spine
from bila.sim.shelf_scene import ShelfConfig, make_episode

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


def test_robust_fit_ignores_stray_points_on_the_next_spine():
    """The leaning spine above plus a 4 x 4 mm patch of points 10 mm past its
    top right corner (a learned mask bleeding onto the next book): the plain
    fit stretches to reach it, the robust one doesn't, and with no stray
    points both agree."""
    lean = math.radians(20)
    across, up = np.array([math.cos(lean), 0, -math.sin(lean)]), np.array([math.sin(lean), 0, math.cos(lean)])
    a, b = np.meshgrid(np.linspace(0, 0.03, 40), np.linspace(0, 0.25, 300))
    origin = np.array([0.1, 0.1, 0.5])
    spine = origin + a.reshape(-1, 1) * across + b.reshape(-1, 1) * up
    c, d = np.meshgrid(np.linspace(0.04, 0.044, 5), np.linspace(0.20, 0.204, 5))
    stray = origin + c.reshape(-1, 1) * across + d.reshape(-1, 1) * up
    p = np.vstack([spine, stray])
    normals = np.tile([0.0, 1.0, 0.0], (len(p), 1))
    plain = book_from_spine(fit_spine(p, normals, pixel_size=0.001, grow_px=0), -1, 0.17, 1.0)
    robust = book_from_spine(fit_spine(p, normals, pixel_size=0.001, grow_px=0, robust=True), -1, 0.17, 1.0)
    assert plain.thickness > 0.04
    assert robust.thickness == pytest.approx(0.03, abs=1e-3)
    assert robust.lean == pytest.approx(lean, abs=2e-3)
    clean = fit_spine(spine, normals[:len(spine)], pixel_size=0.001, grow_px=0, robust=True)
    assert np.allclose(clean, fit_spine(spine, normals[:len(spine)], pixel_size=0.001, grow_px=0), atol=1e-4)


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


@pytest.fixture(scope="module")
def sweep():
    """Images from 30 cm either side of the slot and in front of it, of a row
    where the right-hand view sees a book's side past its neighbour (clear 4),
    with the target book in the model (parked off the shelf)."""
    from bila.control import insert_book
    model, data, row = insert_book.make_episode(ShelfConfig(), "clear", np.random.default_rng(4), PANDA)
    try:
        cam = WristCamera(model, noise=DepthNoise(), rng=np.random.default_rng(4))
    except Exception as e:
        pytest.skip(f"can't render: {e}")
    perception = RGBDPerception(model, row, camera=cam)
    frames, centre = [], None
    for dx in (-0.3, 0.0, 0.3):
        q = solve_inspection(model, data, row, dx)[0]
        centre = q if dx == 0 else centre
        look_from(model, data, q)
        frames.append(perception.capture(data))
    look_from(model, data, centre)
    yield model, data, row, perception, frames
    perception.close()


def test_sweep_sees_the_whole_row_and_names_it(sweep):
    model, data, row, perception, frames = sweep
    est = perception.estimate_from_frames(frames)
    truth = GroundTruthPerception(model, row).estimate(data)
    by_geom = {model.geom(b.name + "_geom").id: t for b, t in zip(row.books, truth.books)}
    assert [b.slot_offset for b in est.books] == [by_geom[g].slot_offset for g in perception.last_labels]
    assert len(est.books) == len(row.books) and None not in est.panel_x
    far = [b.thickness - by_geom[g].thickness for b, g in zip(est.books, perception.last_labels) if abs(b.slot_offset) >= 4]
    assert np.median(far) < 2.5 * MM and np.max(np.abs(far)) < 8 * MM      # from one view: 4.6 mm too thick (median)


def test_a_book_put_into_the_slot_is_offset_0(sweep):
    import mujoco
    from bila.control.inspection import slot_x
    model, data, row, perception, frames = sweep
    b, x = row.target_book, slot_x(row)
    qa = model.jnt_qposadr[model.joint(b.name + "_joint").id]
    centre = perception.shelf.to_world([x, row.shelf_depth / 2 - 0.015 - b.depth / 2, row.target_board_top + b.height / 2])
    data.qpos[qa:qa + 7] = np.concatenate([centre, [math.cos(row.shelf_yaw / 2), 0, 0, math.sin(row.shelf_yaw / 2)]])
    mujoco.mj_forward(model, data)
    est = perception.estimate_from_frames([perception.capture(data)], placed_x=x)
    assert est.book(0) is not None and abs(est.book(0).thickness - b.thickness) < 3 * MM
    truth = GroundTruthPerception(model, row).estimate(data)
    for k in (-2, -1, 1, 2):
        assert np.linalg.norm(est.book(k).spine_corners - truth.book(k).spine_corners, axis=1).max() < 8 * MM


def test_noisy_depth_stays_close(noisy):
    est, truth, truth_books, row = noisy
    assert [b.slot_offset for b in est.books] == [t.slot_offset for t in truth_books]
    for b, t in zip(est.books, truth_books):
        if abs(t.slot_offset) == 1:
            assert np.linalg.norm(b.spine_corners - t.spine_corners, axis=1).max() < 8 * MM
            assert abs(b.lean - t.lean) < math.radians(1.5)
    h = row.target_book.height
    assert est.min_gap_width(h) == pytest.approx(truth.min_gap_width(h), abs=6 * MM)


def test_fiducials_are_found_and_correct_the_camera_pose():
    """Every marker the camera sees whole is found with its map id, and after
    the correction the chosen marker lands where the map has it."""
    from bila.control.insert_book import ArmIK, DEFAULT_PANDA, ShelfFrame, make_episode
    from bila.control.inspection import look_from, slot_x, solve_inspection
    from bila.sim.shelf_scene import PANDA_HOME, ShelfConfig, fiducials
    from bila.vision.camera import CalibrationError, WristCamera
    from bila.vision.fiducials import FiducialReference
    cfg = ShelfConfig()
    model, data, row = make_episode(cfg, "tight", np.random.default_rng(0), DEFAULT_PANDA)
    look_from(model, data, solve_inspection(model, data, row, 0.0, ArmIK(model, PANDA_HOME[:7]))[0])
    ref = FiducialReference(fiducials(cfg, row), ShelfFrame(row), slot_x(row))
    with WristCamera(model, rng=np.random.default_rng(1), calibration=CalibrationError(translation=0.004)) as cam:
        frame = cam.capture(data)
    shown = {int(model.geom(g).name.split("_")[1]) for g in np.unique(frame.seg)
             if g >= 0 and model.geom(g).name.startswith("fiducial_")}
    found = ref.detect(frame.rgb)
    assert len(found) >= 3 and set(found) <= shown
    before = ref.shift(frame, ref.markers[min(found, key=lambda i: abs(ref.markers[i].centre[0] - slot_x(row)))],
                       found[min(found, key=lambda i: abs(ref.markers[i].centre[0] - slot_x(row)))])
    fixed = ref(frame)
    i, _ = ref.last
    after = ref.shift(fixed, ref.markers[i], found[i])
    assert np.linalg.norm(before) > 0.002 and np.linalg.norm(after) < 0.0002
