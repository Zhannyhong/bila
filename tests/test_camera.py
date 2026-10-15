"""Checks that camera.py's images, camera model and 3D points match the scene.

    MUJOCO_GL=egl uv run pytest tests/test_camera.py      # in WSL
"""

import math

import mujoco
import numpy as np
import pytest

from camera import DepthNoise, WristCamera, intrinsics, pixels, points, project, robot_mask
from inspection import look_from, solve_inspection
from shelf_scene import ShelfConfig, make_episode

PANDA = "mujoco_menagerie/franka_emika_panda/panda.xml"


@pytest.fixture(scope="module")
def shot():
    """The wrist camera's image of a leaning row, from the inspection pose."""
    model, data, row, _ = make_episode(ShelfConfig(), "leaning", np.random.default_rng(0), panda_xml=PANDA)
    q, _ = solve_inspection(model, data, row)
    look_from(model, data, q)
    try:
        cam = WristCamera(model)
    except Exception as e:                       # no OpenGL here (e.g. headless without EGL)
        pytest.skip(f"can't render: {e}")
    with cam:
        frame = cam.capture(data)
    return model, data, row, frame


def erode(mask, k=2):
    """Drop pixels within k of the mask's edge (they can mix two surfaces)."""
    out = mask.copy()
    for _ in range(k):
        out[1:] &= out[:-1].copy()
        out[:-1] &= out[1:].copy()
        out[:, 1:] &= out[:, :-1].copy()
        out[:, :-1] &= out[:, 1:].copy()
    return out


def distance_to_box(model, data, geom, p):
    """Distance from points p (N, 3) to the surface of a box geom."""
    local = (p - data.geom_xpos[geom]) @ data.geom_xmat[geom].reshape(3, 3)
    q = np.abs(local) - model.geom_size[geom]
    outside = np.linalg.norm(np.maximum(q, 0), axis=1)
    inside = np.minimum(q.max(axis=1), 0)
    return np.abs(outside + inside)


def test_intrinsics_match_the_field_of_view():
    K = intrinsics(58.0, 1280, 720)
    assert K[0, 0] == K[1, 1] == pytest.approx(360 / math.tan(math.radians(29)))
    assert (K[0, 2], K[1, 2]) == (639.5, 359.5)
    assert 2 * math.degrees(math.atan(640 / K[0, 0])) == pytest.approx(89.1, abs=0.1)   # like a D405 (87)


def test_image_is_upright(shot):
    model, data, row, frame = shot
    assert frame.pose[2, 1] < 0                                  # image "down" points down in the world
    board = model.geom(f"board_{row.target_level}").id
    rows = np.nonzero(frame.seg == board)[0]
    assert rows.mean() > frame.size[1] / 2                       # the board the camera looks along is below the middle


def test_points_lie_on_the_books(shot):
    """Every pixel of a book, turned into a 3D point, is on that book's box:
    checks the intrinsics, the pose, the 180 deg turn and the depth together."""
    model, data, row, frame = shot
    for b in row.books[max(0, row.target_index - 3):row.target_index + 3]:
        g = model.geom(b.name + "_geom").id
        mask = erode(frame.seg == g)
        if mask.sum() < 50:
            continue
        d = distance_to_box(model, data, g, points(frame, mask))
        assert np.percentile(d, 99) < 0.002, f"{b.name}: 99% of its points within {1000 * np.percentile(d, 99):.1f} mm"


def test_project_inverts_points(shot):
    _, _, _, frame = shot
    uv = pixels(frame)
    u, v, z = project(frame, points(frame))
    np.testing.assert_allclose(np.column_stack([u, v]), uv, atol=1e-6)
    np.testing.assert_allclose(z, frame.depth[uv[:, 1], uv[:, 0]], atol=1e-6)


def test_robot_mask_and_no_reading_beyond_range(shot):
    model, _, row, frame = shot
    robot = robot_mask(model, frame)
    assert robot.mean() < 0.005                                  # at most a strip at the image edge
    assert np.all(frame.depth[frame.seg == -1] == 0)             # background: no reading
    books = [model.geom(b.name + "_geom").id for b in row.books]
    assert np.all(frame.depth[np.isin(frame.seg, books)] > 0)    # every book pixel has one


def test_depth_noise_grows_with_distance_and_drops_edges():
    depth = np.full((200, 200), 0.5, dtype=np.float32)
    depth[:, 100:] = 0.8                                         # a step: an object edge
    depth[:10, :10] = 0.03                                       # too close for the sensor
    noise = DepthNoise()
    a = noise.apply(depth, np.random.default_rng(0))
    b = noise.apply(depth, np.random.default_rng(0))
    np.testing.assert_array_equal(a, b)
    expected = lambda z: z ** 2 * noise.subpixel / (noise.focal_px * noise.baseline)
    near, far = a[20:, 20:90], a[20:, 110:]
    assert np.std(near) == pytest.approx(expected(0.5), rel=0.1)  # ~1.7 mm at 50 cm
    assert np.std(far) == pytest.approx(expected(0.8), rel=0.1)
    assert np.all(a[:, 99:101] == 0)                             # no reading at the edge
    assert np.all(a[:10, :10] == 0)                              # nor too close


def test_plot_frame_saves_a_picture(shot, tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    from perception import GroundTruthPerception
    from visualize import plot_frame
    model, data, row, frame = shot
    path = tmp_path / "camera.png"
    plot_frame(frame, GroundTruthPerception(model, row).estimate(data), model, str(path))
    assert path.stat().st_size > 10_000
