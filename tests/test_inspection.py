"""Checks that inspection.py puts the wrist camera where it should be, and that
the slot's neighbours are fully in its view.

    uv run pytest tests/test_inspection.py
"""

import math

import mujoco
import numpy as np
import pytest

from bila.control.insert_book import ShelfFrame
from bila.control.inspection import CAMERA, STANDOFFS, camera_target, look_from, solve_inspection
from bila.vision.perception import GroundTruthPerception
from bila.sim.shelf_scene import ShelfConfig, make_episode

PANDA = "mujoco_menagerie/franka_emika_panda/panda.xml"


@pytest.fixture(scope="module", params=[("leaning", 0), ("leaning", 1), ("collapsed", 1)])
def scene(request):
    scenario, seed = request.param
    model, data, row, _ = make_episode(ShelfConfig(), scenario, np.random.default_rng(seed), panda_xml=PANDA)
    view = solve_inspection(model, data, row)
    assert view is not None, "no arm pose reaches any inspection view"
    q, standoff = view
    look_from(model, data, q)
    return model, data, row, standoff


def test_camera_is_at_the_target(scene):
    model, data, row, standoff = scene
    assert standoff in STANDOFFS
    cam = model.camera(CAMERA).id
    pos, rot = camera_target(row, standoff=standoff)
    assert np.linalg.norm(data.cam_xpos[cam] - pos) < 1e-3
    R = data.cam_xmat[cam].reshape(3, 3)
    angle = math.degrees(math.acos(np.clip((np.trace(rot.T @ R) - 1) / 2, -1, 1)))
    assert angle < 0.5


def test_sideways_views_are_reachable(scene):
    model, data, row, _ = scene
    for dx in (-0.2, 0.2):
        assert solve_inspection(model, data, row, dx) is not None


W, H = 640, 360


def pixel(model, data, p):
    """Pixel (u, v) of world point p in the wrist camera's image, and its
    depth along the camera axis. MuJoCo cameras look along their -z, y up."""
    cam = model.camera(CAMERA).id
    pc = data.cam_xmat[cam].reshape(3, 3).T @ (p - data.cam_xpos[cam])
    f = (H / 2) / math.tan(math.radians(model.cam_fovy[cam]) / 2)
    return W / 2 + f * pc[0] / -pc[2], H / 2 - f * pc[1] / -pc[2], -pc[2]


def test_neighbour_spines_fully_seen_and_robot_out_of_the_way(scene):
    """Each neighbour's whole spine face is in the image and nothing is in
    front of it: points just inside its corners project into the image, and
    the depth image there shows the spine, not something nearer. Part of the
    arm may show at the image edge, but not over a neighbour's spine."""
    model, data, row, _ = scene
    try:
        renderer = mujoco.Renderer(model, H, W)
    except Exception as e:                       # no OpenGL here (e.g. headless without EGL)
        pytest.skip(f"can't render: {e}")
    renderer.enable_depth_rendering()
    renderer.update_scene(data, CAMERA)
    depth = renderer.render()
    renderer.disable_depth_rendering()
    renderer.enable_segmentation_rendering()
    renderer.update_scene(data, CAMERA)
    seg = renderer.render()
    renderer.close()
    ids = np.where(seg[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM), seg[..., 0], -1)
    robot_px = np.isin(ids, [g for g in range(model.ngeom)
                             if model.body(model.geom_bodyid[g]).name.startswith("panda/")])
    assert robot_px.mean() < 0.02, f"{robot_px.mean():.1%} of the image is robot"   # the 12 cm fingertips show at the top edge

    est = GroundTruthPerception(model, row).estimate(data)
    shelf = ShelfFrame(row)
    for offset in (-1, +1):
        c = est.book(offset).spine_corners
        us, vs = [], []
        for p in c + 0.3 * (c.mean(axis=0) - c):                 # well inside each corner (>= 3 px on thin spines)
            u, v, z = pixel(model, data, shelf.to_world(p))
            assert 0 <= u < W and 0 <= v < H, f"neighbour {offset:+d} is cut off by the image edge"
            assert abs(depth[int(v), int(u)] - z) < 0.005, f"something is in front of neighbour {offset:+d}"
            us.append(u)
            vs.append(v)
        box = robot_px[int(min(vs)):int(max(vs)) + 1, int(min(us)):int(max(us)) + 1]
        assert not box.any(), f"the robot covers neighbour {offset:+d}"
