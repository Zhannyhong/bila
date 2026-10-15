"""
Pictures of what the robot knows about the target row: a ShelfEstimate drawn
as a front and top view (matplotlib), or as see-through boxes over the scene in
the MuJoCo viewer. Both read only the ShelfEstimate, so they show any
perception source the same way: ground truth, noisy, and later the camera.
And what the wrist camera sees, with the estimate drawn over its image.

    uv run python visualize.py --scenario leaning --seed 0            # saves scenes/estimate_leaning_seed0.png
    uv run python visualize.py --scenario leaning --seed 0 --noisy    # noisy estimate, ground truth dashed
    uv run python visualize.py --scenario leaning --seed 0 --camera   # wrist camera image -> scenes/camera_*.png
    uv run python view_scene.py --scenario leaning --seed 0 --show-estimate   # the same in 3D, in the viewer

In the pictures
  - Books: the slot's neighbours (-1, +1) orange, the others blue, labelled with
    their place relative to the slot and, if they lean, the lean in degrees.
  - Gap: a bar across the slot at each height where it is measured. Green where
    the book to insert fits, red where it doesn't, grey above that book's top.
    A '?' where the gap wasn't seen.
  - Top view: each book's box as the robot models it, `depth` back from the
    spine. The depth is a prior, so the backs are a guess, not a measurement.
"""

from __future__ import annotations

import argparse
import math
import os
from typing import Optional, Tuple

import mujoco
import numpy as np

from camera import CameraFrame, WristCamera, project, robot_mask
from perception import BookEstimate, GroundTruthPerception, NoisyPerception, ShelfEstimate, gap_edges
from shelf_scene import SCENARIOS, ShelfConfig, make_episode

HERE = os.path.dirname(os.path.abspath(__file__))

NEIGHBOUR_RGBA = (0.91, 0.51, 0.23, 1.0)    # orange
OTHER_RGBA = (0.30, 0.47, 0.66, 1.0)        # blue
FITS_RGBA = (0.17, 0.63, 0.17, 1.0)         # green
TOO_NARROW_RGBA = (0.84, 0.15, 0.16, 1.0)   # red
UNUSED_RGBA = (0.6, 0.6, 0.6, 1.0)          # grey


def book_rgba(b: BookEstimate):
    return NEIGHBOUR_RGBA if abs(b.slot_offset) == 1 else OTHER_RGBA


def gap_rgba(h: float, width: float, target: Optional[Tuple[float, float]]):
    """Colour of the gap bar at height h, for a book to insert of size
    target = (thickness, height)."""
    if target is None or h > target[1]:
        return UNUSED_RGBA
    return FITS_RGBA if width >= target[0] else TOO_NARROW_RGBA


def book_box(b: BookEstimate):
    """The book's box as the robot models it, in the shelf frame: centre,
    rotation (columns: across the spine, out of the spine, up) and half-sizes.
    The box runs `depth` back from the spine face."""
    c = b.spine_corners
    across = (c[1] - c[0]) + (c[2] - c[3])
    up = (c[3] - c[0]) + (c[2] - c[1])
    x = across / np.linalg.norm(across)
    z = up - (up @ x) * x                     # square up the axes: noisy corners aren't a perfect rectangle
    z /= np.linalg.norm(z)
    y = np.cross(z, x)                        # out of the spine, toward the robot
    centre = c.mean(axis=0) - y * b.depth / 2
    return centre, np.column_stack([x, y, z]), np.array([b.thickness, b.depth, b.height]) / 2


def gap_bars(est: ShelfEstimate):
    """(height above the board, left x, width) of the gap at each measured
    height; left and width are NaN where the gap wasn't seen."""
    left, _ = gap_edges(est.books, est.panel_x, est.board_top, est.gap_heights)
    return list(zip(est.gap_heights, left, est.gap_widths))


# ---------------------------------------------------------------------------
# 1. Front and top view (matplotlib)
# ---------------------------------------------------------------------------
def plot_estimate(est: ShelfEstimate, truth: Optional[ShelfEstimate] = None,
                  target: Optional[Tuple[float, float]] = None, path: Optional[str] = None, span: int = 6):
    """Front view (x along the row, height above the board) and top view (x,
    depth) of the estimate, in cm. `truth`, if given, is drawn as dashed
    outlines. `target` = (thickness, height) of the book to insert colours the
    gap. Shows the slot and `span` books each side. Saves to `path`, or opens
    a window."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon, Rectangle

    cm = 100.0
    front_z = (-1.0, est.clear_height * cm + 1)                                  # plotted ranges, cm
    top_y = (-est.shelf_depth / 2 * cm - 1, est.front_y * cm + 3)
    # Panels as tall as their ranges, so at equal scale both views get the same width and line up in x.
    fig, (front, top) = plt.subplots(2, 1, figsize=(11, 8), gridspec_kw={
        "height_ratios": [front_z[1] - front_z[0], top_y[1] - top_y[0]]})

    def front_xy(corners):
        return np.column_stack([corners[:, 0], corners[:, 2] - est.board_top]) * cm

    def top_xy(b):
        centre, R, half = book_box(b)
        quad = [centre + R @ [sx * half[0], sy * half[1], 0] for sx, sy in ((-1, 1), (1, 1), (1, -1), (-1, -1))]
        return np.array(quad)[:, :2] * cm

    # Books: estimate filled, truth dashed
    for b in est.books:
        rgba = book_rgba(b)
        front.add_patch(Polygon(front_xy(b.spine_corners), closed=True, fc=rgba[:3] + (0.45,), ec=rgba, lw=1))
        top.add_patch(Polygon(top_xy(b), closed=True, fc=rgba[:3] + (0.3,), ec=rgba, lw=1))
        xz = front_xy(b.spine_corners)
        front.text(xz[2:, 0].mean(), xz[:, 1].max() + 0.5, f"{b.slot_offset:+d}",     # above the book's top
                   ha="center", va="bottom", fontsize=8, clip_on=True)
        if abs(b.lean) > math.radians(2):
            x, z = xz.mean(axis=0)
            front.text(x, z, f"{math.degrees(b.lean):+.0f}°", ha="center", va="center", fontsize=8, clip_on=True)
    if truth is not None:
        for b in truth.books:
            front.add_patch(Polygon(front_xy(b.spine_corners), closed=True, fill=False, ec="k", ls="--", lw=0.8))
            top.add_patch(Polygon(top_xy(b), closed=True, fill=False, ec="k", ls="--", lw=0.8))

    # Gap bars, with their width in mm
    bars = gap_bars(est)
    seen = [l + w / 2 for _, l, w in bars if not math.isnan(l)]
    x_slot = np.mean(seen) if seen else 0.0
    for h, l, w in bars:
        if math.isnan(l) or math.isnan(w):
            front.text(x_slot * cm, h * cm, "?", ha="center", va="center", color="r", fontsize=10)
            continue
        front.plot([l * cm, (l + w) * cm], [h * cm, h * cm], color=gap_rgba(h, w, target), lw=3, solid_capstyle="butt")
        front.text((l + w) * cm + 0.3, h * cm, f"{1000 * w:.0f}", va="center", fontsize=7, clip_on=True)

    # Shelf: board, board above, side panels, front edge and back panel
    for ax, (lo, hi) in ((front, (0, est.clear_height)), (top, (-est.shelf_depth / 2, est.front_y))):
        for px, side in zip(est.panel_x, (-1, 1)):
            if px is not None:
                ax.add_patch(Rectangle(((px if side > 0 else px - 0.02) * cm, lo * cm), 2, (hi - lo) * cm, color="0.75"))
    front.axhline(0, color="0.4", lw=2)
    front.axhline(est.clear_height * cm, color="0.4", lw=2, ls=":")
    top.axhline(est.front_y * cm, color="0.4", lw=1, ls=":")
    top.axhline(-est.shelf_depth / 2 * cm, color="0.4", lw=2)

    # Frame the slot and `span` books each side
    near = [b for b in est.books if abs(b.slot_offset) <= span] or est.books
    xs = np.concatenate([b.spine_corners[:, 0] for b in near]) * cm
    for ax, lim in ((front, front_z), (top, top_y)):
        ax.set_xlim(xs.min() - 2, xs.max() + 2)
        ax.set_ylim(*lim)
        ax.set_aspect("equal")                            # true proportions
    front.set_ylabel("height above board (cm)")
    top.set_ylabel("depth, + out of shelf (cm)")
    top.set_xlabel("along the row (cm)")
    title = f"{est.source} estimate at t = {est.stamp:.2f} s"
    if target is not None:
        title += (f": narrowest gap up to {target[1] * cm:.0f} cm is {1000 * est.min_gap_width(target[1]):.0f} mm, "
                  f"book is {1000 * target[0]:.0f} mm")
    front.set_title(title + ("  (dashed: ground truth)" if truth is not None else ""), fontsize=10)
    top.set_title("top view: book depth is assumed, not measured", fontsize=9)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150)
        plt.close(fig)
    else:
        plt.show()


# ---------------------------------------------------------------------------
# 2. Overlay in the MuJoCo viewer
# ---------------------------------------------------------------------------
def draw_in_viewer(scn: mujoco.MjvScene, est: ShelfEstimate, target: Optional[Tuple[float, float]] = None):
    """Add the estimate to a scene's geoms: `viewer.user_scn` of a passive
    viewer (inside `viewer.lock()`), or `renderer.scene` after
    `update_scene`. Each book is a see-through box running `depth` back from
    its spine, with its spine corners as dots, and a bar crosses the slot at
    each gap height. Geoms past the scene's capacity are left out."""
    R_ws, p_ws = est.shelf_pose[:3, :3], est.shelf_pose[:3, 3]

    def new_geom():
        if scn.ngeom >= scn.maxgeom:
            return None
        scn.ngeom += 1
        return scn.geoms[scn.ngeom - 1]

    def add(kind, half, pos, R, rgba):
        g = new_geom()
        if g is not None:
            mujoco.mjv_initGeom(g, kind, np.asarray(half, dtype=float), R_ws @ pos + p_ws,
                                (R_ws @ R).flatten(), np.asarray(rgba, dtype=np.float32))

    for b in est.books:
        centre, R, half = book_box(b)
        add(mujoco.mjtGeom.mjGEOM_BOX, half, centre, R, book_rgba(b)[:3] + (0.35,))
        for c in b.spine_corners:
            add(mujoco.mjtGeom.mjGEOM_SPHERE, [0.003] * 3, c, np.eye(3), (0.1, 0.1, 0.1, 1))

    neighbours = [b.spine_y for b in est.books if abs(b.slot_offset) == 1]
    y = (max(neighbours) if neighbours else est.front_y) + 0.005      # just in front of the spines
    for h, l, w in gap_bars(est):
        if math.isnan(l) or math.isnan(w) or w <= 0:
            continue
        g = new_geom()
        if g is None:
            break
        z = est.board_top + h
        mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3), np.eye(3).flatten(),
                            np.asarray(gap_rgba(h, w, target), dtype=np.float32))
        mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, 0.002,
                             R_ws @ [l, y, z] + p_ws, R_ws @ [l + w, y, z] + p_ws)


# ---------------------------------------------------------------------------
# 3. The camera's image, with the estimate over it
# ---------------------------------------------------------------------------
def plot_frame(frame: CameraFrame, est: Optional[ShelfEstimate] = None, model: Optional[mujoco.MjModel] = None,
               path: Optional[str] = None, span: int = 6):
    """The camera's RGB image and depth image. Over the RGB: the estimate's
    spine outlines (the slot's neighbours orange, others blue, within `span`
    of the slot) and, given the model, the robot's own pixels in red. Where the
    outlines sit on the spines, the camera pose and model agree with the
    estimate. Saves to `path`, or opens a window."""
    import matplotlib.pyplot as plt

    fig, (ax_rgb, ax_depth) = plt.subplots(1, 2, figsize=(16, 4.8))
    ax_rgb.imshow(frame.rgb)
    if model is not None:
        robot = robot_mask(model, frame)
        ax_rgb.imshow(np.dstack([robot, np.zeros_like(robot), np.zeros_like(robot), 0.6 * robot]).astype(float))
    if est is not None:
        R, t = est.shelf_pose[:3, :3], est.shelf_pose[:3, 3]
        for b in est.books:
            if abs(b.slot_offset) > span:
                continue
            u, v, z = project(frame, b.spine_corners @ R.T + t)
            if np.all(z > 0):
                ax_rgb.fill(u, v, fill=False, ec=book_rgba(b), lw=1.5)
                ax_rgb.text(u.mean(), v.min() - 6, f"{b.slot_offset:+d}", color="w", fontsize=8, ha="center")
    W, H = frame.size
    ax_rgb.set_xlim(0, W)
    ax_rgb.set_ylim(H, 0)
    ax_rgb.set_title("wrist camera" + (": estimate outlined" if est is not None else ""), fontsize=10)
    shown = np.where(frame.depth > 0, frame.depth, np.nan)
    im = ax_depth.imshow(shown, cmap="viridis")
    fig.colorbar(im, ax=ax_depth, label="depth (m); blank = no reading")
    ax_depth.set_title("depth", fontsize=10)
    for ax in (ax_rgb, ax_depth):
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=110)
        plt.close(fig)
    else:
        plt.show()


# ---------------------------------------------------------------------------
# CLI: estimate a settled row and save the picture
# ---------------------------------------------------------------------------
def estimate_row(model, data, row, noisy: bool = False, seed: int = 0):
    """(estimate, ground truth to compare it with, or None if the estimate is
    the ground truth)."""
    gt = GroundTruthPerception(model, row)
    if not noisy:
        return gt.estimate(data), None
    return NoisyPerception(gt, np.random.default_rng(seed)).estimate(data), gt.estimate(data)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenario", choices=SCENARIOS, default="leaning")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--noisy", action="store_true", help="show NoisyPerception, with the ground truth dashed")
    ap.add_argument("--out", help="image path (default scenes/estimate_<scenario>_seed<seed>.png)")
    ap.add_argument("--show", action="store_true", help="open a window instead of saving")
    ap.add_argument("--camera", action="store_true",
                    help="the wrist camera's image from the inspection pose, with the estimate outlined")
    a = ap.parse_args()

    panda = os.path.join(HERE, "mujoco_menagerie", "franka_emika_panda", "panda.xml") if a.camera else None
    model, data, row, _ = make_episode(ShelfConfig(), a.scenario, np.random.default_rng(a.seed), panda_xml=panda)
    est, truth = estimate_row(model, data, row, a.noisy, a.seed)
    path = None
    if not a.show:
        kind = "camera" if a.camera else "estimate"
        path = a.out or os.path.join(HERE, "scenes", f"{kind}_{a.scenario}_seed{a.seed}.png")
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    if a.camera:
        from inspection import look_from, solve_inspection
        view = solve_inspection(model, data, row)
        if view is None:
            raise SystemExit("no arm pose reaches the inspection view in this scene")
        look_from(model, data, view[0])
        with WristCamera(model) as cam:
            plot_frame(cam.capture(data), est, model, path)
    else:
        plot_estimate(est, truth, (row.target_book.thickness, row.target_book.height), path)
    if path:
        print(f"saved {path}")


if __name__ == "__main__":
    main()
