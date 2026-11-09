"""
Pictures of what the robot knows about the target row: a ShelfEstimate drawn
as a front and top view (matplotlib), or as see-through boxes over the scene in
the MuJoCo viewer. Both read only the ShelfEstimate, so they show any
perception source the same way: ground truth, noisy, and later the camera.
And what the wrist camera sees, with the estimate drawn over its image.

    uv run python -m bila.viz.visualize --scenario leaning --seed 0            # saves scenes/estimate_leaning_seed0.png
    uv run python -m bila.viz.visualize --scenario leaning --seed 0 --noisy    # noisy estimate, ground truth dashed
    uv run python -m bila.viz.visualize --scenario leaning --seed 0 --camera   # wrist camera image -> scenes/camera_*.png
    uv run python -m bila.viz.view_scene --scenario leaning --seed 0 --show-estimate   # the same in 3D, in the viewer

What the trained models see (`uv sync --group rl`, weights in ~/bila_data; ~1-2 min, MUJOCO_GL=egl in WSL):

    uv run python -m bila.viz.visualize --scenario leaning --seed 3 --learned          # scenes/learned_*.png, estimate_learned_*.png
    uv run python -m bila.viz.visualize --scenario tight --seed 1 --learned --cart     # scenes/stickers_*.png

  --learned: the first look's three views with the learned segmenter's masks
    in colour, books it missed outlined red, masks that are no book magenta,
    each target-row book's mask IoU, and the fitted spines; then the front and
    top view of its estimate, the one from the simulation's masks dotted.
  --learned --cart: row_cam, tray_cam, a wrist close-up at the slot and a
    wrist view of the tray, each sticker the label reader found labelled with
    what it read and how sure it is (green right, red wrong, grey unreadable),
    stickers it missed dashed.

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

from bila.vision.camera import CameraFrame, WristCamera, project, robot_mask
from bila.vision.perception import BookEstimate, GroundTruthPerception, NoisyPerception, ShelfEstimate, book_box, gap_edges
from bila.sim.shelf_scene import SCENARIOS, ShelfConfig, make_episode

from bila import ROOT as HERE

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


def gap_bars(est: ShelfEstimate):
    """(height above the board, left x, width) of the gap at each measured
    height; left and width are NaN where the gap wasn't seen."""
    left, _ = gap_edges(est.books, est.panel_x, est.board_top, est.gap_heights)
    return list(zip(est.gap_heights, left, est.gap_widths))


# ---------------------------------------------------------------------------
# 1. Front and top view (matplotlib)
# ---------------------------------------------------------------------------
def plot_estimate(est: ShelfEstimate, truth: Optional[ShelfEstimate] = None,
                  target: Optional[Tuple[float, float]] = None, path: Optional[str] = None, span: int = 6,
                  compare: Optional[ShelfEstimate] = None, note: str = ""):
    """Front view (x along the row, height above the board) and top view (x,
    depth) of the estimate, in cm. `truth`, if given, is drawn as dashed
    outlines, and `compare` (another estimate) as dotted grey ones. `target` =
    (thickness, height) of the book to insert colours the gap. Shows the slot
    and `span` books each side; `note` is a second title line. Saves to
    `path`, or opens a window."""
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
    if compare is not None:
        for b in compare.books:
            front.add_patch(Polygon(front_xy(b.spine_corners), closed=True, fill=False, ec="0.45", ls=":", lw=1.2))
            top.add_patch(Polygon(top_xy(b), closed=True, fill=False, ec="0.45", ls=":", lw=1.2))

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
    front.set_title(title + ("  (dashed: ground truth)" if truth is not None else "") + (f"\n{note}" if note else ""),
                    fontsize=10)
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
# 4. What the learned models see (--learned)
# ---------------------------------------------------------------------------
WEIGHTS = "~/bila_data"            # segmenter/segmenter.pt, labels/finder.pt and labels/reader.pt (see README)
MISSED_RGBA = (0.95, 0.1, 0.1, 1.0)       # red: a book (or sticker) the model missed
EXTRA_RGBA = (0.9, 0.2, 0.9, 1.0)         # magenta: a mask (or sticker) that is no book
READ_RIGHT_RGBA = (0.1, 0.85, 0.2, 1.0)   # green
UNREADABLE_RGBA = (0.75, 0.75, 0.75, 1.0)  # grey
MIN_SHOWN_PX = 500                 # a missed book showing fewer pixels than this isn't marked (slivers at the edge)
MIN_STICKER_PX = 200               # ... nor a missed sticker (the bookcase's far in the background of the tray views)


def outline(ax, mask: np.ndarray, **kw):
    """Draw a mask's outer outline(s) on an image axis."""
    import cv2
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        c = c[:, 0]
        ax.plot(np.r_[c[:, 0], c[0, 0]], np.r_[c[:, 1], c[0, 1]], **kw)


def iou(a: np.ndarray, b: np.ndarray) -> float:
    return float((a & b).sum()) / max(float((a | b).sum()), 1.0)


def plot_learned_views(frames, learned, oracle, est: ShelfEstimate, target_labels, path: Optional[str] = None,
                       span: int = 6):
    """One image per view: the learned segmenter's masks in colour; outlined
    red, books it missed (the simulation's masks; only those showing at
    least MIN_SHOWN_PX); magenta, masks that are no book; on each target-row
    book, its mask's IoU with the true one; and the fitted spines (`est`),
    orange for the slot's neighbours, blue for the others. learned, oracle:
    per frame, the segments (learned ones labelled by OracleLabelled)."""
    import matplotlib.pyplot as plt
    colours = np.random.default_rng(0).uniform(0.2, 1.0, (256, 3))
    fig, axes = plt.subplots(len(frames), 1, figsize=(12, 6.9 * len(frames)), squeeze=False)
    R, t = est.shelf_pose[:3, :3], est.shelf_pose[:3, 3]
    for ax, frame, segs, truth in zip(axes[:, 0], frames, learned, oracle):
        shown = frame.rgb.astype(float) / 255.0
        for k, s in enumerate(segs):
            shown[s.mask] = 0.5 * shown[s.mask] + 0.5 * colours[k % len(colours)]
        ax.imshow(shown)
        found = {s.label for s in segs if s.label is not None}
        true_mask = {s.label: s.mask for s in truth}
        missed = 0
        for s in truth:
            if s.label not in found and s.mask.sum() >= MIN_SHOWN_PX:
                outline(ax, s.mask, color=MISSED_RGBA, lw=1.8)
                missed += 1
        for s in segs:
            if s.label is None:
                outline(ax, s.mask, color=EXTRA_RGBA, lw=1.8)
            elif s.label in target_labels:
                v, u = np.nonzero(s.mask)
                ax.text(u.mean(), v.max() + 4, f"{iou(s.mask, true_mask[s.label]):.2f}", color="w", fontsize=7,
                        ha="center", va="top", bbox=dict(fc="k", alpha=0.5, pad=1, lw=0))
        for b in est.books:
            if abs(b.slot_offset) > span:
                continue
            u, v, z = project(frame, b.spine_corners @ R.T + t)
            if np.all(z > 0):
                ax.fill(u, v, fill=False, ec=book_rgba(b), lw=1.2)
                ax.text(u.mean(), v.min() - 6, f"{b.slot_offset:+d}", color="w", fontsize=8, ha="center")
        W, H = frame.size
        ax.set_xlim(0, W)
        ax.set_ylim(H, 0)
        ax.set_xticks([])
        ax.set_yticks([])
        extra = sum(s.label is None for s in segs)
        ax.set_title(f"{len(segs)} learned masks; red: {missed} books missed; magenta: {extra} masks that are no "
                     f"book; numbers: target-row IoU; outlines: fitted spines", fontsize=9)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=100)
        plt.close(fig)
    else:
        plt.show()


def learned_shelf(scenario: str, seed: int, weights: str, path_for):
    """The first look's views with the learned segmenter, as eval_perception's
    learned kind takes them (realistic camera, robust spine fit): saves
    plot_learned_views, and the estimate's front and top view with the
    estimate from the simulation's masks (same images, same fit) dotted and
    the ground truth dashed."""
    from bila.vision import learned_segmenter as ls                         # imports torch._dynamo, before any render
    from bila.vision.camera import CalibrationError, DepthNoise
    from bila.control.insert_book import SWEEP
    from bila.control.inspection import look_from, solve_inspection
    from bila.vision.rgbd_perception import OracleSegmenter, RGBDPerception

    panda = os.path.join(HERE, "mujoco_menagerie", "franka_emika_panda", "panda.xml")
    model, data, row, _ = make_episode(ShelfConfig(), scenario, np.random.default_rng(seed), panda_xml=panda)
    views = [v for dx in SWEEP if (v := solve_inspection(model, data, row, dx))]
    if not views:
        raise SystemExit("no arm pose reaches the inspection view in this scene")
    camera = WristCamera(model, noise=DepthNoise.d405(), calibration=CalibrationError(), rng=np.random.default_rng(seed))
    frames = []
    for q, _ in views:
        look_from(model, data, q)
        frames.append(camera.capture(data))
    segmenter = ls.OracleLabelled(ls.LearnedSegmenter(os.path.join(weights, "segmenter", "segmenter.pt")), model)
    est = RGBDPerception(model, row, camera=camera, segmenter=segmenter, robust_fit=True).estimate_from_frames(frames)
    with_oracle = RGBDPerception(model, row, camera=camera, robust_fit=True).estimate_from_frames(frames)
    camera.close()
    truth = GroundTruthPerception(model, row).estimate(data)
    oracle = OracleSegmenter(model)
    plot_learned_views(frames, [segmenter(f) for f in frames], [oracle(f) for f in frames], est,
                       {model.geom(b.name + "_geom").id for b in row.books}, path_for("learned"))
    h = row.target_book.height
    gap = lambda e: 1000 * (e.min_gap_width(h) - truth.min_gap_width(h))
    note = (f"learned masks; dotted: the simulation's masks. Narrowest gap error {gap(est):+.1f} mm (dotted "
            f"{gap(with_oracle):+.1f}); books {len(est.books)} (dotted {len(with_oracle.books)}, true {len(truth.books)})")
    print(note)
    plot_estimate(est, truth, (row.target_book.thickness, h), path_for("estimate_learned"), compare=with_oracle, note=note)


def sticker_verdicts(readings, stickers):
    """([(reading, verdict, index of its true sticker or None)], indices of
    the stickers in clear view not found). Verdicts: "right", "wrong",
    "unreadable", or "no sticker" (found where there is none). A reading and
    a true sticker match if each one's centre is inside the other."""
    from bila.vision.label_reader import inside
    used, out = set(), []
    for r in readings:
        truth = next((i for i, s in enumerate(stickers) if i not in used and inside(r.centre, s["corners"])
                      and inside(np.mean(s["corners"], axis=0), r.corners)), None)
        if truth is None:
            out.append((r, "no sticker", None))
            continue
        used.add(truth)
        verdict = "unreadable" if r.text is None else "right" if r.text == stickers[truth]["text"] else "wrong"
        out.append((r, verdict, truth))
    missed = [i for i, s in enumerate(stickers)
              if i not in used and s["seen"] >= 0.95 and s["pixels"] >= MIN_STICKER_PX]
    return out, missed


def plot_stickers(shots, path: Optional[str] = None):
    """One image per camera: every sticker the label reader found, outlined
    and labelled with what it read and its confidence: green read right, red
    read wrong (with the true text), grey unreadable, magenta no sticker
    there; dashed red, stickers in view (95%+ of them, MIN_STICKER_PX or
    more) it didn't find.
    shots: [(camera name, frame, readings, true stickers as
    make_vision_dataset.sticker_labels gives them)]."""
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(len(shots), 1, figsize=(12, 6.9 * len(shots)), squeeze=False)
    one_line = lambda text: text.replace("\n", " ")
    colour = {"right": READ_RIGHT_RGBA, "wrong": MISSED_RGBA, "unreadable": UNREADABLE_RGBA, "no sticker": EXTRA_RGBA}
    for ax, (name, frame, readings, stickers) in zip(axes[:, 0], shots):
        ax.imshow(frame.rgb)
        verdicts, missed_ids = sticker_verdicts(readings, stickers)
        counts = {v: sum(x[1] == v for x in verdicts) for v in colour}
        for r, verdict, truth in verdicts:
            label = {"right": one_line(r.text or ""), "unreadable": "unreadable", "no sticker": "no sticker",
                     "wrong": f"{one_line(r.text or '')} (is {one_line(stickers[truth]['text']) if truth is not None else ''})"}[verdict]
            q = np.vstack([r.corners, r.corners[:1]])
            ax.plot(q[:, 0], q[:, 1], color=colour[verdict], lw=1.5)
            ax.text(q[:, 0].mean(), q[:, 1].max() + 3, f"{label} {r.confidence:.3f}", color=colour[verdict], fontsize=6,
                    ha="center", va="top", bbox=dict(fc="k", alpha=0.5, pad=1, lw=0))
        for i in missed_ids:
            c = np.asarray(stickers[i]["corners"])
            ax.plot(np.r_[c[:, 0], c[0, 0]], np.r_[c[:, 1], c[0, 1]], color=MISSED_RGBA, lw=1.2, ls="--")
        missed = len(missed_ids)
        W, H = frame.size
        ax.set_xlim(0, W)
        ax.set_ylim(H, 0)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"{name}: {len(readings)} stickers found ({counts['right']} read right, {counts['wrong']} wrong, "
                     f"{counts['unreadable']} unreadable); {missed} in clear view not found (dashed)", fontsize=9)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=100)
        plt.close(fig)
    else:
        plt.show()


def learned_cart(scenario: str, seed: int, weights: str, path):
    """The cart scene cart_demo.py makes for this scenario and seed, through
    the label reader: row_cam (arm leaned out of its view), tray_cam, a wrist
    close-up where the shelf map puts the slot, and a wrist view of the whole
    tray; saves plot_stickers."""
    from bila.vision import label_reader as lr                              # imports torch._dynamo, before any render
    from bila.control import cart_demo
    from bila.vision.camera import DepthNoise
    from bila.control.insert_book import ArmIK, ShelfFrame
    from bila.control.inspection import look_from, slot_x
    from bila.vision.make_vision_dataset import cart_numbers, sticker_labels
    from bila.sim.shelf_scene import PANDA_HOME, settle

    cfg, row, tray, _ = cart_demo.sample(scenario, seed)
    numbers = cart_numbers(row, tray, seed)
    model, tray = cart_demo.build(cfg, row, tray, numbers)
    data = mujoco.MjData(model)
    settle(model, data, row)
    books = row.books + tray + [b for fl in row.filler_levels for b in fl.books]
    texts = {b.name: cart_demo.sticker_text(numbers[b.name], cart_demo.cutter(b)) for b in books
             if b.name in numbers and mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, b.name + "_label") >= 0}
    reader = lr.LabelReader(os.path.join(weights, "labels", "finder.pt"), os.path.join(weights, "labels", "reader.pt"),
                            catalogue=sorted(set(texts.values())))
    cams = {n: WristCamera(model, name=n, noise=DepthNoise.d405(), rng=np.random.default_rng([seed, i]))
            for i, n in enumerate(("row_cam", "tray_cam", "panda/wrist_cam"))}
    shots = []

    def shoot(name, camera):
        frame = cams[camera].capture(data)
        shots.append((name, frame, reader.read(frame.rgb), sticker_labels(model, data, frame, texts)))

    look_from(model, data, cart_demo.SCAN_POSE)
    shoot("row_cam", "row_cam")
    look_from(model, data, PANDA_HOME[:7])
    shoot("tray_cam", "tray_cam")
    ik, rng = ArmIK(model, PANDA_HOME[:7]), np.random.default_rng(seed)
    point = ShelfFrame(row).to_world(np.array([slot_x(row), row.shelf_depth / 2, row.target_board_top + cart_demo.STICKER_UP]))
    q = cart_demo.row_closeup(model, data, ik, row, point)
    if q is not None:
        look_from(model, data, q)
        shoot("wrist camera, close up where the map puts the slot", "panda/wrist_cam")
    lift = model.jnt_qposadr[model.joint("lift_z").id]
    keep = float(data.qpos[lift])
    centre = data.geom_xpos[model.geom("tray").id].copy()
    centre[2] = np.median([data.geom_xpos[model.geom(b.name + "_label").id][2] for b in tray])
    plan = cart_demo.tray_closeup(model, data, ik, centre, rng, above=cart_demo.OVERVIEW_ABOVE,
                                  yaws=(math.pi / 2, -math.pi / 2))
    if plan is not None:
        data.qpos[lift] = plan[0]
        look_from(model, data, plan[1])
        shoot("wrist camera, over the tray", "panda/wrist_cam")
        data.qpos[lift] = keep
        mujoco.mj_forward(model, data)
    for c in cams.values():
        c.close()
    plot_stickers(shots, path)


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
    ap.add_argument("--noisy", action="store_true",
                    help="NoisyPerception (or, with --camera, D405-like depth noise), with the ground truth dashed")
    ap.add_argument("--camera", action="store_true",
                    help="estimate from the wrist camera at the inspection pose: saves its image with the "
                         "estimate outlined, and the estimate's front and top view with the ground truth dashed")
    ap.add_argument("--learned", action="store_true",
                    help="what the trained models see: the learned segmenter's masks and the estimate from them "
                         "(with --cart: the label reader's stickers in the cart scene); needs their weights")
    ap.add_argument("--cart", action="store_true", help="with --learned: the cart scene's stickers, as cart_demo.py reads them")
    ap.add_argument("--weights", default=WEIGHTS, help="with --learned: folder with segmenter/ and labels/")
    ap.add_argument("--out", help="image path (default scenes/<kind>_<scenario>_seed<seed>.png)")
    ap.add_argument("--show", action="store_true", help="open windows instead of saving")
    a = ap.parse_args()

    def path_for(kind):
        if a.show:
            return None
        own = a.out and kind != "camera" and not a.learned
        path = a.out if own else os.path.join(HERE, "scenes", f"{kind}_{a.scenario}_seed{a.seed}.png")
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        print(f"saving {path}")
        return path

    if a.learned:
        weights = os.path.expanduser(a.weights)
        if a.cart:
            if a.scenario == "collapsed":
                raise SystemExit("the cart scene has no collapsed scenario")
            learned_cart(a.scenario, a.seed, weights, path_for("stickers"))
        else:
            learned_shelf(a.scenario, a.seed, weights, path_for)
        return

    panda = os.path.join(HERE, "mujoco_menagerie", "franka_emika_panda", "panda.xml") if a.camera else None
    model, data, row, _ = make_episode(ShelfConfig(), a.scenario, np.random.default_rng(a.seed), panda_xml=panda)
    target = (row.target_book.thickness, row.target_book.height)
    if not a.camera:
        est, truth = estimate_row(model, data, row, a.noisy, a.seed)
        plot_estimate(est, truth, target, path_for("estimate"))
        return

    from bila.vision.camera import DepthNoise
    from bila.control.inspection import look_from, solve_inspection
    from bila.vision.rgbd_perception import RGBDPerception
    view = solve_inspection(model, data, row)
    if view is None:
        raise SystemExit("no arm pose reaches the inspection view in this scene")
    look_from(model, data, view[0])
    camera = WristCamera(model, noise=DepthNoise() if a.noisy else None, rng=np.random.default_rng(a.seed))
    with RGBDPerception(model, row, camera=camera) as perception:
        frame = camera.capture(data)
        est = perception.estimate_from_frame(frame)
    plot_frame(frame, est, model, path_for("camera"))
    plot_estimate(est, GroundTruthPerception(model, row).estimate(data), target, path_for("estimate_rgbd"))


if __name__ == "__main__":
    main()
