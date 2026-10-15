"""
What the robot knows about the target row: a ShelfEstimate, as a wrist RGB-D
camera could report it.

A ShelfEstimate holds only what a camera looking at the row from the front can
measure, plus what the shelf map gives. Each book is its spine face (the four
corners), and the values derived from it: thickness, height, lean, swivel and
how far out the spine sits. The gap is measured at the spine plane, at a few
heights above the board, as free width between the books on either side.

What the camera can't see stays out:
  - A book's depth (front to back) is a prior, never the true value: the backs
    are hidden behind the spines.
  - Mass, friction, contact forces and velocities. They are privileged
    simulator state, for the critic and the reward only.

Sources of an estimate (each returns a ShelfEstimate):
  GroundTruthPerception   spine faces of the simulated books, exact
  NoisyPerception         the ground truth with the wrist camera's measured errors: cheap, for training
  RGBDPerception          (rgbd_perception.py) spine faces fitted to the wrist camera's depth image

Every source produces spine corners and derives the rest with the same
functions (book_from_spine, gap_profile), so noise on the corners carries
through to every derived value consistently.

For now the robot looks once, from in front of the row, before it pushes: an
estimate is a snapshot, and `stamp` says when it was taken. While the hand is
in the gap the camera can't see it, so the policy relies on force and joint
feedback until it looks again.

Frame: the shelf frame of shelf_scene / insert_book (x along the row, y out of
the shelf, z up, origin on the floor at the bookcase's centre). The board's
front edge is at y = shelf_depth / 2.

    from perception import GroundTruthPerception
    est = GroundTruthPerception(model, row).estimate(data)
    left = est.book(-1)                        # the slot's left neighbour
    print(left.lean, est.min_gap_width(0.25))  # its lean, the narrowest gap up to 25 cm
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import List, Optional, Protocol, Tuple

import mujoco
import numpy as np

from insert_book import ShelfFrame
from shelf_scene import Row, ShelfConfig

# Heights above the target board where the gap is measured (m). Books are
# 17 to 30 cm tall, so these cover any target book; above a book's top the
# width is just whatever is beyond it.
GAP_HEIGHTS = np.arange(0.01, 0.30, 0.04)          # 0.01, 0.05, ..., 0.29
# A book's depth is never measured: the camera only sees spines. Assume this,
# capped so the book fits the shelf (the middle of the generator's range).
DEPTH_PRIOR = 0.17
DEPTH_CLEARANCE = ShelfConfig.depth_clearance      # every book at least this much shallower than the shelf
FACING_COS = 0.5                                   # spine counts as visible if it faces within 60 deg of the camera side


# ---------------------------------------------------------------------------
# 1. The estimate
# ---------------------------------------------------------------------------
@dataclass
class BookEstimate:
    """One book as the camera sees it: its spine face, and a box model."""
    slot_offset: int            # place relative to the slot: -1 = left neighbour, +1 = right, -2, +2, ...
    spine_corners: np.ndarray   # (4, 3) shelf frame: bottom-left, bottom-right, top-right, top-left
    thickness: float            # across the spine (m)
    height: float               # along the spine (m)
    lean: float                 # rad, about the shelf's y axis; + = top toward +x
    yaw: float                  # rad, swivel about vertical; + = right edge of the spine further out
    spine_y: float              # how far out the spine's centre sits (shelf y)
    depth: float                # front to back: ASSUMED (DEPTH_PRIOR), not measured
    visible: float              # fraction of the spine face in view, 0..1


@dataclass
class ShelfEstimate:
    """What perception reports about the target row at one moment."""
    shelf_pose: np.ndarray      # (4, 4) world <- shelf frame (from the fiducial)
    board_top: float            # z of the target board's top surface
    front_y: float              # y of the board's front edge
    clear_height: float         # board top to the board above
    shelf_depth: float          # from the shelf map, not the camera
    panel_x: Tuple[Optional[float], Optional[float]]   # inner faces of the left and right side panels
    books: List[BookEstimate]   # books seen, left to right
    gap_heights: np.ndarray     # (N,) heights above the board where the gap is measured
    gap_widths: np.ndarray      # (N,) free width at the spine plane; NaN where not seen
    stamp: float                # sim time of the image it came from
    source: str                 # "ground_truth", "noisy", ...

    def book(self, slot_offset: int) -> Optional[BookEstimate]:
        """The book at this place relative to the slot, or None if not seen."""
        return next((b for b in self.books if b.slot_offset == slot_offset), None)

    def slot_x(self) -> float:
        """Where the slot is along the row: between its neighbours' bottom
        corners. NaN if a neighbour wasn't seen."""
        left, right = self.book(-1), self.book(+1)
        if left is None or right is None:
            return math.nan
        return (left.spine_corners[1, 0] + right.spine_corners[0, 0]) / 2

    def min_gap_width(self, up_to: float) -> float:
        """Narrowest free width of the gap from the board up to `up_to` metres
        above it: what a book that tall must get through. Heights not seen are
        skipped: there nothing in view on one side reaches that high, so the gap
        runs past the camera's view and can't be the narrowest. NaN if no
        height up to there was seen."""
        w = self.gap_widths[self.gap_heights <= up_to]
        w = w[~np.isnan(w)]
        return float(np.min(w)) if len(w) else math.nan


# ---------------------------------------------------------------------------
# 2. Derived values, shared by every source
# ---------------------------------------------------------------------------
def book_from_spine(corners, slot_offset: int, depth: float, visible: float) -> BookEstimate:
    """A BookEstimate from the spine face's corners (bottom-left, bottom-right,
    top-right, top-left; shelf frame). Thickness and height are averaged over
    the two opposite edges, and so are the angles."""
    c = np.asarray(corners, dtype=float)
    across = ((c[1] - c[0]) + (c[2] - c[3])) / 2         # bottom edge and top edge, left to right
    up = ((c[3] - c[0]) + (c[2] - c[1])) / 2             # left edge and right edge, bottom to top
    return BookEstimate(
        slot_offset=slot_offset,
        spine_corners=c,
        thickness=float(np.linalg.norm(across)),
        height=float(np.linalg.norm(up)),
        lean=math.atan2(up[0], up[2]),
        yaw=math.atan2(across[1], across[0]),
        spine_y=float(c[:, 1].mean()),
        depth=depth,
        visible=visible,
    )


def slice_x(corners, z: float) -> Optional[Tuple[float, float]]:
    """Where the spine face crosses height z, as (x_min, x_max), seen from the
    front (the face projected on the x-z plane). None if it doesn't reach z."""
    xs = []
    for p, q in zip(corners, np.roll(corners, -1, axis=0)):      # the four edges
        lo, hi = sorted((p[2], q[2]))
        if not lo <= z <= hi:
            continue
        if hi - lo < 1e-9:                                       # horizontal edge at z
            xs += [p[0], q[0]]
        else:
            xs.append(p[0] + (z - p[2]) / (q[2] - p[2]) * (q[0] - p[0]))
    return (min(xs), max(xs)) if xs else None


def gap_edges(books: List[BookEstimate], panel_x, board_top: float, heights) -> Tuple[np.ndarray, np.ndarray]:
    """x of the slot's left and right edges at each height above the board:
    the right-most edge of anything left of the slot, and the left-most edge
    of anything right of it, side panels included. Every book counts, not just
    the two neighbours, so a run of books leaning over the slot narrows it too.
    NaN where a side has nothing seen at that height, and on a whole side if
    its neighbour (-1 or +1) wasn't seen: the gap there is unknown, and
    measuring to a book further out would overstate it."""
    seen = {b.slot_offset for b in books}
    lefts, rights = [], []
    for h in heights:
        z = board_top + h
        left = [s[1] for b in books if b.slot_offset < 0 and (s := slice_x(b.spine_corners, z))]
        right = [s[0] for b in books if b.slot_offset > 0 and (s := slice_x(b.spine_corners, z))]
        if panel_x[0] is not None:
            left.append(panel_x[0])
        if panel_x[1] is not None:
            right.append(panel_x[1])
        lefts.append(max(left) if left and -1 in seen else math.nan)
        rights.append(min(right) if right and +1 in seen else math.nan)
    return np.array(lefts), np.array(rights)


def gap_profile(books: List[BookEstimate], panel_x, board_top: float, heights) -> np.ndarray:
    """Free width of the slot at each height above the board (see gap_edges).
    NaN where a side has nothing seen at that height."""
    left, right = gap_edges(books, panel_x, board_top, heights)
    return np.maximum(0.0, right - left)                  # NaN stays NaN


# ---------------------------------------------------------------------------
# 3. Sources
# ---------------------------------------------------------------------------
class Perception(Protocol):
    def estimate(self, data: mujoco.MjData) -> ShelfEstimate: ...


class GroundTruthPerception:
    """Exact spine faces of the simulated target row. Uses the shelf map (pose
    and size of the bookcase, which slot is the target) from `row`, and the
    books' poses from the simulation. Never reads a book's true depth, mass or
    friction."""

    def __init__(self, model: mujoco.MjModel, row: Row, depth_prior: float = DEPTH_PRIOR,
                 gap_heights=GAP_HEIGHTS):
        self.m, self.row = model, row
        self.shelf = ShelfFrame(row)
        self.geoms = [model.geom(b.name + "_geom").id for b in row.books]
        self.depth = min(depth_prior, row.shelf_depth - DEPTH_CLEARANCE)
        self.gap_heights = np.asarray(gap_heights, dtype=float)
        ti = row.target_index                     # the slot is between books[ti - 1] and books[ti]
        self.offsets = [i - ti if i < ti else i - ti + 1 for i in range(len(row.books))]

    def spine_face(self, data: mujoco.MjData, geom: int) -> Tuple[np.ndarray, float]:
        """Spine corners in the shelf frame, and whether the spine faces the
        camera side. A book's box has its thickness along local x, depth along
        y (spine at +y) and height along z."""
        t, d, h = self.m.geom_size[geom]
        local = np.array([[-t, d, -h], [t, d, -h], [t, d, h], [-t, d, h]])
        R = data.geom_xmat[geom].reshape(3, 3)
        corners = self.shelf.to_shelf(local @ R.T + data.geom_xpos[geom])
        facing = (self.shelf.R.T @ R[:, 1])[1]                   # spine normal's shelf-y component
        return corners, float(facing > FACING_COS)

    def estimate(self, data: mujoco.MjData) -> ShelfEstimate:
        row = self.row
        books = []
        for geom, offset in zip(self.geoms, self.offsets):
            corners, visible = self.spine_face(data, geom)
            books.append(book_from_spine(corners, offset, self.depth, visible))
        pose = np.eye(4)
        pose[:3, :3], pose[:3, 3] = self.shelf.R, self.shelf.p
        panels = (-row.shelf_width / 2, row.shelf_width / 2)
        return ShelfEstimate(
            shelf_pose=pose, board_top=row.target_board_top, front_y=row.shelf_depth / 2,
            clear_height=row.clear_height, shelf_depth=row.shelf_depth, panel_x=panels,
            books=books, gap_heights=self.gap_heights.copy(),
            gap_widths=gap_profile(books, panels, row.target_board_top, self.gap_heights),
            stamp=float(data.time), source="ground_truth")


# What the wrist camera's estimate gets wrong, measured by eval_perception.py
# (`calibrate`) over 48 scenes, 12 of each scenario: camera ~50 cm from the
# spines, D405-like depth noise. Per band of |slot offset|, each quantity's
# error as (median, std), in m or rad; the std is set so that 95% of errors fall
# where 95% of the camera's do (its errors have heavier tails than a Gaussian).
CAMERA_ERRORS = {
    (1, 1): {"x": (0.00006, 0.00062), "z": (-0.00095, 0.00027), "spine_y": (-0.00000, 0.00002),
             "thickness": (-0.00045, 0.00148), "height": (0.00070, 0.00047), "lean": (-0.00001, 0.00270),
             "yaw": (-0.00009, 0.00598)},
    (2, 3): {"x": (-0.00006, 0.00092), "z": (-0.00089, 0.00029), "spine_y": (-0.00000, 0.00002),
             "thickness": (0.00215, 0.00151), "height": (0.00082, 0.00049), "lean": (0.00003, 0.00356),
             "yaw": (0.00026, 0.00989)},
    (4, 99): {"x": (-0.00004, 0.00177), "z": (-0.00085, 0.00058), "spine_y": (-0.00001, 0.00004),
              "thickness": (0.00464, 0.00294), "height": (0.00090, 0.00090), "lean": (0.00022, 0.00523),
              "yaw": (-0.00024, 0.01371)},
}
VIEW_REACH = 0.53   # m: the camera sees books whose outer edge is within this of the slot, along the row


def rotation(axis: str, angle: float) -> np.ndarray:
    """Rotation by angle (rad) about the shelf's y or z axis."""
    c, s = math.cos(angle), math.sin(angle)
    if axis == "y":
        return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def perturb_spine(corners, err: dict) -> np.ndarray:
    """Spine corners with errors added: thickness and height grown by
    err["thickness"] and err["height"]; turned about the spine's centre by
    err["lean"] (about the shelf's y axis) and err["yaw"] (about vertical); the
    centre moved by err["x"], err["spine_y"] and err["z"]."""
    c = np.asarray(corners, dtype=float)
    centre = c.mean(axis=0)
    across = ((c[1] - c[0]) + (c[2] - c[3])) / 2
    up = ((c[3] - c[0]) + (c[2] - c[1])) / 2
    across *= 1 + err["thickness"] / np.linalg.norm(across)
    up *= 1 + err["height"] / np.linalg.norm(up)
    turn = rotation("z", err["yaw"]) @ rotation("y", err["lean"])
    across, up = turn @ across, turn @ up
    centre = centre + np.array([err["x"], err["spine_y"], err["z"]])
    return np.array([centre - across / 2 - up / 2, centre + across / 2 - up / 2,
                     centre + across / 2 + up / 2, centre - across / 2 + up / 2])


class NoisyPerception:
    """Wraps another source (normally the ground truth) and makes its estimate
    look like the wrist camera's, at a fraction of the cost (no rendering):
      - each book's position, size, lean and swivel get the camera's measured
        error for its distance from the slot (CAMERA_ERRORS): its bias, plus
        Gaussian noise with its spread times `scale`;
      - books, and side panels, further along the row from the slot than the
        camera sees (VIEW_REACH) are left out.
    Everything derived (thickness, angles, the gap) is recomputed from the
    changed corners. scale > 1 trains on worse estimates than the camera's."""

    def __init__(self, source: Perception, rng: np.random.Generator, scale: float = 1.0,
                 errors: dict = CAMERA_ERRORS, reach: float = VIEW_REACH):
        self.source, self.rng, self.scale = source, rng, scale
        self.errors, self.reach = errors, reach

    def error_for(self, slot_offset: int) -> dict:
        """A random error for a book this far from the slot."""
        band = next(b for b in self.errors if b[0] <= abs(slot_offset) <= b[1])
        return {k: mean + self.scale * std * self.rng.normal() for k, (mean, std) in self.errors[band].items()}

    def estimate(self, data: mujoco.MjData) -> ShelfEstimate:
        est = self.source.estimate(data)
        x_slot = est.slot_x()
        books = []
        for b in est.books:
            outer = b.spine_corners[0, 0] if b.slot_offset < 0 else b.spine_corners[1, 0]
            if abs(outer - x_slot) > self.reach:
                continue
            corners = perturb_spine(b.spine_corners, self.error_for(b.slot_offset))
            books.append(book_from_spine(corners, b.slot_offset, b.depth, b.visible))
        panels = tuple(x if x is not None and abs(x - x_slot) <= self.reach else None for x in est.panel_x)
        widths = gap_profile(books, panels, est.board_top, est.gap_heights)
        return replace(est, books=books, panel_x=panels, gap_widths=widths, source="noisy")
