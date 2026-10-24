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
  NoisyPerception         wraps another source, adds camera-like errors
  (later) an RGB-D source fitting spine faces to the wrist camera's depth image

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

    def min_gap_width(self, up_to: float) -> float:
        """Narrowest free width of the gap from the board up to `up_to` metres
        above it: what a book that tall must get through. NaN if not seen."""
        w = self.gap_widths[self.gap_heights <= up_to]
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


def gap_profile(books: List[BookEstimate], panel_x, board_top: float, heights) -> np.ndarray:
    """Free width of the slot at each height above the board: from the
    right-most edge of anything left of the slot to the left-most edge of
    anything right of it, side panels included. Every book counts, not just
    the two neighbours, so a run of books leaning over the slot narrows it too.
    NaN where a side has nothing seen at that height."""
    widths = []
    for h in heights:
        z = board_top + h
        left = [s[1] for b in books if b.slot_offset < 0 and (s := slice_x(b.spine_corners, z))]
        right = [s[0] for b in books if b.slot_offset > 0 and (s := slice_x(b.spine_corners, z))]
        if panel_x[0] is not None:
            left.append(panel_x[0])
        if panel_x[1] is not None:
            right.append(panel_x[1])
        widths.append(max(0.0, min(right) - max(left)) if left and right else math.nan)
    return np.array(widths)


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


class NoisyPerception:
    """Wraps another source and adds camera-like errors. Each spine corner is
    off by Gaussian noise (2 mm by default: about +-1 deg of lean on a 20 cm
    book and +-5 deg of swivel on a 3 cm spine), and gaps narrower than the
    camera resolves read as closed. Everything derived is recomputed from the
    noisy corners. The noise levels are a starting guess, to be set from the
    RGB-D source's real errors once it exists."""

    def __init__(self, source: Perception, rng: np.random.Generator,
                 corner_std: float = 0.002, min_gap: float = 0.004):
        self.source, self.rng = source, rng
        self.corner_std, self.min_gap = corner_std, min_gap

    def estimate(self, data: mujoco.MjData) -> ShelfEstimate:
        est = self.source.estimate(data)
        books = [book_from_spine(b.spine_corners + self.rng.normal(0.0, self.corner_std, (4, 3)),
                                 b.slot_offset, b.depth, b.visible) for b in est.books]
        widths = gap_profile(books, est.panel_x, est.board_top, est.gap_heights)
        widths[widths < self.min_gap] = 0.0                      # NaN (not seen) stays NaN
        return replace(est, books=books, gap_widths=widths, source="noisy")
