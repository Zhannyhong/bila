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
  NoisyPerception         the ground truth with the wrist camera's measured errors (each book's,
                          and one rigid shift of the whole row per episode): cheap, for training
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


FREE_SPACE_MIN = 0.010   # m: a space between books this wide or more is room to slide or tip into
FREE_SPACE_CAP = 6       # books: look no further out than this (a push doesn't carry further)


@dataclass
class FreeSpace:
    """Room on one side of the slot, behind its neighbour, measured at the board."""
    books: int        # books between the slot and the first free space (FREE_SPACE_CAP if none that close)
    width: float      # m: how wide that free space is (0 if none within the cap)
    resolved: bool    # False if the camera didn't see far enough to tell


def free_space(est: ShelfEstimate, side: int, min_width: float = FREE_SPACE_MIN,
               cap: int = FREE_SPACE_CAP) -> FreeSpace:
    """Walk out from the slot on one side (-1 left, +1 right), book by book,
    to the first space between books at least `min_width` wide, measured
    between their spines' bottom corners. The books packed in front of it
    move together when the neighbour is pushed that way, and the space is
    where they can slide or tip into: the cascade risk. After the last book
    seen, the side panel closes the row if it is in view; if it isn't, what
    lies beyond is unknown (resolved=False)."""
    by_offset = {b.slot_offset: b for b in est.books}
    inner, outer = (1, 0) if side < 0 else (0, 1)        # bottom corner facing the slot, and facing away
    for k in range(1, cap + 1):
        book = by_offset.get(side * k)
        if book is None:
            return FreeSpace(cap, 0.0, False)
        edge = book.spine_corners[outer, 0]
        beyond = by_offset.get(side * (k + 1))
        if beyond is not None:
            width = side * (beyond.spine_corners[inner, 0] - edge)
        elif est.panel_x[0 if side < 0 else 1] is not None:
            width = side * (est.panel_x[0 if side < 0 else 1] - edge)
            if width < min_width:                        # packed up to the panel: no room on this side
                return FreeSpace(cap, 0.0, True)
        else:
            return FreeSpace(cap, 0.0, False)            # the rest of the row wasn't seen
        if width >= min_width:
            return FreeSpace(k, float(width), True)
    return FreeSpace(cap, 0.0, True)


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
# (`calibrate`) over 80 scenes (seeds 1000-1019 of clear, tight, leaning and
# packed) with the realistic camera insert_book.py --perception camera uses:
# a D405's stereo errors (DepthNoise.d405) and a calibration error
# (CalibrationError), the first look's three views fused, ~48 cm from the
# board's edge. Each quantity's error as (median, std), in m or rad; the std is
# set so that 95% of errors fall where 95% of the camera's do (its errors have
# heavier tails than a Gaussian).
#
# CAMERA_SYSTEMATIC: the rigid motion of the whole row, fitted per scene (see
# SYSTEMATIC_PARAMS), over the scenes. It is most of the camera's error: a few
# mm each way, where each book's own is under 1 mm. Its rotation about the row
# (rx: the spines seen tipped back, tops further in) is a steady bias of the
# D405 model, not random.
CAMERA_SYSTEMATIC = {"x": (-0.00047, 0.00256), "y": (-0.00146, 0.00313), "z": (-0.00013, 0.00249),
                     "rx": (-0.01216, 0.00492), "ry": (0.00057, 0.00250), "rz": (0.00025, 0.00105)}
# CAMERA_ERRORS: each book's own error once its scene's rigid motion is taken
# out, per band of |slot offset|.
CAMERA_ERRORS = {
    (1, 1): {"x": (-0.00001, 0.00064), "z": (-0.00011, 0.00059), "spine_y": (0.00004, 0.00041),
             "thickness": (0.00252, 0.00143), "height": (0.00177, 0.00168), "lean": (0.00006, 0.00468),
             "yaw": (0.00133, 0.02279)},
    (2, 3): {"x": (-0.00026, 0.00097), "z": (-0.00016, 0.00069), "spine_y": (-0.00004, 0.00067),
             "thickness": (0.00399, 0.00193), "height": (0.00165, 0.00171), "lean": (0.00022, 0.00519),
             "yaw": (0.00212, 0.03945)},
    (4, 99): {"x": (-0.00001, 0.00118), "z": (-0.00000, 0.00109), "spine_y": (-0.00001, 0.00061),
              "thickness": (0.00348, 0.00236), "height": (0.00185, 0.00192), "lean": (-0.00012, 0.00536),
              "yaw": (-0.00554, 0.04176)},
}
VIEW_REACH = 0.58   # m: the camera sees books whose outer edge is within this of the slot, along the row


def rotation(axis: str, angle: float) -> np.ndarray:
    """Rotation by angle (rad) about the shelf's y or z axis."""
    c, s = math.cos(angle), math.sin(angle)
    if axis == "y":
        return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


# ---------------------------------------------------------------------------
# The camera's systematic error: one rigid motion of everything it sees
# ---------------------------------------------------------------------------
# A hand-eye calibration error (where the robot thinks the camera sits on the
# hand) and the D405's per-camera depth offset don't make each book wrong in
# its own way: they move the whole estimate together, by the same small
# rotation and shift, for as long as the camera keeps them (an episode). The
# controller plans every push from that estimate, so a shared error matters
# far more than the same amount of independent noise would: every book, and
# the gap, is off in the same direction. It is written as a shift of the
# slot's point (slot_pivot) and a rotation vector about it, in the shelf frame.
SYSTEMATIC_PARAMS = ("x", "y", "z", "rx", "ry", "rz")


def slot_pivot(est: ShelfEstimate) -> np.ndarray:
    """The point the systematic error turns the row about: the slot, at the
    board's top and front edge."""
    return np.array([est.slot_x(), est.front_y, est.board_top])


def matrix_from_rotvec(w) -> np.ndarray:
    """Rotation matrix for a rotation vector (axis times angle, rad)."""
    w = np.asarray(w, dtype=float)
    angle = np.linalg.norm(w)
    if angle < 1e-12:
        return np.eye(3)
    k = w / angle
    K = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return np.eye(3) + math.sin(angle) * K + (1 - math.cos(angle)) * K @ K


def rotvec_from_matrix(R) -> np.ndarray:
    """Rotation vector (axis times angle, rad) of a rotation matrix (angles below 180 deg)."""
    angle = math.acos(min(1.0, max(-1.0, (np.trace(R) - 1) / 2)))
    if angle < 1e-12:
        return np.zeros(3)
    axis = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2 * math.sin(angle))
    return axis * angle


def rigid_shift(params: dict) -> Tuple[np.ndarray, np.ndarray]:
    """(rotation, translation) for a systematic error given as SYSTEMATIC_PARAMS."""
    return (matrix_from_rotvec([params["rx"], params["ry"], params["rz"]]),
            np.array([params["x"], params["y"], params["z"]]))


def apply_shift(points, R, t, pivot) -> np.ndarray:
    """Points moved by the rigid motion (R, t) about `pivot`: R (p - pivot) + pivot + t."""
    return (np.asarray(points) - pivot) @ R.T + pivot + t


def fit_rigid(src, dst, pivot) -> Tuple[np.ndarray, np.ndarray]:
    """The rigid motion (R, t) about `pivot` that best maps points src onto dst,
    in the least-squares sense (Kabsch): dst ~ apply_shift(src, R, t, pivot)."""
    src, dst = np.asarray(src) - pivot, np.asarray(dst) - pivot
    cs, cd = src.mean(axis=0), dst.mean(axis=0)
    U, _, Vt = np.linalg.svd((src - cs).T @ (dst - cd))
    flip = np.diag([1.0, 1.0, np.sign(np.linalg.det(Vt.T @ U.T))])     # a rotation, never a mirror
    R = Vt.T @ flip @ U.T
    return R, cd - R @ cs


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
        Gaussian noise with its spread times `scale`, drawn anew every estimate;
      - then everything seen moves by one rigid motion (CAMERA_SYSTEMATIC): the
        camera's hand-eye calibration and depth offset. It is drawn once, when
        this object is made, and kept for every estimate after: make one
        NoisyPerception per episode, as each episode is a new camera;
      - books, and side panels, further along the row from the slot than the
        camera sees (VIEW_REACH) are left out.
    Everything derived (thickness, angles, the gap) is recomputed from the
    changed corners. scale > 1 trains on worse estimates than the camera's.
    systematic=None leaves out the rigid motion (the books' errors are then
    independent, as the basic camera's are)."""

    def __init__(self, source: Perception, rng: np.random.Generator, scale: float = 1.0,
                 errors: dict = CAMERA_ERRORS, reach: float = VIEW_REACH,
                 systematic: Optional[dict] = CAMERA_SYSTEMATIC):
        self.source, self.rng, self.scale = source, rng, scale
        self.errors, self.reach = errors, reach
        # Drawn from its own stream, split off rng without drawing from it (as
        # WristCamera does), so the books' noise for a seed is the same with or without it.
        shift_rng, = rng.spawn(1)
        self.shift = None if systematic is None else {
            k: mean + scale * std * shift_rng.normal() for k, (mean, std) in systematic.items()}
        self.pivot = None          # slot_pivot of the first estimate: the camera's error is fixed, not the slot

    def error_for(self, slot_offset: int) -> dict:
        """A random error for a book this far from the slot."""
        band = next(b for b in self.errors if b[0] <= abs(slot_offset) <= b[1])
        return {k: mean + self.scale * std * self.rng.normal() for k, (mean, std) in self.errors[band].items()}

    def estimate(self, data: mujoco.MjData) -> ShelfEstimate:
        est = self.source.estimate(data)
        x_slot = est.slot_x()
        if self.pivot is None:
            self.pivot = slot_pivot(est)
        R, t = rigid_shift(self.shift) if self.shift is not None else (np.eye(3), np.zeros(3))
        books = []
        for b in est.books:
            outer = b.spine_corners[0, 0] if b.slot_offset < 0 else b.spine_corners[1, 0]
            if abs(outer - x_slot) > self.reach:
                continue
            corners = apply_shift(perturb_spine(b.spine_corners, self.error_for(b.slot_offset)), R, t, self.pivot)
            books.append(book_from_spine(corners, b.slot_offset, b.depth, b.visible))
        panels = tuple(x if x is not None and abs(x - x_slot) <= self.reach else None for x in est.panel_x)
        widths = gap_profile(books, panels, est.board_top, est.gap_heights)
        return replace(est, books=books, panel_x=panels, gap_widths=widths, source="noisy")
