"""
A ShelfEstimate from the wrist camera's RGB-D image: what the robot knows
about the target row once it has looked at it.

    from rgbd_perception import RGBDPerception
    look_from(model, data, solve_inspection(model, data, row)[0])    # the arm at the inspection pose
    with RGBDPerception(model, row) as perception:
        est = perception.estimate(data)                              # a ShelfEstimate, like the others

Steps, for one image (estimate_from_frame):
  1. Segment: one pixel mask per book. Here OracleSegmenter reads them off the
     simulation's object ids; on a real robot a learned segmenter would. The
     rest doesn't depend on where the masks come from.
  2. Turn depth into 3D points in the shelf frame and keep a book's points
     only if they are on the target level, inside the bookcase.
  3. Keep the spine: points whose surface faces out of the shelf (normals from
     the depth image), dropping the tops and sides seen at an angle.
  4. Fit a plane to them: facing the median of their surface normals (a
     spine is a long, narrow strip, so a least-squares plane could turn freely
     about its long side), through the median of their positions. Then the
     smallest rectangle around them in that plane, grown by the pixels lost at
     its edges (the normals need NORMAL_STEP pixels of surface on each side). Its corners are the
     spine corners; book_from_spine derives the rest.
  5. Name the books by where they stand relative to the slot. The shelf map
     says, as a real robot would know it from the catalogue, how many books
     stand left of the slot and about where it is along the row. If the
     camera sees the whole row (both side panels, and as many books as the
     map lists), count them. Otherwise, as books may have shifted since the
     map was made, the slot is the boundary between two adjacent books (by
     where their bottoms stand) closest to the map's place for it. In a few
     leaning rows the books settle 3 to 7 cm from where the map has them,
     and that names every book one place off.
  6. The gap profile from the books (gap_profile), as for every source.

Several images (estimate_from_frames), e.g. from views along the row: steps 1-4
for each, then each book is taken from the view whose camera is nearest it
along the row (if two nearly tie, the one that has it whole in view), and only
books those views missed from the others; then steps 5-6 once. Seen from far
to the side, books come out too thick, and one that sticks out past its
neighbour shows its side, which the fit can take for a thin spine further
back. After the robot has put a book into the slot, `placed_x` (where it let
go) names the book seen there slot offset 0, and the others around it.

What the camera can't see stays as in every ShelfEstimate: book depth is a
prior, a book cut off by the image edge or by the robot is still fitted, from
what is in view, with `visible` < 1, and a side panel out of view is None (so
the gap isn't measured to it past books the camera didn't see).

The shelf map (bookcase pose and size, target slot) comes from `row`, the
books only from the image.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional

import cv2
import mujoco
import numpy as np

from camera import CameraFrame, WristCamera, project, robot_mask
from insert_book import ShelfFrame
from inspection import slot_x
from perception import (DEPTH_PRIOR, DEPTH_CLEARANCE, GAP_HEIGHTS, BookEstimate, ShelfEstimate,
                        book_from_spine, gap_profile)
from shelf_scene import Row

FACING_COS = math.cos(math.radians(30))   # a point is on a spine if its surface faces within 30 deg of out of the shelf
NORMAL_STEP = 2                           # px: normals from the points this far to each side
NORMAL_SMOOTH = 5                         # px: normals from depth averaged over this square (depth noise)
PLANE_TOL = 0.003                         # m: points further than this from the fitted spine plane are dropped,
PLANE_TOL_SPREAD = 3.0                    # ... or than this many times the points' robust spread, if larger
MIN_POINTS = 100                          # fewer spine points than this: not a usable view of the book
WHOLE = 0.99                              # a book with this share of its outline a real edge is wholly in view
NEAREST_TIE = 0.01                        # m: cameras this close to equally near a book both count as nearest


@dataclass
class Sighting:
    """One book's spine fitted in one image (shelf frame)."""
    corners: np.ndarray          # (4, 3) bottom-left, bottom-right, top-right, top-left
    visible: float               # share of its outline that is a real edge of the book (boundary_visible)
    off_axis: float              # m along the row between the camera and the book: how slanted the view was
    label: Optional[int] = None  # segment label, for checking

    @property
    def x(self) -> float:
        """Where its bottom stands along the row."""
        return float((self.corners[0, 0] + self.corners[1, 0]) / 2)

    def overlap(self, other: "Sighting") -> float:
        """Share of the narrower bottom edge, along the row, that the two have
        in common: near 1 for the same book (also when one view cuts it off),
        about 0 for neighbours."""
        a, b = sorted(self.corners[:2, 0]), sorted(other.corners[:2, 0])
        return max(0.0, min(a[1], b[1]) - max(a[0], b[0])) / max(min(a[1] - a[0], b[1] - b[0]), 1e-6)

    def better_than(self, other: "Sighting") -> bool:
        return (self.visible >= WHOLE, -self.off_axis) > (other.visible >= WHOLE, -other.off_axis)


@dataclass
class Segment:
    """One book's pixels in an image. `label` identifies it if the segmenter
    knows (the MuJoCo geom id here), for checking; nothing else uses it."""
    mask: np.ndarray            # (H, W) bool
    label: Optional[int] = None


class OracleSegmenter:
    """Book masks read off the simulation's object ids: every geom whose name
    contains 'book'. Stands in for a learned segmenter."""

    def __init__(self, model: mujoco.MjModel):
        self.books = [g for g in range(model.ngeom) if "book" in (model.geom(g).name or "")]

    def __call__(self, frame: CameraFrame) -> List[Segment]:
        present = np.intersect1d(np.unique(frame.seg), self.books)
        return [Segment(frame.seg == g, int(g)) for g in present]


# ---------------------------------------------------------------------------
# Geometry on the image
# ---------------------------------------------------------------------------
def organized_points(frame: CameraFrame) -> np.ndarray:
    """(H, W, 3) camera-frame point for every pixel, NaN where there is no depth."""
    W, H = frame.size
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    z = np.where(frame.depth > 0, frame.depth, np.nan).astype(np.float64)
    K = frame.K
    return np.dstack([(u - K[0, 2]) * z / K[0, 0], (v - K[1, 2]) * z / K[1, 1], z])


def smoothed(frame: CameraFrame, size: int = NORMAL_SMOOTH) -> CameraFrame:
    """The frame with its depth averaged over size x size pixels, leaving out
    pixels without a reading (they stay without one)."""
    valid = (frame.depth > 0).astype(np.float32)
    total = cv2.boxFilter(frame.depth * valid, -1, (size, size), normalize=False)
    count = cv2.boxFilter(valid, -1, (size, size), normalize=False)
    depth = np.where(valid > 0, total / np.maximum(count, 1), 0.0).astype(np.float32)
    return CameraFrame(frame.rgb, depth, frame.seg, frame.K, frame.pose, frame.stamp)


def surface_normals(xyz: np.ndarray, step: int = NORMAL_STEP) -> np.ndarray:
    """(H, W, 3) unit normal of the surface at each pixel, from the points
    `step` pixels to each side, facing the camera. NaN at the image border, at
    pixels without depth around them, and so across depth jumps."""
    n = np.full_like(xyz, np.nan)
    s = step
    du = xyz[s:-s, 2 * s:] - xyz[s:-s, :-2 * s]               # left to right
    dv = xyz[2 * s:, s:-s] - xyz[:-2 * s, s:-s]               # top to bottom
    c = np.cross(du, dv)
    c /= np.linalg.norm(c, axis=2, keepdims=True)
    c *= -np.sign(np.sum(c * xyz[s:-s, s:-s], axis=2, keepdims=True))   # toward the camera
    n[s:-s, s:-s] = c
    return n


def fit_spine(p: np.ndarray, normals: np.ndarray, pixel_size: float, grow_px: float):
    """Spine corners (bottom-left, bottom-right, top-right, top-left; shelf
    frame) fitted to spine points p (N, 3) with their surface normals: a plane
    facing their median normal, through their median offset, then the smallest
    rectangle around the points near it, grown by grow_px pixels on every side.
    None if too few points are near the plane."""
    normal = np.median(normals, axis=0)
    normal /= np.linalg.norm(normal)
    offset = p @ normal
    spread = 1.4826 * np.median(np.abs(offset - np.median(offset)))     # robust std
    near = np.abs(offset - np.median(offset)) < max(PLANE_TOL, PLANE_TOL_SPREAD * spread)
    if near.sum() < MIN_POINTS:
        return None
    p = p[near]
    centre = p.mean(axis=0)
    up = np.array([0.0, 0.0, 1.0]) - normal[2] * normal       # shelf "up", in the plane
    up /= np.linalg.norm(up)
    across = np.cross(normal, up)                             # along the row (+x for a spine facing +y)
    flat = np.column_stack([(p - centre) @ across, (p - centre) @ up]).astype(np.float32)
    box = cv2.boxPoints(cv2.minAreaRect(flat)).astype(np.float64)    # 4 corners, in order around
    mid = box.mean(axis=0)
    side_a, side_b = box[1] - box[0], box[2] - box[1]
    tall, wide = (side_a, side_b) if np.linalg.norm(side_a) > np.linalg.norm(side_b) else (side_b, side_a)
    tall = tall if tall[1] > 0 else -tall                     # the long side is the book's height: point it up
    wide = wide if wide[0] > 0 else -wide                     # its thickness: point it along +x
    grow = grow_px * pixel_size
    tall = tall + 2 * grow * tall / np.linalg.norm(tall)
    wide = wide + 2 * grow * wide / np.linalg.norm(wide)
    corners2 = [mid - wide / 2 - tall / 2, mid + wide / 2 - tall / 2,
                mid + wide / 2 + tall / 2, mid - wide / 2 + tall / 2]
    return np.array([centre + c[0] * across + c[1] * up for c in corners2])


def boundary_visible(mask: np.ndarray, hidden: np.ndarray) -> float:
    """Share of a mask's outline that is a real edge of the book: not at the
    image border and not next to a pixel in `hidden` (e.g. the robot)."""
    inner = mask.copy()
    inner[1:] &= mask[:-1]
    inner[:-1] &= mask[1:]
    inner[:, 1:] &= mask[:, :-1]
    inner[:, :-1] &= mask[:, 1:]
    edge = mask & ~inner
    if not edge.any():
        return 0.0
    blocked = np.zeros_like(mask)
    blocked[0, :] = blocked[-1, :] = blocked[:, 0] = blocked[:, -1] = True
    grown = hidden.copy()
    grown[1:] |= hidden[:-1]
    grown[:-1] |= hidden[1:]
    grown[:, 1:] |= hidden[:, :-1]
    grown[:, :-1] |= hidden[:, 1:]
    return 1.0 - float((edge & (blocked | grown)).sum()) / float(edge.sum())


# ---------------------------------------------------------------------------
# The estimate
# ---------------------------------------------------------------------------
class RGBDPerception:
    """ShelfEstimates from the wrist camera. Uses the shelf map from `row`
    (bookcase pose and size, where the target slot is) and, for the books,
    only the image. estimate(data) takes an image from wherever the camera is;
    put the arm at the inspection pose first."""

    def __init__(self, model: mujoco.MjModel, row: Row, camera: Optional[WristCamera] = None,
                 segmenter=None, depth_prior: float = DEPTH_PRIOR, gap_heights=GAP_HEIGHTS):
        self.m, self.row = model, row
        self.shelf = ShelfFrame(row)
        self.camera = camera or WristCamera(model)
        self.segment = segmenter or OracleSegmenter(model)
        self.depth = min(depth_prior, row.shelf_depth - DEPTH_CLEARANCE)
        self.gap_heights = np.asarray(gap_heights, dtype=float)
        self.last_labels: List[Optional[int]] = []   # segment label of each book in the last estimate, for checking

    def capture(self, data: mujoco.MjData) -> CameraFrame:
        return self.camera.capture(data)

    def estimate(self, data: mujoco.MjData) -> ShelfEstimate:
        return self.estimate_from_frame(self.capture(data))

    def estimate_from_frame(self, frame: CameraFrame) -> ShelfEstimate:
        return self.estimate_from_frames([frame])

    def estimate_from_frames(self, frames: List[CameraFrame], x_slot: Optional[float] = None,
                             placed_x: Optional[float] = None) -> ShelfEstimate:
        """One estimate from several images of the row. x_slot: where the slot
        is along the row (default: the shelf map's place for it). placed_x:
        where the robot let go of a book it put into the slot."""
        cams = [self.shelf.to_shelf(f.pose[:3, 3])[0] for f in frames]
        views = [self.sightings(f) for f in frames]

        def nearest(k, s):
            return abs(s.x - cams[k]) <= min(abs(s.x - c) for c in cams) + NEAREST_TIE

        kept = []                                                    # (view, sighting)
        for k, view in enumerate(views):                             # each book from the view nearest it
            for s in (s for s in view if nearest(k, s)):
                same = next((i for i, (j, o) in enumerate(kept) if j != k and s.overlap(o) > 0.5), None)
                if same is None:
                    kept.append((k, s))
                elif s.better_than(kept[same][1]):
                    kept[same] = (k, s)
        for k, view in enumerate(views):                             # then books those views missed
            for s in (s for s in view if not nearest(k, s)):
                if all(j == k or s.overlap(o) <= 0.1 for j, o in kept):
                    kept.append((k, s))
        row, shelf = self.row, self.shelf
        W_half, z0 = row.shelf_width / 2, row.target_board_top
        panels = tuple(x if any(self.panel_in_view(f, x) for f in frames) else None for x in (-W_half, W_half))
        whole_row = None not in panels and len(kept) == len(row.books)
        books, labels = self.name([s for _, s in kept], slot_x(row) if x_slot is None else x_slot, placed_x,
                                  n_left=row.target_index if whole_row else None)
        self.last_labels = labels

        pose = np.eye(4)
        pose[:3, :3], pose[:3, 3] = shelf.R, shelf.p
        return ShelfEstimate(
            shelf_pose=pose, board_top=z0, front_y=row.shelf_depth / 2, clear_height=row.clear_height,
            shelf_depth=row.shelf_depth, panel_x=panels, books=books, gap_heights=self.gap_heights.copy(),
            gap_widths=gap_profile(books, panels, z0, self.gap_heights), stamp=frames[-1].stamp, source="rgbd")

    def sightings(self, frame: CameraFrame) -> List[Sighting]:
        """Steps 1-4 for one image: the spine of every book in it."""
        row, shelf = self.row, self.shelf
        xyz_cam = organized_points(frame)
        R_sc = shelf.R.T @ frame.pose[:3, :3]                     # camera -> shelf rotation
        xyz = shelf.to_shelf(xyz_cam @ frame.pose[:3, :3].T + frame.pose[:3, 3])
        normals = surface_normals(organized_points(smoothed(frame))) @ R_sc.T
        facing = normals[..., 1] > FACING_COS
        W_half, z0 = row.shelf_width / 2, row.target_board_top
        on_level = ((np.abs(xyz[..., 0]) < W_half) & (xyz[..., 1] > -row.shelf_depth / 2)
                    & (xyz[..., 1] < row.shelf_depth / 2 + 0.05)
                    & (xyz[..., 2] > z0 - 0.002) & (xyz[..., 2] < z0 + row.clear_height))
        robot = robot_mask(self.m, frame)
        f = frame.K[0, 0]
        cam_x = shelf.to_shelf(frame.pose[:3, 3])[0]

        out = []
        for seg in self.segment(frame):
            keep = seg.mask & facing & on_level & ~robot
            if keep.sum() < MIN_POINTS:
                continue
            pixel_size = float(np.median(xyz_cam[keep][:, 2])) / f
            corners = fit_spine(xyz[keep], normals[keep], pixel_size, grow_px=NORMAL_STEP)
            if corners is not None:
                off_axis = abs((corners[0, 0] + corners[1, 0]) / 2 - cam_x)
                out.append(Sighting(corners, boundary_visible(seg.mask, robot), off_axis, seg.label))
        return out

    def name(self, seen: List[Sighting], x_slot: float, placed_x: Optional[float], n_left: Optional[int] = None):
        """Step 5: BookEstimates named by where their bottoms stand relative
        to the slot, left to right, and their segment labels. The slot is
        after the first n_left books if given, else the boundary between
        adjacent books closest to x_slot; or, if a book is seen within half
        its thickness of placed_x, that book (offset 0)."""
        seen = sorted(seen, key=lambda s: s.x)
        bottoms = [s.x for s in seen]
        placed = None
        if placed_x is not None and seen:
            i = int(np.argmin([abs(x - placed_x) for x in bottoms]))
            if abs(bottoms[i] - placed_x) < np.linalg.norm(seen[i].corners[1] - seen[i].corners[0]) / 2:
                placed = i
        if placed is not None:
            offsets = [i - placed for i in range(len(seen))]
        else:
            if n_left is None:
                if not bottoms or x_slot < bottoms[0]:
                    n_left = 0
                elif x_slot > bottoms[-1]:
                    n_left = len(bottoms)
                else:
                    boundaries = [(a + b) / 2 for a, b in zip(bottoms, bottoms[1:])]
                    n_left = 1 + int(np.argmin([abs(x - x_slot) for x in boundaries]))
            offsets = [i - n_left if i < n_left else i - n_left + 1 for i in range(len(seen))]
        books = [book_from_spine(s.corners, k, self.depth, s.visible) for s, k in zip(seen, offsets)]
        return books, [s.label for s in seen]

    def panel_in_view(self, frame: CameraFrame, x: float) -> bool:
        """Whether the side panel's inner face at x (shelf frame) is in the
        image, at the spines and halfway up the level. Occluded by books or
        not: if the books next to it are in view, the gap can be measured to it."""
        row = self.row
        p = self.shelf.to_world(np.array([x, row.shelf_depth / 2 - 0.02, row.target_board_top + row.clear_height / 2]))
        u, v, z = project(frame, p)
        W, H = frame.size
        return bool(z[0] > 0 and 0 <= u[0] < W and 0 <= v[0] < H)

    def close(self):
        self.camera.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
