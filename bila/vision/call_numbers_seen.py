"""
Call numbers read off the spines' stickers in the wrist camera's images, and
the slot they put the held book in: between the two books whose numbers
bracket its own, as a librarian finds it.

    numbers, texts = catalogue(row)                   # what the library knows: every book's number and sticker text
    reader = OracleStickers(model, numbers)           # or LearnedStickers("~/bila_data/labels", texts)
    eyes = ReadingEyes(make_eyes(model, row, seed), reader, row, numbers[row.target_book.name])
    Episode(model, data, row, eyes=eyes).reshelve()   # its first look reads the row and picks the slot

The scene needs stickers: insert_book.make_episode(..., stickers=True).

A sticker read is a call number and where the sticker is along the row (the
median of the depth image's points on it, shelf frame), no book identity: the
slot is the midpoint between the two bracketing stickers, and RGBDPerception
names the books from there (read_slot), in place of the shelf map's place
for the slot. Only the first look decides; later looks follow the slot as
the controller tracks it. If the numbers on both sides can't be read, the
shelf map's place is kept.

  OracleStickers   a sticker is read if at least MIN_STICKER_PX of it shows;
                   its number comes from the simulation (cart_demo's rule).
  LearnedStickers  label_reader.LabelReader finds and reads them in the
                   colour image (train it first: label_reader.py).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import mujoco
import numpy as np

from bila.vision.camera import CameraFrame, points
from bila.control.insert_book import ShelfFrame

MIN_STICKER_PX = 200        # oracle: a sticker showing fewer pixels than this (1280 x 720) can't be read


@dataclass
class Sticker:
    number: int
    x: float                 # along the row, shelf frame (m)
    corners: np.ndarray      # (4, 2) px in its image, for drawing


def catalogue(row) -> Tuple[Dict[str, int], Dict[str, int]]:
    """({book name: call number}, {sticker text: call number}) for every book
    in the bookcase, the held one included."""
    from bila.control.cart_demo import call_numbers, cutter, sticker_text
    numbers = call_numbers(row)[0]
    books = row.books + [row.target_book] + [b for fl in row.filler_levels for b in fl.books]
    return numbers, {sticker_text(numbers[b.name], cutter(b)): numbers[b.name] for b in books if b.name in numbers}


class OracleStickers:
    """Stickers read off the simulation's object ids: [(number, mask, corners)]."""

    def __init__(self, model: mujoco.MjModel, numbers: Dict[str, int]):
        self.number_of = {}
        for name, n in numbers.items():
            g = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name + "_label")
            if g >= 0:
                self.number_of[g] = n

    def __call__(self, frame: CameraFrame):
        ids, counts = np.unique(frame.seg, return_counts=True)
        out = []
        for g, c in zip(ids, counts):
            if int(g) in self.number_of and c >= MIN_STICKER_PX:
                mask = frame.seg == g
                v, u = np.nonzero(mask)
                corners = np.array([[u.min(), v.min()], [u.max(), v.min()], [u.max(), v.max()], [u.min(), v.max()]])
                out.append((self.number_of[int(g)], mask, corners))
        return out


class LearnedStickers:
    """Stickers found and read by label_reader.LabelReader: [(number, mask, corners)]."""

    def __init__(self, weights: str, texts: Dict[str, int]):
        from bila.vision.label_reader import LabelReader                 # imports torch (torch._dynamo first, before any render)
        weights = os.path.expanduser(weights)
        self.reader = LabelReader(os.path.join(weights, "finder.pt"), os.path.join(weights, "reader.pt"),
                                  catalogue=list(texts))
        self.number_of = texts

    def __call__(self, frame: CameraFrame):
        out = []
        for r in self.reader.read(frame.rgb):
            if r.text is None:                               # found, but not readable
                continue
            mask = np.zeros(frame.depth.shape, np.uint8)
            cv2.fillConvexPoly(mask, np.round(r.corners).astype(np.int32), 1)
            out.append((self.number_of[r.text], mask > 0, r.corners))
        return out


def slot_between(stickers: List[Sticker], number: int) -> Optional[Tuple[Sticker, Sticker]]:
    """The two stickers read whose numbers bracket `number` (largest below,
    smallest above), or None if a side has none. One sticker per number:
    the same one read in several images counts once (its x averaged)."""
    by_number: Dict[int, List[Sticker]] = {}
    for s in stickers:
        by_number.setdefault(s.number, []).append(s)
    merged = [Sticker(n, float(np.mean([s.x for s in ss])), ss[0].corners) for n, ss in by_number.items()]
    below = [s for s in merged if s.number < number]
    above = [s for s in merged if s.number > number]
    if not below or not above:
        return None
    # ponytail: an unread book between the two leaves the slot at the boundary nearest their midpoint;
    # count the spines between them (RGBDPerception's sightings) if that turns out to misplace it.
    return max(below, key=lambda s: s.number), min(above, key=lambda s: s.number)


class ReadingEyes:
    """Episode's eyes (an RGBDPerception) that also read the stickers in every
    look, and on the first look put the slot where the numbers say."""

    def __init__(self, eyes, reader, row, number: int, log=print):
        self.eyes, self.reader, self.row, self.number, self.log = eyes, reader, row, number, log
        self.shelf = ShelfFrame(row)
        self.segment = eyes.segment
        self.seen: List[List[Sticker]] = []     # the last look's stickers, per image
        self.slot: Optional[Tuple[Sticker, Sticker]] = None
        self.decided = False

    def read(self, frame: CameraFrame) -> List[Sticker]:
        """The stickers read in one image that are on the target level."""
        z0, out = self.row.target_board_top, []
        for number, mask, corners in self.reader(frame):
            xyz = points(frame, mask)
            if len(xyz) == 0:
                continue
            p = self.shelf.to_shelf(np.median(xyz, axis=0))
            if z0 < p[2] < z0 + self.row.clear_height:       # not a sticker on the level below or above
                out.append(Sticker(number, float(p[0]), np.asarray(corners)))
        return out

    def capture(self, data):
        return self.eyes.capture(data)

    def estimate_from_frames(self, frames, x_slot=None, placed_x=None):
        self.seen = [self.read(f) for f in frames]
        read_slot = None
        if not self.decided and x_slot is None and placed_x is None:        # the first look
            from bila.control.cart_demo import dewey
            self.decided = True
            stickers = [s for ss in self.seen for s in ss]
            self.slot = slot_between(stickers, self.number)
            if self.slot is None:
                self.log(f"         read {len({s.number for s in stickers})} call numbers, none on one side of "
                         f"{dewey(self.number)}: going by the shelf map")
            else:
                lo, hi = self.slot
                read_slot = (lo.x + hi.x) / 2
                self.log(f"         read {len({s.number for s in stickers})} call numbers: {dewey(self.number)} goes "
                         f"between {dewey(lo.number)} and {dewey(hi.number)}, at x = {1000 * read_slot:.0f} mm")
        return self.eyes.estimate_from_frames(frames, x_slot=x_slot, placed_x=placed_x, read_slot=read_slot)

    def close(self):
        self.eyes.close()
