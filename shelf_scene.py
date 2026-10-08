"""
MuJoCo bookshelf scene generator, ported from ContactExplorer's Isaac Gym
singulation task (src/tasks/allegro_singulation.py).

What it builds
  - A static bookcase standing on the floor: plinth, 3-5 shelf boards,
    top board, side panels and back panel (the single cubby of the Isaac
    Gym version, stacked into a full bookcase).
  - A target level: a row of free-moving books (boxes) with a target slot,
    in one of four scenario families: clear, tight, leaning, collapsed.
  - Filler rows on the other levels (static by default, for speed).
  - Optional Franka Panda (from MuJoCo Menagerie) with a wrist camera
    and a 6-axis force/torque sensor, plus a fixed scene camera.

Frame convention (same as the Isaac Gym task)
  x = along the row, y = depth (back panel at -y, open front at +y), z = up.
  The shelf body's origin is on the floor, at the centre of the bookcase.

Usage
  # 1) Just write the XML (no mujoco install needed for this step)
  python shelf_scene.py --scenario leaning --seed 3 --out scenes/

  # 2) Build, settle and check in Python (needs `pip install mujoco`)
  from shelf_scene import ShelfConfig, make_episode
  model, data, row, report = make_episode(
      ShelfConfig(), "collapsed", np.random.default_rng(0),
      panda_xml="mujoco_menagerie/franka_emika_panda/panda.xml")
"""

from __future__ import annotations

import argparse
import json
import math
import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field, asdict
from typing import List, Optional

import numpy as np

SCENARIOS = ("clear", "tight", "leaning", "collapsed")


# ---------------------------------------------------------------------------
# 1. Configuration: every randomized quantity has a (low, high) range.
#    REPO_DEFAULTS reproduces the original Isaac Gym cubby exactly.
# ---------------------------------------------------------------------------
@dataclass
class ShelfConfig:
    # Bookcase (inner dimensions, metres)
    width: tuple = (0.60, 1.00)          # repo: 0.344
    depth: tuple = (0.22, 0.32)          # repo: 0.26
    board_thickness: float = 0.02        # repo: _table_thickness
    num_levels: tuple = (3, 5)           # shelf boards above the plinth (ints)
    level_height: tuple = (0.25, 0.40)   # clear height of each level (repo: 0.53)
    plinth_height: tuple = (0.05, 0.10)  # kick board under the bottom board
    target_band: tuple = (0.40, 1.10)    # target board top must be this high off the floor
    fill_other_levels: tuple = (0.5, 1.0)  # how full the non-target rows are
    static_other_levels: bool = True     # freeze non-target books (fast); False = free bodies
    pose_jitter_xy: float = 0.01         # +-1 cm
    pose_jitter_yaw_deg: float = 2.0     # +-2 deg
    base_friction: tuple = (0.3, 0.8)    # shelf-board friction

    # Robot base: in front of the shelf; its height follows the target level
    # (like a lift), so the arm always sees the same geometry relative to it.
    robot_xy: tuple = (0.0, 0.62)
    robot_z_offset: float = -0.21        # robot base z minus target board top

    # Books (repo boxes: 0.04 x 0.16 x 0.24, density 500-1000)
    thickness: tuple = (0.015, 0.06)
    book_depth: tuple = (0.12, 0.22)
    book_height: tuple = (0.17, 0.30)
    density: tuple = (400.0, 900.0)      # kg/m^3
    book_friction: tuple = (0.25, 0.7)
    spacing: tuple = (0.0, 0.003)        # gap between neighbouring books
    end_slack: tuple = (0.03, 0.15)      # free space left at one row end

    # Contact settings randomized per episode
    solref_timeconst: tuple = (0.005, 0.02)

    # Scenario knobs
    clear_gap_factor: tuple = (1.2, 2.0)     # gap / target thickness
    tight_gap_factor: tuple = (0.6, 1.1)
    lean_angle_deg: tuple = (12.0, 35.0)     # leaning: how far the neighbour leans
    leaning_reserve: float = 0.18            # row space reserved for the leaning gap
    collapsed_reserve: float = 0.15          # row space reserved for the hole next to the run
    collapsed_run: tuple = (3, 6)            # books that topple into the hole


REPO_DEFAULTS = ShelfConfig(   # one cubby whose board sits 0.5 m up, as on the repo's table
    width=(0.344, 0.344), depth=(0.26, 0.26), num_levels=(1, 1), level_height=(0.53, 0.53),
    plinth_height=(0.49, 0.49), target_band=(0.0, 2.0),
    pose_jitter_xy=0.0, pose_jitter_yaw_deg=0.0, base_friction=(1.0, 1.0),
    thickness=(0.04, 0.04), book_depth=(0.16, 0.16), book_height=(0.24, 0.24),
    density=(500.0, 500.0), book_friction=(0.8, 0.8), spacing=(0.005, 0.005),
    end_slack=(0.0, 0.0),
)


# ---------------------------------------------------------------------------
# 2. Sampling a row
# ---------------------------------------------------------------------------
@dataclass
class Book:
    name: str
    thickness: float
    depth: float
    height: float
    mass: float
    friction: float
    x: float                  # centre along the row (shelf frame)
    rgba: tuple = (0.5, 0.5, 0.5, 1.0)


@dataclass
class FillerLevel:
    level: int                # which shelf board these books stand on
    books: List[Book]


@dataclass
class Row:
    scenario: str
    shelf_width: float
    shelf_depth: float
    clear_height: float       # clear height of the target level
    base_friction: float
    shelf_pos: tuple          # world position of the bookcase origin (on the floor)
    shelf_yaw: float          # radians
    plinth_height: float
    level_heights: List[float]  # clear height of each level, bottom to top
    target_level: int         # index into level_heights
    target_board_top: float   # height of the target board's top surface, from the floor
    solref_timeconst: float
    books: List[Book]
    target_index: int         # the new book goes between books[i-1] and books[i]
    target_book: Book         # the book the robot holds (not placed in the row)
    gap_left: float           # x of the gap's left edge (shelf frame)
    gap_right: float          # x of the gap's right edge
    nudges: List[dict] = field(default_factory=list)  # pushes applied while settling
    filler_levels: List[FillerLevel] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)


def _u(rng, lohi):
    lo, hi = lohi
    return float(lo if lo == hi else rng.uniform(lo, hi))


def _make_book(cfg, rng, name):
    t = _u(rng, cfg.thickness)
    d = _u(rng, cfg.book_depth)
    h = _u(rng, cfg.book_height)
    rho = _u(rng, cfg.density)
    colour = tuple(float(c) for c in rng.uniform(0.15, 0.9, 3)) + (1.0,)
    return Book(name, t, d, h, rho * t * d * h, _u(rng, cfg.book_friction), 0.0, colour)


def board_tops(cfg: ShelfConfig, plinth: float, level_heights: List[float]) -> List[float]:
    """Height of each shelf board's top surface, from the floor."""
    t, z, tops = cfg.board_thickness, plinth, []
    for h in level_heights:
        tops.append(z + t)
        z += t + h
    return tops


def sample_levels(cfg: ShelfConfig, rng: np.random.Generator, tries: int = 50):
    """Pick the number of levels, their heights, and a target level whose
    board top lies in cfg.target_band (within the robot's reach)."""
    for _ in range(tries):
        n = int(rng.integers(cfg.num_levels[0], cfg.num_levels[1] + 1))
        heights = [_u(rng, cfg.level_height) for _ in range(n)]
        plinth = _u(rng, cfg.plinth_height)
        tops = board_tops(cfg, plinth, heights)
        ok = [k for k, z in enumerate(tops) if cfg.target_band[0] <= z <= cfg.target_band[1]]
        if ok:
            return plinth, heights, int(rng.choice(ok)), tops
    raise ValueError("No level falls inside target_band; widen it or change level_height")


def sample_filler(cfg: ShelfConfig, rng: np.random.Generator, level: int,
                  H: float, W: float, D: float) -> FillerLevel:
    """An ordinary upright row on a non-target level, packed against a random
    side panel and filled to a random fraction of the width."""
    fill = _u(rng, cfg.fill_other_levels)
    books, used = [], 0.0
    while True:
        b = _make_book(cfg, rng, f"l{level}_book_{len(books)}")
        b.height = min(b.height, H - 0.02)
        b.depth = min(b.depth, D - 0.01)
        s = _u(rng, cfg.spacing)
        if used + b.thickness + s > fill * W:
            break
        books.append(b)
        used += b.thickness + s
    from_left = bool(rng.integers(0, 2))
    x = -W / 2 if from_left else W / 2 - used
    for b in books:
        b.x = x + b.thickness / 2
        x += b.thickness + _u(rng, cfg.spacing)
    return FillerLevel(level, books)


def sample_row(cfg: ShelfConfig, scenario: str, rng: np.random.Generator) -> Row:
    """Sample one bookcase + target row of books for a scenario.

    Books are laid out left to right. The gap for the target book is placed
    somewhere in the middle of the row. Leaning and collapsed rows are made
    physically: the books start upright and are pushed during settling
    (see `nudges`), so every lean is a real, stable resting pose.
    """
    assert scenario in SCENARIOS, scenario
    W = _u(rng, cfg.width)
    D = _u(rng, cfg.depth)
    plinth, level_heights, target_level, tops = sample_levels(cfg, rng)
    H = level_heights[target_level]
    target = _make_book(cfg, rng, "target_book")
    target.height = min(target.height, H - 0.03)
    target.depth = min(target.depth, D - 0.01)

    # Gap width at the target slot.
    if scenario == "clear":
        gap = target.thickness * _u(rng, cfg.clear_gap_factor)
    elif scenario == "tight":
        gap = target.thickness * _u(rng, cfg.tight_gap_factor)
    elif scenario == "leaning":
        gap = cfg.leaning_reserve          # final width is set below, from the lean angle
    else:  # collapsed: the slot is closed, but a hole opens next to the run
        gap = 0.0
    hole = cfg.collapsed_reserve if scenario == "collapsed" else 0.0   # final width set below

    # Fill the row with books until the shelf is (nearly) full.
    # Any space left over ends up as extra room at the far end of the row.
    slack = _u(rng, cfg.end_slack)
    budget = W - gap - hole - slack
    books: List[Book] = []
    used = 0.0
    while True:
        b = _make_book(cfg, rng, f"book_{len(books)}")
        b.height = min(b.height, H - 0.02)       # must fit under the top board
        b.depth = min(b.depth, D - 0.01)
        s = _u(rng, cfg.spacing)
        if used + b.thickness + s > budget:
            break
        books.append(b)
        used += b.thickness + s
    n = len(books)
    if n < 4:
        raise ValueError("Shelf too narrow for this sample; widen ShelfConfig.width")

    # Target slot roughly in the middle third of the row.
    ti = int(rng.integers(max(1, n // 3), max(2, 2 * n // 3) + 1))
    run = int(rng.integers(cfg.collapsed_run[0], cfg.collapsed_run[1] + 1)) if scenario == "collapsed" else 0
    run = max(1, min(run, n - ti - 2))   # keep >= 2 books after the hole to catch the run

    # Leaning: size the gap so the neighbour comes to rest at the chosen angle.
    # A book only stays leaning if its centre of mass passes the bottom edge it
    # pivots on, i.e. tan(angle) > thickness / height (we require 1.5x that).
    # The book pivots on its gap-side bottom corner, so its top corner moves
    # h*sin(a) sideways: that is the gap needed to rest at angle a.
    sides = []
    if scenario == "leaning":
        # One neighbour leans. (Pushing both at once tested badly: one falls flat
        # over the other. Use "collapsed" for several leaning books.)
        sides = [int(rng.choice([-1, 1]))]
        angle = math.radians(_u(rng, cfg.lean_angle_deg))

        def lean_gap(side):
            b = books[ti - 1 if side == -1 else ti]          # the leaner
            far = books[ti if side == -1 else ti - 1]        # what it leans on
            a_min = math.atan(1.5 * b.thickness / b.height)  # else it springs back upright
            # Its top must touch the far book below that book's top edge, or it tips over it.
            a_min = max(a_min, math.acos(min(1.0, 0.95 * far.height / b.height)))
            return b.height * math.sin(max(angle, a_min))

        if lean_gap(sides[0]) > cfg.leaning_reserve:          # try the other neighbour
            sides = [-sides[0]]
        gap = min(lean_gap(sides[0]), cfg.leaning_reserve)

    # Collapsed: the same rule sizes the hole the run falls into. Every book in
    # the run must lean far enough not to spring back, and the last one must
    # land on the book across the hole below that book's top (or the side panel).
    if scenario == "collapsed":
        angle = math.radians(_u(rng, cfg.lean_angle_deg))
        run_books = books[ti:ti + run]
        last = run_books[-1]
        a_min = max(math.atan(1.5 * b.thickness / b.height) for b in run_books)
        if ti + run < n:
            far = books[ti + run]
            a_min = max(a_min, math.acos(min(1.0, 0.95 * far.height / last.height)))
        hole = min(last.height * math.sin(max(angle, a_min)), cfg.collapsed_reserve)

    # Lay books out. Slack (spare row space) goes to one row end. When books
    # are tipped over, it goes to the end away from where they fall, so the
    # books that catch them are packed against a side panel and can't slide away.
    if scenario == "collapsed":
        slack_left = True                  # run falls toward +x
    elif scenario == "leaning":
        slack_left = sides[0] == -1        # left neighbour leans right -> keep the right side packed
    else:
        slack_left = bool(rng.integers(0, 2))
    x = -W / 2 + (slack if slack_left else 0.0)
    gap_left = gap_right = None
    for i, b in enumerate(books):
        if i == ti:
            gap_left = x
            x += gap
            gap_right = x
        if scenario == "collapsed" and i == ti + run:
            x += hole                       # hole after the run; run will fall into it
        b.x = x + b.thickness / 2
        x += b.thickness + _u(rng, cfg.spacing)

    jitter = cfg.pose_jitter_xy
    pos = (float(rng.uniform(-jitter, jitter)) if jitter else 0.0,
           float(rng.uniform(-jitter, jitter)) if jitter else 0.0,
           0.0)                                    # bookcase stands on the floor
    yaw = math.radians(float(rng.uniform(-cfg.pose_jitter_yaw_deg, cfg.pose_jitter_yaw_deg))) \
        if cfg.pose_jitter_yaw_deg else 0.0

    # Pushes applied while settling: a sideways force at the top of a book,
    # stored as force + torque at its centre (world frame). The force is
    # 3x what it takes to tip that book over its bottom edge.
    def nudge(idx, direction, duration):
        b = books[idx]
        f = 3.0 * b.mass * 9.81 * b.thickness / b.height
        fx, fy = direction * f * math.cos(yaw), direction * f * math.sin(yaw)
        r = b.height / 2
        return {"book": idx, "body": b.name, "force": [fx, fy, 0.0],
                "torque": [-r * fy, r * fx, 0.0], "start": 0.05, "duration": duration}

    nudges = []
    if scenario == "leaning":
        for side in sides:
            nudges.append(nudge(ti - 1 if side == -1 else ti, -side, 0.25))
    elif scenario == "collapsed":
        for k in range(run):        # domino: the run tips toward the hole (+x)
            nudges.append(nudge(ti + k, +1, 0.35))

    fillers = [sample_filler(cfg, rng, k, h, W, D)
               for k, h in enumerate(level_heights) if k != target_level]

    return Row(scenario, W, D, H, _u(rng, cfg.base_friction), pos, yaw,
               plinth, level_heights, target_level, tops[target_level],
               _u(rng, cfg.solref_timeconst), books, ti, target,
               float(gap_left), float(gap_right), nudges, fillers)


# ---------------------------------------------------------------------------
# 3. Panda with wrist camera and F/T sensor (a modified copy of Menagerie's)
# ---------------------------------------------------------------------------
def make_panda_with_sensors(panda_xml: str) -> str:
    """Write `panda_shelf.xml` next to Menagerie's panda.xml (so mesh paths
    still resolve) with a wrist camera, an F/T sensor site and no keyframe.
    Returns the new file's path."""
    tree = ET.parse(panda_xml)
    root = tree.getroot()
    for kf in root.findall("keyframe"):
        root.remove(kf)                       # the scene writes its own keyframe
    hand = root.find(".//body[@name='hand']")
    if hand is None:
        raise RuntimeError("No body named 'hand' in " + panda_xml)
    # F/T sensor site at the flange (hand origin). MuJoCo's force/torque
    # sensors report the wrench between the hand subtree and link7.
    ET.SubElement(hand, "site", name="ft_site", pos="0 0 0", size="0.005", rgba="1 0 0 0.3", group="4")
    # Wrist camera (RealSense D405-like): 9 cm off-axis, looking along the
    # fingers and pitched ~15 deg inward (placement checked in renders). MuJoCo cameras look down their -z.
    ET.SubElement(hand, "camera", name="wrist_cam", pos="0.09 0 0.08",
                  xyaxes="0 1 0 0.97 0 0.26", fovy="58")
    sensor = root.find("sensor")
    if sensor is None:
        sensor = ET.SubElement(root, "sensor")
    ET.SubElement(sensor, "force", name="ft_force", site="ft_site")
    ET.SubElement(sensor, "torque", name="ft_torque", site="ft_site")
    out = os.path.join(os.path.dirname(os.path.abspath(panda_xml)), "panda_shelf.xml")
    tree.write(out)
    return out


PANDA_HOME = [0, -0.6, 0, -2.6, 0, 2.0, 0.7853, 0.04, 0.04]  # tucked 'ready' pose: hand ~10 cm in front of the shelf.
# (Menagerie's own 'home' pose reaches 0.55 m forward and hits the shelf's top board.)


# ---------------------------------------------------------------------------
# 4. MJCF builder
# ---------------------------------------------------------------------------
def _f(*v):
    return " ".join(f"{x:.6g}" for x in v)


def build_scene(cfg: ShelfConfig, row: Row, panda_xml: Optional[str] = None) -> str:
    """Return a complete MJCF string. If `panda_xml` (Menagerie's panda.xml)
    is given, the Panda is attached in front of the shelf facing -y.
    The XML uses an absolute path for the robot, so it can be loaded from
    any directory with mujoco.MjModel.from_xml_string / from_xml_path."""
    t = cfg.board_thickness
    W, D, H = row.shelf_width, row.shelf_depth, row.clear_height
    tops = board_tops(cfg, row.plinth_height, row.level_heights)
    target_top = tops[row.target_level]               # in shelf-body frame (= from the floor)
    tr = row.solref_timeconst
    robot_pos = (cfg.robot_xy[0], cfg.robot_xy[1], target_top + cfg.robot_z_offset)

    m = ET.Element("mujoco", model=f"bookshelf_{row.scenario}")
    ET.SubElement(m, "compiler", angle="radian", autolimits="true")
    ET.SubElement(m, "option", timestep="0.002", integrator="implicitfast",
                  cone="elliptic", impratio="10")
    ET.SubElement(m, "size", memory="64M")
    vis = ET.SubElement(m, "visual")
    ET.SubElement(vis, "global", offwidth="1280", offheight="960")
    ET.SubElement(vis, "headlight", ambient="0.4 0.4 0.4", diffuse="0.6 0.6 0.6")

    d = ET.SubElement(m, "default")
    dd = ET.SubElement(d, "default", **{"class": "book"})
    # condim 4 adds torsional friction so books don't spin on the base.
    ET.SubElement(dd, "geom", type="box", condim="4", solref=_f(tr, 1.0),
                  solimp="0.95 0.99 0.001", margin="0.0005")
    ds = ET.SubElement(d, "default", **{"class": "shelf"})
    ET.SubElement(ds, "geom", type="box", condim="4", rgba="0.55 0.4 0.25 1",
                  solref=_f(tr, 1.0), solimp="0.95 0.99 0.001")

    asset = ET.SubElement(m, "asset")
    ET.SubElement(asset, "texture", name="grid", type="2d", builtin="checker",
                  rgb1="0.85 0.85 0.85", rgb2="0.75 0.75 0.75", width="512", height="512")
    ET.SubElement(asset, "material", name="floor", texture="grid", texrepeat="8 8")
    if panda_xml:
        ET.SubElement(asset, "model", name="panda", file=make_panda_with_sensors(panda_xml))

    wb = ET.SubElement(m, "worldbody")
    ET.SubElement(wb, "light", pos="0 1.5 2.5", dir="0 -0.5 -1", diffuse="0.6 0.6 0.6")
    ET.SubElement(wb, "geom", name="floor", type="plane", size="3 3 0.05", material="floor")

    # Robot first, so its 9 joints come first in qpos.
    if panda_xml:
        rb = ET.SubElement(wb, "body", name="robot_mount", pos=_f(*robot_pos),
                           euler=_f(0, 0, -math.pi / 2))      # face -y, toward the shelf
        ET.SubElement(rb, "geom", type="cylinder", size=_f(0.09, robot_pos[2] / 2),
                      pos=_f(0, 0, -robot_pos[2] / 2), rgba="0.3 0.3 0.3 1",
                      group="2")                       # same visual group as the Panda's meshes
        ET.SubElement(rb, "attach", model="panda", body="link0", prefix="panda/")

    # Shelf: one static body standing on the floor. The boards are the repo's
    # table/upper shelf stacked into levels; the panels are its walls, made
    # full height.
    sp = row.shelf_pos
    shelf = ET.SubElement(wb, "body", name="shelf", pos=_f(*sp), euler=_f(0, 0, row.shelf_yaw))
    fric = _f(row.base_friction, 0.005, 0.0001)
    for k, top in enumerate(tops):                     # shelf boards, bottom to top
        ET.SubElement(shelf, "geom", name=f"board_{k}", **{"class": "shelf"},
                      size=_f(W / 2, D / 2, t / 2), pos=_f(0, 0, top - t / 2), friction=fric)
    total_h = tops[-1] + row.level_heights[-1] + t     # top surface of the top board
    ET.SubElement(shelf, "geom", name="top_board", **{"class": "shelf"},
                  size=_f(W / 2 + t, D / 2, t / 2), pos=_f(0, 0, total_h - t / 2))
    for side, name in ((1, "side_panel_pos_x"), (-1, "side_panel_neg_x")):
        ET.SubElement(shelf, "geom", name=name, **{"class": "shelf"},
                      size=_f(t / 2, D / 2, (total_h - t) / 2),
                      pos=_f(side * (W / 2 + t / 2), 0, (total_h - t) / 2),
                      friction="1.0 0.005 0.0001")
    ET.SubElement(shelf, "geom", name="back_panel", **{"class": "shelf"},
                  size=_f(W / 2 + t, t / 2, total_h / 2), pos=_f(0, -D / 2 - t / 2, total_h / 2),
                  friction="1.0 0.005 0.0001")
    ET.SubElement(shelf, "geom", name="plinth", **{"class": "shelf"},
                  size=_f(W / 2, t / 2, row.plinth_height / 2),
                  pos=_f(0, D / 2 - t / 2, row.plinth_height / 2))
    # Fiducial marker: a flat, non-colliding plate on the front edge of the
    # target board. Swap rgba for an AprilTag texture when you add perception.
    ET.SubElement(shelf, "geom", name="fiducial", type="box", contype="0", conaffinity="0",
                  size="0.025 0.0005 0.025", pos=_f(-W / 2 + 0.04, D / 2 + 0.0006, target_top - t / 2),
                  rgba="0 0 0 1")
    # Sites mark the target level: the row frame and the centre of the gap.
    ET.SubElement(shelf, "site", name="shelf_frame", pos=_f(0, 0, target_top), size="0.01", group="4")
    ET.SubElement(shelf, "site", name="target_slot",
                  pos=_f((row.gap_left + row.gap_right) / 2, 0, target_top), size="0.008", rgba="0 1 0 0.6")
    # Fixed scene camera (RealSense D435-like): in front of the shelf and off
    # to one side, so the robot never blocks it. It aims at the target row.
    ET.SubElement(shelf, "body", name="view_target", pos=_f(0, 0, target_top + H / 3))
    ET.SubElement(wb, "camera", name="scene_cam", mode="targetbody", target="view_target",
                  pos=_f(sp[0] + 0.6, sp[1] + 0.9, sp[2] + target_top + H + 0.5), fovy="45")

    # Books: back face 5 mm from the back panel (like the repo's
    # object_near_edge=False layout, pushed back for a realistic shelf).
    c, s = math.cos(row.shelf_yaw), math.sin(row.shelf_yaw)

    def add_books(books, z_base, free):
        """Place books standing on the board whose top is at z_base. Free books
        get a freejoint and are returned (they need keyframe entries); books
        without a joint are welded in place and cost almost nothing to simulate."""
        placed = []
        for b in books:
            lx, ly, lz = b.x, -D / 2 + b.depth / 2 + 0.005, z_base + b.height / 2 + 0.0005
            wx, wy, wz = sp[0] + c * lx - s * ly, sp[1] + s * lx + c * ly, sp[2] + lz
            body = ET.SubElement(wb, "body", name=b.name, pos=_f(wx, wy, wz), euler=_f(0, 0, row.shelf_yaw))
            if free:
                ET.SubElement(body, "freejoint", name=b.name + "_joint")
                placed.append((wx, wy, wz))
            ET.SubElement(body, "geom", name=b.name + "_geom", **{"class": "book"},
                          size=_f(b.thickness / 2, b.depth / 2, b.height / 2), mass=f"{b.mass:.5g}",
                          friction=_f(b.friction, 0.005, 0.0001), rgba=_f(*b.rgba))
            ET.SubElement(body, "site", name=b.name + "_top", pos=_f(0, 0, b.height / 2), size="0.003", group="4")
        return placed

    # Target-level books first, so their qpos entries come right after the robot's.
    book_bodies = add_books(row.books, target_top, free=True)
    for fl in row.filler_levels:
        book_bodies += add_books(fl.books, tops[fl.level], free=not cfg.static_other_levels)

    # Keyframe "init": robot at home, books where we placed them.
    qpos = list(PANDA_HOME) if panda_xml else []
    qw, qz = math.cos(row.shelf_yaw / 2), math.sin(row.shelf_yaw / 2)
    for (wx, wy, wz) in book_bodies:
        qpos += [wx, wy, wz, qw, 0, 0, qz]
    kf = ET.SubElement(m, "keyframe")
    attrs = {"name": "init", "qpos": _f(*qpos)}
    if panda_xml:
        attrs["ctrl"] = _f(*PANDA_HOME[:7], 255)
    ET.SubElement(kf, "key", **attrs)

    ET.indent(m)
    return ET.tostring(m, encoding="unicode")


# ---------------------------------------------------------------------------
# 5. Settling and stability check (needs the `mujoco` Python package)
# ---------------------------------------------------------------------------
def book_tilt_deg(quat) -> float:
    """Angle between the book's long (z) axis and world up, in degrees."""
    w, x, y, z = quat
    zz = 1 - 2 * (x * x + y * y)               # z-component of the rotated z-axis
    return math.degrees(math.acos(max(-1.0, min(1.0, zz))))


def settle(model, data, row: Row, settle_time=1.5, check_time=5.0):
    """Reset to the 'init' keyframe, apply the scenario's nudges, let physics
    settle, then verify that nothing moves for `check_time` seconds.
    Returns a dict with per-book tilt and the largest drift during the check."""
    import mujoco
    mujoco.mj_resetDataKeyframe(model, data, model.key("init").id)
    ids = [model.body(b.name).id for b in row.books]          # target row (nudges index these)
    others = [model.body(b.name).id for fl in row.filler_levels for b in fl.books]
    check = ids + [i for i in others if model.body_jntnum[i] > 0]   # every book that can move
    dt = model.opt.timestep
    for step in range(int(settle_time / dt)):
        tnow = step * dt
        data.xfrc_applied[:] = 0
        for nd in row.nudges:
            if nd["start"] <= tnow < nd["start"] + nd["duration"]:
                data.xfrc_applied[ids[nd["book"]], :3] = nd["force"]
                data.xfrc_applied[ids[nd["book"]], 3:] = nd["torque"]
        mujoco.mj_step(model, data)
    data.xfrc_applied[:] = 0
    data.qvel[:] = 0                       # start the check from rest
    p0 = data.xpos[check].copy()
    drift = 0.0
    for _ in range(int(check_time / dt)):
        mujoco.mj_step(model, data)
        drift = max(drift, float(np.abs(data.xpos[check] - p0).max()))
    tilts = [book_tilt_deg(data.xquat[i]) for i in check]
    return {"stable": drift < 2e-3 and max(tilts) < 60, "max_drift_m": drift,
            "tilts_deg": tilts, "max_tilt_deg": max(tilts)}


def make_episode(cfg: ShelfConfig, scenario: str, rng: np.random.Generator,
                 panda_xml: Optional[str] = None, max_tries: int = 10):
    """Sample rows until one settles into a stable pose (about 1 in 10
    leaning/collapsed rows keeps sliding slowly and is thrown away).
    Returns (model, data, row, report); `data` holds the settled state, so
    save `data.qpos` as your episode's start state (or a reset distribution)."""
    import mujoco
    for _ in range(max_tries):
        row = sample_row(cfg, scenario, rng)
        model = mujoco.MjModel.from_xml_string(build_scene(cfg, row, panda_xml))
        data = mujoco.MjData(model)
        rep = settle(model, data, row)
        if rep["stable"]:
            return model, data, row, rep
    raise RuntimeError(f"No stable {scenario} row in {max_tries} tries; loosen the config")


# ---------------------------------------------------------------------------
# CLI: write one XML + JSON per scenario
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenario", choices=SCENARIOS + ("all",), default="all")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--panda", help="path to mujoco_menagerie/franka_emika_panda/panda.xml")
    ap.add_argument("--repo-defaults", action="store_true",
                    help="use the original Isaac Gym cubby and box size (0.344 m wide, 4x16x24 cm boxes)")
    ap.add_argument("--out", default="scenes")
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    cfg = REPO_DEFAULTS if a.repo_defaults else ShelfConfig()
    scen = SCENARIOS if a.scenario == "all" else (a.scenario,)
    for sc in scen:
        rng = np.random.default_rng(a.seed)
        row = sample_row(cfg, "clear" if a.repo_defaults else sc, rng)
        xml = build_scene(cfg, row, panda_xml=a.panda)
        base = os.path.join(a.out, f"{sc}_seed{a.seed}")
        with open(base + ".xml", "w") as fh:
            fh.write(xml)
        with open(base + ".json", "w") as fh:
            fh.write(row.to_json())
        print(f"{sc:9s} {len(row.level_heights)} levels, target level {row.target_level} "
              f"(board top {row.target_board_top:.2f} m), "
              f"{len(row.books):2d} books, shelf {row.shelf_width:.2f} m, "
              f"gap {1000*(row.gap_right-row.gap_left):.0f} mm (target {1000*row.target_book.thickness:.0f} mm) -> {base}.xml")
        try:
            import mujoco
            model = mujoco.MjModel.from_xml_string(xml)
            rep = settle(model, mujoco.MjData(model), row)
            print(f"          settled: stable={rep['stable']} drift={1000*rep['max_drift_m']:.2f} mm "
                  f"max tilt={rep['max_tilt_deg']:.1f} deg")
        except ImportError:
            pass


if __name__ == "__main__":
    main()
