"""
MuJoCo bookshelf scene generator, ported from ContactExplorer's Isaac Gym
singulation task (src/tasks/allegro_singulation.py).

What it builds
  - A static bookcase standing on the floor: plinth, 3-5 shelf boards,
    top board, side panels and back panel (the single cubby of the Isaac
    Gym version, stacked into a full bookcase).
  - A target level: a row of free-moving books (boxes) with a target slot,
    in one of five scenario families: clear, tight, leaning, collapsed, packed.
  - Filler rows on the other levels (static by default, for speed).
  - Optional Franka Panda (from MuJoCo Menagerie) with thin 10 mm fingers,
    a wrist camera and a 6-axis force/torque sensor, plus a fixed scene camera.
  - Optional book covers: run dae_to_obj.py once to convert the repo's Collada
    book meshes, then set ShelfConfig(book_meshes=True) to draw them on the
    boxes as non-colliding visuals. Physics stays on the boxes either way.
  - Optional articulated books: ShelfConfig(book_pages=N) splits every book's
    pages into N sections that hinge on the spine, as in the repo's
    urdf/articulated_*.xacro, so the fore edges can fan open.
  - Optional real books: ShelfConfig(catalogue=True) fills the rows with the
    repo's books (CATALOGUE), each with its own size, mass and cover, and the
    Voynich and Lagemann books hinged into their own page sections.

Frame convention (same as the Isaac Gym task)
  x = along the row, y = depth (back panel at -y, open front at +y), z = up.
  The shelf body's origin is on the floor, at the centre of the bookcase.

Usage
  # 1) Just write the XML (no mujoco install needed for this step)
  python -m bila.sim.shelf_scene --scenario leaning --seed 3 --out scenes/

  # 2) Build, settle and check in Python (needs `pip install mujoco`)
  from bila.sim.shelf_scene import ShelfConfig, make_episode
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
import zlib
from dataclasses import dataclass, field, asdict
from typing import List, Optional

import numpy as np

from bila import ROOT

SCENARIOS = ("clear", "tight", "leaning", "collapsed", "packed")


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
    # Placed so the hand can point level into the shelf (needed to slide a
    # book in): the Panda can only do that with its hand about 0.5 m or more
    # above its base, and at 0.35 to 0.9 m in front of it.
    robot_xy: tuple = (0.0, 0.80)
    robot_z_offset: float = -0.40        # robot base z minus target board top (keeps the base above the floor for target_band >= 0.40)

    # Books (repo boxes: 0.04 x 0.16 x 0.24, density 500-1000)
    thickness: tuple = (0.015, 0.06)
    book_depth: tuple = (0.12, 0.22)
    depth_clearance: float = 0.030       # every book at least this much shallower than the shelf, adjustable
    spine_inset: tuple = (0.010, 0.030)  # each spine this far behind the shelf's front edge, adjustable
    book_height: tuple = (0.17, 0.30)
    density: tuple = (400.0, 900.0)      # kg/m^3
    book_friction: tuple = (0.25, 0.7)
    spacing: tuple = (0.0, 0.003)        # gap between neighbouring books
    end_slack: tuple = (0.03, 0.15)      # free space left at one row end
    book_meshes: bool = False            # draw the repo's book covers (dae_to_obj.py) on the boxes
    book_pages: int = 0                  # split every book's pages into this many hinged sections (0 or 1 = one rigid box)
    catalogue: bool = False              # real books from the repo (CATALOGUE: size, mass, cover, pages) instead of
                                         # sampled ones; thickness/book_depth/book_height/density/book_pages then unused

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
    end_slack=(0.0, 0.0), spine_inset=(0.095, 0.095),   # boxes 5 mm off the back panel
)


# ---------------------------------------------------------------------------
# 2. Sampling a row
# ---------------------------------------------------------------------------
@dataclass
class PageSection:
    """One section of a book's pages: the pages between two hinges. Every
    section hinges on the spine (see add_book_parts), so the fore edges can fan
    open. The repo splits its articulated books this way
    (urdf/articulated_voynich.xacro), one section per block of pages."""
    name: str                 # unique; its hinges and geoms are named after it
    thickness: float
    mass: float


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
    inset: float = 0.0        # spine this far behind the shelf's front edge
    parts: List[PageSection] = field(default_factory=list)  # [] = one rigid box
    asset: str = ""           # converted mesh it wears (CATALOGUE books); "" = picked from its name


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


def _page_sections(name: str, thickness: float, mass: float, count: int) -> List[PageSection]:
    """Split a book's pages into `count` sections of equal thickness, from the
    front cover to the back (front cover first, as the repo numbers its parts).
    A section's share of the book is the pages in it, so equal sections carry
    equal mass, as the repo's page-counted books do."""
    return [PageSection(f"{name}_p{i + 1}", thickness / count, mass / count) for i in range(count)]


# The repo's books (tams_books_world/urdf/book_*.xacro and books_demo.xacro), as
# (mesh, thickness, depth front to back, height, kg). The repo's xsize is our
# depth and its ysize our thickness. Its book_memory.xacro and book_rojas.xacro
# reuse the goedel mesh; books_demo.xacro gives them their own, used here.
CATALOGUE = [
    ("book-ai86", 0.029, 0.153, 0.230, 0.828),
    ("book-bits", 0.010, 0.148, 0.207, 0.500),
    ("book-bobh", 0.010, 0.126, 0.184, 0.210),
    ("book-dyson", 0.018, 0.130, 0.198, 0.228),
    ("book-goedel", 0.055, 0.162, 0.234, 1.100),
    ("book-hamburg", 0.005, 0.108, 0.185, 0.062),
    ("book-haykin", 0.032, 0.177, 0.230, 1.316),
    ("book-ifip", 0.022, 0.150, 0.225, 0.828),
    ("book-javaff", 0.034, 0.157, 0.235, 0.906),
    ("book-jekyll", 0.007, 0.110, 0.180, 0.056),
    ("book-khoros", 0.025, 0.173, 0.213, 0.700),
    ("book-knuth", 0.037, 0.167, 0.240, 1.080),
    ("book-knuth-40", 0.014, 0.161, 0.240, 0.400),
    ("book-knuth-41", 0.017, 0.161, 0.240, 0.498),
    ("book-knuth-42", 0.009, 0.161, 0.240, 0.274),
    ("book-lakoff", 0.035, 0.168, 0.240, 0.977),
    ("book-logic", 0.020, 0.173, 0.245, 0.623),
    ("book-lskm", 0.024, 0.217, 0.250, 0.978),
    ("book-mead-conway", 0.025, 0.200, 0.243, 0.972),
    ("book-memory", 0.028, 0.148, 0.227, 0.666),
    ("book-minsky", 0.019, 0.215, 0.280, 0.764),
    ("book-moeller", 0.023, 0.153, 0.236, 0.592),
    ("book-pdp1", 0.032, 0.150, 0.228, 0.897),
    ("book-pdp2", 0.036, 0.150, 0.228, 1.008),
    ("book-ri53", 0.012, 0.124, 0.190, 0.280),
    ("book-rice", 0.045, 0.122, 0.171, 0.528),
    ("book-rojas", 0.030, 0.155, 0.232, 0.666),
    ("book-schmidt", 0.038, 0.200, 0.275, 1.764),
    ("book-selfish", 0.020, 0.130, 0.196, 0.246),
    ("book-titan", 0.058, 0.103, 0.153, 0.828),
    ("book-ttl2", 0.046, 0.148, 0.210, 1.370),
    ("book-twki", 0.017, 0.125, 0.195, 0.283),
    ("book-watson", 0.010, 0.115, 0.190, 0.114),
    ("book-weizen", 0.020, 0.108, 0.175, 0.300),
    ("book-wiering", 0.035, 0.155, 0.235, 0.964),
    ("book-winograd", 0.037, 0.168, 0.240, 0.777),
    ("book-zuse", 0.021, 0.172, 0.247, 0.764),
]
# The repo's articulated books (urdf/articulated_*.xacro): (mesh, depth, height,
# [(thickness, kg) per page section, front cover first]). Their hinges come
# from add_book_parts. The caterpillar (31 cm deep) fits no shelf here.
ARTICULATED = [
    ("book-voynich", 0.160, 0.230, [(0.023, 0.80), (0.014, 0.50), (0.008, 0.28), (0.012, 0.42)]),
    ("book-lagemann", 0.165, 0.242, [(0.003, 0.092), (0.002, 0.055), (0.003, 0.092), (0.009, 0.313)]),
]


def _catalogue_book(cfg, rng, name, max_h, max_d, target):
    """A copy of one of the repo's books (picked uniformly, so a row can hold
    the same book more than once) that fits under max_h and within max_d. The
    target book also keeps to cfg.thickness, the range the scenarios and the
    fingers are tuned for. Friction and colour are still sampled."""
    books = [(m, t, d, h, kg, []) for m, t, d, h, kg in CATALOGUE] + \
            [(m, sum(t for t, _ in p), d, h, sum(kg for _, kg in p), p) for m, d, h, p in ARTICULATED]
    fits = [e for e in books if e[3] <= max_h and e[2] <= max_d
            and (not target or cfg.thickness[0] <= e[1] <= cfg.thickness[1])]
    mesh, t, d, h, kg, sections = fits[int(rng.integers(len(fits)))]
    colour = tuple(float(c) for c in rng.uniform(0.15, 0.9, 3)) + (1.0,)
    b = Book(name, t, d, h, kg, _u(rng, cfg.book_friction), 0.0, colour, asset=mesh)
    b.parts = [PageSection(f"{name}_p{i + 1}", pt, pm) for i, (pt, pm) in enumerate(sections)]
    return b


def _make_book(cfg, rng, name, max_h=math.inf, max_d=math.inf, target=False):
    if cfg.catalogue:
        return _catalogue_book(cfg, rng, name, max_h, max_d, target)
    t = _u(rng, cfg.thickness)
    d = _u(rng, cfg.book_depth)
    h = _u(rng, cfg.book_height)
    rho = _u(rng, cfg.density)
    colour = tuple(float(c) for c in rng.uniform(0.15, 0.9, 3)) + (1.0,)
    b = Book(name, t, d, h, rho * t * d * h, _u(rng, cfg.book_friction), 0.0, colour)
    if cfg.book_pages > 1:                     # one section is just a rigid book
        b.parts = _page_sections(name, b.thickness, b.mass, cfg.book_pages)
    return b


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
        b = _make_book(cfg, rng, f"l{level}_book_{len(books)}", H - 0.02, D - cfg.depth_clearance)
        b.height = min(b.height, H - 0.02)
        b.depth = min(b.depth, D - cfg.depth_clearance)
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
    target = _make_book(cfg, rng, "target_book", H - 0.03, D - cfg.depth_clearance, target=True)
    target.height = min(target.height, H - 0.03)
    target.depth = min(target.depth, D - cfg.depth_clearance)

    # Gap width at the target slot.
    if scenario == "clear":
        gap = target.thickness * _u(rng, cfg.clear_gap_factor)
    elif scenario == "tight":
        gap = target.thickness * _u(rng, cfg.tight_gap_factor)
    elif scenario == "leaning":
        gap = cfg.leaning_reserve          # final width is set below, from the lean angle
    else:  # collapsed: the slot is closed, but a hole opens next to the run.
        gap = 0.0                          # packed: the slot's neighbours touch, upright
    hole = cfg.collapsed_reserve if scenario == "collapsed" else 0.0   # final width set below

    # Fill the row with books until the shelf is (nearly) full.
    # Any space left over ends up as extra room at the far end of the row.
    slack = _u(rng, cfg.end_slack)
    budget = W - gap - hole - slack
    books: List[Book] = []
    used = 0.0
    while True:
        b = _make_book(cfg, rng, f"book_{len(books)}", H - 0.02, D - cfg.depth_clearance)
        b.height = min(b.height, H - 0.02)       # must fit under the top board
        b.depth = min(b.depth, D - cfg.depth_clearance)
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
            if scenario == "packed":       # no spacing either: the neighbours' faces touch
                x = books[ti - 1].x + books[ti - 1].thickness / 2
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

    # Spines line up roughly with the shelf's front edge, as on a library
    # shelf. Drawn last, so it leaves the rest of a seed's row unchanged.
    for b in books + [bk for fl in fillers for bk in fl.books]:
        b.inset = _u(rng, cfg.spine_inset)

    return Row(scenario, W, D, H, _u(rng, cfg.base_friction), pos, yaw,
               plinth, level_heights, target_level, tops[target_level],
               _u(rng, cfg.solref_timeconst), books, ti, target,
               float(gap_left), float(gap_right), nudges, fillers)


# ---------------------------------------------------------------------------
# 3. Panda with wrist camera and F/T sensor (a modified copy of Menagerie's)
# ---------------------------------------------------------------------------
# Thin fingers for the Franka Hand, in place of Menagerie's (21 mm wide, too
# wide for the narrower tight gaps): plain bars, 10 mm wide across their travel
# and 12 mm thick out from the gripping face, 12 cm long (Menagerie's are
# 5.4 cm: the long ones reach down past taller neighbours to grasp a book from
# above, e.g. in the return tray), with a pad strip along all of the finger
# that reaches past the palm (Menagerie's pad is 17 mm), so the narrow finger
# still stops a book turning in the grasp. insert_book.py holds books right up
# against the palm.
FINGER_HALF_W = 0.005       # half-width across the finger's travel
FINGER_THICK = 0.012        # thickness out from the gripping face
FINGER_LEN = 0.12           # length from where it is mounted (Menagerie's: 0.0539)
PAD_LEN = FINGER_LEN - 0.008  # pad strip on the gripping side, from the tip back to the palm


def make_panda_with_sensors(panda_xml: str) -> str:
    """Write `panda_shelf.xml` next to Menagerie's panda.xml (so mesh paths
    still resolve) with a wrist camera, an F/T sensor site, thin fingers and
    no keyframe. Returns the new file's path."""
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
    # Thin fingers: a bar, recessed 2 mm on the gripping side, with a 2 mm pad
    # strip there at its tip. In a finger's frame, z runs along it from where
    # it is mounted and y points away from the other finger (Menagerie turns
    # the right one 180 deg about z, so the same shape serves both).
    bar = ((FINGER_HALF_W, (FINGER_THICK - 0.002) / 2, FINGER_LEN / 2),
           (0, 0.002 + (FINGER_THICK - 0.002) / 2, FINGER_LEN / 2))
    pad = ((FINGER_HALF_W, 0.001, PAD_LEN / 2), (0, 0.001, FINGER_LEN - PAD_LEN / 2))
    for name in ("left_finger", "right_finger"):
        finger = root.find(f".//body[@name='{name}']")
        for g in finger.findall("geom"):
            finger.remove(g)
        for half, pos in (bar, pad):
            ET.SubElement(finger, "geom", type="box", size=_f(*half), pos=_f(*pos), material="black",
                          **{"class": "visual"})
            ET.SubElement(finger, "geom", type="box", size=_f(*half), pos=_f(*pos), **{"class": "collision"})
    sensor = root.find("sensor")
    if sensor is None:
        sensor = ET.SubElement(root, "sensor")
    ET.SubElement(sensor, "force", name="ft_force", site="ft_site")
    ET.SubElement(sensor, "torque", name="ft_torque", site="ft_site")
    out = os.path.join(os.path.dirname(os.path.abspath(panda_xml)), "panda_shelf.xml")
    # Several processes may build scenes at once (parallel trials, RL workers):
    # leave an identical file alone, and swap a new one in whole, so no reader
    # ever sees it half written.
    xml = ET.tostring(root)
    try:
        with open(out, "rb") as fh:
            if fh.read() == xml:
                return out
    except FileNotFoundError:
        pass
    tmp = f"{out}.{os.getpid()}.tmp"
    with open(tmp, "wb") as fh:
        fh.write(xml)
    os.replace(tmp, out)
    return out


PANDA_HOME = [0, -0.6, 0, -2.6, 0, 2.0, 0.7853, 0.04, 0.04]  # tucked 'ready' pose: hand ~25 cm in front of the shelf.
# (Menagerie's own 'home' pose reaches 0.55 m forward. It hit the shelf when the
# base stood 0.62 m out, and clears it now, but the tucked pose leaves more room.)


# ---------------------------------------------------------------------------
# 4. MJCF builder
# ---------------------------------------------------------------------------
def _f(*v):
    return " ".join(f"{x:.6g}" for x in v)


VIEW_DISTANCE = 0.9   # m from the gap to where a viewer's camera starts

# Book visuals: MuJoCo cannot read the repo's Collada meshes, so dae_to_obj.py
# converts each book into six flat, textured face quads under this directory
# (gitignored, so a checkout without the conversion silently keeps the boxes).
# They are pure decoration: non-colliding, massless geoms on the box bodies.
BOOK_ASSET_DIR = os.path.join(ROOT, "generated_assets", "books")
# A book's DAE has x along its depth, y through its thickness and z up its
# height, while the box geoms have x through the thickness: turning the faces
# this far about z puts the DAE's axes back onto the box's.
VISUAL_YAW = -math.pi / 2

# An articulated book (see add_book_parts) keeps a non-colliding box called
# <book>_geom as its outer envelope, so the controllers and perception still
# find the book's box under its own name. That box carries no mass, but a body
# with a freejoint must not be massless - MuJoCo then gets the body's centre of
# mass wrong, and the book topples. This is small enough to be inert (1 mg on a
# 0.7 kg book) and only exists to keep the body's mass positive.
ENVELOPE_MASS = 1e-6

_book_assets = None


def book_assets() -> List[dict]:
    """The meshes dae_to_obj.py wrote: [{name, faces: [{obj, texture}]}] with
    absolute paths, or [] if it has not been run. A face whose image the repo
    does not ship (book-ai86, book-logic1) is dropped, and a book with none
    left is left out, so it is drawn as a plain coloured box instead."""
    global _book_assets
    if _book_assets is None:
        _book_assets = []
        try:
            with open(os.path.join(BOOK_ASSET_DIR, "index.json")) as fh:
                entries = json.load(fh)["assets"]
        except (OSError, ValueError, KeyError):
            entries = []
        for e in entries:
            faces = [{"obj": os.path.join(BOOK_ASSET_DIR, f["obj"]),
                      "texture": os.path.join(BOOK_ASSET_DIR, f["texture"])}
                     for f in e.get("faces") or [] if f.get("texture")]
            if faces:
                _book_assets.append({"name": e["name"], "faces": faces})
    return _book_assets


def is_page_section(asset_name: str) -> bool:
    """True for the converted mesh of one page section (book-voynich-part1).
    Those draw a single hinged section, never a whole book's cover, so a book
    must not pick one as the mesh it wears."""
    return "-part" in asset_name


def page_section_assets(asset_name: str) -> List[dict]:
    """The converted meshes of a book's page sections, in section order. An
    articulated book keeps one mesh per section, named after the book
    (book-voynich-part1 ... part4), or [] if dae_to_obj.py has not converted
    them (it then falls back to box-coloured sections)."""
    prefix = asset_name + "-part"
    return sorted([a for a in book_assets() if a["name"].startswith(prefix)],
                  key=lambda a: a["name"])


def book_asset(name: str, sections: int = 0) -> Optional[dict]:
    """The converted mesh a book named `name` wears, or None if the conversion
    has not been run. Picked from the name, so that every caller (shelf_scene,
    insert_book) reaches for the same one. `sections` asks for a book that was
    converted into that many page sections: only those can be drawn one mesh
    per section (see add_page_visual_assets), so an articulated book picks
    among them; without it the page sections are left out, because one of them
    is not a whole book."""
    assets = book_assets()
    if sections:
        assets = [a for a in assets if len(page_section_assets(a["name"])) == sections] or assets
    else:
        assets = [a for a in assets if not is_page_section(a["name"])]
    return assets[zlib.crc32(name.encode()) % len(assets)] if assets else None


def _declare_faces(asset, declared: set, asset_name: str, faces, scale: str, mesh_stem: str) -> List[str]:
    """Declare the textures, materials and meshes that draw one converted mesh
    under the names `<mesh_stem>_f<k>`, and return the material to use per
    face. Textures and materials belong to the converted mesh (`asset_name`),
    so books wearing the same mesh share them; meshes belong to the geom, which
    holds its own scale, so every book (and every page section) gets its own."""
    for k, f in enumerate(faces):
        tex = f"{asset_name}_f{k}_tex"
        if tex not in declared:
            ET.SubElement(asset, "texture", name=tex, type="2d", file=f["texture"])
            ET.SubElement(asset, "material", name=f"{asset_name}_f{k}_mat", texture=tex)
            declared.add(tex)
        # A face is flat (no volume), so MuJoCo needs inertia="shell".
        ET.SubElement(asset, "mesh", name=f"{mesh_stem}_f{k}", file=f["obj"],
                      inertia="shell", scale=scale)
    return [f"{asset_name}_f{k}_mat" for k in range(len(faces))]


def add_book_visual_assets(asset, books) -> dict:
    """Declare the textures, materials and meshes for the visual geoms of
    `books`, and return {book name: [material name per face]}. Books that share
    a converted mesh share its textures and materials; each gets its own meshes,
    because a mesh asset holds the scale that turns the DAE's unit cube into
    that book's box. Returns {} if dae_to_obj.py has not been run.
    Articulated books are drawn per page section instead (see
    add_page_visual_assets) and are left out here."""
    if not book_assets():
        return {}
    # Textures already declared count as shared: insert_book calls this again
    # on the asset block build_scene filled in.
    out, declared = {}, {t.get("name") for t in asset.findall("texture")}
    for b in books:
        if b.parts:
            continue
        # A CATALOGUE book wears its own mesh, or none if it was not converted.
        a = next((x for x in book_assets() if x["name"] == b.asset), None) if b.asset else book_asset(b.name)
        if a is None:
            continue
        scale = _f(b.depth / 2, b.thickness / 2, b.height / 2)   # see VISUAL_YAW
        out[b.name] = _declare_faces(asset, declared, a["name"], a["faces"], scale, b.name)
    return out


def add_page_visual_assets(asset, books) -> dict:
    """Declare the meshes that draw the page sections of the articulated
    `books`, and return {section name: [material name per face]}. Each section
    is scaled to its own thickness, so it needs its own meshes. Sections of a
    book whose converted meshes have no page sections are left out and stay
    plain coloured boxes (section names are unique, so [] would be an error)."""
    if not book_assets():
        return {}
    out, declared = {}, {t.get("name") for t in asset.findall("texture")}
    for b in books:
        if not b.parts:
            continue
        name = b.asset or book_asset(b.name, sections=len(b.parts))["name"]
        for part, section in zip(b.parts, page_section_assets(name)):
            scale = _f(b.depth / 2, part.thickness / 2, b.height / 2)   # see VISUAL_YAW
            out[part.name] = _declare_faces(asset, declared, section["name"], section["faces"],
                                            scale, part.name)
    return out


def add_book_parts(body, b: Book, page_materials: Optional[dict] = None) -> None:
    """Give an articulated book its pages (the repo's design, see
    urdf/book_part.xacro): one box per section, hinged on the spine so the fore
    edges can fan open. The sections are nested from the back forward, as in
    the repo's joint chain: the back one is welded to `body` (the repo's fixed
    back-cover joint) and each further one hinges about z on the section before
    it. A hinge sits on the vertical spine line at the section's far (+x) face,
    turned negative to open, exactly as the repo places its joint origins."""
    mats = page_materials or {}
    parent, previous = body, 0.0
    for k, part in enumerate(reversed(b.parts)):
        # The back cover starts on the spine, at the book's +x face; the rest
        # sit one section further forward, relative to the section they hinge on.
        pos = (b.thickness / 2, b.depth / 2, 0.0) if not k else (-previous, 0.0, 0.0)
        node = ET.SubElement(parent, "body", name=part.name, pos=_f(*pos))
        if k:
            ET.SubElement(node, "joint", name=part.name + "_joint", type="hinge",
                          axis="0 0 1", range=_f(-math.pi, 0.0))
        # The box hangs half a section behind its hinge and half the depth in
        # front of the spine line: the repo's (xsize/2, -ysize/2, 0) offset.
        offset = _f(-part.thickness / 2, -b.depth / 2, 0.0)
        faces = mats.get(part.name, [])
        attrs = {"class": "book", "group": "3"} if faces else {"class": "book"}
        ET.SubElement(node, "geom", name=part.name + "_geom", pos=offset,
                      size=_f(part.thickness / 2, b.depth / 2, b.height / 2), mass=f"{part.mass:.5g}",
                      friction=_f(b.friction, 0.005, 0.0001), rgba=_f(*b.rgba), **attrs)
        for j, material in enumerate(faces):
            ET.SubElement(node, "geom", name=f"{part.name}_visual{j}", **{"class": "book_visual"},
                          mesh=f"{part.name}_f{j}", material=material, pos=offset, euler=_f(0, 0, VISUAL_YAW))
        parent, previous = node, part.thickness


def book_children(b: Book) -> List[str]:
    """The bodies an articulated book brings with it (see add_book_parts): its
    page sections, back first, the order their joints come in qpos. [] for a
    rigid book. A book's `<book>_geom` is only a non-colliding envelope, so
    whatever has to see where a book really is - its contacts with the hand and
    the other books - has to reach for these bodies' geoms too."""
    return [p.name for p in reversed(b.parts)]


def add_book_geoms(body, b: Book, visual_materials: Optional[dict] = None,
                   page_materials: Optional[dict] = None) -> None:
    """Add one book to `body`: its collision box (the only thing physics sees),
    plus the six textured face geoms that draw it if `visual_materials` names
    the book. The box is left out of the render groups then, so that the faces
    covering it exactly do not fight it for the depth buffer.

    An articulated book (b.parts) keeps that box as a non-colliding envelope
    with the same name and size - the controllers reconstruct a book's corners
    from `<name>_geom`'s size, and perception reads it - and takes its pages as
    hinged sections instead (see add_book_parts). Those carry the mass and the
    collision, so the row still weighs what it did."""
    if b.parts:
        ET.SubElement(body, "geom", name=b.name + "_geom", size=_f(b.thickness / 2, b.depth / 2, b.height / 2),
                      mass=_f(ENVELOPE_MASS), contype="0", conaffinity="0", rgba=_f(*b.rgba),
                      **{"class": "book", "group": "3"})
        add_book_parts(body, b, page_materials)
        return
    mats = (visual_materials or {}).get(b.name, [])
    attrs = {"class": "book", "group": "3"} if mats else {"class": "book"}
    ET.SubElement(body, "geom", name=b.name + "_geom",
                  size=_f(b.thickness / 2, b.depth / 2, b.height / 2), mass=f"{b.mass:.5g}",
                  friction=_f(b.friction, 0.005, 0.0001), rgba=_f(*b.rgba), **attrs)
    for k, material in enumerate(mats):
        ET.SubElement(body, "geom", name=f"{b.name}_visual{k}", **{"class": "book_visual"},
                      mesh=f"{b.name}_f{k}", material=material, euler=_f(0, 0, VISUAL_YAW))


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

    # MuJoCo's default free camera looks at the gap: at its front, a third of
    # the way up the level, from VIEW_DISTANCE away, above the arm and off to
    # the gap's side so the robot isn't in the way. The viewer uses it when it
    # opens the scene as a file and from its Align button; view_scene.py and
    # insert_book.py --view apply it themselves. It sits 1.5 x extent away. A
    # small extent would also shrink the shadows' reach, so shadowclip is
    # raised to keep it at ~3 m.
    gap_x = (row.gap_left + row.gap_right) / 2
    side = 1 if gap_x >= 0 else -1
    c, s = math.cos(row.shelf_yaw), math.sin(row.shelf_yaw)
    lx, ly, lz = gap_x, D / 2 - 0.05, target_top + H / 3
    look = (row.shelf_pos[0] + c * lx - s * ly, row.shelf_pos[1] + s * lx + c * ly, row.shelf_pos[2] + lz)
    ET.SubElement(m, "statistic", center=_f(*look), extent=_f(VIEW_DISTANCE / 1.5))
    vis = ET.SubElement(m, "visual")
    ET.SubElement(vis, "global", offwidth="1280", offheight="960",
                  azimuth=_f(math.degrees(row.shelf_yaw) - 90 - side * 30), elevation="-25")
    ET.SubElement(vis, "map", shadowclip=_f(3.0 * 1.5 / VIEW_DISTANCE))
    ET.SubElement(vis, "headlight", ambient="0.4 0.4 0.4", diffuse="0.6 0.6 0.6")

    d = ET.SubElement(m, "default")
    dd = ET.SubElement(d, "default", **{"class": "book"})
    # condim 4 adds torsional friction so books don't spin on the base.
    ET.SubElement(dd, "geom", type="box", condim="4", solref=_f(tr, 1.0),
                  solimp="0.95 0.99 0.001", margin="0.0005")
    ds = ET.SubElement(d, "default", **{"class": "shelf"})
    ET.SubElement(ds, "geom", type="box", condim="4", rgba="0.55 0.4 0.25 1",
                  solref=_f(tr, 1.0), solimp="0.95 0.99 0.001")
    dv = ET.SubElement(d, "default", **{"class": "book_visual"})
    # Decoration only: the massless, non-colliding face meshes of a book
    # (see add_book_visual_assets). Group 2 is drawn by default; group 3, where
    # the box they cover ends up, is not.
    ET.SubElement(dv, "geom", type="mesh", contype="0", conaffinity="0", group="2", mass="0")

    asset = ET.SubElement(m, "asset")
    ET.SubElement(asset, "texture", name="grid", type="2d", builtin="checker",
                  rgb1="0.85 0.85 0.85", rgb2="0.75 0.75 0.75", width="512", height="512")
    ET.SubElement(asset, "material", name="floor", texture="grid", texrepeat="8 8")
    if panda_xml:
        ET.SubElement(asset, "model", name="panda", file=make_panda_with_sensors(panda_xml))
    all_books = row.books + [b for fl in row.filler_levels for b in fl.books]
    visual_materials = add_book_visual_assets(asset, all_books) if cfg.book_meshes else {}
    page_materials = add_page_visual_assets(asset, all_books) if cfg.book_meshes else {}

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

    # Books: spine `inset` behind the shelf's front edge. With every book at
    # least depth_clearance (>= the largest inset) shallower than the shelf,
    # no back goes past the back panel.
    # qpos for the "init" keyframe, in the model's joint order: the robot's
    # joints first, then each book's - its freejoint if it has one, followed by
    # the hinge of every page section after the welded back one (closed, 0).
    qpos = list(PANDA_HOME) if panda_xml else []
    qw, qz = math.cos(row.shelf_yaw / 2), math.sin(row.shelf_yaw / 2)

    def add_books(books, z_base, free):
        """Place books standing on the board whose top is at z_base. Free books
        get a freejoint and are appended to `qpos`; books without a joint are
        welded in place and cost almost nothing to simulate. An articulated
        book adds its page hinges to `qpos` too, closed."""
        for b in books:
            lx, ly, lz = b.x, D / 2 - b.inset - b.depth / 2, z_base + b.height / 2 + 0.0005
            wx, wy, wz = sp[0] + c * lx - s * ly, sp[1] + s * lx + c * ly, sp[2] + lz
            body = ET.SubElement(wb, "body", name=b.name, pos=_f(wx, wy, wz), euler=_f(0, 0, row.shelf_yaw))
            if free:
                ET.SubElement(body, "freejoint", name=b.name + "_joint")
                qpos.extend([wx, wy, wz, qw, 0, 0, qz])
            qpos.extend([0.0] * max(0, len(b.parts) - 1))
            add_book_geoms(body, b, visual_materials, page_materials)
            ET.SubElement(body, "site", name=b.name + "_top", pos=_f(0, 0, b.height / 2), size="0.003", group="4")

    # Target-level books first, so their qpos entries come right after the robot's.
    add_books(row.books, target_top, free=True)
    for fl in row.filler_levels:
        add_books(fl.books, tops[fl.level], free=not cfg.static_other_levels)

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
    ap.add_argument("--book-meshes", action="store_true",
                    help="draw the converted repo book meshes (run dae_to_obj.py first)")
    ap.add_argument("--book-pages", type=int, default=0, metavar="N",
                    help="split every book's pages into N sections hinged on the spine (0 = one rigid box)")
    ap.add_argument("--catalogue", action="store_true",
                    help="fill the rows with the repo's real books (size, mass, cover, pages) instead of sampled ones")
    ap.add_argument("--out", default="scenes")
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    cfg = REPO_DEFAULTS if a.repo_defaults else ShelfConfig()
    cfg.book_meshes = a.book_meshes
    cfg.book_pages = a.book_pages
    cfg.catalogue = a.catalogue
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
