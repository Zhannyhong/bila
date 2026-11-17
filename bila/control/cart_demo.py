"""
Book-return cart: the Panda on a lift, on a wheeled cart with a return tray of
real books standing spine-up (one per slot) and a row camera on the lift, in
front of a bookcase of the repo's real books, each with a call-number sticker
on its spine.

    uv run python -m bila.control.cart_demo --scenario tight             # headless: prints each step and the score
    uv run mjpython -m bila.control.cart_demo --scenario tight --view    # watch it (macOS needs mjpython)
    uv run python -m bila.control.cart_demo --scenario leaning --seed 3  # clear, tight, leaning or packed
    uv run python -m bila.control.cart_demo --scenario tight --segmenter ~/bila_data/segmenter/segmenter.pt   # learned book masks
    uv run python -m bila.control.cart_demo --scenario tight --policy runs/hl/q.npz   # learned high-level steps (high_level.py)

What the robot does, as a librarian would, with one camera, on its wrist:
  1. Read the call numbers in the tray: a wrist view down onto the whole
     tray, then a close-up of each sticker it couldn't read from there.
  2. Look each one up in the shelf map, which it knows: every row's board
     height and the first and last call number on it. That names the row.
  3. Raise the lift to that row. The episode's first look (insert_book's
     sweep of wrist views along the row) reads the spines' stickers, and the
     slot is between the two books whose numbers bracket the book's
     (call_numbers_seen.ReadingEyes); the row is estimated from the same
     images (RGBDPerception).
  4. Make room there with the empty hand (insert_book's skills: widen a tight
     gap, stand a leaning neighbour up, shove a packed row aside).
  5. Pick the book out of the tray and insert it. The tray holds it spine-up,
     so the hand comes down onto its spine exactly as insert_book holds a book
     for insertion; the arm then turns it to face the shelf. This replaces
     insert_book's "hand-over (not modelled)".

Reading a sticker here means enough of it is visible in the camera image; its
number then comes from the simulation. With `--labels learned` the numbers
are read from the images instead (label_reader.py; train it first).
Books the robot can't handle (shelvable()) go to the librarian, not the tray.
Only the scenario's target row has a slot to fill: the other tray books are
read and looked up, but stay in the tray.

Cart frame (the old robot mount, which faces the shelf): +x toward the shelf,
+y to the tray side, z up, origin at the Panda's base at the target row's
usual height. The cart does not drive yet. Its parts are visual except the
tray, its dividers, the lift rail and the deck, which the arm must not hit.
"""

import argparse
import os
import time
import xml.etree.ElementTree as ET

import numpy as np
import mujoco
import mujoco.viewer
from PIL import Image, ImageDraw, ImageFont

from bila.control import insert_book
from bila.sim.shelf_scene import (PANDA_HOME, ShelfConfig, _catalogue_book, _f, add_book_geoms, add_book_visual_assets,
                         board_tops, build_scene, sample_row, settle)
from bila.control.insert_book import (DEFAULT_PANDA, FINGERTIP_Z, GAP_MARGIN, GRIP_CLOSED, GRIP_FORCE, GRIP_OPEN, MIN_MARGIN,
                         PAD_INSET, PALM_GAP, PALM_Z,
                         WAYPOINT_TRIES, ArmIK, Episode, Runner, compile_robot_scene, report, segment)

from bila import ROOT as HERE
LABEL_DIR = os.path.join(HERE, "generated_assets", "labels")   # git-ignored, made on demand

LIFT_MAX = 0.70             # lift travel: the Panda's base this far above the deck at most
DECK = 0.16                 # deck top above the floor
RAIL_X = -0.30              # the lift rail's centre, behind the Panda's base (cart x)
REACH_BELOW = 0.40          # the Panda's base stands this far below the board it works on (as shelf_scene)
TRAY = (-0.12, 0.50, 0.72)  # tray centre (cart x, y) and its floor's height above the floor
TRAY_HALF = (0.27, 0.15)
SLOT_CLEAR = 0.025          # room each side of a book in its slot: finger 12 mm + opening 8 mm + 5 mm
DIVIDER_H = 0.08
TRAY_BOOKS = 5              # books the tray is loaded with (if they fit)
OPEN_SPARE = 0.008          # the fingers open this much wider than the book, each side
LIFT_SPEED = 0.10           # m/s, peak, of the lift's moves
CARRY_JOINT_SPEED = 0.3     # rad/s, peak, of the arm's joint moves while it carries a book
CARRY_CHECK = 0.01          # rad: ... checked for collisions this often along the way (~4 mm at the book's far end)
PAD_TORSION = 0.05          # torsional friction of the finger pads (m): resists a held book twisting
CARRY_OUT = (0.12, 0.09, 0.06, 0.04, 0.02)   # the book comes down on the lift this far in front of its pre-insertion pose
APPROACH_ABOVE = 0.012      # ... and comes in over the board this much higher (if the level has room), then down
MAX_KG = 1.5                # assumption: heavier returns go to the librarian, whatever the grip could do
# Carried at its spine end, a book's weight twists it in the pinch: kg g (depth/2 - PAD_INSET) about the
# fingers' axis. The pads' torsional friction holds about 2 GRIP_FORCE PAD_TORSION of it: in the slip
# runs (tests/test_grasp_slip.py, pad torsion 0.005-0.02) a 20 cm book began to creep in the grip at
# 0.8-1.4 times that, and turned 10-75 deg at 2. So the book's twist is kept to TWIST_SAFETY of it.
# Only as good as PAD_TORSION, which is not measured on the real pads yet.
TWIST_SAFETY = 0.5
# Carrying it from the tray stops if it turns or moves this much in the grip. Normal carries (seeds 0-9)
# drift up to 0.8 deg / 4.3 mm; a 3 kg book with pads of torsion 0.01 crept 7.4 deg / 11.5 mm.
SLIP_TURN_DEG, SLIP_MM = 5.0, 10.0
STEP = 10                   # call numbers go up in tens along the shelves, so a returned book fits between two


def shelvable(t, d, h, kg) -> list:
    """Why the robot can't shelve a book this size ([] = it can)."""
    why = []
    if t > 2 * (0.04 - OPEN_SPARE):
        why.append("too thick for the gripper")
    if t < ShelfConfig.thickness[0]:
        why.append("thinner than the insertion skills are tuned for")
    if h > 2 * TRAY_HALF[1] - 0.016:
        why.append("too tall for the tray")
    if d < FINGERTIP_Z - PALM_Z - PALM_GAP + 0.01:
        why.append("too shallow: the fingertips would reach the slot floor")
    if kg > MAX_KG:
        why.append("heavier than the robot is allowed to carry")
    if twist(d, kg) > TWIST_SAFETY * 2 * GRIP_FORCE * PAD_TORSION:
        why.append("its weight would twist it in the grip")
    return why


def twist(d, kg) -> float:
    """N m a book's weight puts about the fingers' axis, held at its spine end and level."""
    return kg * 9.81 * max(0.0, d / 2 - PAD_INSET)


# ---------------------------------------------------------------------------
# Call numbers and the shelf map
# ---------------------------------------------------------------------------
def call_numbers(row):
    """{book name: call number} for every book in the bookcase, in reading
    order: top row first, each row left to right as seen from the front (the
    shelf's -x side is on the right). The held book gets the number between
    its two neighbours'. Returns (numbers, shelf map: [(level, board top,
    first, last)], top row first)."""
    levels = {row.target_level: row.books} | {fl.level: fl.books for fl in row.filler_levels}
    tops = board_tops(ShelfConfig(), row.plinth_height, row.level_heights)
    numbers, shelf_map, n = {}, [], 100
    for level in sorted(levels, reverse=True):
        if not levels[level]:
            continue
        first = n
        for b in sorted(levels[level], key=lambda b: -b.x):      # left to right from the front
            numbers[b.name] = n
            n += STEP
        shelf_map.append((level, tops[level], first, n - STEP))
    ti = row.target_index
    numbers[row.target_book.name] = (numbers[row.books[ti - 1].name] + numbers[row.books[ti].name]) // 2
    return numbers, shelf_map


def row_for(number, shelf_map):
    """The row (level, board top) a call number belongs on, from the shelf map."""
    return next(((level, top) for level, top, first, last in shelf_map if first <= number <= last), None)


def dewey(number) -> str:
    """A call number's Dewey class, as the sticker prints it: 100 -> "010.0".
    Tenths keep the in-between numbers distinct (105 -> "010.5")."""
    return f"{number / 10:05.1f}"


def cutter(b) -> str:
    """The author mark under the class on a spine label: the first three
    letters of the catalogue book's name (book-knuth -> KNU)."""
    return b.asset[5:8].upper() if b.asset else ""


STICKER_PX = 600            # sticker texture height; its width follows the sticker's own aspect


def sticker(number, mark, aspect) -> str:
    """A spine-label texture as libraries print them: the Dewey class above the
    author mark, horizontal, black on white (made once, cached on disk).
    `aspect` = width / height, so the text isn't stretched on the box."""
    h = STICKER_PX
    w = max(16, round(h * aspect))
    path = os.path.join(LABEL_DIR, f"{number:05d}{mark}_{w}.png")
    if not os.path.exists(path):
        os.makedirs(LABEL_DIR, exist_ok=True)
        img = Image.new("RGB", (w, h), "white")
        draw = ImageDraw.Draw(img)
        draw.rectangle([0, 0, w - 1, h - 1], outline="black", width=max(2, w // 40))
        lines = [dewey(number)] + ([mark] if mark else [])
        size = int(min(h / 4, 0.9 * w / (0.6 * max(map(len, lines)))))     # widest line fills ~90% of it
        font = ImageFont.load_default(size=size)
        for i, line in enumerate(lines):
            draw.text((w / 2, h * (0.25 + 0.22 * i)), line, fill="black", anchor="mm", font=font)
        img.save(path)
    return path


def add_stickers(root, books, numbers):
    """Call-number stickers near the bottom of the spine of every book in
    `books` that has a number: visual only (massless, no contacts), a
    `<book>_label` geom on the book's body, so book_of_geom counts it as part
    of the book."""
    asset, by_name = root.find("asset"), {b.name: b for b in books}
    for body in root.find("worldbody").iter("body"):
        b = by_name.get(body.get("name"))
        if b is None or b.name not in numbers:
            continue
        n, hx = numbers[b.name], min(0.85 * b.thickness / 2, 0.012)
        ET.SubElement(asset, "texture", name=f"label{n}", type="2d", hflip="true", file=sticker(n, cutter(b), hx / 0.03))
        ET.SubElement(asset, "material", name=f"label{n}", texture=f"label{n}")
        ET.SubElement(body, "geom", name=b.name + "_label", type="box", material=f"label{n}", group="2",
                      size=_f(hx, 0.0005, 0.03),
                      pos=_f(0, b.depth / 2 + 0.0008, -b.height / 2 + STICKER_UP), contype="0", conaffinity="0", mass="0")


# ---------------------------------------------------------------------------
# The scene
# ---------------------------------------------------------------------------
def sample(scenario, seed):
    """A row whose target book the robot can shelve, the tray loaded with it
    and other returned books (depths close enough that the palm clears every
    neighbour's spine), and the books sent to the librarian instead.
    Returns (cfg, row, tray books, [(book, reasons)])."""
    # The target book at least 20 mm thick: a tight gap is 0.6-1.1x its thickness, and the fingers need 12 mm.
    cfg = ShelfConfig(catalogue=True, book_meshes=True, target_band=(DECK + REACH_BELOW + 0.01, 1.10),
                      thickness=(0.020, 0.060))
    rng = np.random.default_rng(seed)
    while True:
        row = sample_row(cfg, scenario, rng)
        b = row.target_book
        if not b.parts and not shelvable(b.thickness, b.depth, b.height, b.mass):
            break
    tray, librarian = [b], []
    for _ in range(40):
        if len(tray) == TRAY_BOOKS:
            break
        o = _catalogue_book(cfg, rng, f"return_{len(tray)}", 1.0, 1.0, False)
        why = shelvable(o.thickness, o.depth, o.height, o.mass) or (["hinged pages"] if o.parts else [])
        if why:
            if o.asset not in [x.asset for x, _ in librarian]:
                librarian.append((o, why))
        elif abs(o.depth - b.depth) < PALM_GAP - 0.015:         # its spine stays under the palm
            tray.append(o)
    rng.shuffle(tray)
    return cfg, row, tray, librarian


def spine_up_quat():
    """World orientation of a book standing spine-up in the tray: its
    thickness (local x) along the tray, its spine normal (local y) up, its
    height (local z) across the tray. The cart is turned -90 deg about z."""
    R_cart = np.column_stack([[1, 0, 0], [0, 0, 1], [0, -1, 0]]).astype(float)
    Rz = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]], dtype=float)   # cart x -> world -y, cart y -> world +x
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, (Rz @ R_cart).ravel())
    return q


def build(cfg, row, tray, numbers):
    """The cart scene, compiled: (model, the tray books that fit)."""
    root = ET.fromstring(build_scene(cfg, row, DEFAULT_PANDA))
    wb, asset = root.find("worldbody"), root.find("asset")
    cart = wb.find("body[@name='robot_mount']")
    for g in cart.findall("geom"):
        cart.remove(g)                                   # the pillar: the lift holds the arm up now
    bx, by, zb = map(float, cart.get("pos").split())
    fz, deck = -zb, DECK - zb                            # floor and deck top, in the cart frame
    to_world = lambda p: (bx + p[1], by - p[0], zb + p[2])   # the mount is turned -90 deg about z

    def geom(parent, name, rgba, solid=True, **kw):
        ET.SubElement(parent, "geom", name=name, rgba=rgba, group="2",
                      **({} if solid else {"contype": "0", "conaffinity": "0"}), **kw)

    def box(name, half, pos, rgba, solid=True, parent=cart):
        geom(parent, name, rgba, solid, type="box", size=_f(*half), pos=_f(*pos))

    grey, dark, wood = "0.55 0.57 0.6 1", "0.15 0.15 0.15 1", "0.62 0.45 0.28 1"
    x0, x1, y0, y1 = -0.50, 0.15, -0.32, 0.78
    box("deck", ((x1 - x0) / 2, (y1 - y0) / 2, 0.03), ((x0 + x1) / 2, (y0 + y1) / 2, deck - 0.03), grey)
    for i, (x, y) in enumerate([(x1 - 0.07, y1 - 0.07), (x1 - 0.07, y0 + 0.07), (x0 + 0.07, y1 - 0.07), (x0 + 0.07, y0 + 0.07)]):
        geom(cart, f"wheel{i}", dark, False, type="cylinder", size="0.06 0.02", pos=_f(x, y, fz + 0.06), euler="1.5708 0 0")

    # Lift: a rail behind the arm, and a carriage on the lift_z joint carrying the Panda. The rail stands
    # clear of the arm's elbow in the tucked home pose (at -0.22 m it ran link3 into it).
    box("lift_rail", (0.035, 0.09, (LIFT_MAX + 0.15) / 2), (RAIL_X, 0, deck + (LIFT_MAX + 0.15) / 2), "0.25 0.25 0.28 1")
    lift = ET.Element("body", name="lift", pos=_f(0, 0, deck))
    ET.SubElement(lift, "joint", name="lift_z", type="slide", axis="0 0 1", range=_f(0, LIFT_MAX), damping="3000")
    box("carriage", (0.11, 0.10, 0.015), (0, 0, -0.015), "0.35 0.35 0.38 1", False, lift)
    box("carriage_bracket", ((RAIL_X + 0.09) / -2, 0.09, 0.03), ((RAIL_X - 0.09) / 2, 0, -0.03), "0.35 0.35 0.38 1", False, lift)
    attach = cart.find("attach")
    cart.remove(attach); lift.append(attach); cart.append(lift)
    act = root.find("actuator")
    if act is None:
        act = ET.SubElement(root, "actuator")
    ET.SubElement(act, "position", name="lift", joint="lift_z", kp="60000", ctrlrange=_f(0, LIFT_MAX))

    # Return tray on the +y side: one slot per book, books spine-up, dividers between the slots.
    tx, ty, tz = TRAY[0], TRAY[1], TRAY[2] - zb
    hx, hy = TRAY_HALF
    box("tray", (hx, hy, 0.008), (tx, ty, tz), wood)
    for name, half, p in [("lip_back", (0.008, hy, 0.04), (tx - hx, ty, tz + 0.04)), ("lip_front", (0.008, hy, 0.04), (tx + hx, ty, tz + 0.04)),
                          ("lip_in", (hx, 0.008, 0.04), (tx, ty - hy, tz + 0.04)), ("lip_out", (hx, 0.008, 0.04), (tx, ty + hy, tz + 0.04))]:
        box(name, half, p, wood)
    for i, (x, y) in enumerate([(tx - hx + 0.02, ty - hy + 0.02), (tx - hx + 0.02, ty + hy - 0.02),
                                (tx + hx - 0.02, ty - hy + 0.02), (tx + hx - 0.02, ty + hy - 0.02)]):
        geom(cart, f"tray_leg{i}", grey, type="cylinder", size="0.012", fromto=_f(x, y, deck, x, y, tz))
    slots, x = [], tx - hx + 0.01
    for b in tray:
        pitch = b.thickness + 2 * SLOT_CLEAR
        if x + pitch > tx + hx:
            break
        slots.append((b, x + pitch / 2))
        x += pitch
        box(f"divider{len(slots)}", (0.003, hy - 0.01, DIVIDER_H / 2), (x, ty, tz + 0.008 + DIVIDER_H / 2), wood)

    # Tray books: free bodies in the world, spine-up, their fore-edges on the tray floor (after all else in qpos).
    key = root.find("keyframe/key")
    qpos = [_f(-deck)] + key.get("qpos").split()         # lift_z comes first: the base at its usual height
    tray_books = [b for b, _ in slots]
    materials = add_book_visual_assets(asset, tray_books)
    q = spine_up_quat()
    for b, sx in slots:
        pos = to_world((sx, ty, tz + 0.008 + b.depth / 2 + 0.001))
        body = ET.SubElement(wb, "body", name=b.name, pos=_f(*pos), quat=_f(*q))
        ET.SubElement(body, "freejoint", name=b.name + "_joint")
        add_book_geoms(body, b, materials)
        qpos += [*map(_f, pos), *map(_f, q)]
    key.set("qpos", " ".join(qpos))
    key.attrib.pop("ctrl")                               # set by actuator name below: the lift's comes first

    add_stickers(root, row.books + tray_books + [b for fl in row.filler_levels for b in fl.books], numbers)

    model = compile_robot_scene(ET.tostring(root, encoding="unicode"))
    # Rubber pads resist a book twisting in the grip: give the fingers torsional friction (MuJoCo's
    # default point contacts have none) and stiffer contacts. Carrying a book from the tray, held at
    # its spine end, it twisted 50 mm without; about 10 mm with.
    for g in range(model.ngeom):
        if model.body(model.geom_bodyid[g]).name in ("panda/left_finger", "panda/right_finger"):
            model.geom_condim[g] = 6
            model.geom_friction[g, 1] = PAD_TORSION
            model.geom_solref[g] = (2 * model.opt.timestep, 1.0)
    # ... and frictional contacts that hold rather than creep (MuJoCo's soft friction lets a book carried
    # for half a minute twist a few degrees; this cart scene only, insert_book's scenes are unchanged).
    model.opt.impratio, model.opt.cone = 50.0, mujoco.mjtCone.mjCONE_ELLIPTIC
    want = {f"panda/actuator{i + 1}": v for i, v in enumerate(PANDA_HOME[:7])} | {"panda/actuator8": 255, "lift": -deck}
    model.key_ctrl[0] = [want[model.actuator(i).name] for i in range(model.nu)]
    return model, tray_books


class OracleLabels:
    """Call numbers "read" by the simulation: a sticker is read if at least
    `min_pixels` of it show in the wrist camera's image, and its number is
    the one the scene gave it. Same interface as LearnedLabels."""

    def __init__(self, model, numbers: dict, min_pixels=25):
        self.m, self.number_of, self.min_pixels = model, numbers, min_pixels
        self.label = {model.geom(n + "_label").id: n for n in numbers
                      if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, n + "_label") >= 0}
        self.cam = None
        self.numbers, self.where = {}, {}

    def look(self, data, camera="panda/wrist_cam") -> tuple:
        """(stickers showing, stickers read) in `camera`'s image now."""
        from bila.vision.camera import WristCamera
        if self.cam is None:
            self.cam = WristCamera(self.m, name=camera)
        ids, counts = np.unique(self.cam.capture(data).seg, return_counts=True)
        shown = {self.label[int(g)]: c for g, c in zip(ids, counts) if int(g) in self.label}
        for n, c in shown.items():
            self.where[n] = data.geom_xpos[self.m.geom(n + "_label").id].copy()
            if c >= self.min_pixels:
                self.numbers[n] = (self.number_of[n], 1.0)
        return len(shown), sum(c >= self.min_pixels for c in shown.values())

    def read(self, names) -> dict:
        return {n: self.numbers[n][0] for n in names if n in self.numbers}

    def close(self):
        if self.cam is not None:
            self.cam.close()


# ---------------------------------------------------------------------------
# Reading the stickers from the images (--labels learned)
# ---------------------------------------------------------------------------
CLOSE_STANDOFFS = (0.18, 0.22, 0.14)   # m: wrist close-ups of the row's stickers, this far in front of the board (row_closeup)
CLOSE_ABOVE = (0.20, 0.25)             # m: wrist close-ups of a tray sticker, this far above it
OVERVIEW_ABOVE = (0.30, 0.27, 0.33)    # m: a wrist view of the whole tray from this far above the stickers
STICKER_UP = 0.045                     # m: stickers sit this far up from a spine's bottom (build)
CLOSE_LIFTS = (0.45, 0.35, 0.55, 0.25, 0.65)   # lift heights tried for a view down into the tray
MAX_CLOSEUPS = 4                       # close-ups per step at most


def sticker_text(number, mark) -> str:
    """What a sticker says, its lines joined by a newline: '031.0\\nPDP'."""
    return f"{dewey(number)}\n{mark}".strip()


class LearnedLabels:
    """Call numbers read from camera images by label_reader.LabelReader, in
    place of OracleLabels' "visible enough, so the simulation's number". The
    catalogue (every sticker's text and its number) is what the library
    knows. A sticker read is tied to the book it is on by the simulation's
    object ids under it: bookkeeping only, since the demo names books by
    their bodies (a real robot would go by where the sticker is, which is
    also kept: the sticker's centre in the world, from the depth image)."""

    def __init__(self, model, weights: str, catalogue: dict):
        from bila.vision.label_reader import LabelReader                 # imports torch (and torch._dynamo, before any render)
        from bila.vision.rgbd_perception import book_of_geom
        self.m = model
        self.reader = LabelReader(os.path.join(weights, "finder.pt"), os.path.join(weights, "reader.pt"),
                                  catalogue=list(catalogue))
        self.number_of = catalogue
        self.book_of = {g: model.geom(b).name[:-len("_geom")] for g, b in book_of_geom(model).items()}
        self.cams = {}
        self.numbers = {}           # book -> (call number, confidence): the surest reading of it so far
        self.where = {}             # book -> its sticker's centre (world), read or not

    @staticmethod
    def available(weights: str) -> bool:
        return all(os.path.exists(os.path.join(weights, f)) for f in ("finder.pt", "reader.pt"))

    def camera(self, name):
        from bila.vision.camera import DepthNoise, WristCamera
        if name not in self.cams:
            self.cams[name] = WristCamera(self.m, name=name, noise=DepthNoise.d405(),
                                          rng=np.random.default_rng(len(self.cams)))
        return self.cams[name]

    def look(self, data, camera) -> tuple:
        """Find and read every sticker `camera` sees now: (stickers found, read)."""
        frame = self.camera(camera).capture(data)
        readings = self.reader.read(frame.rgb)
        read = 0
        for r in readings:
            book, point = self.locate(frame, r.corners)
            if book is None:
                continue
            if point is not None:
                self.where[book] = point
            if r.text is not None:
                read += 1
                if book not in self.numbers or r.confidence > self.numbers[book][1]:
                    self.numbers[book] = (self.number_of[r.text], r.confidence)
        return len(readings), read

    def locate(self, frame, corners):
        """(the book under a sticker's corners, the sticker's centre in the
        world or None without depth there)."""
        import cv2
        inside = np.zeros(frame.depth.shape, np.uint8)
        cv2.fillConvexPoly(inside, np.round(corners).astype(np.int32), 1)
        inside = inside > 0
        books = [b for b in (self.book_of.get(int(g)) for g in frame.seg[inside]) if b]
        if not books:
            return None, None
        book = max(set(books), key=books.count)
        d = frame.depth[inside & (frame.depth > 0)]
        if len(d) == 0:
            return book, None
        (u, v), z, K = np.mean(corners, axis=0), float(np.median(d)), frame.K
        p = np.array([(u - K[0, 2]) * z / K[0, 0], (v - K[1, 2]) * z / K[1, 1], z])
        return book, frame.pose[:3, :3] @ p + frame.pose[:3, 3]

    def read(self, names) -> dict:
        """{book: call number} for the books in `names` read so far."""
        return {n: self.numbers[n][0] for n in names if n in self.numbers}

    def close(self):
        for c in self.cams.values():
            c.close()


def tray_closeup(model, data, ik, point, rng, above=CLOSE_ABOVE, yaws=(0.0, np.pi / 2, np.pi, -np.pi / 2)):
    """(lift height, arm joint angles) for a wrist view straight down onto
    `point` (a tray sticker) from one of `above` over it, the image's right
    turned by one of `yaws` from the world's x; None if the arm reaches none."""
    from bila.control.inspection import looking, solve_camera
    lift = model.jnt_qposadr[model.joint("lift_z").id]
    keep = float(data.qpos[lift])
    try:
        for h in CLOSE_LIFTS:
            data.qpos[lift] = h
            mujoco.mj_forward(model, data)
            for up in above:
                for yaw in yaws:
                    rot = looking((0.0, 0.0, -1.0), (np.cos(yaw), np.sin(yaw), 0.0))
                    q = solve_camera(model, data, ik, np.asarray(point) + (0.0, 0.0, up), rot, rng, seeds=8)
                    if q is not None:
                        return h, q
    finally:
        data.qpos[lift] = keep
        mujoco.mj_forward(model, data)
    return None


def row_closeup(model, data, ik, row, point):
    """Arm joint angles for a wrist view square onto the row at a sticker's
    `point` (world), close enough to read it, or None."""
    from bila.control.inspection import slot_x, solve_view
    p = insert_book.ShelfFrame(row).to_shelf(np.asarray(point))
    for standoff in CLOSE_STANDOFFS:
        q = solve_view(model, data, row, p[0] - slot_x(row), ik, standoff=standoff, height=p[2] - row.target_board_top)
        if q is not None:
            return q
    return None


def read_tray(labels, model, data, run, ik, tray, rng, log=print):
    """A wrist view of the whole tray from above (the tray's place on the
    cart is known), then a close-up of each sticker found but not read."""
    def look_down(point, what, **kw):
        plan = tray_closeup(model, data, ik, point, rng, **kw)
        if plan is None:
            log(f"  no arm pose looks down on {what}")
            return
        data.ctrl[model.actuator("lift").id] = plan[0]
        run.wait(f"lift to {plan[0]:.2f} m to look into the tray", 1.0)
        run.move_joints(f"look down on {what}", plan[1])
        found, read = labels.look(data, "panda/wrist_cam")
        log(f"  finds {found} stickers, reads {read}")

    centre = data.geom_xpos[model.geom("tray").id].copy()
    centre[2] += 0.20                                       # about the spines' tops
    look_down(centre, "the whole tray", above=OVERVIEW_ABOVE, yaws=(np.pi / 2, -np.pi / 2))
    for o in [o for o in tray if o.name not in labels.numbers and o.name in labels.where][:MAX_CLOSEUPS]:
        look_down(labels.where[o.name], f"{o.asset[5:]}'s sticker")
    run.move_joints("tuck the arm", PANDA_HOME[:7])


# ---------------------------------------------------------------------------
# Insertion with a real pick from the tray
# ---------------------------------------------------------------------------
class CartEpisode(Episode):
    """insert_book's episode, with the book picked out of the tray instead of
    handed over: take_book() is the only step that changes."""

    def __init__(self, model, data, row, viewer=None, log=print, eyes=None):
        super().__init__(model, data, row, viewer, log, eyes)
        self.lift_act = model.actuator("lift").id
        self.lift_q = model.jnt_qposadr[model.joint("lift_z").id]
        self.lift_row = float(data.qpos[self.lift_q])     # where the lift stands for the target row
        self.carrying = False                             # a book swings with the hand (joint_route checks closer)
        self.next_grip_check = 0.0                        # sim time of check_grip's next look
        self.ik.robot -= {g for g in self.ik.robot if model.body(model.geom_bodyid[g]).name == "panda/link0"}
        self.tray_bodies = [(i, model.body_geomadr[i]) for i in range(model.nbody)   # (body, its box) of the tray books
                            if model.body(i).name.startswith("return_") or i == self.book_body]

    def place_arm(self, q):
        """Move the arm to q from wherever the scan left it (Episode's version
        sets qpos, which is fine at an episode's start but snaps the arm here)."""
        m, d = self.m, self.d
        self.run = Runner(m, d, self.ik, d.qpos[self.ik.qadr].copy(), GRIP_OPEN, self.viewer, self.log)
        self.ik.sync(d)
        self.reposition(q)                                   # collision-checked along the way, not just at q

    def drive_lift(self, q, name):
        """Move the lift to q along a min-jerk path (a step in its setpoint
        jolts the arm hard enough to shake a held book loose), then settle."""
        run, d = self.run, self.d
        run.log(f"  {run.t:5.2f} s  {name}")
        q0 = float(d.ctrl[self.lift_act])
        n = int(round(max(0.5, 1.875 * abs(q - q0) / LIFT_SPEED) / self.m.opt.timestep))
        for k in range(1, n + 1):
            u = k / n
            d.ctrl[self.lift_act] = q0 + (q - q0) * u ** 3 * (10 - 15 * u + 6 * u * u)
            run.step()
        run.wait("let the lift settle", 0.3)
        self.ik.sync(self.d)
        self.run.halt()                                  # re-read the hand pose at the new height

    def take_book(self, grasp_sign, centre, yaw=0.0, name="close the gripper on the book"):
        """The hand waits empty at the pose that holds the book for insertion.
        Go to the tray, grasp the book spine-up with the hand in that same pose
        relative to the book, bring it back, and carry on."""
        d, ik, run = self.d, self.ik, self.run
        goal_q, goal_pos, goal_rot = run.q_cmd.copy(), run.pos.copy(), run.rot.copy()   # insert()'s waiting pose
        hand_in_book = self.shelf.rot(yaw).T @ self.grasp_pose(grasp_sign, centre, yaw)[1]
        tip = self.tip_y((run.pos, run.rot))
        if self.D / 2 + 0.10 > tip:
            run.move("take the hand out of the shelf", run.pos + self.ys * (self.D / 2 + 0.10 - tip))

        # The grasp in the tray, and a lift height from which it is reachable without hitting anything.
        R_b = d.xmat[self.book_body].reshape(3, 3)
        R_h = R_b @ hand_in_book
        p_h = d.xpos[self.book_body] + R_b @ self.pad_in_book - R_h @ [0, 0, insert_book.PAD_Z]
        up = np.array([0.0, 0.0, 1.0])
        opening = self.t / 2 + OPEN_SPARE
        seeds = [goal_q] + [np.array(PANDA_HOME[:7]) + [a, 0, 0, 0, 0, 0, 0] for a in (0.8, 1.6, -0.8)]
        # It comes out of its slot until its fore-edge clears the tallest spine in the tray, so it can
        # swing; the hand comes down onto it from that height, so check the arm there. Where that is out
        # of reach at every lift height, out less: the swing's collision check (joint_route) takes the
        # tray's books in.
        floor = d.xpos[self.book_body][2] - self.depth / 2
        spines = max(d.xpos[i][2] + d.xmat[i].reshape(3, 3)[2, 1] * self.m.geom_size[g][1] for i, g in self.tray_bodies)
        clear = spines - floor + 0.06

        def grasp_at(q_lift, rise):
            ik.sync(d, finger=opening)
            ik.d.qpos[self.lift_q] = q_lift
            for seed in seeds:
                q, ok = ik.solve_far(seed, p_h, R_h)
                if ok and not ik.collides(q, allowed={self.book_geom}):
                    q_pre, ok = ik.solve_far(q, p_h + rise * up, R_h)
                    if ok and not ik.collides(q_pre):
                        return q_pre
            return None

        q_lift, rise, q_pre = next(((h, r, q) for r in (clear, clear - 0.03, clear - 0.06) if r > 0.10
                                    for h in (0.45, 0.35, 0.55, 0.25, 0.65) if (q := grasp_at(h, r)) is not None),
                                   (None, None, None))
        if q_pre is None:
            raise RuntimeError("the book in the tray is out of reach at every lift height")
        # Where to carry it: in front of the slot, CARRY_OUT short of the waiting pose. The lift only
        # translates the arm, so the same joints put the hand there with the lift up, dz higher.
        # Found by backing out from the waiting pose in the arm configuration insert() chose and checked
        # for the whole insertion, so the moves back in end in that configuration (solved afresh, it may
        # be another, never checked for the slide and the push).
        ik.sync(d)
        ik.d.qpos[self.lift_q] = self.lift_row
        carry = None
        for out in CARRY_OUT:                                        # as far out as the arm allows
            ik.q_rest = goal_q
            q, worst, ok = ik.follow(goal_q, segment((goal_pos, goal_rot), (goal_pos + self.ys * out, goal_rot), 20),
                                     allowed=frozenset())
            if ok and worst >= MIN_MARGIN and not ik.collides(q):
                carry = (out, q)
                break
        posture = goal_q if carry is not None else None
        if carry is None:                                            # none in that posture: any that reaches it
            carry = next(((out, q) for out in CARRY_OUT for seed in seeds
                          for q, ok in [ik.solve_far(seed, goal_pos + self.ys * out, goal_rot)]
                          if ok and not ik.collides(q)), None)
        if carry is None:
            raise RuntimeError("no arm configuration to carry the book in front of the slot")
        out, q_carry = carry

        self.drive_lift(q_lift, "raise the lift to the tray")
        self.reposition(q_pre)                                       # collision-checked, by way of home if need be
        run.wait("open just past the book's thickness", 0.4, grip=255.0 * opening / 0.04)
        run.move("lower onto its spine", p_h, R_h, speed=0.05)
        run.wait(name.replace("(the hand-over is not modelled)", "out of the tray"), 0.8, grip=GRIP_CLOSED)
        self.measure_in_hand()
        run.move("lift it out of its slot", p_h + rise * up, R_h, speed=0.08)
        # From here on the book swings with the hand: check it for collisions too, and swing it gently.
        # The pads hold only its spine end, so a fast swing twists it in the grip.
        collides, ik.collides = ik.collides, self.collides_carrying
        move_joints = run.move_joints
        run.move_joints = lambda name, q, speed=CARRY_JOINT_SPEED: move_joints(name, q, speed)
        step = run.step
        run.step = lambda: (step(), self.check_grip(run))
        self.carrying = True
        try:
            ik.sync(d)
            if self.joint_route(run.q_cmd, q_carry) is not None:
                self.reposition(q_carry)                             # turn it to face the shelf, lift still up
                self.drive_lift(self.lift_row, "lower the lift back to the row, the book in front of the slot")
            else:
                # No way there with the lift up (the book swings into the bookcase, or the next book in the
                # tray): tuck it in, lower the lift, and then take it out in front of the slot.
                home = np.array(PANDA_HOME[:7])
                for h in np.linspace(float(d.qpos[self.lift_q]), self.lift_row, 12):
                    ik.d.qpos[self.lift_q] = h
                    if ik.collides(home):
                        raise RuntimeError("no way to carry the book from the tray to the slot")
                self.reposition(home, "tuck the book in")
                self.drive_lift(self.lift_row, "lower the lift back to the row")
                self.reposition(q_carry, "take the book out in front of the slot")
        finally:
            ik.collides = collides
            del run.move_joints, run.step
            self.carrying = False
        if posture is not None:
            ik.q_rest = posture                                      # and come back in to insert()'s configuration
        # The book has shifted a little in the grip on the way: from here on place the book, not the
        # hand (as insert_book does), so the hand tilts to hold it straight.
        self.measure_in_hand()
        # Come in a little above where insert() holds it (HOVER over the board): carried in, the book is
        # held less exactly than handed over, and its back, caught on the board's front edge, turned it
        # nose-up in the grip.
        above = max(0.0, min(APPROACH_ABOVE, self.H - self.h - insert_book.HOVER - 0.01))
        run.move("line the book up in front of the slot", *self.hand_for_book(centre + [0, out, above], yaw), speed=0.05)
        run.move("bring it up to the slot", *self.hand_for_book(centre + [0, 0, above], yaw), speed=0.05)
        if above:
            run.move("lower it to just above the board", *self.hand_for_book(centre, yaw), speed=0.02)
        self.measure_in_hand()

    def joint_route(self, q0, q):
        """Episode.joint_route, checked every CARRY_CHECK rad while the arm
        carries a book: its 25 checks along a joint move leave gaps of
        several cm at the far end of a book held at its spine, and the book
        swung out of the tray hit the one in the next slot through one."""
        if not self.carrying:
            return super().joint_route(q0, q)
        ik, home = self.ik, np.array(PANDA_HOME[:7])

        def clear(a, b):
            n = max(25, int(np.ceil(np.max(np.abs(b - a)) / CARRY_CHECK)))
            return not any(ik.collides(a + s * (b - a)) for s in np.linspace(0, 1, n + 1)[1:])

        if clear(q0, q):
            return [q]
        if clear(q0, home) and clear(home, q):
            return [home, q]
        rng = np.random.default_rng(0)
        for _ in range(WAYPOINT_TRIES):
            w = np.clip((q0 + q) / 2 + rng.normal(0, 0.6, 7), ik.lo, ik.hi)
            if not ik.collides(w) and clear(q0, w) and clear(w, q):
                return [w, q]
        return None

    def slip(self):
        """(deg, mm) the held book has turned and moved in the hand since measure_in_hand."""
        d, ik = self.d, self.ik
        R_h = d.xmat[ik.hand].reshape(3, 3)
        R = R_h.T @ d.xmat[self.book_body].reshape(3, 3)
        p = R_h.T @ (d.xpos[self.book_body] - d.xpos[ik.hand])
        turn = np.degrees(np.arccos(np.clip((np.trace(self.in_hand_R.T @ R) - 1) / 2, -1, 1)))
        return float(turn), 1000 * float(np.linalg.norm(p - self.in_hand_p))

    def check_grip(self, run):
        """While carrying: stop if the book is slipping (SLIP_TURN_DEG, SLIP_MM). Read from the
        simulation; a robot would compare its wrist torque with the book's expected twist instead."""
        if run.d.time < self.next_grip_check:
            return
        self.next_grip_check = run.d.time + 0.05
        turn, moved = self.slip()
        if turn > SLIP_TURN_DEG or moved > SLIP_MM:
            raise RuntimeError(f"the book slipped in the grip on the way from the tray "
                               f"({turn:.1f} deg, {moved:.0f} mm): hand it to the librarian")

    def measure_in_hand(self):
        d, ik = self.d, self.ik
        R_h = d.xmat[ik.hand].reshape(3, 3)
        self.in_hand_p = R_h.T @ (d.xpos[self.book_body] - d.xpos[ik.hand])
        self.in_hand_R = R_h.T @ d.xmat[self.book_body].reshape(3, 3)

    def collides_carrying(self, q, allowed=(), clearance=0.0):
        """ArmIK.collides, with the held book moved along with the hand and
        counted as part of the arm (its contacts with the fingers aside),
        clearance included."""
        m, ik = self.m, self.ik
        ik.d.qpos[ik.qadr] = q
        mujoco.mj_kinematics(m, ik.d)
        R_h = ik.d.xmat[ik.hand].reshape(3, 3)
        qa = m.jnt_qposadr[self.book_joint]
        ik.d.qpos[qa:qa + 3] = ik.d.xpos[ik.hand] + R_h @ self.in_hand_p
        ik.d.qpos[qa + 3:qa + 7] = insert_book.mat2quat(R_h @ self.in_hand_R)
        book = {self.book_geom} | {g for g, b in ik.book_of.items() if b == self.book_geom}
        ik.forward(clearance, also=book)
        held = ik.robot | book
        for c in ik.d.contact[:ik.d.ncon]:
            if (c.geom1 in held) == (c.geom2 in held):
                continue
            other = c.geom2 if c.geom1 in held else c.geom1
            if c.dist < -0.001 or (c.dist < clearance and other in ik.structure):
                return True
        return False


# ---------------------------------------------------------------------------
def setup(scenario, seed, log=print):
    """The cart scene for a scenario and seed, settled: (model, data, row,
    tray books, {book: call number}, shelf map). Returned books that belong
    on other rows get spare numbers of those rows."""
    cfg, row, tray, librarian = sample(scenario, seed)
    numbers, shelf_map = call_numbers(row)
    spare = [n for level, _, first, last in shelf_map if level != row.target_level
             for n in range(first + STEP // 2, last, STEP)]
    others = [o for o in tray if o is not row.target_book]
    for o, n in zip(others, np.random.default_rng(seed).choice(spare, len(others), replace=False)):
        numbers[o.name] = int(n)
    model, tray = build(cfg, row, tray, numbers)
    data = mujoco.MjData(model)
    log(f"settled: {settle(model, data, row)['stable']}")
    log("shelf map (the robot knows this):")
    for level, top, first, last in shelf_map:
        log(f"  row {level}: board top {top:.2f} m, call numbers {dewey(first)}-{dewey(last)}")
    for o, why in librarian:
        log(f"  to the librarian, not the tray: {o.asset[5:]} ({'; '.join(why)})")
    return model, data, row, tray, numbers, shelf_map


def run(model, data, row, tray, shelf_map, labels, eyes_for, viewer=None, log=print, seed=0, numbers=None,
        policy=None):
    """Steps 1-5 with the wrist camera: read the tray (`labels`: OracleLabels
    or LearnedLabels), find the row, lift to it, and reshelve the book with
    eyes_for(its call number) as the episode's eyes. Returns the episode's
    result (insert_book.report formats it). `numbers` (the true ones) only
    flags misreads."""
    # 1-2. Read the tray and look each book up in the shelf map.
    ik = ArmIK(model, PANDA_HOME[:7])
    runner = Runner(model, data, ik, data.qpos[ik.qadr].copy(), 255.0, viewer, log)
    try:
        read_tray(labels, model, data, runner, ik, tray, np.random.default_rng(seed), log)
    finally:
        labels.close()
    read = labels.read([o.name for o in tray])
    log(f"read {len(read)}/{len(tray)} tray spines:")
    todo = None
    for o in tray:
        if o.name not in read:
            log(f"  {o.asset[5:]}: its sticker can't be read from above")
            continue
        level, top = row_for(read[o.name], shelf_map)
        here = level == row.target_level
        wrong = f"  MISREAD (it says {dewey(numbers[o.name])})" if numbers and read[o.name] != numbers[o.name] else ""
        log(f"  {dewey(read[o.name])} {cutter(o)} ({o.asset[5:]}) belongs on row {level} (board {top:.2f} m)"
            f"{'  <- shelve this one' if here else ''}{wrong}")
        todo = o if here else todo
    if todo is None:
        raise RuntimeError("no readable book in the tray belongs on the simulated row")

    # 3. Lift to that row (the cart already stands for it); the episode's first look reads the row's spines.
    n = read[todo.name]
    level, top = row_for(n, shelf_map)
    data.ctrl[model.actuator("lift").id] = top - REACH_BELOW - DECK
    runner.wait(f"lift to row {level}: {top - REACH_BELOW - DECK:.2f} m", 1.0)

    # 4-5. Make room there and insert the book, picked out of the tray.
    eyes = eyes_for(n)
    try:
        ep = CartEpisode(model, data, row, viewer, log, eyes)
        if policy is None:                                   # insert_book's rules choose the steps
            return ep.reshelve(GAP_MARGIN)
        from bila.control import high_level                  # a learned high-level policy does (high_level.py)
        return high_level.run(ep, high_level.QPolicy(os.path.expanduser(policy)))
    finally:
        eyes.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenario", choices=("clear", "tight", "leaning", "packed"), default="tight")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--view", action="store_true", help="watch in the MuJoCo viewer (run with mjpython on macOS)")
    ap.add_argument("--labels", choices=("oracle", "learned"), default="oracle",
                    help="read call numbers from the simulation (oracle) or the camera images (learned; label_reader.py)")
    ap.add_argument("--label-weights", default="~/bila_data/labels", help="folder with label_reader's finder.pt and reader.pt")
    ap.add_argument("--policy", help="a fitted high-level policy (high_level.py fit's q.npz) instead of the rules")
    ap.add_argument("--segmenter", help="learned segmenter weights (segmenter.pt) for the book masks; "
                                        "default: the simulation's (notebooks/04_train_segmenter.ipynb trains one)")
    a = ap.parse_args()

    weights = os.path.expanduser(a.label_weights)
    learned = a.labels == "learned" and LearnedLabels.available(weights)
    if a.labels == "learned" and not learned:
        print(f"no label reader in {weights}: reading call numbers from the simulation")
    if learned:
        from bila.vision import label_reader  # noqa: F401  (imports torch._dynamo, before any render)
    segmenter = None
    if a.segmenter:
        from bila.vision.learned_segmenter import LearnedSegmenter    # before any rendering (see learned_segmenter.py)
        segmenter = LearnedSegmenter(os.path.expanduser(a.segmenter))
    model, data, row, tray, numbers, shelf_map = setup(a.scenario, a.seed)
    labels, eyes_for = readers(model, row, tray, numbers, a.seed, weights if learned else None, segmenter)

    viewer = None
    if a.view:
        viewer = mujoco.viewer.launch_passive(model, data)
        with viewer.lock():
            mujoco.mjv_defaultFreeCamera(model, viewer.cam)
    try:
        print(report(run(model, data, row, tray, shelf_map, labels, eyes_for, viewer, seed=a.seed, numbers=numbers,
                         policy=a.policy)))
    except RuntimeError as e:                        # no safe way to do it in this scene, or it jammed
        print(f"  FAIL: {e}")
    if viewer is not None:
        print("Close the viewer window to exit.")
        while viewer.is_running():
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(model.opt.timestep)


def readers(model, row, tray, numbers, seed, weights=None, segmenter=None, noise="realistic", fiducials=True):
    """(tray labels, eyes_for): what reads the tray's stickers, and the
    episode's eyes for a call number (the wrist camera's RGBDPerception, with
    the row's stickers read in every look: the slot from them on the first,
    and each book read sized from the library's catalogue). weights: label_reader's
    folder, to read the stickers from the images; None: the simulation's."""
    from bila.vision.call_numbers_seen import LearnedStickers, OracleStickers, ReadingEyes, catalogue
    from bila.vision.rgbd_perception import make_eyes
    texts, database = catalogue(numbers, row.books + tray + [b for fl in row.filler_levels for b in fl.books])
    labels = LearnedLabels(model, weights, texts) if weights else OracleLabels(model, numbers)
    stickers = LearnedStickers(weights, texts) if weights else OracleStickers(model, numbers)
    return labels, lambda n: ReadingEyes(make_eyes(model, row, seed, noise, segmenter, fiducials), stickers, row, n,
                                         database=database)


if __name__ == "__main__":
    main()
