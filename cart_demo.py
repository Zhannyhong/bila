"""
Book-return cart: the Panda on a lift, on a wheeled cart with a return tray of
real books standing spine-up (one per slot) and a row camera on the lift, in
front of a bookcase of the repo's real books, each with a call-number sticker
on its spine.

    uv run python cart_demo.py --scenario tight             # headless: prints each step and the score
    uv run mjpython cart_demo.py --scenario tight --view    # watch it (macOS needs mjpython)
    uv run python cart_demo.py --scenario leaning --seed 3  # clear, tight, leaning or packed

What the robot does, as a librarian would:
  1. Read the call numbers in the tray (tray_cam, looking down on the spines).
  2. Look each one up in the shelf map, which it knows: every row's board
     height and the first and last call number on it. That names the row.
  3. Raise the lift to that row, lean the arm out of row_cam's view, and read
     the spines along the row, to find the slot: between the two books whose
     numbers bracket the book's.
  4. Make room there with the empty hand (insert_book's skills: widen a tight
     gap, stand a leaning neighbour up, shove a packed row aside).
  5. Pick the book out of the tray and insert it. The tray holds it spine-up,
     so the hand comes down onto its spine exactly as insert_book holds a book
     for insertion; the arm then turns it to face the shelf. This replaces
     insert_book's "hand-over (not modelled)".

Reading a sticker here means enough of it is visible in the camera image; its
number then comes from the simulation. With `--labels learned` the numbers
are read from the images instead (label_reader.py; train it first): tray_cam
and row_cam find the stickers and read what they can, and the wrist camera
takes a close look (~20 cm) at those they can't, in the tray, and in the row
until the two books whose numbers bracket the book's are read side by side.
Those close-ups are collision-checked joint moves (move_checked); one with no
clear way there is skipped.
Books the robot can't handle (shelvable()) go to the librarian, not the tray.
Only the scenario's target row has a slot to fill: the other tray books are
read and looked up, but stay in the tray.

Cart frame (the old robot mount, which faces the shelf): +x toward the shelf,
+y to the tray side, z up, origin at the Panda's base at the target row's
usual height. The cart does not drive yet. Its parts are visual except the
tray, its dividers, the lift rail and the deck, which the arm must not hit.
The cameras' mounts are visual too: the cameras only show the scene and read
the labels, so the arm may pass through them.
"""

import argparse
import os
import time
import xml.etree.ElementTree as ET
from types import SimpleNamespace

import numpy as np
import mujoco
import mujoco.viewer
from PIL import Image, ImageDraw, ImageFont

import insert_book
from shelf_scene import (PANDA_HOME, ShelfConfig, _catalogue_book, _f, add_book_geoms, add_book_visual_assets,
                         board_tops, build_scene, sample_row, settle)
from insert_book import (DEFAULT_PANDA, FINGERTIP_Z, GAP_MARGIN, GRIP_CLOSED, GRIP_OPEN, PALM_GAP, PALM_Z, ArmIK, Episode,
                         Runner, compile_robot_scene, report, report_collisions)

HERE = os.path.dirname(os.path.abspath(__file__))
LABEL_DIR = os.path.join(HERE, "generated_assets", "labels")   # git-ignored, made on demand

LIFT_MAX = 0.70             # lift travel: the Panda's base this far above the deck at most
DECK = 0.16                 # deck top above the floor
REACH_BELOW = 0.40          # the Panda's base stands this far below the board it works on (as shelf_scene)
TRAY = (-0.12, 0.50, 0.72)  # tray centre (cart x, y) and its floor's height above the floor
TRAY_HALF = (0.27, 0.15)
SLOT_CLEAR = 0.025          # room each side of a book in its slot: finger 12 mm + opening 8 mm + 5 mm
DIVIDER_H = 0.08
TRAY_CAM_H = 1.0            # tray camera's boom this high above the tray floor: above the wrist (link 6) as the arm
                            # lifts the deepest book out of its slot (the boom is visual only; at 0.75 m the wrist
                            # showed passing through it)
RAIL_X = -0.40              # lift rail's centre behind the Panda's base: clear of the elbow at home and on the way to
                            # SCAN_POSE (at -0.22 the tucked arm stood 10 mm inside it on low rows)
TRAY_BOOKS = 5              # books the tray is loaded with (if they fit)
OPEN_SPARE = 0.008          # the fingers open this much wider than the book, each side
LIFT_SPEED = 0.10           # m/s, peak, of the lift's moves
CARRY_JOINT_SPEED = 0.3     # rad/s, peak, of the arm's joint moves while it carries a book
PAD_TORSION = 0.05          # torsional friction of the finger pads (m): resists a held book twisting
CARRY_OUT = (0.12, 0.09, 0.06, 0.04, 0.02)   # the book comes down on the lift this far in front of its pre-insertion pose
MAX_KG = 1.5                # heaviest book the grip is trusted with (a guess, to be measured)
STEP = 10                   # call numbers go up in tens along the shelves, so a returned book fits between two
CAM_MOUNT = (-0.14, -0.27, 0.55)   # row camera (D435-like) on the carriage, from the Panda's base
CAM_AIM = (0.75, 0.0, 0.45)        # where it looks: the target row's front, ~0.40 m + a third of a level above the base
SCAN_POSE = [0.6, -1.2, 0, -1.5, 0, 2.0, 0.7853]  # arm leaned back and turned to the tray side, out of row_cam's view


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
        why.append("too heavy to trust the grip")
    return why


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

    # Lift: a rail behind the arm, and a carriage on the lift_z joint carrying the Panda, cantilevered
    # out from the rail on a bracket.
    box("lift_rail", (0.035, 0.09, (LIFT_MAX + 0.15) / 2), (RAIL_X, 0, deck + (LIFT_MAX + 0.15) / 2), "0.25 0.25 0.28 1")
    lift = ET.Element("body", name="lift", pos=_f(0, 0, deck))
    ET.SubElement(lift, "joint", name="lift_z", type="slide", axis="0 0 1", range=_f(0, LIFT_MAX), damping="3000")
    box("carriage", (0.11, 0.10, 0.015), (0, 0, -0.015), "0.35 0.35 0.38 1", False, lift)
    bracket = (-0.11, RAIL_X + 0.035)                    # from the carriage's back edge to the rail's face
    box("carriage_bracket", ((bracket[0] - bracket[1]) / 2, 0.09, 0.03), (sum(bracket) / 2, 0, -0.03),
        "0.35 0.35 0.38 1", False, lift)
    attach = cart.find("attach")
    cart.remove(attach); lift.append(attach); cart.append(lift)
    act = root.find("actuator")
    if act is None:
        act = ET.SubElement(root, "actuator")
    ET.SubElement(act, "position", name="lift", joint="lift_z", kp="60000", ctrlrange=_f(0, LIFT_MAX))

    # Row camera on a post on the carriage, beside the arm on the side away from the tray. It rides the lift
    # with the arm, and the arm's base tracks the target board, so it sees every row from the same angle.
    cx, cy, cz = CAM_MOUNT
    # Its mount is visual only, like the tray camera's: the cameras show the scene and read the labels, they
    # aren't part of the robot, so the arm may pass through them.
    box("cam_arm", (0.015, (0.09 - cy) / 2 - 0.09, 0.015), (cx, (cy - 0.09) / 2, -0.03), grey, False, lift)
    geom(lift, "cam_post", grey, False, type="cylinder", size="0.015", fromto=_f(cx, cy, -0.03, cx, cy, cz - 0.02))
    box("cam_head", (0.0125, 0.045, 0.0125), (cx, cy, cz), dark, False, lift)
    fwd = np.subtract(CAM_AIM, CAM_MOUNT) / np.linalg.norm(np.subtract(CAM_AIM, CAM_MOUNT))
    right = np.array([fwd[1], -fwd[0], 0.0]) / np.hypot(fwd[0], fwd[1])
    ET.SubElement(lift, "camera", name="row_cam", pos=_f(cx + 0.015, cy, cz), xyaxes=_f(*right, *np.cross(right, fwd)), fovy="58")

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

    # Tray camera on a boom over the tray, looking straight down at the spines. Its fov sees the tray as from
    # 0.75 m, as the images label_reader.py was trained on do.
    px, py, ptop = tx - hx - 0.06, ty + hy + 0.04, tz + TRAY_CAM_H
    geom(cart, "tray_cam_post", grey, False, type="cylinder", size="0.015", fromto=_f(px, py, deck, px, py, ptop))
    geom(cart, "tray_cam_boom", grey, False, type="cylinder", size="0.012", fromto=_f(px, py, ptop, tx, ty, ptop))
    ET.SubElement(cart, "camera", name="tray_cam", pos=_f(tx, ty, ptop - 0.02), xyaxes="1 0 0 0 1 0",
                  fovy=_f(2 * np.degrees(np.arctan(np.tan(np.radians(30)) * 0.73 / (TRAY_CAM_H - 0.02)))))

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

    # Call-number stickers near the bottom of every spine.
    books = {b.name: b for b in row.books + tray_books + [b for fl in row.filler_levels for b in fl.books]}
    for body in wb.iter("body"):
        b = books.get(body.get("name"))
        if b is None or b.name not in numbers:
            continue
        n, hx = numbers[b.name], min(0.85 * b.thickness / 2, 0.012)
        ET.SubElement(asset, "texture", name=f"label{n}", type="2d", hflip="true", file=sticker(n, cutter(b), hx / 0.03))
        ET.SubElement(asset, "material", name=f"label{n}", texture=f"label{n}")
        ET.SubElement(body, "geom", name=b.name + "_label", type="box", material=f"label{n}", group="2",
                      size=_f(hx, 0.0005, 0.03),
                      pos=_f(0, b.depth / 2 + 0.0008, -b.height / 2 + 0.045), contype="0", conaffinity="0", mass="0")

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


def read_labels(model, data, camera, names, min_pixels=25):
    """The books in `names` whose sticker shows clearly enough in `camera`'s image to read."""
    # Without multisampling: anti-aliased edges blend the id colours into ids of no object (an IndexError).
    samples, model.vis.quality.offsamples = model.vis.quality.offsamples, 0
    try:
        r = mujoco.Renderer(model, 480, 640)
    finally:
        model.vis.quality.offsamples = samples
    with r:
        r.enable_segmentation_rendering()
        r.update_scene(data, camera=camera)
        seg = r.render()[..., 0]
    return [n for n in names if (seg == model.geom(n + "_label").id).sum() >= min_pixels]


# ---------------------------------------------------------------------------
# Reading the stickers from the images (--labels learned)
# ---------------------------------------------------------------------------
CLOSE_STANDOFFS = (0.18, 0.22, 0.14)   # m: wrist close-ups of the row's stickers, this far in front of the board
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
    place of read_labels' "visible enough, so the simulation's number". The
    catalogue (every sticker's text and its number) is what the library
    knows. A sticker read is tied to the book it is on by the simulation's
    object ids under it: bookkeeping only, since the demo names books by
    their bodies (a real robot would go by where the sticker is, which is
    also kept: the sticker's centre in the world, from the depth image)."""

    def __init__(self, model, weights: str, catalogue: dict):
        from label_reader import LabelReader                 # imports torch (and torch._dynamo, before any render)
        from rgbd_perception import book_of_geom
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
        from camera import DepthNoise, WristCamera
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


def move_checked(run, ik, data, name, q) -> bool:
    """A joint move to a camera view, collision-checked as the episode's
    are (Episode.joint_route: straight there, by way of home, or by way of
    another configuration); False, without moving, if no way is clear. (The
    views come before the episode, so there is no Episode to reposition with:
    joint_route needs only the planner.)"""
    ik.sync(data)                                       # the scene as it is now: the lift's height, the books
    route = Episode.joint_route(SimpleNamespace(ik=ik, placed=None, book_geom=None), run.q_cmd,
                                np.asarray(q, dtype=float))
    if route is None:
        return False
    for w in route[:-1]:
        run.move_joints("tuck the arm" if np.allclose(w, PANDA_HOME[:7]) else "move the arm out of the way", w)
    run.move_joints(name, q)
    return True


def move_back(run, ik, data, name, q):
    """From a close-up back to a pose the demo goes on from: checked as
    move_checked, else straight there (as the demo's other joint moves go)."""
    if not move_checked(run, ik, data, name, q):
        print(f"  no clear way to {name}: going straight there")
        run.move_joints(name, q)


def tray_closeup(model, data, ik, point, rng, above=CLOSE_ABOVE, yaws=(0.0, np.pi / 2, np.pi, -np.pi / 2)):
    """(lift height, arm joint angles) for a wrist view straight down onto
    `point` (a tray sticker) from one of `above` over it, the image's right
    turned by one of `yaws` from the world's x; None if the arm reaches none."""
    from inspection import looking, solve_camera
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
    from inspection import slot_x, solve_view
    p = insert_book.ShelfFrame(row).to_shelf(np.asarray(point))
    for standoff in CLOSE_STANDOFFS:
        q = solve_view(model, data, row, p[0] - slot_x(row), ik, standoff=standoff, height=p[2] - row.target_board_top)
        if q is not None:
            return q
    return None


def bracket(row, read, n):
    """(index of the book with the largest number below n, of the one with
    the smallest above it), among the row's books read; None for a side with
    none read."""
    below = [i for i, b in enumerate(row.books) if b.name in read and read[b.name] < n]
    above = [i for i, b in enumerate(row.books) if b.name in read and read[b.name] > n]
    return (max(below, key=lambda i: read[row.books[i].name]) if below else None,
            min(above, key=lambda i: read[row.books[i].name]) if above else None)


def read_tray(labels, model, data, run, ik, tray, rng):
    """tray_cam; then, if it left any tray book unread, a wrist view of the
    whole tray from above (the tray's place on the cart is known), and a
    close-up of each sticker still unread that was found."""
    def look_down(point, what, **kw):
        plan = tray_closeup(model, data, ik, point, rng, **kw)
        if plan is None:
            print(f"  no arm pose looks down on {what}")
            return
        data.ctrl[model.actuator("lift").id] = plan[0]
        run.wait(f"lift to {plan[0]:.2f} m to look into the tray", 1.0)
        if not move_checked(run, ik, data, f"look down on {what}", plan[1]):
            print(f"  no clear way to look down on {what}")
            return
        found, read = labels.look(data, "panda/wrist_cam")
        print(f"  finds {found} stickers, reads {read}")

    unread = lambda: [o for o in tray if o.name not in labels.numbers]
    found, read = labels.look(data, "tray_cam")
    print(f"tray_cam finds {found} stickers, reads {read}")
    if unread():
        centre = data.geom_xpos[model.geom("tray").id].copy()
        tops = [labels.where[o.name][2] for o in tray if o.name in labels.where]
        centre[2] = np.median(tops) if tops else centre[2] + 0.20         # the spines' tops, or about
        look_down(centre, "the whole tray", above=OVERVIEW_ABOVE, yaws=(np.pi / 2, -np.pi / 2))
    for o in [o for o in unread() if o.name in labels.where][:MAX_CLOSEUPS]:
        look_down(labels.where[o.name], f"{o.asset[5:]}'s sticker")
    move_back(run, ik, data, "tuck the arm", PANDA_HOME[:7])


def read_row(labels, model, data, run, ik, row, n):
    """row_cam, then close-ups near where the book goes until the two books
    whose numbers bracket n are read and stand side by side."""
    from inspection import slot_x
    found, read = labels.look(data, "row_cam")
    print(f"row_cam finds {found} stickers, reads {read}")
    shelf, looked = insert_book.ShelfFrame(row), set()
    for _ in range(MAX_CLOSEUPS):
        i_lo, i_hi = bracket(row, labels.read([b.name for b in row.books]), n)
        if i_lo is not None and i_hi is not None and abs(i_lo - i_hi) == 1:
            break
        lo, hi = sorted(i for i in (i_lo, i_hi) if i is not None) if i_lo is not None and i_hi is not None else (-1, len(row.books))
        unread = [i for i, b in enumerate(row.books) if lo < i < hi and b.name not in labels.numbers
                  and b.name in labels.where and b.name not in looked]
        if unread:                                  # the unread sticker nearest the map's place for the slot
            i = min(unread, key=lambda i: abs(shelf.to_shelf(labels.where[row.books[i].name])[0] - slot_x(row)))
            what, point = f"book {i}'s sticker", labels.where[row.books[i].name]
            looked.add(row.books[i].name)
        elif "slot" not in looked:                  # none found there: the map's place for the slot itself
            what = "the map's place for the slot"
            point = shelf.to_world(np.array([slot_x(row), row.shelf_depth / 2, row.target_board_top + STICKER_UP]))
            looked.add("slot")
        else:
            break
        q = row_closeup(model, data, ik, row, point)
        if q is None:
            print(f"  no arm pose for a close-up of {what}")
            continue
        if not move_checked(run, ik, data, f"close-up of {what}", q):
            print(f"  no clear way to a close-up of {what}")
            continue
        found, read = labels.look(data, "panda/wrist_cam")
        print(f"  close-up finds {found}, reads {read}")
    move_back(run, ik, data, "lean the arm out of row_cam's view", SCAN_POSE)


# ---------------------------------------------------------------------------
# Insertion with a real pick from the tray
# ---------------------------------------------------------------------------
class CartEpisode(Episode):
    """insert_book's episode, with the book picked out of the tray instead of
    handed over: take_book() is the only step that changes."""

    def __init__(self, model, data, row, viewer=None, log=print):
        super().__init__(model, data, row, viewer, log)
        self.lift_act = model.actuator("lift").id
        self.lift_q = model.jnt_qposadr[model.joint("lift_z").id]
        self.lift_row = float(data.qpos[self.lift_q])     # where the lift stands for the target row
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
        goal_q = run.q_cmd.copy()
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

        def grasp_at(q_lift):
            ik.sync(d, finger=opening)
            ik.d.qpos[self.lift_q] = q_lift
            for seed in seeds:
                q, ok = ik.solve_far(seed, p_h, R_h)
                if ok and not ik.collides(q, allowed={self.book_geom}):
                    q_pre, ok = ik.solve_far(q, p_h + 0.15 * up, R_h)
                    if ok and not ik.collides(q_pre):
                        return q_pre
            return None

        q_lift, q_pre = next(((h, q) for h in (0.45, 0.35, 0.55, 0.25, 0.65) if (q := grasp_at(h)) is not None),
                             (None, None))
        if q_pre is None:
            raise RuntimeError("the book in the tray is out of reach at every lift height")
        # Where to carry it: in front of the slot, CARRY_OUT short of the waiting pose. The lift only
        # translates the arm, so the same joints put the hand there with the lift up, dz higher.
        goal_pos, goal_rot = run.pos.copy(), run.rot.copy()
        ik.sync(d)
        ik.d.qpos[self.lift_q] = self.lift_row
        carry = next(((out, q) for out in CARRY_OUT for seed in seeds        # as far out as the arm (and
                      for q, ok in [ik.solve_far(seed, goal_pos + self.ys * out, goal_rot)]   # row_cam's post) allow
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
        # Out of its slot until its fore-edge clears the tallest spine in the tray, so it can swing.
        floor = d.xpos[self.book_body][2] - self.depth / 2
        spines = max(d.xpos[i][2] + d.xmat[i].reshape(3, 3)[2, 1] * self.m.geom_size[g][1]
                     for i, g in self.tray_bodies)
        run.move("lift it out of its slot", run.pos + (spines - floor + 0.06) * up, speed=0.08)
        # From here on the book swings with the hand: check it for collisions too, and swing it gently.
        # The pads hold only its spine end, so a fast swing twists it in the grip.
        ik.collides = self.collides_carrying
        move_joints = run.move_joints
        run.move_joints = lambda name, q, speed=CARRY_JOINT_SPEED: move_joints(name, q, speed)
        try:
            self.reposition(q_carry)                                 # turn it to face the shelf, lift still up
        finally:
            del ik.collides, run.move_joints
        self.drive_lift(self.lift_row, "lower the lift back to the row, the book in front of the slot")
        # The book has shifted a little in the grip on the way: from here on place the book, not the
        # hand (as insert_book does), so the hand tilts to hold it straight.
        self.measure_in_hand()
        run.move("line the book up in front of the slot", *self.hand_for_book(centre + [0, out, 0], yaw), speed=0.05)
        run.move("bring it up to the slot", *self.hand_for_book(centre, yaw), speed=0.05)
        self.measure_in_hand()

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
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenario", choices=("clear", "tight", "leaning", "packed"), default="tight")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--view", action="store_true", help="watch in the MuJoCo viewer (run with mjpython on macOS)")
    ap.add_argument("--labels", choices=("oracle", "learned"), default="oracle",
                    help="read call numbers from the simulation (oracle) or the camera images (learned; label_reader.py)")
    ap.add_argument("--label-weights", default="~/bila_data/labels", help="folder with label_reader's finder.pt and reader.pt")
    a = ap.parse_args()

    cfg, row, tray, librarian = sample(a.scenario, a.seed)
    numbers, shelf_map = call_numbers(row)
    spare = [n for level, _, first, last in shelf_map if level != row.target_level
             for n in range(first + STEP // 2, last, STEP)]
    others = [o for o in tray if o is not row.target_book]
    for o, n in zip(others, np.random.default_rng(a.seed).choice(spare, len(others), replace=False)):
        numbers[o.name] = int(n)                                     # returned books that belong on other rows
    model, tray = build(cfg, row, tray, numbers)
    data = mujoco.MjData(model)
    labels = None
    if a.labels == "learned":
        weights = os.path.expanduser(a.label_weights)
        if LearnedLabels.available(weights):
            books = row.books + tray + [b for fl in row.filler_levels for b in fl.books]
            labels = LearnedLabels(model, weights, {sticker_text(numbers[b.name], cutter(b)): numbers[b.name]
                                                    for b in books if b.name in numbers})
        else:
            print(f"no label reader in {weights}: reading call numbers from the simulation")
    print(f"settled: {settle(model, data, row)['stable']}")
    print("shelf map (the robot knows this):")
    for level, top, first, last in shelf_map:
        print(f"  row {level}: board top {top:.2f} m, call numbers {dewey(first)}-{dewey(last)}")
    for o, why in librarian:
        print(f"  to the librarian, not the tray: {o.asset[5:]} ({'; '.join(why)})")

    viewer = None
    if a.view:
        viewer = mujoco.viewer.launch_passive(model, data)
        with viewer.lock():
            mujoco.mjv_defaultFreeCamera(model, viewer.cam)

    # 1-2. Read the tray and look each book up in the shelf map.
    if labels is None:
        seen = read_labels(model, data, "tray_cam", [o.name for o in tray])
        read = {o.name: numbers[o.name] for o in tray if o.name in seen}
        print(f"tray_cam reads {len(seen)}/{len(tray)} spines:")
    else:
        ik = ArmIK(model, PANDA_HOME[:7])
        run = Runner(model, data, ik, data.qpos[ik.qadr].copy(), 255.0, viewer)
        read_tray(labels, model, data, run, ik, tray, np.random.default_rng(a.seed))
        read = labels.read([o.name for o in tray])
        print(f"read {len(read)}/{len(tray)} tray spines:")
    todo = None
    for o in tray:
        if o.name not in read:
            print(f"  {o.asset[5:]}: its sticker can't be read from above")
            continue
        level, top = row_for(read[o.name], shelf_map)
        here = level == row.target_level
        wrong = f"  MISREAD (it says {dewey(numbers[o.name])})" if read[o.name] != numbers[o.name] else ""
        print(f"  {dewey(read[o.name])} {cutter(o)} ({o.asset[5:]}) belongs on row {level} (board {top:.2f} m)"
              f"{'  <- shelve this one' if here else ''}{wrong}")
        todo = o if here else todo
    if todo is None:
        raise SystemExit("no readable book in the tray belongs on the simulated row")

    # 3. Lift to that row (the cart already stands for it), lean the arm aside, and scan the row for the slot.
    n = read[todo.name]
    level, top = row_for(n, shelf_map)
    ik = ArmIK(model, PANDA_HOME[:7])
    run = Runner(model, data, ik, data.qpos[ik.qadr].copy(), 255.0, viewer)
    data.ctrl[model.actuator("lift").id] = top - REACH_BELOW - DECK
    run.wait(f"lift to row {level}: {top - REACH_BELOW - DECK:.2f} m", 1.0)
    run.move_joints("lean the arm out of row_cam's view", SCAN_POSE)
    if labels is None:
        visible = read_labels(model, data, "row_cam", [b.name for b in row.books])
        print(f"row_cam reads {len(visible)}/{len(row.books)} spines on row {level}")
        read = {b.name: numbers[b.name] for b in row.books if b.name in visible}
    else:
        read_row(labels, model, data, run, ik, row, n)
        read = labels.read([b.name for b in row.books])
        wrong = [b.name for b in row.books if b.name in read and read[b.name] != numbers[b.name]]
        print(f"read {len(read)}/{len(row.books)} spines on row {level}" + (f"; MISREAD {wrong}" if wrong else ""))
    i_lo, i_hi = bracket(row, read, n)
    if i_lo is None or i_hi is None:
        raise SystemExit(f"can't read call numbers on both sides of where {dewey(n)} goes")
    if abs(i_lo - i_hi) != 1:
        raise SystemExit(f"{dewey(n)} goes somewhere between books {i_lo} and {i_hi}: look closer")
    print(f"{dewey(n)} goes between {dewey(read[row.books[i_lo].name])} and {dewey(read[row.books[i_hi].name])}: "
          f"slot {max(i_lo, i_hi)} (the scene's: {row.target_index})")
    run.move_joints("tuck the arm", PANDA_HOME[:7])
    if labels is not None:
        labels.close()

    # 4-5. Make room there and insert the book, picked out of the tray: insert_book's controller
    # (Episode.reshelve) takes whatever steps the slot needs, and its insert picks the book (take_book).
    ep = CartEpisode(model, data, row, viewer)
    try:
        print(report(ep.reshelve(GAP_MARGIN)))
    except RuntimeError as e:                        # no safe way to do it in this scene, or it jammed
        print(f"  FAIL{' [' + ep.info['method'] + ']' if 'method' in ep.info else ''}: {e}"
              f"{report_collisions(ep.collisions())}")
    if viewer is not None:
        print("Close the viewer window to exit.")
        while viewer.is_running():
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(model.opt.timestep)


if __name__ == "__main__":
    main()
