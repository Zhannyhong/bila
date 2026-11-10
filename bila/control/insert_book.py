"""
Scripted insertion of the target book: one controller for the clear, tight,
leaning and packed scenarios.

    uv run python -m bila.control.insert_book --scenario clear --seed 0              # one episode, headless
    uv run python -m bila.control.insert_book --scenario tight --seed 0 --trials 50  # success rate over seeds 0..49
    uv run mjpython -m bila.control.insert_book --scenario tight --seed 0 --view     # watch it (macOS needs mjpython)
    uv run mjpython -m bila.control.insert_book --scenario leaning --seed 12 --view  # a leaner it has to clamp
    uv run mjpython -m bila.control.insert_book --scenario packed --seed 7 --view    # a packed row
    uv run python -m bila.control.insert_book --scenario tight --seed 0 --perception camera   # decide from the wrist camera
    uv run mjpython -m bila.control.insert_book --scenario tight --seed 0 --perception camera --view   # ... and see what it saw

The controller (Episode.reshelve) looks at the slot, takes the high-level step
it needs next, and looks again. The scenario only picks the row; every row
goes through the same rules (shelf frame: x along the row, y out of the
shelf, z up):

    a neighbour leans into the slot (> LEAN_MIN_DEG)               -> stand_up
    else the gap is short of the book + GAP_MARGIN (at the spines, or at
    its narrowest after a shove or a stand-up, which can leave a neighbour
    turned):
        the fingers, stacked (10 mm along the row), fit in it      -> widen
        they don't (the neighbours touch)                          -> shove
    else (or nothing more would widen it, and the book fits)       -> insert

Each step starts from wherever the arm is and leaves the hand out of the
shelf, so they can follow one another in any order, and a step that fell
short is simply taken again: a second stand-up; a second widening on the
other side; a third on the same side as the first, with the arm configured
for force, if the book still doesn't fit. A clear row is just insert, a
tight one widen then insert, a leaning one stand_up then (widen,) insert, a
packed one shove then (widen,) insert.

High-level steps
  stand_up(side)  Lift the leaning neighbour along an arc about the bottom
                  corner it pivots on, with the closed fingers (stacked) under
                  it as far up its face as they fit, pressing square on it so
                  it turns rather than slides, to just short of upright (within
                  its tipping angle, so it stands up the rest of the way by
                  itself). If the fingers can't get under it at any height,
                  clamp it instead: pinch it across its thickness near the
                  spine, with the hand rolled to its lean, and swing it up about
                  its bottom edge. A book later let go in a gap more than twice
                  its thickness stands SIDE_GAP off the taller upright neighbour.
  widen(margin)   Put the stacked fingers into the gap and push one neighbour
                  aside, as hard as the arm can, until the gap at the spines is
                  the book plus margin: the side with room for all of it first,
                  re-pushing by whatever the arm's give left short. The fingers
                  only reach ~4 cm past the spines, so the neighbour swivels
                  rather than slides: the gap widens at the spines but less at
                  the back, and the book itself widens the rest as it slides in.
                  Books the palm would sweep past that stand well proud of the
                  neighbour (it went back for a shove) go back level with it
                  first, or the fingertips would get none of its face.
  shove(margin)   Where the neighbours touch, the room can only come from the
                  row's free space at one end, so every book between the slot
                  and there has to slide. Push the books beside the neighbour on
                  that side straight back (push_back), so it sticks out of the
                  row, pinch it (hand at its yaw, closing at CLOSE_SPEED), and
                  shove it and every book beyond into the room, with the arm
                  configuration chosen for force (choose_strong); pinch again,
                  level or pointing down, up to SHOVE_STAGES times while it
                  gives. A run of SEGMENT_MIN books or more is shoved in two,
                  its far half first, if the neighbour turns TURN_STOP_DEG
                  (the first shove stops there) or the gap is still short. If
                  the books beside it can't go back far enough to pinch it,
                  first walk it out of the row (walk_out): pinch the little
                  that sticks out, pull 10 mm, pinch deeper.
  insert()        Take the book at the pre-insertion pose (the hand-over is
                  not modelled): lined up with the gap, its back 1 cm over the
                  board's front edge and 3 mm above it, pinched 11.5 cm above
                  its bottom with its spine 2 mm off the palm. Rest its back on
                  the board and slide it in, holding it, until the hand would
                  come within 5 mm of another book's spine (the fingers can't
                  follow it in). If it runs into a neighbour first (a swivelled
                  or turned one narrows the slot toward the back), wedge it in:
                  on slowly and harder, its back corner pushing that face aside
                  deep in the shelf. It goes in along the middle of the straight
                  way in, or, if that is narrower than the book plus
                  CHANNEL_CLEAR, where it gets deepest before meeting a face
                  (wedge_slot). Let go, turn the hand 90 deg (fingers stacked)
                  and push the spine low down with the lower fingertip until
                  the book is in place, measuring again after each push, as
                  the book drags its neighbours back. In place means flush
                  (ALIGN_TOL) with the neighbour that sticks out more, or, for
                  a book deeper than that allows, EDGE_IN behind the front edge.
  Smaller steps they are built from: push_back(i, y), walk_out(k),
  shove_stage (one pinch), lift_up / clamp_up, fingertip_push(side, d).

Residual (Episode.residual, None by default): a learned correction of the hand
position during the gap-creation contact moves only (widen's pushes, the lift
and the clamp's swing, the shove), as gap_env.py trains it. Every
RESIDUAL_EVERY steps it adds up to RESIDUAL_STEP per shelf axis, ramped in,
RESIDUAL_MAX in all, inside each move's bounds (never past the row's room);
after the move it fades out over the next one. The rules, the plans and every
other move stay the script's. Without one the controller runs exactly as
without the hook.

Score: upright within 5 deg, in place, and released, plus how far the other
books moved. The slide and the pushes stop when the wrist force sensor feels
the book stop giving way, rather than forcing a jammed book.

Perception (--perception): where the controller's knowledge of the row comes from.
  - truth (default): the simulation, read whenever it is needed.
  - camera: the wrist camera (rgbd_perception). --camera-noise realistic (the
    default) gives it a D405's stereo errors and a calibration error drawn per
    episode (camera.py: DepthNoise.d405, CalibrationError); basic only
    independent depth noise per pixel.
    Before touching the row, the arm sweeps along it and looks from three
    inspection views (SWEEP), and the images are fused. It looks again, from
    the middle view, each time the hand has left the shelf after changing the
    row: after each widening, lift, clamp, push back, pinch of the shove and
    pull of the walk-out, after letting go of the book and after each push of
    it. The arm really moves there and back: from a view, in joint space to
    just in front of the next step, then straight in. While the hand is on a
    book, where it is comes from where the hand is (joint angles): the camera
    can't see a face the hand is on. The fingertips pushing a neighbour aside
    give its face; a book pinched and shoved moves and turns with the hand,
    which is what stops a shove at TURN_STOP_DEG. Arm paths are planned around
    the books where the last look saw them, and once let go of, the inserted
    book is in the way of the arm's joint moves. Known without looking: the
    shelf map, the book being inserted (its size, and how it is held); the
    other books' depth is a prior (perception.DEPTH_PRIOR). The score is
    always from the simulation. With --view, the viewer draws what the last
    look saw over the scene (visualize.py's draw_in_viewer: see-through boxes,
    spine corners, the gap's width).

Control: Menagerie's position actuators, fed joint targets by differential IK
along a min-jerk Cartesian path, with gravity compensation on the arm. Each
step's arm configuration is chosen so its whole path stays clear of the joint
limits and the shelf, and runs with the IK posture it was planned with. The
gripper is made as strong as the real Franka Hand (70 N continuous):
Menagerie's squeezes a book with only 1-2 N. The fingers are shelf_scene's
thin ones, not Menagerie's.
"""

import argparse
import math
import os
import time
import xml.etree.ElementTree as ET
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import mujoco

from bila.sim.shelf_scene import (FINGER_HALF_W, FINGER_LEN, FINGER_THICK, PANDA_HOME, ShelfConfig, _f, add_book_geoms,
                         add_book_visual_assets, add_page_visual_assets, book_children, book_tilt_deg,
                         build_scene, sample_row, settle)

from bila import ROOT as HERE
DEFAULT_PANDA = os.path.join(HERE, "mujoco_menagerie", "franka_emika_panda", "panda.xml")

# Panda hand geometry, with the thin fingers shelf_scene gives it (Menagerie's
# hand, measured from its collision mesh). Hand frame: z = approach (along the
# fingers), y = finger travel.
FINGERTIP_Z = 0.0584 + FINGER_LEN   # end of the fingers (Menagerie mounts them 58.4 mm out)
PAD_Z = FINGERTIP_Z - 0.0097   # where the pads hold the book: as far behind the tips as Menagerie's pad centre
PALM_Z = 0.066              # face of the hand between the fingers
FINGER_OPEN_SPAN = 0.04 + FINGER_THICK       # outer face of a fully open finger, from the hand axis
FINGER_OPEN_MID = 0.04 + FINGER_THICK / 2    # centre of a fully open finger, from the hand axis
HAND_HALF_T = 0.0316        # hand half-thickness, across finger travel
HAND_HALF_L = 0.105         # hand half-length, along finger travel

# Grasp and sequence settings
GRIP_FORCE = 70.0           # N on each side of the book (the real Franka Hand's continuous force)
PALM_GAP = FINGERTIP_Z - PALM_Z - 0.044   # the book's spine sits this far off the palm: 4.4 cm short of the
                            # fingertips (2 mm off the palm with Menagerie-length fingers). The fingers can't
                            # follow the book in between its neighbours, so the tips must not reach far past
                            # its spine, or the slide stops with the book still off the board.
PAD_INSET = PAD_Z - PALM_Z - PALM_GAP   # ... so the pads hold it this far behind its spine
PAD_HEIGHT = 0.115          # ... and this high above the book's bottom. Low, so sliding the book on
                            # the board can't tip it: board friction at its back-bottom corner tips it
                            # about the pads if friction x PAD_HEIGHT > (pads to book back). Not lower,
                            # because the arm can only hold its hand level >= ~0.5 m above its base.
PRE_OVERLAP = 0.01          # pre-insertion: the book's back is this far over the board's front edge
HOVER = 0.003               # ... and this high above the board
CLEAR = 0.005               # gap kept between the hand and any other book or panel
BACK_GAP = 0.005            # deepest a book goes: back face this far off the back panel (as the generator places books)
PUSH_LIFT = 0.01            # pushing: bottom of the turned hand this far above the board
SLIDE_STOP_FORCE = 35.0     # N change at the wrist that stops the slide (the book has run into a neighbour);
                            # above the board's drag on the heaviest book (~0.8 x 35 N)
PUSH_STOP_FORCE = 40.0      # N change at the wrist that stops the final push (the book is stuck)
TRANSIT_SPEED = 0.10        # m/s, peak of each min-jerk move
INSERT_SPEED = 0.06
PUSH_SPEED = 0.05
TURN_SPEED = 1.5            # rad/s
JOINT_SPEED = 1.0           # rad/s, peak, for joint-space moves away from the shelf (changing arm configuration)
IK_SEEDS = 20               # random starts per round when choosing the arm configuration
IK_ROUNDS = 3               # rounds tried before giving up (most scenes need only the first)
IK_EXTRA_ROUNDS = 3         # ... and more if none of them found any configuration at all
WAYPOINT_TRIES = 40         # random waypoints tried for a joint move that can't go direct or by way of home
MIN_MARGIN = 0.05           # rad: the chosen path must keep every joint this far from its limits
CLOSE_SPEED = 0.05          # m/s per finger when closing on a book from a few mm away

# Step: insert
CHANNEL_CLEAR = 0.003       # slide the book in along the middle of the straight way in if it is this much wider,
WEDGE_HUG = 0.001           # ... else wedge it in (wedge_slot): its face this far off the straighter neighbour's,
WEDGE_RUN_UP = 0.015        # slid in as usual until its back is this far from the face it will meet,
WEDGE_SPEED = 0.04          # then slowly,
WEDGE_STOP_FORCE = 60.0     # stopping only above this (more, and it slips back in the fingers),
WEDGE_PUSH_FORCE = 100.0    # and once let go, pushed home until the wrist feels this
SIDE_GAP = 0.003            # a book going into a wide gap after a stand-up stands this far off a neighbour

# Step: widen (the fingers fit in the gap)
GAP_MARGIN = 0.006          # widen the gap to the book's thickness plus this
FIT_CLEAR = 0.001           # fingers need this much room each side to enter the gap
MIN_REACH = 0.010           # fingertips must get this far behind a neighbour's spine to push it
ROOM_KEEP = 0.002           # leave this much of a side's free space unused
LEVEL_PROUD = 0.010         # books standing this much proud of the neighbour to push go back level with it first
WIDEN_STEPS = 3             # widenings at most: the other side next, then the first side again, harder,
HARD_BELOW = 0.002          # ... that only while the gap is narrower than the book plus this
GAP_PUSH_SPEED = 0.03

# Step: shove (the neighbours touch)
SHOVE_BACK = 0.035          # push the books beside the neighbour back until their spines are this far behind its spine
SHOVE_BACK_MIN = 0.012      # ... and shove only if they can go back at least this far
PINCH_Z = 0.050             # pinch the neighbour this high above the board
PINCH_OPEN = 0.004          # the fingers open this far off its faces
TIP_CLEAR = 0.007           # fingertips this far in front of the books pushed back (the arm gives way under load),
TIP_CLEAR_MIN = 0.004       # ... or down to this close, to pinch PINCH_WANT of the neighbour's depth
PALM_CLEAR = 0.012          # the palm this far in front of the other books' spines (it rubs them along otherwise)
PINCH_WANT = 0.012
MIN_PINCH = 0.008           # pinch at least this much of it
SHOVE_STEP = 0.015          # a shove may stop after this far, if the arm pushes much harder for it
PINCH_PITCHES = tuple(math.radians(a) for a in (0, 15, 30))   # the hand pinches level, or pointing down this much
SHOVE_STAGES = 5            # pinches at most
SEGMENT_MIN = 5             # if a shove stalls, shove a run of this many books or more again in two halves,
TURN_STOP_DEG = 15.0        # ... stopping the first shove of such a run once the neighbour turns this much
WALK_CLEAR = 0.0015         # walking a neighbour out: fingertips this far in front of the books beside it,
WALK_MIN = 0.002            # ... pinching at least this much of it,
WALK_STEP = 0.010           # ... pulling it this far out each time,
WALK_TARGET = 0.035         # ... until it is this far out
WALK_TRIES = 6
STRONG_MARGIN = 0.02        # rad: joint margin an arm configuration chosen for force must keep
STRONG_SEEDS = 60           # random starts per round when choosing it
YAW_LEVER = 0.06            # ... counting the pushed books' resistance this far behind the point pushed

# Step: stand_up (a neighbour leans into the slot)
LEAN_MIN_DEG = 5.0          # a neighbour leaning further than this into the slot gets stood back up
STAND_STEPS = 2             # stand-ups at most,
STAND_TRIES = 3             # ... each with up to this many lifts
STAND_TO = (0.5, 0.2, -0.2)  # lift it to this fraction of its tipping angle short of upright, per try
LIFT_SPEED = 0.06           # m/s, peak, of the fingers lifting it

# Looking with the wrist camera (--perception camera)
SWEEP = (-0.3, 0.0, 0.3)    # m along the row from the slot: the views of the first look, in order
HOLD_STILL = 0.2            # s the arm holds still at a view before the image is taken
STILL = (0.003, math.radians(1.5))   # a book seen again within this (m, rad) of before hasn't moved

# Residual correction (Runner.contact): a learned policy may nudge the hand during the
# gap-creation contact moves. Without one (the default) the script runs exactly as it is.
RESIDUAL_EVERY = 20         # physics steps per decision (25 Hz at the 2 ms step)
RESIDUAL_STEP = 0.003       # m per shelf axis per decision, at most, ramped in over the steps to the next
RESIDUAL_MAX = 0.025        # m per shelf axis in all
RESIDUAL_FADE = 0.3         # s: once a contact move ends, the correction fades out over the next move's first this
RESIDUAL_FLOOR = 0.002      # m: the correction keeps the hand this much of its clearance (board, books pushed back)

# Scoring
ALIGN_TOL = 0.010          # in place: spine within this of the more protruding neighbour's spine, or
EDGE_IN = 0.010             # ... if the book is deeper than that neighbour, at least this behind the front edge
EDGE_AIM = 0.005            # push a deeper book this much further in than EDGE_IN needs
UPRIGHT_DEG = 5.0           # upright within 5 deg (project proposal)
COLLISION_MIN_FORCE = 1.0   # N: the robot pressing on the shelf's structure, the cart or the floor harder than this
                            # is a collision (lighter contact is grazing)


# ---------------------------------------------------------------------------
# 1. Scene: the generated bookcase plus the target book
# ---------------------------------------------------------------------------
def build_model(cfg: ShelfConfig, row, panda_xml: str, stickers: bool = False) -> mujoco.MjModel:
    """The scene from shelf_scene.build_scene, plus the target book parked on
    the floor beside the bookcase (out of the way while the row settles), a
    stronger gripper, and gravity compensation on the arm. stickers: a
    call-number sticker on every book's spine (cart_demo.call_numbers)."""
    root = ET.fromstring(build_scene(cfg, row, panda_xml))
    b = row.target_book
    park = (row.shelf_pos[0] + row.shelf_width / 2 + 0.5, 0.4, b.height / 2 + 0.0005)
    body = ET.SubElement(root.find("worldbody"), "body", name=b.name, pos=_f(*park))
    ET.SubElement(body, "freejoint", name=b.name + "_joint")
    asset = root.find("asset")
    visual_materials = add_book_visual_assets(asset, [b]) if cfg.book_meshes else {}
    page_materials = add_page_visual_assets(asset, [b]) if cfg.book_meshes else {}
    add_book_geoms(body, b, visual_materials, page_materials)
    key = root.find("keyframe/key")
    # Its joints come last in qpos: the freejoint, then one hinge per page
    # section after the welded back one, all closed.
    hinges = _f(*([0.0] * max(0, len(b.parts) - 1)))
    key.set("qpos", key.get("qpos") + " " + _f(*park, 1, 0, 0, 0) + (" " + hinges if hinges else ""))
    if stickers:
        from bila.control.cart_demo import add_stickers, call_numbers                  # it imports this module
        add_stickers(root, row.books + [b] + [o for fl in row.filler_levels for o in fl.books], call_numbers(row)[0])

    return compile_robot_scene(ET.tostring(root, encoding="unicode"))


def compile_robot_scene(xml: str) -> mujoco.MjModel:
    """Compile a scene with the Panda in it, with gravity compensation on the
    arm and a gripper as strong as the real Franka Hand."""
    # Gravity compensation has to be set before compiling: MuJoCo skips it
    # when the compiled model has no compensated bodies.
    spec = mujoco.MjSpec.from_string(xml)
    for body in spec.bodies:
        if body.name.startswith("panda/"):
            body.gravcomp = 1.0
    model = spec.compile()

    # Gripper: stiff enough that closing on any book saturates at GRIP_FORCE.
    # ctrl 0..255 still means finger opening 0..0.04 m. The tendon force is
    # split between the two fingers, so it is twice the squeeze per side.
    a = model.actuator("panda/actuator8").id
    kp = 20000.0
    model.actuator_gainprm[a, 0] = kp * 0.04 / 255
    model.actuator_biasprm[a, :3] = (0.0, -kp, -0.01 * kp)
    model.actuator_forcerange[a] = (-2 * GRIP_FORCE, 2 * GRIP_FORCE)
    return model


class ShelfFrame:
    """Shelf frame: x along the row, y out of the shelf, z up, origin on the floor."""

    def __init__(self, row):
        c, s = math.cos(row.shelf_yaw), math.sin(row.shelf_yaw)
        self.R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        self.p = np.array(row.shelf_pos, dtype=float)

    def rot(self, yaw=0.0):
        """World rotation of something turned by `yaw` about z in the shelf frame."""
        c, s = math.cos(yaw), math.sin(yaw)
        return self.R @ np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])

    def to_world(self, p):
        return self.p + self.R @ p

    def to_shelf(self, p):
        return (p - self.p) @ self.R          # works on (3,) and (n, 3)

    def box(self, model, data, geom: int) -> np.ndarray:
        """Corners of a box geom in the shelf frame, (8, 3)."""
        s = model.geom_size[geom]
        local = np.array([[i * s[0], j * s[1], k * s[2]] for i in (-1, 1) for j in (-1, 1) for k in (-1, 1)])
        return self.to_shelf(local @ data.geom_xmat[geom].reshape(3, 3).T + data.geom_xpos[geom])


def obstacles(model, data, shelf: ShelfFrame, row):
    """(x_lo, x_hi, y_front) in the shelf frame of everything on the target
    level the hand must not run into: the other books and the side panels."""
    out = []
    for name in [bk.name + "_geom" for bk in row.books] + ["side_panel_pos_x", "side_panel_neg_x"]:
        c = shelf.box(model, data, model.geom(name).id)
        out.append((c[:, 0].min(), c[:, 0].max(), c[:, 1].max()))
    return out


def tip_limit(obs, x_mid: float, parts) -> float:
    """Smallest y the fingertips may reach. Each part of the hand is
    (half-width along the row, how far it sits behind the fingertips), and
    must stay CLEAR in front of every obstacle it overlaps along the row."""
    lim = -np.inf
    for half_w, behind in parts:
        for lo, hi, front in obs:
            if hi > x_mid - half_w - CLEAR and lo < x_mid + half_w + CLEAR:
                lim = max(lim, front + CLEAR - behind)
    return lim


# ---------------------------------------------------------------------------
# 2. Arm control: differential IK feeding the position actuators
# ---------------------------------------------------------------------------
def mat2quat(R):
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, R.reshape(9))
    return q


def quat2mat(q):
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, q)
    return R.reshape(3, 3)


def rotation_between(R0, R1):
    """Rotation vector v with R0 * exp(v) = R1 (in R0's frame)."""
    v = np.zeros(3)
    mujoco.mju_subQuat(v, mat2quat(R1), mat2quat(R0))
    return v


def interpolate(p0, R0, p1, R1, s):
    """Pose a fraction s of the way from (p0, R0) to (p1, R1): straight line, slerp."""
    q = mat2quat(R0)
    mujoco.mju_quatIntegrate(q, rotation_between(R0, R1), s)
    return p0 + s * (p1 - p0), quat2mat(q)


class ArmIK:
    """Damped least-squares IK for the hand frame, on its own MjData. The
    redundant 7th DoF is pulled gently toward a rest posture (set it to the
    configuration chosen for the task, so the arm stays in that family)."""

    def __init__(self, model, q_rest):
        self.m, self.d = model, mujoco.MjData(model)
        self.hand = model.body("panda/hand").id
        jids = [model.joint(f"panda/joint{i}").id for i in range(1, 8)]
        self.qadr = model.jnt_qposadr[jids]
        self.dadr = model.jnt_dofadr[jids]
        self.lo, self.hi = model.jnt_range[jids].T
        self.q_rest = np.array(q_rest, dtype=float)
        self.jacp, self.jacr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
        body = lambda g: model.body(model.geom_bodyid[g]).name
        self.robot = {g for g in range(model.ngeom) if body(g).startswith("panda/")}
        self.fingers = {g for g in self.robot if body(g) in ("panda/left_finger", "panda/right_finger")}
        self.finger_qadr = [model.jnt_qposadr[model.joint(j).id] for j in ("panda/finger_joint1", "panda/finger_joint2")]
        self.ignore = set()                 # geoms the arm may touch anywhere (the book it handles)
        self.belief = []                    # (qpos address, pose): books to plan around where they are thought to be
        self.book_of = {}                   # page-section geom -> its book's geom: see book_children

    def sync(self, data, finger=None):
        """Copy the scene (books, fingers) from the simulation, for collision
        checks, with the books in `belief` where they are thought to be
        instead; with `finger`, plan with each finger that far open instead."""
        self.d.qpos[:] = data.qpos
        for adr, pose in self.belief:
            self.d.qpos[adr:adr + 7] = pose
        if finger is not None:
            self.d.qpos[self.finger_qadr] = finger

    def collides(self, q, allowed=()):
        """Whether the arm at q is more than 1 mm into anything, other than
        the geoms in `allowed`, which its fingers may press on."""
        self.d.qpos[self.qadr] = q
        mujoco.mj_fwdPosition(self.m, self.d)
        for c in self.d.contact[:self.d.ncon]:
            if (c.geom1 in self.robot) == (c.geom2 in self.robot) or c.dist > -0.001:
                continue
            arm, other = (c.geom1, c.geom2) if c.geom1 in self.robot else (c.geom2, c.geom1)
            other = self.book_of.get(other, other)      # a page section counts as its book
            if other not in self.ignore and not (arm in self.fingers and other in allowed):
                return True
        return False

    def fk(self, q):
        self.d.qpos[self.qadr] = q
        mujoco.mj_kinematics(self.m, self.d)
        mujoco.mj_comPos(self.m, self.d)
        return self.d.xpos[self.hand].copy(), self.d.xmat[self.hand].reshape(3, 3).copy()

    def error(self, q, pos, rot):
        p, R = self.fk(q)
        return np.concatenate([pos - p, 0.5 * sum(np.cross(R[:, i], rot[:, i]) for i in range(3))])

    def solve(self, q, pos, rot, iters=2, max_step=0.2, pull=0.05):
        """`pull`: fraction of the way toward the rest posture per iteration.
        It sets how fast the posture settles, not where."""
        q = np.array(q, dtype=float)
        for _ in range(iters):
            e = self.error(q, pos, rot)
            mujoco.mj_jacBody(self.m, self.d, self.jacp, self.jacr, self.hand)
            J = np.vstack([self.jacp[:, self.dadr], self.jacr[:, self.dadr]])
            free = np.ones(7, dtype=bool)
            for _ in range(7):
                # Joints pinned at a limit and pushed further into it are taken
                # out, so the others make up for them instead of stalling.
                Jf = J * free
                Jinv = Jf.T @ np.linalg.inv(Jf @ Jf.T + 1e-4 * np.eye(6))
                # The posture pull goes through an exact null-space projector:
                # a damped one leaks into the task and leaves a steady error.
                null = np.eye(7) - np.linalg.pinv(Jf, rcond=1e-3) @ Jf
                dq = Jinv @ e + null @ (pull * (self.q_rest - q) * free)
                pinned = free & (((q <= self.lo + 1e-6) & (dq < 0)) | ((q >= self.hi - 1e-6) & (dq > 0)))
                if not pinned.any():
                    break
                free &= ~pinned
            q = np.clip(q + np.clip(dq, -max_step, max_step), self.lo, self.hi)
        return q

    def margin(self, q) -> float:
        """Distance of the nearest joint from its limit, in rad."""
        return float(np.min(np.minimum(q - self.lo, self.hi - q)))

    def solve_far(self, q, pos, rot):
        """Solve from a distant seed (pulling toward that seed). Returns (q, converged)."""
        self.q_rest = np.array(q, dtype=float)
        q = self.solve(q, pos, rot, iters=400, max_step=0.05)
        e = self.error(q, pos, rot)
        return q, bool(np.linalg.norm(e[:3]) < 1e-4 and np.linalg.norm(e[3:]) < 1e-3)

    def follow(self, q, poses, floor=-np.inf, allowed=None):
        """Track a list of (pos, rot) hand poses from q, warm-started like the
        runner does. Returns (end q, worst joint-limit margin on the way, reached
        the end). Gives up early once the margin drops below `floor`, or, if
        `allowed` is given, once the arm runs into something else.
        The runner re-solves every step, hundreds of times between two of
        these poses, so its posture pull has settled by each one. A few
        solves per pose at the runner's pull leave the posture lagging behind
        (by up to 0.1 rad), so the margins came out wrong. A stronger pull
        settles it in those few at the same posture."""
        worst = self.margin(q)
        if allowed is not None and self.collides(q, allowed):
            return q, worst, False
        for k, (pos, rot) in enumerate(poses):
            q = self.solve(q, pos, rot, iters=4, pull=0.5)
            worst = min(worst, self.margin(q))
            if worst < floor or (allowed is not None and k % 3 == 2 and self.collides(q, allowed)):
                return q, worst, False
        e = self.error(q, *poses[-1])
        return q, worst, bool(np.linalg.norm(e[:3]) < 1e-3 and np.linalg.norm(e[3:]) < 1e-2)


@dataclass
class Contact:
    """A gap-creation contact move the residual may correct (Runner.contact).
    lo, hi: bounds on the correction (shelf frame, m), inside +-RESIDUAL_MAX.
    x_limit: (side, x): the correction never takes the hand (its origin, shelf
    frame) past x toward side: the row's room, as the script knows it."""
    phase: str                                  # "widen", "lift", "clamp" or "shove"
    lo: np.ndarray
    hi: np.ndarray
    x_limit: Optional[Tuple[int, float]] = None
    steps: int = 0                              # physics steps of hand moves taken in it so far


class Runner:
    """Steps the simulation while moving the commanded hand pose along
    min-jerk paths. Each step re-solves IK from the last command.

    A residual (any object with decide(runner, contact) -> 3 m, shelf frame)
    may correct the hand's position during the moves marked with contact():
    IK then aims at the script's pose plus `offset`. self.pos stays the
    script's pose throughout, so the script's next move starts from where it
    planned. With residual=None nothing changes."""

    def __init__(self, model, data, ik: ArmIK, q_cmd, grip, viewer=None, log=print, residual=None, shelf=None):
        self.m, self.d, self.ik = model, data, ik
        self.q_cmd, self.grip = np.array(q_cmd), grip
        self.pos, self.rot = ik.fk(self.q_cmd)
        self.viewer, self.log = viewer, log
        self.residual, self.shelf = residual, shelf  # shelf: a ShelfFrame, needed with a residual
        self.offset = np.zeros(3)                   # the residual's correction of the hand position, shelf frame
        self.rate = np.zeros(3)                     # ... added to it each step until the next decision
        self.touching = None                        # the Contact being made, if any
        self.fade = None                            # (correction as the contact ended, steps left to fade it)
        self.fade_steps = int(round(RESIDUAL_FADE / model.opt.timestep))
        self.t0 = data.time
        self.wall0, self.next_frame = time.perf_counter(), 0.0
        self.ft = int(model.sensor("panda/ft_force").adr[0])       # wrist force sensor (hand frame)
        # By name: a scene may add actuators of its own (a cart's lift) ahead of the arm's.
        self.arm_act = [model.actuator(f"panda/actuator{i}").id for i in range(1, 8)]
        self.grip_act = model.actuator("panda/actuator8").id
        # Unintended contact: the robot (its base aside, which sits on its mount) touching the shelf's
        # structure, the cart or the floor. Books don't count: pushing them is what the skills are for,
        # and score() measures what that does to them.
        name = lambda g: model.body(model.geom_bodyid[g]).name
        self.arm_geom = np.array([name(g).startswith("panda/") and name(g) != "panda/link0" for g in range(model.ngeom)])
        self.structure_geom = np.array([name(g) in ("world", "shelf", "robot_mount", "lift") for g in range(model.ngeom)])
        self.collision = {"steps": 0, "max_force_n": 0.0, "impulse_ns": 0.0, "parts": set(), "during": set()}
        self.doing = ""
        self._f6 = np.zeros(6)

    def wrist_force(self):
        return self.d.sensordata[self.ft:self.ft + 3].copy()

    def halt(self):
        """Hold the hand where it actually is, so the arm stops pressing on
        whatever it ran into. With a correction on, the script's pose is
        where the hand is less the correction, which stays (and stops
        growing)."""
        self.q_cmd = self.d.qpos[self.ik.qadr].copy()
        self.pos, self.rot = self.ik.fk(self.q_cmd)
        if self.offset.any():
            self.pos = self.pos - self.shelf.R @ self.offset
        self.rate[:] = 0.0

    # --- The residual's correction -----------------------------------------
    @contextmanager
    def contact(self, phase, lo=(-RESIDUAL_MAX,) * 3, hi=(RESIDUAL_MAX,) * 3, x_limit=None):
        """Mark the hand moves inside as the gap-creation contact move `phase`
        (see Contact for the bounds): the residual, if any, corrects them. Its
        correction outlives the block and fades out over the next move (the
        hand may still hold what it moved: a clamped book)."""
        if self.residual is None:
            yield None
            return
        lo = np.maximum(np.asarray(lo, dtype=float), -RESIDUAL_MAX)
        hi = np.minimum(np.asarray(hi, dtype=float), RESIDUAL_MAX)
        self.touching, self.fade = Contact(phase, lo, hi, x_limit), None
        self.rate[:] = 0.0
        try:
            yield self.touching
        finally:
            self.touching = None
            self.rate[:] = 0.0
            if self.offset.any():
                self.fade = (self.offset.copy(), self.fade_steps)

    def bounded(self, offset, c: Contact):
        """The correction kept inside the contact's bounds. The room limit only
        holds the correction back: it never pulls back a hand the script
        itself takes that far."""
        offset = np.clip(offset, c.lo, c.hi)
        if c.x_limit is not None:
            side, x = c.x_limit
            room = max(0.0, side * (x - self.shelf.to_shelf(self.pos)[0]))
            offset[0] = side * min(side * offset[0], room)
        return offset

    def correct(self):
        """Advance the correction by one physics step of a hand move: in a
        contact, every RESIDUAL_EVERY steps ask the residual for an increment
        (at most RESIDUAL_STEP per axis) and ramp it in over the steps to the
        next decision; after one, fade it out (min-jerk) over fade_steps."""
        c = self.touching
        if c is not None:
            if c.steps % RESIDUAL_EVERY == 0:
                inc = np.clip(np.asarray(self.residual.decide(self, c), dtype=float), -RESIDUAL_STEP, RESIDUAL_STEP)
                self.rate = (self.bounded(self.offset + inc, c) - self.offset) / RESIDUAL_EVERY
            c.steps += 1
            self.offset = self.bounded(self.offset + self.rate, c)
        elif self.fade is not None:
            start, left = self.fade
            u = (left - 1) / self.fade_steps
            self.offset = start * (u ** 3 * (10 - 15 * u + 6 * u * u))
            self.fade = (start, left - 1) if left > 1 else None

    def hand_target(self):
        """Where IK aims the hand this step: the script's position plus the correction."""
        if self.residual is None:
            return self.pos
        self.correct()
        return self.pos + self.shelf.R @ self.offset if self.offset.any() else self.pos

    @property
    def t(self):
        return self.d.time - self.t0

    def step(self):
        self.d.ctrl[self.arm_act] = self.q_cmd
        self.d.ctrl[self.grip_act] = self.grip
        mujoco.mj_step(self.m, self.d)
        self.watch_contacts()
        if self.viewer is not None and self.t >= self.next_frame:     # draw at 60 fps, in real time
            self.next_frame += 1 / 60
            if not self.viewer.is_running():
                raise SystemExit
            self.viewer.sync()
            lag = self.t - (time.perf_counter() - self.wall0)
            if lag > 0:
                time.sleep(lag)

    def watch_contacts(self):
        """Record this step's unintended contacts (over COLLISION_MIN_FORCE) in self.collision."""
        m, d, n = self.m, self.d, self.d.ncon
        if not n:
            return
        g = d.contact.geom[:n]
        hit = np.flatnonzero((self.arm_geom[g[:, 0]] & self.structure_geom[g[:, 1]]) |
                             (self.arm_geom[g[:, 1]] & self.structure_geom[g[:, 0]]))
        worst = 0.0
        for i in hit:
            mujoco.mj_contactForce(m, d, int(i), self._f6)
            if self._f6[0] > COLLISION_MIN_FORCE:
                worst = max(worst, self._f6[0])
                arm, other = g[i] if self.arm_geom[g[i, 0]] else g[i][::-1]
                self.collision["parts"].add(f"{m.body(m.geom_bodyid[arm]).name[6:]} vs "
                                            f"{m.geom(other).name or m.body(m.geom_bodyid[other]).name}")
        if worst:
            c = self.collision
            c["steps"] += 1
            c["max_force_n"] = max(c["max_force_n"], worst)
            c["impulse_ns"] += worst * m.opt.timestep
            c["during"].add(self.doing)

    def move(self, name, pos, rot=None, speed=TRANSIT_SPEED, grip=None, stop_force=None, stop_when=None):
        """Move the hand frame to (pos, rot) in a straight line, with a slerp
        on its orientation. Peak speeds are `speed` and TURN_SPEED. With
        `stop_force`, stop and hold as soon as the wrist force along the hand's
        approach axis changes by more than that (N): what it pushes has stopped
        giving way. (The other components change as the board takes over the
        weight of a held book.) The force is smoothed over ~50 ms first, as a
        book chattering on the board's edge gives short spikes of 20-30 N.
        Returns False then. Likewise if `stop_when()` (checked every 20 ms)
        comes true; its docstring says why."""
        self.doing = name
        self.log(f"  {self.t:5.2f} s  {name}")
        if grip is not None:
            self.grip = grip
        rot = self.rot if rot is None else rot
        p0, R0 = self.pos.copy(), self.rot.copy()
        dist, ang = np.linalg.norm(pos - p0), np.linalg.norm(rotation_between(R0, rot))
        T = max(0.3, 1.875 * max(dist / speed, ang / TURN_SPEED))   # min-jerk peak = 1.875 x mean
        n = int(round(T / self.m.opt.timestep))
        f0 = smooth = self.wrist_force()[2]
        blend = self.m.opt.timestep / 0.05
        for k in range(1, n + 1):
            u = k / n
            self.pos, self.rot = interpolate(p0, R0, pos, rot, u ** 3 * (10 - 15 * u + 6 * u * u))
            self.q_cmd = self.ik.solve(self.q_cmd, self.hand_target(), self.rot)
            self.step()
            smooth += blend * (self.wrist_force()[2] - smooth)
            if stop_force is not None and abs(smooth - f0) > stop_force:
                self.halt()
                self.log(f"  {self.t:5.2f} s    stopped: the wrist feels over {stop_force:.0f} N")
                return False
            if stop_when is not None and k % 10 == 0 and stop_when():
                self.halt()
                self.log(f"  {self.t:5.2f} s    stopped: {stop_when.__doc__}")
                return False
        return True

    def follow(self, name, poses, speed=TRANSIT_SPEED):
        """Move the hand through a list of (pos, rot) poses as one min-jerk
        move along the whole path (straight lines and slerps between them)."""
        self.doing = name
        self.log(f"  {self.t:5.2f} s  {name}")
        pts = [(self.pos.copy(), self.rot.copy())] + list(poses)
        seg = [max(np.linalg.norm(b[0] - a[0]), np.linalg.norm(rotation_between(a[1], b[1])) * speed / TURN_SPEED)
               for a, b in zip(pts, pts[1:])]
        cum = np.concatenate([[0.0], np.cumsum(seg)])
        n = int(round(max(0.3, 1.875 * cum[-1] / speed) / self.m.opt.timestep))
        for k in range(1, n + 1):
            u = k / n
            s = cum[-1] * u ** 3 * (10 - 15 * u + 6 * u * u)
            i = min(int(np.searchsorted(cum, s, side="right")) - 1, len(seg) - 1)
            f = 0.0 if seg[i] <= 0 else (s - cum[i]) / seg[i]
            self.pos, self.rot = interpolate(*pts[i], *pts[i + 1], min(max(f, 0.0), 1.0))
            self.q_cmd = self.ik.solve(self.q_cmd, self.hand_target(), self.rot)
            self.step()

    def move_joints(self, name, q_goal, speed=JOINT_SPEED):
        """Move the arm in joint space to q_goal along a min-jerk path (peak
        joint speed `speed`, rad/s). Only for moves clear of the shelf: nothing
        here checks what the hand passes."""
        self.doing = name
        self.log(f"  {self.t:5.2f} s  {name}")
        q0, q_goal = self.q_cmd.copy(), np.asarray(q_goal, dtype=float)
        T = max(0.3, 1.875 * float(np.max(np.abs(q_goal - q0))) / speed)
        n = int(round(T / self.m.opt.timestep))
        for k in range(1, n + 1):
            u = k / n
            self.q_cmd = q0 + (q_goal - q0) * (u ** 3 * (10 - 15 * u + 6 * u * u))
            self.step()
        self.pos, self.rot = self.ik.fk(self.q_cmd)
        self.offset, self.fade = np.zeros(3), None                  # (a joint move takes up any correction left)

    def wait(self, name, duration, grip=None):
        self.doing = name
        self.log(f"  {self.t:5.2f} s  {name}")
        if grip is not None:
            self.grip = grip
        for _ in range(int(round(duration / self.m.opt.timestep))):
            self.step()


# ---------------------------------------------------------------------------
# 3. The scripted sequences
# ---------------------------------------------------------------------------
GRIP_CLOSED, GRIP_OPEN = 0.0, 255.0
# Closed on nothing, 1 mm apart: fully closed, the fingers press on each other with all the
# gripper's force, and a jolt can push one pad into the other, where they stick.
GRIP_SHUT = GRIP_OPEN * 0.0005 / 0.04


def axis_rot(axis, ang):
    """Rotation matrix for `ang` rad about `axis` (Rodrigues)."""
    a = np.asarray(axis, dtype=float) / np.linalg.norm(axis)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + math.sin(ang) * K + (1 - math.cos(ang)) * K @ K


def hand_pose_for_point(point_hand, point_world, rot):
    """Hand position that puts `point_hand` (hand frame) at `point_world`."""
    return point_world - rot @ point_hand


def front_x(c):
    """(x_lo, x_hi) of a box's 4 front corners (corners c, shelf frame): along the row, where its spine is."""
    f = c[np.argsort(c[:, 1])[4:], 0]
    return f.min(), f.max()


def segment(a, b, n):
    """n poses from just after pose a up to pose b (straight line, slerp)."""
    return [interpolate(*a, *b, s) for s in np.linspace(0, 1, n + 1)[1:]]


def still(b, before) -> bool:
    """Whether a book seen again (BookEstimate) stands where it was seen
    before, within what the camera gets wrong (STILL)."""
    bottom = lambda c: (c[0, 0] + c[1, 0]) / 2
    return (abs(bottom(b.spine_corners) - bottom(before.spine_corners)) < STILL[0]
            and abs(b.spine_y - before.spine_y) < STILL[0]
            and abs(b.lean - before.lean) < STILL[1] and abs(b.yaw - before.yaw) < STILL[1])


def resized(b, like):
    """A seen book (BookEstimate) with the thickness and height of `like`, its
    spine face scaled about its centre."""
    from bila.vision.perception import book_from_spine                     # perception imports this module
    c = b.spine_corners
    centre = c.mean(axis=0)
    across = ((c[1] - c[0]) + (c[2] - c[3])) / 2 * like.thickness / b.thickness
    up = ((c[3] - c[0]) + (c[2] - c[1])) / 2 * like.height / b.height
    corners = [centre - across / 2 - up / 2, centre + across / 2 - up / 2,
               centre + across / 2 + up / 2, centre - across / 2 + up / 2]
    return book_from_spine(corners, b.slot_offset, b.depth, b.visible)


class Episode:
    """One insertion attempt in a settled scene: geometry, hand poses, the
    steps of each sequence, and the score."""

    def __init__(self, model, data, row, viewer=None, log=print, eyes=None):
        self.m, self.d, self.row, self.viewer, self.log = model, data, row, viewer, log
        self.shelf = ShelfFrame(row)
        self.xs, self.ys, self.zs = self.shelf.R.T                   # shelf axes in world
        b = row.target_book
        self.t, self.depth, self.h = b.thickness, b.depth, b.height
        self.D, self.H, self.z0 = row.shelf_depth, row.clear_height, row.target_board_top
        if 2 * HAND_HALF_L + PUSH_LIFT + CLEAR > self.H:
            raise ValueError(f"Level is {self.H:.2f} m tall: too low to turn the hand for the push")
        self.book_geom = model.geom(b.name + "_geom").id
        self.book_body = model.body(b.name).id
        self.book_joint = model.joint(b.name + "_joint").id
        self.row_geoms = [model.geom(bk.name + "_geom").id for bk in row.books]
        self.row_bodies = [model.body(bk.name).id for bk in row.books]
        self.start_pos = data.xpos[self.row_bodies].copy()
        self.start_tilt = np.array([book_tilt_deg(data.xquat[i]) for i in self.row_bodies])
        self.spine_seated = -self.D / 2 + BACK_GAP + self.depth      # spine y once seated
        self.pad_in_book = np.array([0, self.depth / 2 - PAD_INSET, PAD_HEIGHT - self.h / 2])
        self.hand_z = self.z0 + HAND_HALF_L + PUSH_LIFT               # hand axis height while pushing
        self.z_rest = self.z0 + self.h / 2 + 0.0005                  # book centre height on the board
        self.ik = ArmIK(model, PANDA_HOME[:7])
        self.ik.ignore = {self.book_geom}
        # An articulated book keeps its box as a non-colliding envelope: the
        # geoms that do touch are the page sections'. Name them after their
        # book, so the sequences below can go on naming a book by its box.
        for bk in [b] + list(row.books):
            for child in book_children(bk):
                self.ik.book_of[model.geom(child + "_geom").id] = model.geom(bk.name + "_geom").id
        self.run = None
        self.residual = None       # corrects the hand in the gap-creation contact moves (Runner.contact); None: the script alone
        self.info = {}
        self.intended = set()      # row books the sequence sets out to move (a leaner it stands up): not disturbance
        self.widened = set()       # sides whose neighbour the fingertips have pushed aside
        self.shoved = None         # row book the shove pinched
        self.turned = False        # it turned in the shove, so the shove stopped
        self.lean_on_neighbour = False   # put a book that could tip over in a wide gap against a neighbour
        self._perception = None
        # Perception: with eyes=None the row is read from the simulation whenever it
        # is needed; else from what a camera saw (an RGBDPerception, see look).
        self.eyes = eyes
        self.est = None            # what the last look saw
        self.felt = {}             # slot faces felt by the fingertips since (-1 left, +1 right): x
        self.carried = None        # (row book, fingertips' (x, y), hand yaw) as the hand pinched it since
        self.moved = {}            # row book: where the hand left it since the look (BookEstimate)
        self.views = None          # [(dx, arm angles)] of the inspection views the arm reaches
        self.far = False           # the arm is at a view: go to the next step in joint space
        self.placed = None         # (position, rotation) where the hand let go of the book in the slot

    # --- Geometry, measured from the simulation (or as the camera saw it) ------
    def obstacles(self):
        if self.eyes is None:
            return obstacles(self.m, self.d, self.shelf, self.row)
        out = [(c[:, 0].min(), c[:, 0].max(), c[:, 1].max())
               for c in (self.seen_box(b) for b in self.seen().books if b.slot_offset != 0)]
        for name in ("side_panel_pos_x", "side_panel_neg_x"):       # on the shelf map: they don't move
            c = self.shelf.box(self.m, self.d, self.m.geom(name).id)
            out.append((c[:, 0].min(), c[:, 0].max(), c[:, 1].max()))
        return out

    def slot(self, truth=False):
        """The slot as it is now: x of the left and right neighbours' faces at
        their spines, and y of their spines (shelf frame). With eyes (unless
        `truth`): as the last look saw them, moved with the hand if it holds
        one, with any face the fingertips have felt since."""
        if self.eyes is not None and not truth:
            ti = self.row.target_index
            L, R = self.book_seen(ti - 1).spine_corners, self.book_seen(ti).spine_corners
            return (self.felt.get(-1, L[:, 0].max()), self.felt.get(+1, R[:, 0].min()), L[:, 1].max(), R[:, 1].max())
        ti = self.row.target_index
        L = self.shelf.box(self.m, self.d, self.row_geoms[ti - 1])
        R = self.shelf.box(self.m, self.d, self.row_geoms[ti])
        spine_end = lambda c: c[np.argsort(c[:, 1])[4:]]            # the 4 corners at the spine
        return spine_end(L)[:, 0].max(), spine_end(R)[:, 0].min(), L[:, 1].max(), R[:, 1].max()

    def free_space(self):
        """Room each side of the slot: from the neighbour's face at its spine
        to the side panel, less the thickness of every book on that side (the
        spaces between them plus the row's end, for upright books). Measured
        from the books' outlines instead, a book pushed until it turned would
        seem to take up more of the row than it does. With eyes: the books the
        camera saw, and if it didn't see the side panel, no room past the last
        book it saw."""
        W, ti, books = self.row.shelf_width, self.row.target_index, self.row.books
        xl, xr, _, _ = self.slot()
        if self.eyes is not None:
            est, rooms = self.seen(), []
            for side, face in ((-1, xl), (1, xr)):
                beyond = [b for b in est.books if side * b.slot_offset > 0]
                end = est.panel_x[0 if side < 0 else 1]
                if end is None:
                    edges = [x for b in beyond for x in b.spine_corners[:, 0]] or [face]
                    end = min(edges) if side < 0 else max(edges)
                rooms.append(side * (end - face) - sum(b.thickness for b in beyond))
            return tuple(rooms)
        return (xl + W / 2 - sum(b.thickness for b in books[:ti]),
                W / 2 - xr - sum(b.thickness for b in books[ti:]))

    def estimate(self):
        """What the robot knows about the row now: with eyes, what the last
        look saw; else what a camera would report (ground truth)."""
        if self.eyes is not None:
            return self.seen()
        from bila.vision.perception import GroundTruthPerception           # perception imports this module
        if self._perception is None:
            self._perception = GroundTruthPerception(self.m, self.row)
        return self._perception.estimate(self.d)

    def gap_at(self, z, est=None):
        """(x_left, x_right) of the slot at height z above the board, at the
        spine plane: the right-most edge of anything left of it and the
        left-most edge of anything right of it (side panels included)."""
        from bila.vision.perception import slice_x
        est = est or self.estimate()
        zz = self.z0 + z
        left = [s[1] for b in est.books if b.slot_offset < 0 and (s := slice_x(b.spine_corners, zz))]
        right = [s[0] for b in est.books if b.slot_offset > 0 and (s := slice_x(b.spine_corners, zz))]
        return max(left + [-self.row.shelf_width / 2]), min(right + [self.row.shelf_width / 2])

    def widen_gap(self):
        """The gap the widening goes by: at the spines, where the fingertips
        push (the neighbour they push swivels, and the book wedges past it);
        after a shove or a stand-up, at its narrowest over the depth the book
        will take up (a neighbour shoved at its front turns, shutting the gap
        at the back, and one stood up may be left skewed)."""
        xl, xr, _, _ = self.slot()
        return xr - xl if self.shoved is None and not self.intended else self.narrowest_gap()

    def finger_gap(self):
        """(x_left, x_right) of the slot, at the spine plane, where the fingers
        go to widen it (narrowest over the heights the stacked fingers take up).
        Wider than slot() at the board where a neighbour leans away from it.
        With eyes, a face the fingertips have pushed since the look moves it."""
        est = self.estimate()
        mid = self.hand_z - self.z0
        gaps = [self.gap_at(z, est) for z in np.linspace(mid - FINGER_OPEN_SPAN, mid + FINGER_OPEN_SPAN, 5)]
        lo, hi = max(g[0] for g in gaps), min(g[1] for g in gaps)
        if self.eyes is not None and self.felt:
            xl, xr, _, _ = self.slot()
            L, R = est.book(-1).spine_corners, est.book(+1).spine_corners
            lo, hi = lo + xl - L[:, 0].max(), hi + xr - R[:, 0].min()
        return lo, hi

    def seat_target(self, truth=False):
        """Where the book's spine should end up (y), and which rule applies:
        'flush' with the spine of the neighbour that sticks out more, if the
        book gets there before its back reaches the back panel; otherwise
        'inside' the shelf's front edge by at least EDGE_IN (aiming EDGE_AIM
        further in, but never deeper than the back panel allows). `truth`:
        from the simulation, even with eyes (to score)."""
        _, _, fl, fr = self.slot(truth)
        flush = max(fl, fr)
        if self.spine_seated <= flush:
            return flush, "flush"
        return max(self.spine_seated, self.D / 2 - EDGE_IN - EDGE_AIM), "inside"

    def book_pre(self, x_slot):
        """Book centre at the pre-insertion pose."""
        return np.array([x_slot, self.D / 2 - PRE_OVERLAP + self.depth / 2, self.z0 + self.h / 2 + HOVER])

    def hold_limit(self, x_slot, obs):
        return tip_limit(obs, x_slot, [(FINGER_OPEN_SPAN, 0.0), (HAND_HALF_L, FINGERTIP_Z - PALM_Z)])

    def push_limit(self, x, obs):
        return tip_limit(obs, x, [(FINGER_HALF_W, 0.0), (HAND_HALF_T, FINGERTIP_Z - PALM_Z)])

    # --- Looking (with eyes) ----------------------------------------------------------
    def seen(self):
        """What the last look saw; the first time, look (the sweep)."""
        if self.est is None:
            self.look()
        return self.est

    def looked(self):
        """With eyes, look again: the hand has left the shelf after changing the row."""
        if self.eyes is not None:
            self.look()

    def look(self, placed_x=None):
        """Take the hand out of the shelf, move the arm to an inspection view
        and estimate the row from the camera. The first look sweeps along the
        row (an image from each view in SWEEP that the arm reaches, fused);
        later ones look from the middle view only, and keep what the sweep saw
        of the books further out. placed_x: where the hand let go of the book
        it inserted. Raises RuntimeError if no view is reachable, or the
        camera doesn't see both of the slot's neighbours."""
        from dataclasses import replace
        from bila.vision.perception import gap_profile                          # perception imports this module
        if self.run is None:                                         # the first look: start from where the arm is
            self.place_arm(self.d.qpos[self.ik.qadr].copy())
        run, t0 = self.run, self.run.t
        if self.views is None:
            self.views = self.inspection_views()
            if not self.views:
                raise RuntimeError("the arm can't reach any view of the row for the camera")
            views, x_slot = self.views, None                         # the slot where the shelf map has it
        else:
            views, x_slot = [min(self.views, key=lambda v: abs(v[0]))], self.slot_now()
        frames = []
        for dx, q in views:
            where = f"{100 * abs(dx):.0f} cm {'left' if dx < 0 else 'right'} of the slot" if dx else "in front of the slot"
            self.reposition(q, f"move the camera {where}")
            run.wait("hold still for the camera", HOLD_STILL)
            frames.append(self.eyes.capture(self.d))
        est = self.eyes.estimate_from_frames(frames, x_slot=x_slot, placed_x=placed_x)
        if self.est is not None:
            # Past the slot's neighbours (which the middle view sees as squarely as the sweep did), a
            # book that hasn't moved keeps what the looks before saw of it, and one that has keeps the
            # size they saw: the sweep saw each book from the view nearest it, and from the middle view
            # books further out come out too thick. The books further out than this view sees are kept
            # as before too.
            old = {b.slot_offset: b for b in self.est.books if abs(b.slot_offset) > 1}
            reach = {s: max([abs(b.slot_offset) for b in est.books if s * b.slot_offset > 0], default=0)
                     for s in (-1, 1)}
            books = [b if b.slot_offset not in old else old[b.slot_offset] if still(b, old[b.slot_offset])
                     else resized(b, old[b.slot_offset]) if old[b.slot_offset].visible >= b.visible else b
                     for b in est.books]
            books += [b for b in old.values() if 0 < reach[1 if b.slot_offset > 0 else -1] < abs(b.slot_offset)]
            books.sort(key=lambda b: b.slot_offset)
            panels = tuple(p if p is not None else old for p, old in zip(est.panel_x, self.est.panel_x))
            est = replace(est, books=books, panel_x=panels,
                          gap_widths=gap_profile(books, panels, est.board_top, est.gap_heights))
        self.info["looks"] = self.info.get("looks", 0) + 1
        self.info["look_time_s"] = self.info.get("look_time_s", 0.0) + run.t - t0
        n = [sum(1 for b in est.books if s * b.slot_offset > 0) for s in (-1, 1)]
        self.log(f"         the camera sees {n[0]} books left of the slot and {n[1]} right"
                 f"{', and the book put into it' if est.book(0) else ''}")
        if est.book(-1) is None or est.book(+1) is None:
            raise RuntimeError("the camera didn't see both of the slot's neighbours")
        self.est, self.felt, self.carried, self.moved, self.far = est, {}, None, {}, True
        self.believe()
        if self.viewer is not None:                                  # draw what the camera saw over the scene
            from bila.viz.visualize import draw_in_viewer
            with self.viewer.lock():
                self.viewer.user_scn.ngeom = 0
                draw_in_viewer(self.viewer.user_scn, est, (self.t, self.h))

    def inspection_views(self):
        """[(dx, arm angles)] of the views in SWEEP the arm reaches, the scene
        as it is now. (Its own method so a training env can cache it.)"""
        from bila.control.inspection import solve_inspection                     # inspection imports this module
        return [(dx, v[0]) for dx in SWEEP if (v := solve_inspection(self.m, self.d, self.row, dx))]

    def slot_now(self):
        """Where the slot is along the row, at the board: between the
        neighbours' bottoms as last seen, moved by however far the hand has
        moved them since (a face the fingertips pushed, a book it shoved)."""
        est = self.est
        faces = {-1: est.book(-1).spine_corners[:, 0].max(), 1: est.book(+1).spine_corners[:, 0].min()}
        xl, xr, _, _ = self.slot()
        return est.slot_x() + (xl - faces[-1] + xr - faces[1]) / 2

    def believe(self):
        """Plan the arm's moves around the books where the last look saw them:
        each spine where it was seen, with the rest of the box (the books'
        real size, in the planner's model) behind it; and as if the books the
        camera didn't see weren't there. The book it let go of in the slot:
        where the camera saw it, else where the hand let go of it."""
        from bila.vision.perception import book_box
        ti, by_offset, belief = self.row.target_index, {b.slot_offset: b for b in self.est.books}, []

        def pose_of(b, geom):
            _, R, _ = book_box(b)
            centre = b.spine_corners.mean(axis=0) - R[:, 1] * self.m.geom_size[geom][1]
            return np.concatenate([self.shelf.to_world(centre), mat2quat(self.shelf.R @ R)])

        for i, (geom, body) in enumerate(zip(self.row_geoms, self.row_bodies)):
            b = by_offset.get(self.offset(i))
            pose = np.array([100.0 + i, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]) if b is None else pose_of(b, geom)
            belief.append((self.m.jnt_qposadr[self.m.body_jntadr[body]], pose))
        if self.placed is not None:
            b = self.est.book(0)
            pose = (pose_of(b, self.book_geom) if b is not None else
                    np.concatenate([self.placed[0], mat2quat(self.placed[1])]))
            belief.append((self.m.jnt_qposadr[self.book_joint], pose))
        self.ik.belief = belief

    def offset(self, i):
        """Row book i's place relative to the slot (BookEstimate.slot_offset)."""
        ti = self.row.target_index
        return i - ti if i < ti else i - ti + 1

    def book_seen(self, i):
        """Row book i (a BookEstimate) as the last look saw it, moved and
        turned with the hand if the hand has pinched it since (carry, let_go);
        None if the camera didn't see it."""
        from bila.vision.perception import book_from_spine                     # perception imports this module
        b = self.seen().book(self.offset(i))
        if i in self.moved:
            return self.moved[i]
        if b is None or self.carried is None or self.carried[0] != i:
            return b
        _, p0, yaw0 = self.carried
        p, yaw = self.fingertips()
        c, s = math.cos(yaw - yaw0), math.sin(yaw - yaw0)
        corners = b.spine_corners.copy()
        corners[:, :2] = p + (corners[:, :2] - p0) @ np.array([[c, s], [-s, c]])
        return book_from_spine(corners, b.slot_offset, b.depth, b.visible)

    def known(self, i):
        """Whether the controller knows where row book i is: always from the
        simulation, with eyes if the camera saw it."""
        return self.eyes is None or self.seen().book(self.offset(i)) is not None

    def fingertips(self):
        """Where the middle of the fingertips is ((x, y), shelf frame) and how
        far the hand is turned about the vertical, from where the hand is
        (joint angles): the yaw of a book it pinches square to its spine."""
        pos, rot = self.ik.fk(self.d.qpos[self.ik.qadr])
        tip = self.shelf.to_shelf(pos + rot @ [0, 0, FINGERTIP_Z])
        a = self.shelf.R.T @ rot[:, 2]                               # approach axis: into the shelf, -y of the book
        return tip[:2], math.atan2(a[0], -a[1])

    def carry(self, k):
        """With eyes: the hand has closed on row book k, which moves and turns
        with it from now until it lets go (let_go), if the fingers (their
        joint angles) stopped on it: more than half its thickness, as seen,
        apart. Closed further, they missed it, and it stays where it was seen."""
        if self.eyes is None:
            return
        opening = float(self.d.qpos[self.ik.finger_qadr].sum())
        if opening > self.half_size(k)[0]:
            self.carried = (k, *self.fingertips())
        else:
            self.log(f"         the fingers closed to {1000 * opening:.0f} mm: they missed it")

    def let_go(self):
        """With eyes: the hand is letting go of the book it pinched, which
        stays where the hand took it, until the next look."""
        if self.eyes is not None and self.carried is not None:
            k = self.carried[0]
            self.moved[k] = self.book_seen(k)
            self.carried = None

    def feel(self, side):
        """With eyes: where the slot's face on `side` is now, from where the
        fingertips pushing it are (joint angles). The camera can't see a face
        the hand is on."""
        if self.eyes is not None:
            pos, rot = self.ik.fk(self.d.qpos[self.ik.qadr])
            self.felt[side] = self.shelf.to_shelf(pos + rot @ [0, 0, FINGERTIP_Z])[0] + side * FINGER_HALF_W

    def seen_box(self, b):
        """Corners (8, 3; shelf frame) of a seen book's box, as the robot models it."""
        from bila.vision.perception import book_box
        centre, R, half = book_box(b)
        return centre + (np.array([[i, j, k] for i in (-1, 1) for j in (-1, 1) for k in (-1, 1)]) * half) @ R.T

    def held_pose(self):
        """World position and rotation of the book in the hand, from where the
        hand is (joint angles) and how it took the book."""
        pos, rot = self.ik.fk(self.d.qpos[self.ik.qadr])
        return pos + rot @ self.in_hand_p, rot @ self.in_hand_R

    def held_box(self):
        """Corners (8, 3; shelf frame) of the book in the hand (held_pose)."""
        return self.book_corners(*self.held_pose())

    def book_corners(self, p, R):
        """Corners (8, 3; shelf frame) of the book to insert at world pose (p, R)."""
        s = np.array([self.t, self.depth, self.h]) / 2
        local = np.array([[i, j, k] for i in (-1, 1) for j in (-1, 1) for k in (-1, 1)]) * s
        return self.shelf.to_shelf(local @ R.T + p)

    # --- Hand poses ----------------------------------------------------------------
    # Holding: the fingers close across the book's thickness, approaching along
    # the book's -y. Pushing: the hand is turned 90 deg about the approach axis
    # so the fingers are stacked vertically; the lower finger pushes low on the
    # spine, so the push can't tip the book over backwards. Both work with the
    # hand either way up (sign).
    def grasp_pose(self, sign, centre, yaw=0.0):
        Rb = self.shelf.rot(yaw)
        rot = np.column_stack([np.cross(sign * Rb[:, 0], -Rb[:, 1]), sign * Rb[:, 0], -Rb[:, 1]])
        pad = self.shelf.to_world(centre) + Rb @ self.pad_in_book
        return hand_pose_for_point([0, 0, PAD_Z], pad, rot), rot

    def push_pose(self, sign, x, tip_y):
        rot = np.column_stack([np.cross(sign * self.zs, -self.ys), sign * self.zs, -self.ys])
        low_tip = [0, -sign * FINGER_OPEN_MID, FINGERTIP_Z]
        return hand_pose_for_point(low_tip, self.shelf.to_world([x, tip_y, self.hand_z - FINGER_OPEN_MID]), rot), rot

    def col_pose(self, sign, x, tip_y, z_axis, yaw=0.0):
        """Hand turned as in push_pose (fingers stacked vertically), with its
        axis at height z_axis and fingertips at (x, tip_y) (shelf frame),
        pointing into the shelf, or along -y of a book turned by yaw.
        push_pose is col_pose at z_axis = hand_z, with the fingers open."""
        ys = self.shelf.rot(yaw)[:, 1]
        rot = np.column_stack([np.cross(sign * self.zs, -ys), sign * self.zs, -ys])
        return self.shelf.to_world([x, tip_y, z_axis]) - rot @ [0, 0, FINGERTIP_Z], rot

    def pinch_pose(self, sign, x, tip_y, z, yaw=0.0, pitch=0.0):
        """Hand as in grasp_pose (fingers closing across a book's thickness)
        for a book turned by yaw, with the middle of its fingertips at
        (x, tip_y, z) (shelf frame), pointing into the shelf, or down into it
        by pitch."""
        Rb = self.shelf.rot(yaw)
        approach = -math.cos(pitch) * Rb[:, 1] - math.sin(pitch) * self.zs
        rot = np.column_stack([np.cross(sign * Rb[:, 0], approach), sign * Rb[:, 0], approach])
        return hand_pose_for_point([0, 0, FINGERTIP_Z], self.shelf.to_world([x, tip_y, z]), rot), rot

    def tip_y(self, pose):
        return self.shelf.to_shelf(pose[0] + pose[1] @ [0, 0, FINGERTIP_Z])[1]

    def push_path(self, start, sign, x, spine_now, tip_goal):
        """From `start` (just let go): back off, turn, push to tip_goal.
        Returns the back-off pose, the turned pose and the whole path."""
        tip_back = max(self.D / 2 + 0.015, spine_now + 0.02)
        back_off = (start[0] + self.ys * (tip_back - self.tip_y(start)), start[1])
        ready = self.push_pose(sign, x, tip_back)
        path = (segment(start, back_off, 15) + segment(back_off, ready, 50)
                + [self.push_pose(sign, x, y) for y in np.linspace(tip_back, tip_goal, 30)])
        return back_off, ready, path

    def insertion_plan(self, grasp_sign, push_sign, x_slot, y_from, obs):
        """Nominal hand poses to rest the book, slide it in, let go and push it
        home, for choosing the arm configuration (refined once it is held)."""
        spine_ahead = PAD_INSET + FINGERTIP_Z - PAD_Z
        target = self.seat_target()[0]
        y_hold = max(target, self.hold_limit(x_slot, obs) + spine_ahead) - self.depth / 2
        slide = [self.grasp_pose(grasp_sign, [x_slot, y, self.z_rest]) for y in np.linspace(y_from, y_hold, 40)]
        tip_goal = max(target, self.push_limit(x_slot, obs))
        return slide + self.push_path(slide[-1], push_sign, x_slot, y_hold + self.depth / 2, tip_goal)[2]

    def choose_arm(self, candidates, allowed=frozenset(), finger=None, reach_from=None):
        """Many arm configurations reach a start pose. Near a joint limit the IK
        swings the other joints round, and the arm can jump to another
        configuration mid-move and knock the row. So follow each candidate's
        whole planned path, drop those where the arm runs into anything (other
        than `allowed` geoms its fingers may press on), and keep the one that
        stays farthest from the joint limits. candidates: (start pose, path, label).
        `finger`: plan with the fingers this far open (default: as they are).
        `reach_from`: only configurations a joint move gets to from these arm
        angles (see joint_route), with the fingers as they are."""
        ik, seed_rng, best = self.ik, np.random.default_rng(0), None

        def reachable(q):
            ik.sync(self.d)
            ok = self.joint_route(reach_from, q) is not None
            ik.sync(self.d, finger)
            return ok

        ik.sync(self.d, finger)
        for rnd in range(IK_ROUNDS + IK_EXTRA_ROUNDS):  # more random starts only if the first ones fall short
            seeds = [seed_rng.uniform(ik.lo, ik.hi) for _ in range(IK_SEEDS)]
            if rnd == 0:
                seeds.insert(0, np.array(PANDA_HOME[:7]))
            for start, path, label in candidates:
                for s in seeds:
                    q, ok = ik.solve_far(s, *start)
                    if not ok:
                        continue
                    ik.q_rest = q
                    _, worst, ok = ik.follow(q, path, floor=-np.inf if best is None else best[0], allowed=allowed)
                    if ok and (best is None or worst > best[0]) and (reach_from is None or reachable(q)):
                        best = (worst, q, label)
            if best is not None and best[0] >= MIN_MARGIN:
                break
            if rnd + 1 == IK_ROUNDS and best is not None:     # the extra rounds only when nothing at all was found
                break
        if best is None or best[0] < MIN_MARGIN:
            raise RuntimeError("No arm configuration does the whole sequence without nearing a joint limit "
                               "or running into the shelf")
        ik.q_rest = best[1]
        return best[1], best[2]

    # --- Steps ----------------------------------------------------------------------
    def place_arm(self, q):
        """Start the arm at q with the gripper open."""
        m, d = self.m, self.d
        d.qpos[self.ik.qadr] = q
        for j in ("panda/finger_joint1", "panda/finger_joint2"):
            d.qpos[m.jnt_qposadr[m.joint(j).id]] = 0.04
        d.qvel[:] = 0
        mujoco.mj_forward(m, d)
        self.run = Runner(m, d, self.ik, q, GRIP_OPEN, self.viewer, self.log, self.residual, self.shelf)

    def begin(self, candidates, allowed=frozenset(), finger=None):
        """Get the hand to the start of a step, ready to follow one of the
        candidates (start pose, path, label), and return its label. At the
        start of the episode the arm is placed at the best configuration. Later
        the hand carries on from where it is, if some candidate's start and
        path can be reached from there with joint limits to spare and without
        hitting anything; otherwise the arm moves for real to the best
        configuration (reposition). From an inspection view it always moves in
        joint space, to just in front of the start, and then straight in, so
        the arm doesn't swing through the books (or the one it let go of) on
        its way. Raises RuntimeError if none is possible."""
        run, ik = self.run, self.ik
        if run is not None and self.far:
            fronts = [f for _, _, f in self.obstacles()] + [self.D / 2]
            if self.placed is not None:
                b = self.est.book(0)
                fronts.append(b.spine_corners[:, 1].max() if b is not None else self.book_corners(*self.placed)[:, 1].max())
            out = max(fronts) + 0.08
            staged = []
            for start, path, label in candidates:
                back = max(0.0, out - self.tip_y(start))
                ahead = (start[0] + self.ys * back, start[1])
                staged.append((ahead, segment(ahead, start, 10) + list(path), (label, start, back)))
            q, (label, start, back) = self.choose_arm(staged, allowed, finger, reach_from=run.q_cmd.copy())
            self.reposition(q)
            if back > 0:
                run.move("move in to the start of the next step", *start)
            return label
        if run is not None:
            ik.sync(self.d, finger)
            for start, path, label in candidates:
                # Plan as it will run: the move to the start keeps the arm's posture, and the step
                # then keeps the one it starts in. (Kept at the posture before the move, which may
                # have turned the hand a quarter turn, the IK's pull swung the arm toward a
                # singularity during the step.)
                ik.q_rest = run.q_cmd.copy()
                q, worst, ok = ik.follow(run.q_cmd, segment((run.pos, run.rot), start, 30), allowed=allowed)
                if not ok:
                    continue
                ik.q_rest = q
                _, worst_step, ok = ik.follow(q, list(path), allowed=allowed)
                if ok and min(worst, worst_step) >= 2 * MIN_MARGIN:   # a little spare: the real path drifts
                    ik.q_rest = run.q_cmd.copy()
                    run.move("move to the start of the next step", *start)
                    ik.q_rest = run.q_cmd.copy()
                    return label
        q, label = self.choose_arm(candidates, allowed, finger)
        if run is None:
            self.place_arm(q)
        else:
            self.reposition(q)                                       # q puts the hand at that start pose
        return label

    def reposition(self, q, name="move the arm to its next configuration"):
        """Take the hand straight out of the shelf, then move the arm in joint
        space to q, by way of the tucked home pose if the direct move would hit
        anything: the next step may need another arm configuration."""
        run, ik = self.run, self.ik
        tip = self.tip_y((run.pos, run.rot))
        out = max(tip, self.D / 2 + 0.08)
        if out > tip:
            # Keep the arm's posture on the way out: choosing q set the IK's rest posture to q,
            # and its pull would swing the arm round (and the hand into the shelf) mid-move.
            ik.q_rest = run.q_cmd.copy()
            run.move("take the hand out of the shelf", run.pos + self.ys * (out - tip))
        ik.sync(self.d)
        route = self.joint_route(run.q_cmd, q)
        if route is None:
            raise RuntimeError("No collision-free joint move to the next arm configuration")
        if len(route) > 1:
            tucked = np.allclose(route[0], PANDA_HOME[:7])
            run.move_joints("tuck the arm" if tucked else "move the arm out of the way", route[0])
        run.move_joints(name, q)
        ik.q_rest = np.array(q, dtype=float)
        self.far = False                                             # (a look sets it again once at a view)

    def joint_route(self, q0, q):
        """Waypoints of a joint move from q0 to q that runs into nothing in the
        planner's scene: [q] straight there, [home, q] by way of the tucked
        home pose, or [w, q] by way of some other configuration sampled around
        the two ends; None if none is clear. A book let go of in the slot is
        in the way too (the hand may only push it, in a straight move)."""
        ik, home, ignore = self.ik, np.array(PANDA_HOME[:7]), self.ik.ignore

        def clear(a, b):
            return not any(ik.collides(a + s * (b - a)) for s in np.linspace(0, 1, 25)[1:])

        if self.placed is not None:
            ik.ignore = ignore - {self.book_geom}
        try:
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
        finally:
            ik.ignore = ignore

    def take_book(self, grasp_sign, centre, yaw=0.0, name="close the gripper on the book"):
        """Put the book between the open fingers, held as at (centre, yaw), and
        close them. Where the book comes from is not modelled."""
        m, d, ik = self.m, self.d, self.ik
        R_h = d.xmat[ik.hand].reshape(3, 3)
        R_b = R_h @ self.grasp_pose(grasp_sign, centre, yaw)[1].T @ self.shelf.rot(yaw)
        # Each pad 0.2 mm off the book, inside its 0.5 mm contact margin (shelf_scene), so both
        # contacts act from the first step. Right on the margin, one could start a step late: the
        # finger closed 1.3 mm in that step, and its contact, starting that deep, was too soft to
        # stop it before it was 9 mm into the book (which flew off when let go).
        for j in ("panda/finger_joint1", "panda/finger_joint2"):
            d.qpos[m.jnt_qposadr[m.joint(j).id]] = self.t / 2 + 0.0002
            d.qvel[m.jnt_dofadr[m.joint(j).id]] = 0
        qa, va = m.jnt_qposadr[self.book_joint], m.jnt_dofadr[self.book_joint]
        d.qpos[qa:qa + 3] = d.xpos[ik.hand] + R_h @ [0, 0, PAD_Z] - R_b @ self.pad_in_book
        d.qpos[qa + 3:qa + 7] = mat2quat(R_b)
        d.qvel[va:va + 6] = 0
        mujoco.mj_forward(m, d)
        self.run.wait(name, 0.5, grip=GRIP_CLOSED)
        # From here on, move the book: keep its pose in the hand as measured now.
        R_h = d.xmat[ik.hand].reshape(3, 3)
        self.in_hand_p = R_h.T @ (d.xpos[self.book_body] - d.xpos[ik.hand])
        self.in_hand_R = R_h.T @ d.xmat[self.book_body].reshape(3, 3)

    def hand_for_book(self, centre, yaw=0.0):
        rot = self.shelf.rot(yaw) @ self.in_hand_R.T
        return self.shelf.to_world(centre) - rot @ self.in_hand_p, rot

    def seat(self, x_slot, y_from, push_sign, wedge_from=None):
        """Holding the book straight in front of (or partly in) the slot: rest
        it on the board, slide it in, let go, and push it into place.
        wedge_from: the book centre's y from which to slide it slowly and
        harder, and then push it harder (wedge_slot). Returns (time let go,
        time in place, whether the hand was blocked)."""
        run = self.run
        obs = self.obstacles()
        target = self.seat_target()[0]
        spine_ahead = y_from + self.depth / 2 - self.tip_y(self.hand_for_book([x_slot, y_from, self.z_rest]))
        y_hold = max(target, self.hold_limit(x_slot, obs) + spine_ahead) - self.depth / 2
        if y_hold > self.D / 2 - CLEAR:                             # its centre must be over the board
            raise RuntimeError("Hand can't get the book far enough onto the board to let go")
        # A two-finger pinch can't stop a heavy book from pitching nose-down (its
        # weight acts several cm behind the pads). Resting its back on the board
        # first lets the board carry that, as a person would do it.
        run.move("rest it on the board", *self.hand_for_book([x_slot, y_from, self.z_rest]), speed=0.01)
        # If the book runs into a neighbour (in a tight row, the pushed
        # neighbours' backs still narrow the gap), stop rather than force it:
        # forced, it twists in the fingers and jams. The fingertip push below,
        # low and central on the spine, carries on from there.
        if wedge_from is not None:
            # (If it meets something sooner than the turned face, the slow part carries on from there.)
            if y_hold < wedge_from < y_from:
                run.move("slide it in along the board", *self.hand_for_book([x_slot, wedge_from, self.z_rest]),
                         speed=INSERT_SPEED, stop_force=SLIDE_STOP_FORCE)
            slid = run.move("wedge it in", *self.hand_for_book([x_slot, y_hold, self.z_rest]), speed=WEDGE_SPEED,
                            stop_force=WEDGE_STOP_FORCE)
        else:
            slid = run.move("slide it in along the board", *self.hand_for_book([x_slot, y_hold, self.z_rest]),
                            speed=INSERT_SPEED, stop_force=SLIDE_STOP_FORCE)
        if not slid:
            c = self.shelf.box(self.m, self.d, self.book_geom) if self.eyes is None else self.held_box()
            if c[:, 1].mean() > self.D / 2 - CLEAR:                 # centre not over the board: can't let go
                raise RuntimeError("the book jammed at the slot's entrance")
        run.wait("open the gripper", 0.4, grip=GRIP_OPEN)
        t_release = run.t
        push_force = PUSH_STOP_FORCE if wedge_from is None else WEDGE_PUSH_FORCE
        if self.eyes is not None:
            return (t_release, *self.push_home_seen(x_slot, push_sign, t_release, push_force))

        # Sliding in drags the neighbours back into the shelf, which moves the
        # spine to line up with: aim at where they are now.
        obs, target = self.obstacles(), self.seat_target()[0]
        c = self.shelf.box(self.m, self.d, self.book_geom)
        x_book, spine = c[:, 0].mean(), c[:, 1].max()
        if spine <= target + 0.002:                                 # the slide already got it there
            tip_back = max(self.D / 2 + 0.015, spine + 0.02)
            run.move("back off (it is already in place)",
                     run.pos + self.ys * (tip_back - self.tip_y((run.pos, run.rot))))
            run.wait("let it settle", 1.0)
            return t_release, t_release, False

        # Replan the push from where the book really is. If the planned turn
        # direction now comes too close to a joint limit, try the other one.
        tip_goal = max(target, self.push_limit(x_book, obs))
        blocked = tip_goal > target
        self.ik.sync(self.d)
        for sign in (push_sign, -push_sign):
            back_off, ready, path = self.push_path((run.pos, run.rot), sign, x_book, spine, tip_goal)
            _, worst, ok = self.ik.follow(run.q_cmd, path, allowed=frozenset())
            if ok and worst >= MIN_MARGIN:
                run.move("back off", *back_off)
                run.move("turn the hand to push", *ready)
                break
        else:
            # Neither turn keeps clear of the joint limits from this arm configuration: change it.
            run.move("back off", *back_off)
            tip_back = self.tip_y(back_off)
            sign = self.begin([(self.push_pose(s, x_book, tip_back),
                                [self.push_pose(s, x_book, y) for y in np.linspace(tip_back, tip_goal, 30)], s)
                               for s in (1, -1)])
        run.move("reach the spine", *self.push_pose(sign, x_book, spine + 0.005))
        pushed = run.move("push it into place" if not blocked else
                          f"push it in (nearby spines stop the hand {1000 * (tip_goal - target):.0f} mm short)",
                          *self.push_pose(sign, x_book, tip_goal), speed=PUSH_SPEED, stop_force=push_force)
        # The push drags them back too: push on to where they are now, unless
        # the book is stuck or the hand can't go any deeper.
        for _ in range(3):
            target = self.seat_target()[0]
            spine = self.shelf.box(self.m, self.d, self.book_geom)[:, 1].max()
            goal = max(target, self.push_limit(x_book, self.obstacles()))
            if not pushed or spine <= target + 0.002 or goal >= tip_goal - 0.001:
                break
            tip_goal, blocked = goal, goal > target
            pushed = run.move("push it further (the neighbours went back with it)",
                              *self.push_pose(sign, x_book, tip_goal), speed=PUSH_SPEED, stop_force=push_force)
        t_done = run.t
        run.move("back out", *self.push_pose(sign, x_book, max(self.D / 2, tip_goal) + 0.05))
        run.wait("let it settle", 1.0)
        return t_release, t_done, blocked

    def push_home_seen(self, x_slot, push_sign, t_release, push_force):
        """seat's push, with eyes: back off, look where the book and its
        neighbours are now, and push the book into place from the front if it
        isn't, looking again after each push (as without eyes, it drags the
        neighbours back, which moves where it has to go). Where the hand let
        go of it stands in for the book if the camera doesn't find it there.
        Returns (time in place, whether the hand was blocked)."""
        run, released = self.run, self.held_box()
        self.placed = self.held_pose()
        self.believe()
        tip_back = max(self.D / 2 + 0.015, released[:, 1].max() + 0.02)
        run.move("back off", run.pos + self.ys * (tip_back - self.tip_y((run.pos, run.rot))))
        t_done, blocked, pushed, tip_goal = t_release, False, True, None
        for k in range(5):                                           # a push, then up to 3 more, as without eyes
            self.look(placed_x=x_slot)
            target = self.seat_target()[0]
            book = self.est.book(0)
            c = released if book is None else book.spine_corners
            x_book, spine = c[:, 0].mean(), c[:, 1].max()
            if spine <= target + 0.002:
                self.log("         it is in place")
                break
            goal = max(target, self.push_limit(x_book, self.obstacles()))
            if k == 4 or (k > 0 and (not pushed or goal >= tip_goal - 0.001)):   # stuck, or the hand can't go deeper
                break
            tip_goal, blocked = goal, goal > target
            tip_back = max(self.D / 2 + 0.015, spine + 0.02)
            sign = self.begin([(self.push_pose(s, x_book, tip_back),
                                [self.push_pose(s, x_book, y) for y in np.linspace(tip_back, tip_goal, 30)], s)
                               for s in (push_sign, -push_sign)])
            run.move("reach the spine", *self.push_pose(sign, x_book, spine + 0.005))
            what = "push it into place" if k == 0 else "push it further (the neighbours went back with it)"
            pushed = run.move(what if not blocked else
                              f"{what} (nearby spines stop the hand {1000 * (tip_goal - target):.0f} mm short)",
                              *self.push_pose(sign, x_book, tip_goal), speed=PUSH_SPEED, stop_force=push_force)
            t_done = run.t
            run.move("back out", *self.push_pose(sign, x_book, max(self.D / 2, tip_goal) + 0.05))
        run.wait("let it settle", 1.0)
        return t_done, blocked

    def score(self, t_release, t_done, blocked, **info) -> dict:
        m, d = self.m, self.d
        c = self.shelf.box(m, d, self.book_geom)
        tilt = book_tilt_deg(d.xquat[self.book_body])
        back_gap = c[:, 1].min() + self.D / 2
        spine = c[:, 1].max()                                        # frontmost point of the book
        target, rule = self.seat_target(truth=True)                  # from where the neighbours ended up
        if rule == "flush":
            offset = spine - target                                  # > 0: sticks out past the neighbour
            in_place = abs(offset) <= ALIGN_TOL
        else:
            offset = self.D / 2 - spine                              # how far inside the front edge
            in_place = offset >= EDGE_IN
        robot, book_of = self.ik.robot, self.ik.book_of
        touching = any((book_of.get(con.geom1, con.geom1) == self.book_geom and con.geom2 in robot) or
                       (book_of.get(con.geom2, con.geom2) == self.book_geom and con.geom1 in robot)
                       for con in d.contact[:d.ncon])
        others = [i for i in range(len(self.row_bodies)) if i not in self.intended]
        shifts = np.linalg.norm(d.xpos[self.row_bodies] - self.start_pos, axis=1)[others]
        dtilt = np.abs(np.array([book_tilt_deg(d.xquat[i]) for i in self.row_bodies]) - self.start_tilt)[others]
        res = {
            "upright": tilt <= UPRIGHT_DEG, "in_place": in_place, "released": not touching,
            "rule": rule, "offset_m": offset, "tilt_deg": tilt, "back_gap_m": back_gap, "blocked": blocked,
            "neighbour_shift_m": float(shifts.max()), "total_shift_m": float(shifts.sum()),
            "neighbour_tilt_change_deg": float(dtilt.max()),
            "time_to_release_s": t_release, "time_to_seat_s": t_done,
            **self.info, **info, **self.collisions(),
        }
        res["success"] = res["upright"] and res["in_place"] and res["released"]
        res["safe_success"] = res["success"] and not res["collision_s"]   # ... and never touched the shelf or cart
        return res

    # --- The controller: look at the slot, take the step it needs, look again --------
    def reshelve(self, margin=GAP_MARGIN):
        """Put the book in, whatever the row looks like. Look at the slot, take
        the high-level step it needs next, and look again:
          stand_up  a neighbour leans into the slot: stand it back up;
          widen     the gap is narrower than the book plus `margin`, and the
                    fingers fit in it: push a neighbour aside with them;
          shove     they don't fit (the neighbours touch): pinch a neighbour
                    and shove it, and the books beyond it, into the row's room;
          insert    the gap is wide enough, or nothing more would widen it and
                    the book can still be wedged in.
        Each step starts from wherever the arm is and leaves the hand out of
        the shelf, so they can follow one another in any order. The scenario
        never comes into it: the same steps handle every row."""
        taken = []                                                   # (step, whether it changed anything)
        while True:
            step = self.next_step(margin, taken)
            if step is None:
                return self.skip(self.info.get("why", "no step left that would make room for the book"))
            name, *args = step
            self.info["method"] = ", ".join([s for s, _ in taken] + [name])
            self.log(f"  -- {name.replace('_', ' ')}")
            if name == "insert":
                return self.insert()
            taken.append((name, getattr(self, name)(*args)))

    def next_step(self, margin, taken):
        """The step the slot needs now (name, args), or None if nothing would help."""
        count = lambda name: sum(s == name for s, _ in taken)
        stuck = lambda name: any(s == name and not did for s, did in taken)
        est = self.estimate()
        lean = [s for s in (-1, 1) if est.book(s) is not None and -s * est.book(s).lean > math.radians(LEAN_MIN_DEG)]
        if lean:
            if count("stand_up") >= STAND_STEPS:
                self.info["why"] = self.info.get("stand", "the neighbour fell back into the slot")
                return None
            return "stand_up", lean[0]
        xl, xr, _, _ = self.slot()
        if self.widen_gap() < self.t + margin - 0.001:
            gl, gr = self.finger_gap()
            if gr - gl < 2 * (FINGER_HALF_W + FIT_CLEAR):            # the fingers don't fit: make room first
                if not count("shove"):
                    return "shove", margin
            elif not stuck("widen"):
                harder = count("widen") < WIDEN_STEPS and xr - xl < self.t + HARD_BELOW   # only while it doesn't fit
                if count("widen") < 2 or harder:
                    return "widen", margin
        if xr - xl < self.t:
            self.info.setdefault("why", "the gap stayed narrower than the book")
            return None
        return ("insert",)

    # --- Step: insert -----------------------------------------------------------------
    def insert(self):
        """Take the book (at the pre-insertion pose: the hand-over is not
        modelled), slide it in, let go and push it into place. It goes along
        the middle of the straight way in if that clears it by CHANNEL_CLEAR,
        and is otherwise wedged in past a turned neighbour (wedge_slot)."""
        width, x_mid = self.channel()
        wedge_from = None
        if width >= self.t + CHANNEL_CLEAR:
            x_slot = x_mid
            if self.lean_on_neighbour and width > 2 * self.t:
                # Room for it to lean past its tipping angle, centred: stand it
                # against the taller upright neighbour instead.
                est = self.estimate()
                up = [s for s in (-1, 1) if est.book(s) and abs(est.book(s).lean) < math.radians(3)]
                if up:
                    s = max(up, key=lambda s: est.book(s).height)
                    x_slot = x_mid + s * (width / 2 - self.t / 2 - SIDE_GAP)
        else:
            x_slot, wedge_from = self.wedge_slot()
        pre = self.book_pre(x_slot)
        obs = self.obstacles()
        candidates = [(self.grasp_pose(gs, pre), self.insertion_plan(gs, ps, x_slot, pre[1], obs), (gs, ps))
                      for gs in (1, -1) for ps in (1, -1)]
        first = self.run is None
        gs, ps = self.begin(candidates)
        self.take_book(gs, pre, name="close the gripper on the book at the pre-insertion pose" if first else
                       "take the book (the hand-over is not modelled)")
        return self.score(*self.seat(x_slot, pre[1], ps, wedge_from))

    # --- Step: widen (the fingers fit in the gap) -------------------------------------
    def widen(self, margin):
        """Put the fingers, stacked vertically (10 mm along the row), into the
        gap and push one neighbour aside, as hard as the arm can, until the gap
        is the book plus `margin` (widen_gap), or as far as the room on that
        side allows: a side not pushed yet that has room for all of it, else
        the one with more room. The arm configuration is chosen so it can carry
        on through the insertion, or, pushing a side again (the arm gave way
        the first time), for force. The fingers only reach ~4 cm past the
        spines, so the neighbour swivels rather than slides: the gap widens at
        the spines but less at the back. Returns whether it pushed (else why
        not is in self.info["why"])."""
        need = self.t + margin - self.widen_gap()
        gl, gr = self.finger_gap()
        if gr - gl < 2 * (FINGER_HALF_W + FIT_CLEAR):
            self.info["why"] = "gap narrower than the fingers"
            return False
        room = dict(zip((-1, 1), (r - ROOM_KEEP for r in self.free_space())))

        def reach(side, d):
            """Fingertip depth for pushing `side` d along, and how far that is behind its spine."""
            x_mid, x_end, spine = self.push_span(side, d)
            tip = self.fingertip_depth([x_mid, x_end], self.obstacles())
            return tip, spine - tip

        # A side not pushed yet if its room is enough, else the side with more room (pushed again,
        # the arm configuration is chosen for force: it gave way the first time).
        order = sorted((-1, 1), key=lambda s: (s in self.widened or room[s] < need - 0.001, -room[s]))
        for side in order:
            d = min(need, room[side])
            if d < 0.001:
                continue
            tip_in, behind = reach(side, d)
            if behind < MIN_REACH + LEVEL_PROUD:
                # Books the palm sweeps past stand proud of the neighbour (it went back for a
                # shove), so the fingertips would get little or none of its face: push those
                # books back level with it first.
                if self.level_for_fingers(side, d):
                    tip_in, behind = reach(side, d)
            if behind < MIN_REACH:
                continue
            if self.fingertip_push(side, d, tip_in, strong=side in self.widened):
                self.widened.add(side)
                return True
        self.info.setdefault("why", "the fingers can't reach a neighbour with room behind it")
        return False

    def level_for_fingers(self, side, d):
        """Push back level with the neighbour on `side` every book the palm
        sweeps past, pushing it d along, that stands more than LEVEL_PROUD
        proud of its spine. Returns whether there were any."""
        ti = self.row.target_index
        k = ti if side > 0 else ti - 1
        x_mid, x_end, spine = self.push_span(side, d)
        lo, hi = min(x_mid, x_end) - HAND_HALF_T - CLEAR, max(x_mid, x_end) + HAND_HALF_T + CLEAR
        proud = [i for i in range(len(self.row_geoms)) if i != k and self.known(i) and front_x(self.book_box(i))[1] > lo
                 and front_x(self.book_box(i))[0] < hi and self.book_box(i)[:, 1].max() > spine + LEVEL_PROUD]
        for i in sorted(proud, key=lambda i: abs(i - k)):
            try:
                self.push_back(i, spine, "shoved neighbour" if i == self.shoved else "book beside the gap")
            except RuntimeError as e:                               # (the fingers may still reach)
                self.log(f"         can't push it back level: {e}")
        return bool(proud)

    def push_span(self, side, d):
        """Pushing the neighbour on `side` d along: where the hand starts (x
        of the gap's middle, where the fingers go) and ends, and the y of that
        neighbour's spine."""
        gl, gr = self.finger_gap()
        _, _, fl, fr = self.slot()
        face, spine = (gl, fl) if side < 0 else (gr, fr)
        return (gl + gr) / 2, face + side * (d - FINGER_HALF_W), spine

    def fingertip_push(self, side, d, tip_in, strong=False):
        """Push the neighbour on `side` (-1 left, +1 right) d along the row
        with the fingertips tip_in deep, with an arm configuration that can
        carry on through the insertion, or (strong) the one that pushes
        hardest. Returns False if no arm configuration can do it."""
        obs = self.obstacles()
        x_mid, x_end, _ = self.push_span(side, d)                   # hand x as the push starts and ends
        x_slot = x_mid + side * d / 2                                # slot centre once widened
        lo, hi = min(x_end, x_mid) - HAND_HALF_T - CLEAR, max(x_end, x_mid) + HAND_HALF_T + CLEAR
        tip_out = max(front for a, b, front in obs if b > lo and a < hi) + 0.02
        pre = self.book_pre(x_slot)
        candidates = []
        for ps in (1, -1):
            push = ([self.push_pose(ps, x_mid, y) for y in np.linspace(tip_out, tip_in, 15)[1:]]
                    + [self.push_pose(ps, x, tip_in) for x in np.linspace(x_mid, x_end, 15)[1:]]
                    + [self.push_pose(ps, x, tip_in) for x in np.linspace(x_end, x_slot, 10)[1:]]
                    + [self.push_pose(ps, x_slot, y) for y in np.linspace(tip_in, tip_out, 15)[1:]])
            if strong:
                candidates.append((self.push_pose(ps, x_mid, tip_out), push, set(range(14, 28)), ps))
                continue
            for gs in (1, -1):
                path = push + segment(push[-1], self.grasp_pose(gs, pre), 40) + \
                    self.insertion_plan(gs, ps, x_slot, pre[1], obs)
                candidates.append((self.push_pose(ps, x_mid, tip_out), path, ps))
        ti = self.row.target_index
        # The fingers may touch the neighbours, and the books beyond the one they push: those go
        # ahead of them, but not in the planner's static scene, where a push longer than the
        # neighbour is thick would run them into the next book.
        moving = set(self.row_geoms[ti:] if side > 0 else self.row_geoms[:ti])
        allowed = {self.row_geoms[ti - 1], self.row_geoms[ti]} | moving
        try:
            if strong:
                force, q, ps = self.choose_strong(candidates, allowed, 0.04, side * self.xs,
                                                  lambda ps: np.array([0, -ps * FINGER_OPEN_MID, FINGERTIP_Z]), moving)
                self.log(f"         the arm can push along the row with up to {force:.0f} N")
                self.reposition(q)
            else:
                ps = self.begin(candidates, allowed=allowed, finger=0.04)
        except RuntimeError as e:
            self.info["why"] = str(e)
            return False
        run = self.run
        if run.grip != GRIP_OPEN:
            run.wait("open the gripper", 0.3, grip=GRIP_OPEN)
        run.move("put the fingers into the gap", *self.push_pose(ps, x_mid, tip_in))
        room = self.free_space()[0 if side < 0 else 1]
        xl, xr, _, _ = self.slot()
        face = xr if side > 0 else xl                                # at the board: where the book needs it
        self.push_neighbour(ps, side, face + side * d, face + side * (room - ROOM_KEEP), tip_in,
                            x_end - side * (d - FINGER_HALF_W) - face)
        gl, gr = self.finger_gap()
        x_slot = (gl + gr) / 2
        run.move("back to the middle", *self.push_pose(ps, x_slot, tip_in))
        run.move("take the fingers out", *self.push_pose(ps, x_slot, tip_out))
        self.looked()
        xl, xr, _, _ = self.slot()
        self.log(f"         the gap is now {1000 * (xr - xl):.1f} mm at the spines and "
                 f"{1000 * self.narrowest_gap():.1f} mm at its narrowest, for a {1000 * self.t:.1f} mm book")
        return True

    def fingertip_depth(self, xs, obs):
        """Deepest the fingertips may go (smallest y) while the hand sweeps
        sideways over the hand positions xs: the palm stays clear of the books."""
        lo, hi = min(xs) - HAND_HALF_T, max(xs) + HAND_HALF_T
        return tip_limit(obs, (lo + hi) / 2, [((hi - lo) / 2, FINGERTIP_Z - PALM_Z)])

    def push_neighbour(self, push_sign, side, target, limit, tip_in, lean_off=0.0, tries=3):
        """Push the left (side -1) or right (+1) neighbour until its face, at
        its spine and the board, reaches x = target, never commanding past
        x = limit. Its face where the fingers push is lean_off along from
        there (it leans away from the slot). Under load the arm gives way by
        up to ~2.5 cm (joint 6 tops out at 12 N m), and the neighbour swivels,
        so push again by what is missing."""
        name = "left" if side < 0 else "right"
        cmd = target
        hand_at = lambda x: self.push_pose(push_sign, x + lean_off - side * FINGER_HALF_W, tip_in)
        # The residual may correct the pushes, but never past the room limit, with the fingertips
        # no deeper than keeps the palm clear of the spines, and the hand above the board.
        with self.run.contact("widen", lo=(-RESIDUAL_MAX, -(CLEAR - RESIDUAL_FLOOR), -(PUSH_LIFT - RESIDUAL_FLOOR)),
                              x_limit=(side, self.shelf.to_shelf(hand_at(limit)[0])[0])):
            for k in range(tries):
                self.run.move(f"push the {name} neighbour{' further' if k else ''} to its mark", *hand_at(cmd),
                              speed=GAP_PUSH_SPEED)
                self.feel(side)
                xl, xr, _, _ = self.slot()
                short = side * (target - (xl if side < 0 else xr))
                if short < 0.001 or cmd == limit:
                    return
                cmd = cmd + side * short                               # aim past the mark by what is missing
                if side * (cmd - limit) > 0:
                    cmd = limit

    # --- Step: stand_up (a neighbour leans into the slot) -----------------------------
    def stand_up(self, side):
        """Stand the neighbour on `side` (-1 left, +1 right), which leans into
        the slot, back up: lift it (lift_up), or clamp it (clamp_up) if the
        fingers can't get under it at any height. Returns whether it stands."""
        ti = self.row.target_index
        self.info.setdefault("lean_deg", math.degrees(abs(self.estimate().book(side).lean)))
        self.intended.add(ti - 1 if side < 0 else ti)              # moving it is the point, not a disturbance
        self.lean_on_neighbour = True
        up = self.lift_up(side)
        if up is None:                                              # no way under it: clamp it instead
            up = self.clamp_up(side)
        return bool(up)

    def deepest_tip(self, xs, z_axis):
        """fingertip_depth, and, if the turned hand reaches below the board's
        top, no deeper than keeps its palm in front of the board's edge."""
        tip = self.fingertip_depth(xs, self.obstacles())
        if z_axis - HAND_HALF_L < self.z0 + CLEAR:
            tip = max(tip, self.D / 2 + CLEAR - (FINGERTIP_Z - PALM_Z))
        return tip

    def lift_up(self, side):
        """Stand the neighbour on `side` (-1 left, +1 right), which leans into
        the slot, back up: the closed fingers, stacked (10 mm along the row),
        go under it and lift it along an arc about the bottom corner it pivots
        on, to nearly upright, pressing square on its face, so it turns rather
        than slides. They meet it as far up as they fit. A book within its
        tipping angle (atan(thickness / height)) of upright stands up by
        itself, so this only has to get it near there. Returns True if it
        stays up once the hand has left, False if it falls back, and None if
        the fingers can't get under it at all (nothing has moved then)."""
        ti = self.row.target_index
        geom = self.row_geoms[ti - 1 if side < 0 else ti]
        thick, w = FINGER_THICK, FINGER_HALF_W
        out = -side                                                  # toward the slot, from the leaner
        for attempt in range(STAND_TRIES):
            est = self.estimate()
            b = est.book(side)
            c = b.spine_corners
            pivot = c[1] if side < 0 else c[0]                       # slot-side bottom corner
            lean = -side * b.lean                                    # > 0: into the slot
            # Take it to within its tipping angle of upright, still leaning
            # slightly into the slot, so it lands on its base by itself rather
            # than coasting over the top onto its other side. Closer to
            # upright on each retry.
            beta = -STAND_TO[attempt] * math.atan(b.thickness / b.height)
            plan, why = None, {}
            for r in np.arange(0.9 * b.height, 0.15 * b.height, -0.01):
                # Contact point on its slot-side face, r up from the pivot. The
                # fingers' top corner sits there, the fingers on the slot side.
                cx, cz = pivot[0] + out * r * math.sin(lean), pivot[2] - self.z0 + r * math.cos(lean)
                if cz - 2 * thick < 0.005:
                    continue
                # Room beyond the fingers pressed 1 mm into its face, down to
                # where they come in from (up to 4 mm lower).
                far = [self.gap_at(z, est)[1 if side < 0 else 0] for z in np.linspace(cz - 2 * thick - 0.005, cz, 5)]
                room = (min(far) - cx if side < 0 else cx - max(far)) - (2 * w - 0.001) - FIT_CLEAR
                if room < 0.001:
                    why["no room under it"] = why.get("no room under it", 0) + 1
                    continue                                         # the fingers don't fit under it here
                arc = [(pivot[0] + out * r * math.sin(th) + out * (w - 0.001),
                        self.z0 + r * math.cos(th) - thick - 0.001) for th in np.linspace(lean, -beta, 30)]
                xs = [a[0] for a in arc]
                tip_in = self.deepest_tip([min(xs), max(xs)], min(a[1] for a in arc))
                if b.spine_y - tip_in < MIN_REACH:
                    why["can't reach behind its spine"] = why.get("can't reach behind its spine", 0) + 1
                    continue
                tip_out = max(front for _, _, front in self.obstacles()) + 0.02
                off = min(0.004, room)                               # come in just clear of its face
                start, cands = (arc[0][0] + out * off, arc[0][1] - off), []
                for sign in (1, -1):
                    path = ([self.col_pose(sign, start[0], y, start[1]) for y in np.linspace(tip_out, tip_in, 15)[1:]]
                            + [self.col_pose(sign, x, tip_in, z) for x, z in arc]
                            + [self.col_pose(sign, arc[-1][0], y, arc[-1][1])
                               for y in np.linspace(tip_in, tip_out, 10)[1:]])
                    cands.append((self.col_pose(sign, start[0], tip_out, start[1]), path, sign))
                try:
                    sign = self.begin(cands, allowed={geom}, finger=0.0)
                except RuntimeError:
                    why["no arm path"] = why.get("no arm path", 0) + 1
                    continue
                plan = (r, start, arc, tip_in, tip_out, sign)
                break
            if plan is None:
                self.info["stand"] = "the fingers can't get under the leaning book" + (
                    f" ({max(why, key=why.get)})" if why else "")
                return None if attempt == 0 else False
            r, start, arc, tip_in, tip_out, sign = plan
            self.info["stood_up_by"] = "lift"
            run = self.run
            if run.grip != GRIP_CLOSED:
                run.wait("close the gripper", 0.3, grip=GRIP_CLOSED)
            run.move("put the closed fingers under the leaning book", *self.col_pose(sign, start[0], tip_in, start[1]))
            with run.contact("lift", lo=(-RESIDUAL_MAX, -(CLEAR - RESIDUAL_FLOOR), -RESIDUAL_MAX)):
                run.follow(f"lift it up, {1000 * r:.0f} mm up its face",
                           [self.col_pose(sign, x, tip_in, z) for x, z in arc], speed=LIFT_SPEED)
            run.move("take the fingers out", *self.col_pose(sign, arc[-1][0], tip_out, arc[-1][1]))
            run.wait("let it settle", 0.5)
            self.looked()
            b = self.estimate().book(side)
            self.log(f"         it now leans {math.degrees(-side * b.lean):+.1f} deg into the slot")
            if -side * b.lean < math.radians(LEAN_MIN_DEG):
                return True
        return False

    def clamp_up(self, side):
        """Stand the neighbour on `side` (-1 left, +1 right), which leans into
        the slot, back up by holding it: the fingers pinch it across its
        thickness near the spine, with the hand turned to its lean, and the
        hand swings it about the bottom edge it pivots on to just short of
        upright. Only one finger has to fit between it and the far book. The
        other goes between it and the book on its own side, which is pushed
        along a little. Returns True if it stays up once let go."""
        ti = self.row.target_index
        li = ti - 1 if side < 0 else ti
        geom, body, bk = self.row_geoms[li], self.row_bodies[li], self.row.books[li]
        if self.eyes is None:
            t, depth, h = bk.thickness, bk.depth, bk.height
            R = self.shelf.R.T @ self.d.xmat[body].reshape(3, 3)      # its axes, shelf frame
            c = self.shelf.to_shelf(self.d.xpos[body])
        else:                                                         # its box as the camera saw it (depth: the prior)
            from bila.vision.perception import book_box
            c, R, half = book_box(self.seen().book(side))
            t, depth, h = 2 * half
        out = -side                                                   # toward the slot along x
        xb = R[:, 0] * (1 if R[0, 0] * out > 0 else -1)               # across it, toward the slot
        yb = R[:, 1] * (1 if R[1, 1] > 0 else -1)                     # front to back, toward the spine
        zb = R[:, 2] * (1 if R[2, 2] > 0 else -1)                     # up it
        lean = math.atan2(out * zb[0], zb[2])                         # > 0: into the slot
        pivot = c + xb * t / 2 - zb * h / 2                           # slot-side bottom edge
        turn = axis_rot(yb, 0.01) @ zb                                # which way about yb stands it up
        swing = (1 if math.atan2(out * turn[0], turn[2]) < lean else -1) * (lean - 0.5 * math.atan(t / h))
        opening = min(0.04, t / 2 + 0.004)                            # each finger this far off the axis
        # Grip as deep as PAD_INSET, but keep the palm (and the wide hand)
        # in front of every spine nearby.
        spine = (c + yb * depth / 2)[1]
        front = max(f for lo, hi, f in self.obstacles()
                    if hi > c[0] - HAND_HALF_L - 0.02 and lo < c[0] + HAND_HALF_L + 0.02)
        inset = min(PAD_INSET, spine - (front + CLEAR - (PAD_Z - PALM_Z)))
        if inset < 0.012:
            self.info["stand"] = "neighbours stick out too far for the pads to get onto the leaning book"
            return False
        tip_out = max(f for _, _, f in self.obstacles()) + 0.03
        allowed = {geom, self.row_geoms[ti - 1], self.row_geoms[ti]} | set(
            self.row_geoms[:li] if side < 0 else self.row_geoms[li + 1:])   # its own side is pushed along
        why = "no height where the fingers fit around it"
        for r in np.arange(0.85 * h, 0.3 * h, -0.01):
            g = c - zb * h / 2 + zb * r + yb * (depth / 2 - inset)       # where the pads hold it
            cands = []
            for gs in (1, -1):
                hy, hz = gs * (self.shelf.R @ xb), -(self.shelf.R @ yb)
                rot = np.column_stack([np.cross(hy, hz), hy, hz])

                def pose(ang, out_y=0.0, g=g, rot=rot):
                    Q = axis_rot(yb, ang)
                    Rk = self.shelf.R @ Q @ self.shelf.R.T @ rot
                    return self.shelf.to_world(pivot + Q @ (g - pivot) + yb * out_y) - Rk @ [0, 0, PAD_Z], Rk

                out_y = tip_out - (g[1] - (FINGERTIP_Z - PAD_Z))       # fingertips in front of every spine
                path = ([pose(0.0, out_y * (1 - s)) for s in np.linspace(0, 1, 15)[1:]]
                        + [pose(swing * s) for s in np.linspace(0, 1, 25)[1:]])
                cands.append((pose(0.0, out_y), path, (pose, out_y)))
            try:
                pose, out_y = self.begin(cands, allowed=allowed, finger=opening)
            except RuntimeError as e:
                why = str(e)
                continue
            self.info["stood_up_by"] = "clamp"
            run = self.run
            run.wait("open the gripper to the book's thickness", 0.3, grip=GRIP_OPEN * opening / 0.04)
            run.move("put the fingers around the leaning book", *pose(0.0))
            self.close_on(opening, "clamp it")
            with run.contact("clamp"):
                run.follow(f"swing it up about its bottom edge, held {1000 * r:.0f} mm up",
                           [pose(swing * s) for s in np.linspace(0, 1, 25)[1:]], speed=LIFT_SPEED)
            run.wait("let go", 0.4, grip=GRIP_OPEN * opening / 0.04)
            p_end, R_end = pose(swing)
            run.move("take the fingers out", p_end + R_end @ [0, 0, -0.07])
            run.wait("let it settle", 0.5)
            self.looked()
            b = self.estimate().book(side)
            self.log(f"         it now leans {math.degrees(-side * b.lean):+.1f} deg into the slot")
            if -side * b.lean < math.radians(LEAN_MIN_DEG):
                return True
            self.info["stand"] = "it fell back into the slot after the clamp let go"
            return False
        self.info["stand"] = f"can't clamp the leaning book either ({why})"
        return False

    # --- Step: shove (the neighbours touch: make room) --------------------------------
    def shove(self, margin):
        """Make room where the neighbours touch. The room can only come from
        the row's free space at its ends, so every book between the slot and
        that space has to slide along. On the side with more free space (or
        the other, if the books beside its neighbour can't go back far enough
        to pinch it), push the books beside the neighbour back so it sticks out
        of the row, pinch it, and shove it and every book beyond it into the
        room, until the gap is the book plus `margin` at its narrowest or the
        room runs out (shove_open). If neither side can be pinched, walk the
        neighbour out of the row first (walk_out). A run of SEGMENT_MIN books
        or more is shoved in two, its far half on its own, then the near half
        again, if the neighbour turned TURN_STOP_DEG in the first shove (it
        stops there) or the gap is still short. Returns whether it shoved."""
        room = dict(zip((-1, 1), self.free_space()))
        now = self.narrowest_gap()
        need = self.t + margin - now
        if sum(room[s] - ROOM_KEEP for s in (-1, 1)) < need:
            self.info["why"] = "not enough free space in the row for the book"
            return False
        ti = self.row.target_index
        order = sorted((-1, 1), key=lambda s: -room[s])
        if room[order[0]] - ROOM_KEEP >= need:
            # (side, gap wanted)
            options = [(side, self.t + margin) for side in order if room[side] - ROOM_KEEP >= need]
        else:
            # Shove one side as far as its room allows; widening takes the rest from the other.
            # (Shoving that side too would mean making its neighbour stick out again after it
            # was pushed back for this shove.)
            options = [(order[0], now + room[order[0]] - ROOM_KEEP)]
        why = []
        for side, gap in options:
            reason = self.shove_ready(side)
            if reason is None:
                break
            why.append(reason)
        else:
            # The books beside the neighbour can't go back far enough for a pinch to shove it with:
            # walk it out of the row first.
            side, gap = options[0]
            if not self.walk_out(ti if side > 0 else ti - 1):
                self.info["why"] = "; ".join(why)
                return False
        k = self.shoved = ti if side > 0 else ti - 1
        # A long run that pushes back hard makes the neighbour pivot about its back corner rather
        # than slide it, and the further it is shoved, the more it turns, shutting the gap at the
        # back: stop once it starts turning, and shove the run in two (below).
        run = list(range(k, len(self.row_geoms))) if side > 0 else list(range(k, -1, -1))
        run = [i for i in run if self.known(i)]                     # (with eyes, the books the camera saw)
        long_run = len(run) >= SEGMENT_MIN
        cannot = self.shove_open(side, k, gap, room[side] - ROOM_KEEP, turn_stop=long_run)
        if cannot and self.walked is None and self.walk_out(k):    # (too little to pinch once they went back)
            cannot = self.shove_open(side, k, gap, room[side] - ROOM_KEEP, turn_stop=long_run)
        if cannot:
            self.info["why"] = cannot
            return False
        # If the gap is still short (it turned, or the arm can't push the whole run far enough),
        # shove the far half of the run on its own, then the near half into the gap that leaves.
        if self.narrowest_gap() < gap - 0.001 and long_run:
            for j in sorted(run[2:-1], key=lambda j: abs(abs(j - k) - len(run) // 2))[:3]:
                if self.shove_ready(side, j) is not None:
                    continue
                self.info["shoved_in_two"] = True
                pair = (min(j, j - side), max(j, j - side))
                try:
                    self.shove_open(side, j, gap, self.free_space()[1 if side > 0 else 0] - ROOM_KEEP, pair)
                    self.shove_open(side, k, gap, max(self.narrowest_gap(pair), 0.0) + 0.005)
                except RuntimeError as e:
                    self.log(f"         can't shove the run in two: {e}")
                break
        xl, xr, _, _ = self.slot()
        self.info["narrowest_gap_m"] = self.narrowest_gap()
        self.log(f"         the gap is now {1000 * (xr - xl):.1f} mm at the spines and "
                 f"{1000 * self.info['narrowest_gap_m']:.1f} mm at its narrowest, for a {1000 * self.t:.1f} mm book")
        return True

    def wedge_slot(self):
        """Where to wedge the book in past a neighbour that turned or swivelled
        when it was pushed at its front (its back stayed): (x of the book's
        centre, the centre's y from which to slide it slowly, WEDGE_RUN_UP
        before it meets a neighbour's face). Held, its back corner pushes the
        turned face aside deep in the shelf, near the middle of the books'
        friction, so the neighbour and the run beyond straighten. (Fingers
        pushing a spine only turn it further, and can't reach that deep.) It
        goes in where it meets either face deepest: along the straighter
        neighbour's face if only one turned, between them if both did."""
        ti = self.row.target_index
        xl, xr, _, _ = self.slot()
        top = max(self.book_box(ti - 1)[:, 1].max(), self.book_box(ti)[:, 1].max())
        ys = np.linspace(top, top - self.depth, 60)
        left = np.array([self.face_x(ti - 1, 1, y) for y in ys])
        right = np.array([self.face_x(ti, -1, y) for y in ys])
        xs = np.linspace(xl + self.t / 2 + WEDGE_HUG, xr - self.t / 2 - WEDGE_HUG, 15)
        # how far in each x goes before the book's back corners meet a face
        first = [int(np.argmax(np.append((left > x - self.t / 2 + 0.001) | (right < x + self.t / 2 - 0.001), True)))
                 for x in xs]
        deepest = max(first)
        x_slot = float(np.mean([x for x, f in zip(xs, first) if f == deepest]))
        meet = ys[min(deepest, len(ys) - 1)]
        self.info["wedge_meet_m"] = top - meet
        self.log(f"         wedge the book in {1000 * (x_slot - (xl + xr) / 2):+.0f} mm off the gap's middle: "
                 f"it meets a neighbour {1000 * (top - meet):.0f} mm behind the spines")
        return x_slot, meet + WEDGE_RUN_UP + self.depth / 2

    walked = None                   # (row book walked out, its spine before)

    def walk_out(self, k):
        """Where the books beside neighbour k can't go back far enough for the
        shove's pinch, walk it out of the row: push them back as far as they
        go, pinch what little of it sticks out (the fingertips just in front
        of them: this only pulls, so the arm doesn't sag in), pull it
        WALK_STEP out, and pinch again deeper, until it is WALK_TARGET out.
        Pinches of 2 mm held it. Returns whether it came out."""
        half = self.half_size(k)[0] + PINCH_OPEN
        start = self.book_box(k)[:, 1].max()
        self.walked = (k, start)
        for _ in range(WALK_TRIES):
            c, psi = self.book_box(k), self.book_yaw(k)
            x_k, spine, y_mid = c[:, 0].mean(), c[:, 1].max(), c[:, 1].mean()
            if spine - start >= WALK_TARGET - 0.002:
                break
            for i in sorted(self.under_fingers(k, half, x_k), key=lambda i: abs(i - k)):
                if self.book_box(i)[:, 1].max() > spine - SHOVE_BACK + 0.002 and self.room_behind(i) > 0.002:
                    try:
                        self.push_back(i, spine - SHOVE_BACK, "book beside the neighbour")
                    except RuntimeError as e:
                        self.log(f"         can't push it back: {e}")
            tip = max(self.tip_floor({k}, [(x_k - half - FINGER_THICK - 0.001, x_k + half + FINGER_THICK + 0.001, 0.0,
                                             WALK_CLEAR),
                                            (x_k - HAND_HALF_L - CLEAR, x_k + HAND_HALF_L + CLEAR, FINGERTIP_Z - PALM_Z,
                                             PALM_CLEAR)]),
                      spine + 0.002 - (FINGERTIP_Z - PALM_Z))
            if spine - tip < WALK_MIN:
                self.log(f"         can't walk the neighbour out: only {1000 * (spine - tip):.1f} mm of it to pinch")
                break
            z = self.z0 + max(PINCH_Z, HAND_HALF_T + CLEAR)
            mid = lambda y: x_k - (y - y_mid) * math.tan(psi)
            out = max(f for _, _, f in self.obstacles()) + 0.02
            at = lambda sign, y: self.pinch_pose(sign, mid(y), y, z, psi)
            cands = [(at(sign, out), [at(sign, y) for y in np.linspace(out, tip, 15)[1:]]
                      + [at(sign, y) for y in np.linspace(tip, out + WALK_STEP, 12)[1:]], sign) for sign in (1, -1)]
            try:
                sign = self.begin(cands, allowed={self.row_geoms[k]}, finger=half)
            except RuntimeError as e:
                self.log(f"         can't walk the neighbour out: {e}")
                break
            run = self.run
            run.wait("open the fingers around the neighbour", 0.3, grip=GRIP_OPEN * half / 0.04)
            run.move(f"fingers either side of its front ({1000 * (spine - tip):.1f} mm of it)", *at(sign, tip))
            self.close_on(half, "pinch it")
            self.carry(k)
            run.move(f"pull it {1000 * WALK_STEP:.0f} mm out", *at(sign, tip + WALK_STEP), speed=0.01)
            run.halt()
            self.let_go()
            run.wait("let go of it", 0.3, grip=GRIP_OPEN * (half + 0.002) / 0.04)
            run.move("take the fingers out", *at(sign, tip + WALK_STEP + 0.03))
            self.looked()
        out_by = self.book_box(k)[:, 1].max() - start
        self.info["walked_m"] = out_by
        self.log(f"         walked the neighbour {1000 * out_by:.0f} mm out")
        return out_by > 0.005

    def shove_ready(self, side, k=None):
        """None if row book k (by default the slot's neighbour on `side`) can
        be pinched and shoved (it isn't too thick, and the books a finger goes
        in front of can go back far enough), else why not."""
        k = (self.row.target_index if side > 0 else self.row.target_index - 1) if k is None else k
        if not (self.known(k) and self.known(k - side)):
            return "the camera hasn't seen the book to shove or the one beside it"
        half = self.half_size(k)[0] + PINCH_OPEN
        if half > 0.04:
            return "the neighbour is too thick to pinch"
        back = self.stick_out(k, self.under_fingers(k, half, self.book_box(k)[:, 0].mean()))
        if back < SHOVE_BACK_MIN:
            return f"only {1000 * back:.0f} mm of room behind the books beside the neighbour"
        return None

    def under_fingers(self, k, half, x):
        """The row books (but k) a finger pinching row book k, fingers `half`
        open around x, would be in front of."""
        fingers = ((x - half - FINGER_THICK - 0.002, x - half), (x + half, x + half + FINGER_THICK + 0.002))
        fronts = [(i, front_x(self.book_box(i))) for i in range(len(self.row_geoms)) if i != k and self.known(i)]
        return [i for i, (a, b) in fronts if any(b > lo and a < hi for lo, hi in fingers)]

    def stick_out(self, k, flank):
        """How far row book k can be made to stick out of the books in flank,
        up to SHOVE_BACK, by pushing them back."""
        spine = self.book_box(k)[:, 1].max()
        return min([SHOVE_BACK] + [spine - self.book_box(i)[:, 1].max() + self.room_behind(i) for i in flank])

    def shove_open(self, side, k, need, limit, pair=None, turn_stop=False):
        """Shove row book k, the slot's neighbour on `side`, and every book
        beyond it, away from the slot until the gap is `need` wide where the
        book will go (moving it at most `limit`), then push it back in line.
        Each pinch: push the books a finger will be in front of straight back,
        so k sticks out of them, and shove_stage. pair: the row books (left,
        right) whose gap to open, if not the slot's. turn_stop: stop once k
        turns (shove_stage). Returns None, or why it can't pinch it."""
        other = self.row.target_index - 1 if side > 0 else self.row.target_index
        half = self.half_size(k)[0] + PINCH_OPEN
        line = self.book_box(k)[:, 1].max()
        if self.walked is not None and self.walked[0] == k:          # walked out: back where it stood
            line = min(line, self.walked[1])
        x0 = self.book_box(k)[:, 0].mean()                          # how far k goes is counted from here
        # The strong arm configurations reach joint 5's limit within 1.5 to 2 cm of the push, so
        # the shove may take several pinches, each with the configuration chosen afresh. After
        # each, k may have turned and the books beyond fanned out, so the fingers are over others.
        gap = self.narrowest_gap(pair)
        for stage in range(SHOVE_STAGES):
            c, psi = self.book_box(k), self.book_yaw(k)
            x_c = c[:, 0].mean() - (c[:, 1].max() - PINCH_WANT - c[:, 1].mean()) * math.tan(psi)
            flank = self.under_fingers(k, half, x_c)
            back = self.stick_out(k, flank)
            if back < SHOVE_BACK_MIN:
                if stage == 0:
                    return f"only {1000 * back:.0f} mm of room behind the books beside the neighbour"
                break
            if stage == 0:
                self.info["push_back_m"] = back
            spine = c[:, 1].max()
            # Its fingertips go in front of the books pushed back: further off if they went
            # back far (the arm sags deeper under the load), closer if that leaves too little
            # to pinch.
            clear = min(TIP_CLEAR, max(TIP_CLEAR_MIN, back - PINCH_WANT))
            try:
                for i in sorted(flank, key=lambda i: abs(i - k)):
                    if self.book_box(i)[:, 1].max() > spine - back + 0.002:
                        self.push_back(i, spine - back,
                                       "other neighbour" if i == other else "book beside the one to shove")
                room = limit - side * (self.book_box(k)[:, 0].mean() - x0)
                if room < 0.001:
                    break
                cannot = self.shove_stage(side, k, half, clear, need - gap, room, stage, pair, turn_stop)
            except RuntimeError as e:                               # a later pinch is a bonus: go on without it
                if stage == 0:
                    raise
                self.log(f"         no further pinch: {e}")
                break
            if cannot:
                if stage == 0:
                    return cannot
                break
            before, gap = gap, self.narrowest_gap(pair)
            if gap >= need - 0.001 or (stage and gap < before + 0.002) or self.turned:   # open, no longer gives,
                break                                                                    # or turns
        if self.book_box(k)[:, 1].max() > line + 0.003:
            self.push_back(k, line, "shoved neighbour")
        return None

    def shove_stage(self, side, k, half, clear, short, room, stage, pair=None, turn_stop=False):
        """One pinch of shove_open: pinch row book k with the fingertips `clear`
        in front of the books beside it and the palm clear of the rest, the
        hand turned to its yaw; shove it `short` further away from the slot
        (or as far as a strong arm configuration can), but at most `room`;
        let go. turn_stop: stop as soon as k turns TURN_STOP_DEG (self.turned).
        Returns None, or why it can't pinch it."""
        c, psi = self.book_box(k), self.book_yaw(k)
        x_k, spine, y_mid = c[:, 0].mean(), c[:, 1].max(), c[:, 1].mean()
        mid = lambda y: x_k - (y - y_mid) * math.tan(psi)              # its mid-plane at depth y
        x_c = mid(spine - PINCH_WANT)                                 # (where the fingers will be, once it has turned)
        tip_in = max(self.tip_floor({k}, [(x_c - half - FINGER_THICK - 0.001, x_c + half + FINGER_THICK + 0.001, 0.0,
                                           clear),
                                          (x_c - HAND_HALF_L - CLEAR, x_c + HAND_HALF_L + CLEAR, FINGERTIP_Z - PALM_Z,
                                           PALM_CLEAR)]),
                     spine + 0.002 - (FINGERTIP_Z - PALM_Z))
        held = spine - tip_in
        if stage == 0:
            self.info["pinch_m"] = held
        if held < MIN_PINCH:
            return f"only {1000 * held:.0f} mm of the neighbour to pinch"
        z = self.z0 + max(PINCH_Z, HAND_HALF_T + CLEAR)
        tip_out = max(f for _, _, f in self.obstacles()) + 0.02
        x_in = mid(tip_in)
        moving = {self.row_geoms[i] for i in range(len(self.row_geoms)) if side * (i - k) >= 0}   # the fingers may touch

        def tip_at(pitch):
            """Fingertip depth pitched down by `pitch`: the top of the hand then comes closer
            to the spine than the palm does when level."""
            return max(tip_in, spine + CLEAR - (FINGERTIP_Z - PALM_Z) * math.cos(pitch) + HAND_HALF_T * math.sin(pitch))

        def candidates(push):
            cands = []
            for sign in (1, -1):
                for pitch in (PINCH_PITCHES if stage else (0.0,)):
                    tip = tip_at(pitch)
                    if spine - tip < MIN_PINCH:
                        continue
                    x0 = mid(tip)
                    x_end = x0 + side * push
                    at = lambda x, y: self.pinch_pose(sign, x, y, z, psi, pitch)
                    path = ([at(mid(y), y) for y in np.linspace(tip_out, tip, 20)[1:]]
                            + [at(x, tip) for x in np.linspace(x0, x_end, 15)[1:]]
                            + [at(x_end, y) for y in np.linspace(tip, tip_out, 10)[1:]])
                    cands.append((at(mid(tip_out), tip_out), path, set(range(18, 34)), (sign, pitch)))
            return cands

        # The whole run slides, which takes 30 to 110 N. Depending on the arm's configuration
        # it pushes along the row with 55 N (joint 6 at its 12 N m) to 130 N, and the strong
        # configurations have joint 5 near its limit: choose the configuration for the push, and
        # push less far this time if a much stronger one can do that.
        pushes = sorted({min(short, room), min(SHOVE_STEP, room)} if stage else {min(short, room)}, reverse=True)
        plans = []
        for push in pushes:
            try:
                plans.append((*self.choose_strong(candidates(push), moving, half, side * self.xs,
                                                  np.array([0, 0, FINGERTIP_Z - held / 2]), moving), push))
            except RuntimeError:
                pass
        if plans:
            strongest = max(p[0] for p in plans)
            force, q, (sign, pitch), push = next(p for p in plans if p[0] >= 0.95 * strongest)   # longest that strong
            self.info.setdefault("push_force_N", force)
            self.log(f"         the arm can push along the row with up to {force:.0f} N, for {1000 * push:.0f} mm")
            if self.run is None:
                self.place_arm(q)
            else:
                self.reposition(q)
        else:
            push = min(short, room)
            cands = candidates(push)
            sign, pitch = self.begin([cand[:2] + (cand[3],) for cand in cands], allowed=moving, finger=half)
        tip_in = tip_at(pitch)
        x_in = mid(tip_in)
        run = self.run
        run.wait("open the fingers around the neighbour", 0.3, grip=GRIP_OPEN * half / 0.04)
        run.move("put the fingers either side of it", *self.pinch_pose(sign, x_in, tip_in, z, psi, pitch))
        self.close_on(half, "pinch it")
        self.carry(k)                                                # (with eyes, it turns with the hand from here)
        yaw, gap0 = self.book_yaw(k), self.narrowest_gap(pair)
        goal = x_in + side * push
        def turning():
            """it turns rather than sliding the run"""
            return abs(self.book_yaw(k) - yaw) > math.radians(TURN_STOP_DEG)

        # The residual may correct the shove, but never past the room, with the fingertips no
        # deeper than keeps them clear of the books pushed back, and the hand above the board.
        limit = self.shelf.to_shelf(self.pinch_pose(sign, x_in + side * room, tip_in, z, psi, pitch)[0])[0]
        with run.contact("shove", lo=(-RESIDUAL_MAX, -(clear - RESIDUAL_FLOOR),
                                      -(z - self.z0 - HAND_HALF_T - RESIDUAL_FLOOR)), x_limit=(side, limit)):
            for i in range(2 if stage else 4):                       # the arm gives way under load: push on
                if not run.move(f"shove it and the books beyond{' further' if i else ''}",
                                *self.pinch_pose(sign, goal, tip_in, z, psi, pitch), speed=GAP_PUSH_SPEED,
                                stop_when=turning if turn_stop else None):
                    self.turned = True
                    break
                gained = self.narrowest_gap(pair) - gap0
                if gained >= push - 0.001:
                    break
                goal = x_in + side * min(side * (goal - x_in) + push - gained, push + SHOVE_STEP if stage else room,
                                         room)
        self.info["shove_turn_deg"] = self.info.get("shove_turn_deg", 0.0) + math.degrees(self.book_yaw(k) - yaw)
        # Stop pressing where the arm has got to (it lags under the load, and would spring on
        # when let go), open, ease the fingers off its faces, then take them out.
        run.halt()
        self.let_go()
        run.wait("let go of it", 0.3, grip=GRIP_OPEN * (half + 0.002) / 0.04)
        run.move("ease off its faces", run.pos - side * 0.003 * self.xs)
        run.move("take the fingers out", run.pos + run.rot @ [0, 0, tip_in - tip_out])
        self.looked()
        return None

    def push_back(self, i, goal, what):
        """Push row book i straight back along its own depth (fingers stacked,
        hand square to its spine) until the middle of its spine is at y = goal,
        or as deep as the hand can go."""
        c, psi = self.book_box(i), self.book_yaw(i)
        u = np.array([-math.sin(psi), math.cos(psi)])               # out of its spine
        mid = c[:, :2].mean(axis=0) + u * self.half_size(i)[1]     # middle of its spine
        x = mid[0]
        floor = self.tip_floor({i}, [(x - FINGER_HALF_W - 0.001, x + FINGER_HALF_W + 0.001, 0.0, 0.003),
                                     (x - HAND_HALF_T - CLEAR, x + HAND_HALF_T + CLEAR, FINGERTIP_Z - PALM_Z, CLEAR)])
        dist = (mid[1] - max(goal, floor)) / u[1]
        out = max(f for a, b, f in self.obstacles() if b > x - HAND_HALF_T - CLEAR and a < x + HAND_HALF_T + CLEAR) + 0.02
        away = max((out - mid[1]) / u[1], 0.02)
        at = lambda sign, s: self.col_pose(sign, *(mid - u * s), self.hand_z, psi)   # fingertips s behind its spine
        cands = [(at(sign, -away), [at(sign, s) for s in np.linspace(-away, dist, 20)[1:]]
                  + [at(sign, s) for s in np.linspace(dist, -away, 10)[1:]], sign) for sign in (1, -1)]
        # The book goes back ahead of the fingers, but not in the planner's scene: there a
        # push longer than the fingers (46 mm) would run the hand into it.
        ignore = self.ik.ignore
        self.ik.ignore = ignore | {self.row_geoms[i]}
        try:
            sign = self.begin(cands, allowed={self.row_geoms[i]}, finger=0.0)
        finally:
            self.ik.ignore = ignore
        run = self.run
        if run.grip != GRIP_SHUT:
            run.wait("close the gripper", 0.3, grip=GRIP_SHUT)
        run.move(f"reach the {what}'s spine", *at(sign, -0.005))
        run.move(f"push the {what} {1000 * dist:.0f} mm back", *at(sign, dist), speed=PUSH_SPEED,
                 stop_force=PUSH_STOP_FORCE)
        run.move("take the fingers out", *at(sign, -away))
        self.looked()

    def close_on(self, half, name):
        """Close the fingers from `half` open at CLOSE_SPEED, then squeeze. At
        the simulated gripper's own speed (over 2 m/s from a few mm away) the
        pads go up to 18 mm into a book before their contact stops them, and
        squeeze it out of the grasp. A real Franka Hand closes far slower."""
        run = self.run
        self.log(f"  {run.t:5.2f} s  {name}")
        steps = int(round(half / CLOSE_SPEED / self.m.opt.timestep))
        for k in range(1, steps + 1):
            run.grip = GRIP_OPEN * half * (1 - k / steps) / 0.04
            run.step()
        run.grip = GRIP_CLOSED
        for _ in range(int(round(0.3 / self.m.opt.timestep))):
            run.step()

    def push_capacity(self, q, point, direction):
        """Largest force the arm at q can push with at `point` (world) along
        `direction` before a joint reaches its torque limit, N."""
        ik = self.ik
        ik.d.qpos[ik.qadr] = q
        mujoco.mj_kinematics(ik.m, ik.d)
        mujoco.mj_comPos(ik.m, ik.d)
        jacp, jacr = np.zeros((3, ik.m.nv)), np.zeros((3, ik.m.nv))
        mujoco.mj_jac(ik.m, ik.d, jacp, jacr, point, ik.hand)
        # The books pushed resist behind the point the hand pushes at (deeper in the shelf), so
        # the hand also has to hold their moment about the vertical, or the arm yields and the
        # book turns: count a resistance YAW_LEVER behind the point.
        moment = YAW_LEVER * np.cross(-self.ys, direction)
        torque = np.abs(jacp[:, ik.dadr].T @ direction + jacr[:, ik.dadr].T @ moment)
        return float(np.min(self.m.actuator_forcerange[:7, 1] / np.maximum(torque, 1e-9)))

    def choose_strong(self, candidates, allowed, finger, direction, point_hand, moving=frozenset()):
        """choose_arm for a push that has to be strong: of the configurations
        that follow the whole path within STRONG_MARGIN of the joint limits and
        hit nothing, the one that can push hardest along `direction` at
        `point_hand` (hand frame, or a function of the label giving it) over the path's poses whose indices are each
        candidate's third item. From the first of those on, the geoms in
        `moving` go along with the hand, so it may be anywhere near them in the
        planner's (static) scene. candidates: (start pose, path, pushing pose
        indices, label). Returns (force, arm angles, label)."""
        ik, seed_rng, best = self.ik, np.random.default_rng(0), None
        ik.sync(self.d, finger)
        ignore = ik.ignore
        for _ in range(IK_ROUNDS):
            for start, path, pushing, label in candidates:
                for s in [seed_rng.uniform(ik.lo, ik.hi) for _ in range(STRONG_SEEDS)]:
                    q0, ok = ik.solve_far(s, *start)
                    if not ok or ik.margin(q0) < STRONG_MARGIN or ik.collides(q0, allowed):
                        continue
                    ik.q_rest = q0
                    q, force = q0, np.inf
                    for k, (pos, rot) in enumerate(path):
                        q = ik.solve(q, pos, rot, iters=4)
                        ik.ignore = ignore | moving if k >= min(pushing) else ignore
                        hit = k % 3 == 2 and ik.collides(q, allowed)
                        ik.ignore = ignore
                        if ik.margin(q) < STRONG_MARGIN or hit:
                            break
                        if k in pushing:
                            point = point_hand(label) if callable(point_hand) else point_hand
                            force = min(force, self.push_capacity(q, pos + rot @ point, direction))
                            if best is not None and force <= best[0]:
                                break
                    else:
                        e = ik.error(q, *path[-1])
                        if np.linalg.norm(e[:3]) < 1e-3 and np.linalg.norm(e[3:]) < 1e-2 and (best is None or force > best[0]):
                            best = (force, q0, label)
            if best is not None:
                break
        if best is None:
            raise RuntimeError("No arm configuration does the whole sequence without nearing a joint limit "
                               "or running into the shelf")
        ik.q_rest = best[1]
        return best

    def book_box(self, i):
        """Corners (8, 3; shelf frame) of row book i (with eyes: book_seen's box)."""
        if self.eyes is not None:
            return self.seen_box(self.book_seen(i))
        return self.shelf.box(self.m, self.d, self.row_geoms[i])

    def book_yaw(self, i):
        """How far row book i is turned about the vertical, in the shelf frame (rad)."""
        if self.eyes is not None:
            return self.book_seen(i).yaw
        R = self.shelf.R.T @ self.d.xmat[self.row_bodies[i]].reshape(3, 3)
        return math.atan2(R[1, 0], R[0, 0])

    def half_size(self, i):
        """Half thickness, depth and height of row book i (with eyes: as seen, the depth a prior)."""
        if self.eyes is None:
            return self.m.geom_size[self.row_geoms[i]]
        b = self.book_seen(i)
        return np.array([b.thickness, b.depth, b.height]) / 2

    def room_behind(self, i):
        """How far row book i can go back before its back is BACK_GAP off the back panel."""
        return self.book_box(i)[:, 1].min() + self.D / 2 - BACK_GAP

    def tip_floor(self, skip, parts):
        """tip_limit with a clearance per part: smallest fingertip y for a
        hand whose parts are (x_lo, x_hi, how far behind the fingertips,
        clearance), each that far in front of the spine of every row book (but
        those in skip) and side panel it overlaps along the row. The hand
        stays in front of the spines, so a turned book's back corners, which
        reach further along the row, don't count."""
        boxes = ([self.book_box(i) for i in range(len(self.row_geoms)) if i not in skip and self.known(i)]
                 + [self.shelf.box(self.m, self.d, self.m.geom(name).id) for name in ("side_panel_pos_x", "side_panel_neg_x")])
        lim = -np.inf
        for c in boxes:
            (lo, hi), front = front_x(c), c[:, 1].max()
            for a, b, behind, clear in parts:
                if hi > a and lo < b:
                    lim = max(lim, front + clear - behind)
        return lim

    def face_x(self, i, side, y):
        """x of row book i's face on `side` (-1 its -x face, +1 its +x face) at
        depth y (or at its end nearer y), at the board. From the book's own
        axes: the two corners furthest along the row are the back face's once
        it has turned more than atan(thickness / depth). With eyes: from its
        box as seen (book_seen)."""
        if self.eyes is not None:
            from bila.vision.perception import book_box
            centre, R, (half_t, half_d, half_h) = book_box(self.book_seen(i))
        else:
            g = self.row_geoms[i]
            R = self.shelf.R.T @ self.d.geom_xmat[g].reshape(3, 3)
            half_t, half_d, half_h = self.m.geom_size[g]
            centre = self.shelf.to_shelf(self.d.geom_xpos[g])
        face = centre + side * math.copysign(half_t, R[0, 0]) * R[:, 0] - half_h * R[:, 2]
        s = np.clip((y - face[1]) / R[1, 1], -half_d, half_d)
        return face[0] + s * R[0, 1]

    def channel(self):
        """(width, x of its middle) of the widest straight way into the slot,
        along the shelf's depth, over the depth the book will take up: from
        the furthest in the left neighbour's face comes to the furthest in the
        right one's does. If a neighbour has turned it is narrower than the gap
        at the spines, and off its middle."""
        ti, top = self.row.target_index, self.seat_target()[0]
        ys = np.linspace(top - self.depth, top, 9)
        left, right = max(self.face_x(ti - 1, 1, y) for y in ys), min(self.face_x(ti, -1, y) for y in ys)
        return right - left, (left + right) / 2

    def narrowest_gap(self, pair=None):
        """Narrowest gap between the slot's neighbours over the depth the book
        will take up once in place (narrower than at the spines if a neighbour
        has turned). pair: row books (left, right) instead, over the depth both
        take up."""
        if pair is None:
            ti, top = self.row.target_index, self.seat_target()[0]
            a, b, ys = ti - 1, ti, np.linspace(top - self.depth, top, 9)
        else:
            (a, b), ca, cb = pair, self.book_box(pair[0]), self.book_box(pair[1])
            ys = np.linspace(max(ca[:, 1].min(), cb[:, 1].min()) + 0.005,
                             min(ca[:, 1].max(), cb[:, 1].max()) - 0.005, 9)
        return min(self.face_x(b, -1, y) - self.face_x(a, 1, y) for y in ys)

    def skip(self, reason) -> dict:
        """Leave the book out: a failure, with how far the other books moved
        (only a push that fell short moves them)."""
        self.log(f"  skipped: {reason}")
        others = [i for i in range(len(self.row_bodies)) if i not in self.intended]
        shifts = np.linalg.norm(self.d.xpos[self.row_bodies] - self.start_pos, axis=1)[others]
        return {"success": False, "safe_success": False, "skipped": reason, "neighbour_shift_m": float(shifts.max()),
                "total_shift_m": float(shifts.sum()), **self.info, **self.collisions()}

    def collisions(self) -> dict:
        """What the robot hit that it shouldn't have (Runner.watch_contacts), so far."""
        c = self.run.collision if self.run is not None else {"steps": 0, "max_force_n": 0.0, "impulse_ns": 0.0,
                                                             "parts": (), "during": ()}
        return {"collision_s": c["steps"] * self.m.opt.timestep, "collision_max_force_n": c["max_force_n"],
                "collision_impulse_ns": c["impulse_ns"], "collision_parts": sorted(c["parts"]),
                "collision_during": sorted(c["during"])}


# ---------------------------------------------------------------------------
# 4. Episodes and CLI
# ---------------------------------------------------------------------------
def make_episode(cfg: ShelfConfig, scenario: str, rng, panda_xml: str, max_tries: int = 10, stickers: bool = False):
    """Like shelf_scene.make_episode, on the model with the target book."""
    for _ in range(max_tries):
        row = sample_row(cfg, scenario, rng)
        model = build_model(cfg, row, panda_xml, stickers)
        data = mujoco.MjData(model)
        if settle(model, data, row)["stable"]:
            return model, data, row
    raise RuntimeError(f"No stable {scenario} row in {max_tries} tries")


def make_eyes(model, row, seed, noise="realistic", segmenter=None, fiducials=True):
    """The wrist camera's perception for --perception camera: with `noise`
    realistic, a D405's stereo errors and a calibration error drawn per
    episode; basic, only independent depth noise per pixel. A segmenter
    (e.g. a LearnedSegmenter) replaces the simulation's masks, with the
    robust spine fit learned masks need. With `fiducials`, every image's
    camera pose is corrected from the shelf's markers (vision/fiducials.py)."""
    from bila.vision.camera import CalibrationError, DepthNoise, WristCamera
    from bila.vision.rgbd_perception import RGBDPerception                       # it imports this module
    rng = np.random.default_rng(seed)
    cam = (WristCamera(model, noise=DepthNoise.d405(), rng=rng, calibration=CalibrationError())
           if noise == "realistic" else WristCamera(model, noise=DepthNoise(), rng=rng))
    reference = None
    if fiducials:
        from bila.control.inspection import slot_x
        from bila.vision.fiducials import FiducialReference
        from bila.sim.shelf_scene import fiducials as shelf_markers
        # ponytail: the map is rebuilt from the default ShelfConfig's board thickness, as every scene uses today
        reference = FiducialReference(shelf_markers(ShelfConfig(), row), ShelfFrame(row), slot_x(row))
    return RGBDPerception(model, row, cam, segmenter=segmenter, robust_fit=segmenter is not None, reference=reference)


def report_collisions(res: dict) -> str:
    if not res.get("collision_s"):
        return ""
    return (f"\n  COLLIDED for {res['collision_s']:.2f} s, up to {res['collision_max_force_n']:.0f} N: "
            f"{', '.join(res['collision_parts'][:4])} (during: {'; '.join(res['collision_during'][:3])})")


def report(res: dict) -> str:
    verdict = "SUCCESS" if res["success"] else "FAIL"
    how = f" [{res['method']}]" if res.get("method") else ""
    looked = f" | looked {res['looks']} times ({res['look_time_s']:.0f} s)" if res.get("looks") else ""
    if "skipped" in res:
        return (f"  SKIPPED{how}: {res['skipped']} | other books: most moved {1000 * res['neighbour_shift_m']:.1f} mm"
                f"{looked}" + report_collisions(res))
    place = (f"spine {1000 * res['offset_m']:+.1f} mm from the neighbour's" if res["rule"] == "flush" else
             f"deeper than its neighbours, spine {1000 * res['offset_m']:.1f} mm inside the front edge")
    return (f"  {verdict}{how}: tilt {res['tilt_deg']:.1f} deg, {place}"
            f"{' (hand blocked)' if res['blocked'] else ''}, {'released' if res['released'] else 'STILL TOUCHING'}\n"
            f"  other books: most moved {1000 * res['neighbour_shift_m']:.1f} mm (all together "
            f"{1000 * res['total_shift_m']:.0f} mm), most tilted {res['neighbour_tilt_change_deg']:.1f} deg"
            f" | let go after {res['time_to_release_s']:.1f} s, in place after {res['time_to_seat_s']:.1f} s{looked}"
            + report_collisions(res))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenario", choices=("clear", "tight", "leaning", "packed"), default="clear")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--trials", type=int, default=1, help="run seeds seed .. seed+trials-1 and summarise")
    ap.add_argument("--margin", type=float, default=GAP_MARGIN, help="room to make around the book, m")
    ap.add_argument("--perception", choices=("truth", "camera"), default="truth",
                    help="where the controller's knowledge of the row comes from: the simulation, or the wrist camera")
    ap.add_argument("--camera-noise", choices=("realistic", "basic"), default="realistic",
                    help="camera: a D405's stereo errors and a calibration error (camera.py: DepthNoise.d405, "
                         "CalibrationError), or only independent depth noise per pixel")
    ap.add_argument("--panda", default=DEFAULT_PANDA, help="Menagerie panda.xml")
    ap.add_argument("--book-meshes", action="store_true",
                    help="draw the converted repo book covers (run dae_to_obj.py first)")
    ap.add_argument("--book-pages", type=int, default=0, metavar="N",
                    help="split every book's pages into N sections hinged on the spine (0 = one rigid box)")
    ap.add_argument("--catalogue", action="store_true",
                    help="fill the rows with the repo's real books (size, mass, cover, pages) instead of sampled ones")
    ap.add_argument("--view", action="store_true", help="watch in the MuJoCo viewer (run with mjpython on macOS)")
    a = ap.parse_args()

    results = []
    cfg = ShelfConfig(book_meshes=a.book_meshes, book_pages=a.book_pages, catalogue=a.catalogue)
    for seed in range(a.seed, a.seed + a.trials):
        model, data, row = make_episode(cfg, a.scenario, np.random.default_rng(seed), a.panda)
        b = row.target_book
        print(f"seed {seed}: target level {row.target_level} (board top {row.target_board_top:.2f} m), "
              f"gap {1000 * (row.gap_right - row.gap_left):.0f} mm for a {1000 * b.thickness:.0f} mm book "
              f"({1000 * b.depth:.0f} mm deep, {b.mass:.2f} kg)")
        log = print if a.trials == 1 else (lambda *_: None)
        eyes = make_eyes(model, row, seed, a.camera_noise) if a.perception == "camera" else None
        if a.view:
            import mujoco.viewer
            with mujoco.viewer.launch_passive(model, data) as viewer:
                # The viewer only aims its camera with the scene's defaults (at
                # the gap) when it opens a file, so do it here.
                with viewer.lock():
                    mujoco.mjv_defaultFreeCamera(model, viewer.cam)
                res = Episode(model, data, row, viewer, log, eyes).reshelve(a.margin)
                print(report(res))
                print("Close the viewer window to exit.")
                while viewer.is_running():
                    mujoco.mj_step(model, data)
                    viewer.sync()
                    time.sleep(model.opt.timestep)
        else:
            ep = Episode(model, data, row, None, log, eyes)
            try:
                res = ep.reshelve(a.margin)
                print(report(res))
            except RuntimeError as e:                 # no safe way to do it in this scene, or it jammed
                res = {"success": False, "safe_success": False, "error": str(e), **ep.info, **ep.collisions()}
                print(f"  FAIL{' [' + ep.info['method'] + ']' if 'method' in ep.info else ''}: {e}")
        if eyes is not None:
            eyes.close()
        results.append(res)

    if a.trials > 1:
        n = len(results)
        ok = sum(r["success"] for r in results)
        hit = [r for r in results if r.get("collision_s")]
        print(f"\n{sum(r.get('safe_success', False) for r in results)}/{n} succeeded without touching the shelf or cart; "
              f"{len(hit)}/{n} touched it ({sum(r['success'] for r in hit)} of those still placed the book)")
        print(f"{ok}/{n} succeeded. Of the rest: "
              f"{sum(r.get('blocked', False) and not r['success'] for r in results)} with the hand blocked short, "
              f"{sum(not r.get('upright', True) for r in results)} not upright, "
              f"{sum(not r.get('released', True) for r in results)} not released")
        for err in sorted({r["error"] for r in results if "error" in r}):
            print(f"  {sum(r.get('error') == err for r in results)} stopped: {err}")
        for why in sorted({r["skipped"] for r in results if "skipped" in r}):
            print(f"  {sum(r.get('skipped') == why for r in results)} skipped: {why}")
        for method in sorted({r.get("method") for r in results if r.get("method")}):
            rs = [r for r in results if r.get("method") == method]
            done = [r for r in rs if r["success"]]
            line = f"  {method}: {len(done)}/{len(rs)} succeeded"
            if done:
                line += (f"; when it did, most-moved book {1000 * np.mean([r['neighbour_shift_m'] for r in done]):.1f} mm"
                         f" on average (worst {1000 * max(r['neighbour_shift_m'] for r in done):.1f}),"
                         f" in place after {np.mean([r['time_to_seat_s'] for r in done]):.1f} s")
            print(line)


if __name__ == "__main__":
    main()

