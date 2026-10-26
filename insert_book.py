"""
Scripted insertion of the target book, for the clear, tight, leaning and packed scenarios.

    uv run python insert_book.py --scenario clear --seed 0              # one episode, headless
    uv run python insert_book.py --scenario tight --seed 0 --trials 50  # success rate over seeds 0..49
    uv run mjpython insert_book.py --scenario tight --seed 0 --view     # watch it (macOS needs mjpython)
    uv run mjpython insert_book.py --scenario leaning --seed 12 --view  # a leaner it has to clamp
    uv run mjpython insert_book.py --scenario packed --seed 7 --view    # a packed row it manages
    uv run python insert_book.py --scenario packed --seed 0 --trials 50 --packed stagger   # the older packed baseline

Insertion (shelf frame: x along the row, y out of the shelf, z up)
  1. Hold the book at the pre-insertion pose: lined up with the gap, its back
     1 cm over the board's front edge and 3 mm above it. The fingers pinch its
     thickness 11.5 cm above its bottom, with its spine 2 mm off the palm, so
     the whole pad strip grips it.
  2. Rest its back on the board's edge, then slide it in along the board while
     holding it, until the hand would come within 5 mm of another book's spine.
     Each finger reaches 1.2 cm out from its pad, more than the clearance
     beside the book, so the fingers can't follow it in.
  3. Open the gripper.
  4. Back off, turn the hand 90 deg so the fingers are stacked vertically, and
     push the spine low down with the lower fingertip until the book is in
     place (skipped if the slide already got it there). A book going in
     tightly drags its neighbours back, so where it has to go is measured
     again after the slide, and after each push. In place means:
       - flush with the spine of the neighbour that sticks out more, within
         ALIGN_TOL, if the book can get there without its back reaching the
         back panel;
       - otherwise (it is deeper than that neighbour), at least EDGE_IN behind
         the shelf's front edge.
  5. Back out and score it: upright within 5 deg, in place, and released.

Clear scenario: the book starts already grasped at the pre-insertion pose.

Tight scenario: the gap is narrower than the book, so it is widened first, to
the book's thickness plus GAP_MARGIN.
  - Fingertip (when the fingers fit the gap and reach a neighbour with room
    behind it): the empty hand puts its vertically stacked fingers into the
    gap and pushes the neighbours apart as hard as the arm can, re-pushing by
    whatever the arm's give left short. By default it pushes one side
    (--strategy one-sided), the one with more free space, as far as the
    widening needs or that side's space allows, and pushes the other side too
    only if the gap is still short: on front-aligned rows one side placed
    more books than splitting and was faster, for about the same
    displacement. --strategy split pushes each neighbour
    half as far (fewer books must move at once) but swivels both, so the book
    catches on both sides more often. Then the book is handed over at the
    pre-insertion pose (not modelled) and inserted.
    The fingers only reach ~4 cm past the spines, so the neighbours swivel
    rather than slide: the gap widens at the spines but hardly at the back,
    and the book itself has to widen the rest as it slides in.
  - Skip (otherwise, or if the push leaves the gap at the spines narrower
    than the book): no insertion, reported as a failure. The thin fingers
    (10 mm, see shelf_scene) fit every tight gap the generator makes (12 mm
    and up); Menagerie's 21 mm fingers didn't fit 1 in 5. Nothing else on the
    hand can widen a gap narrower than the fingers: pushed sideways at its
    back corner, the held book slips in the grasp at 10-15 N, short of what
    sliding the books takes. A book wedge (back corner first) pushed the
    neighbours back into the shelf instead of aside, and placed none.

Leaning scenario: a neighbour leans into the slot (more than LEAN_MIN_DEG), so
it is stood back up first, then the row is handled as a tight one.
  - Lift (stand_up): the closed fingers, stacked so they take 10 mm along the
    row, go under the leaning book as far up its face as they fit, and lift it
    along an arc about the bottom corner it pivots on, pressing square on its
    face so it turns rather than slides. They stop within its tipping angle
    (atan(thickness / height)) of upright, still leaning slightly into the
    slot, and it stands up the rest of the way by itself. Up to STAND_TRIES
    lifts, each closer to upright.
  - Clamp (clamp_up, when the fingers can't get under it at any height): the
    fingers pinch it across its thickness near the spine, with the hand
    rolled to its lean, and swing it about its bottom edge to just short of
    upright. Only one finger has to fit between it and the far book. The
    other goes between it and the book on its own side, which is pushed
    along a little.
  - A book going into a gap more than twice its thickness could lean over
    once let go, so it goes SIDE_GAP off the taller upright neighbour instead
    of in the middle.
  On seeds 0..49 every leaner stands up (3 by clamping) and all 50 books go in.

Packed scenario: the neighbours touch, so room has to be made first.
  - Shove (--packed shove, the default; shove_then_insert): the fingers,
    stacked, push the books either side of the neighbour on the side with
    more free space straight back into the shelf, so that neighbour sticks
    out of the row. The fingers pinch it across its thickness, which keeps it
    from swivelling, and shove it, and every book beyond it, into the free
    space until the gap is the book's thickness plus GAP_MARGIN where the
    book will go, with the arm configuration chosen for the force (the whole
    run slides). If the gap is still short, they pinch it again, the hand
    level or pointing down into the shelf, and if it still won't give, the
    far half of the run is shoved first. It is let go, pushed back in line,
    and the book inserted as in a clear row, along the straight way in. If
    the neighbour turned instead (its back stayed, so the gap is shut at the
    back), the other neighbour is pushed aside as in a tight row and the book
    itself is wedged in along the straighter one, its back corner pushing
    the turned one's face aside deep in the shelf (wedge_slot).
  - Stagger (--packed stagger; stagger_open): push the neighbour on the side
    with less room 2 cm straight back into the shelf, so the side of the
    other one shows in front of it, then push that one aside from there until
    the gap is a few mm wider than the fingers. Then as a tight row. Pushed
    at its front corner, the book swivels and comes partly out of the shelf.

The slide and the final push stop if the wrist force sensor feels the book
stop giving way, rather than forcing a jammed book.

Control: Menagerie's position actuators, fed joint targets by differential IK
along a min-jerk Cartesian path, with gravity compensation on the arm. The
gripper is made as strong as the real Franka Hand (70 N continuous): Menagerie's
gripper squeezes a book with only 1-2 N, too little to hold one. The fingers
are shelf_scene's thin ones, not Menagerie's.
"""

import argparse
import math
import os
import time
import xml.etree.ElementTree as ET

import numpy as np
import mujoco

from shelf_scene import (FINGER_HALF_W, FINGER_LEN, FINGER_THICK, PANDA_HOME, ShelfConfig, _f, add_book_geoms,
                         add_book_visual_assets, add_page_visual_assets, book_children, book_tilt_deg,
                         build_scene, sample_row, settle)

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_PANDA = os.path.join(HERE, "mujoco_menagerie", "franka_emika_panda", "panda.xml")

# Panda hand geometry, with the thin fingers shelf_scene gives it (Menagerie's
# hand, measured from its collision mesh). Hand frame: z = approach (along the
# fingers), y = finger travel.
FINGERTIP_Z = 0.0584 + FINGER_LEN   # end of the fingers (Menagerie mounts them 58.4 mm out)
PAD_Z = 0.1026              # where the pads hold the book (where Menagerie's pad centre is)
PALM_Z = 0.066              # face of the hand between the fingers
FINGER_OPEN_SPAN = 0.04 + FINGER_THICK       # outer face of a fully open finger, from the hand axis
FINGER_OPEN_MID = 0.04 + FINGER_THICK / 2    # centre of a fully open finger, from the hand axis
HAND_HALF_T = 0.0316        # hand half-thickness, across finger travel
HAND_HALF_L = 0.105         # hand half-length, along finger travel

# Grasp and sequence settings
GRIP_FORCE = 70.0           # N on each side of the book (the real Franka Hand's continuous force)
PALM_GAP = 0.002            # the book's spine sits this far off the palm, deep in the grasp
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

# Gap creation (tight scenario)
GAP_MARGIN = 0.006          # widen the gap to the book's thickness plus this
FIT_CLEAR = 0.001           # fingers need this much room each side to enter the gap
MIN_REACH = 0.010           # fingertips must get this far behind a neighbour's spine to push it
ROOM_KEEP = 0.002           # leave this much of a side's free space unused
GAP_PUSH_SPEED = 0.03

# Gap creation (packed scenario: no gap at all)
STAGGER = 0.020             # push the other neighbour back until the face to push shows this deep in front of it
STAGGER_OPEN = 0.004        # open the gap this much wider than the fingers, then widen as in a tight row
EASE_OFF = 0.004            # back the fingertips off a pushed face this far before taking them out
SHOVE_BACK = 0.035          # shove: push the books beside the neighbour back until their spines are this far behind its spine
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
SEGMENT_MIN = 5             # if a shove stalls, shove a run of this many books or more again in two halves
CHANNEL_CLEAR = 0.003       # insert straight after the shove if the way in is this much wider than the book
WALK_CLEAR = 0.0015         # walking a neighbour out: fingertips this far in front of the books beside it,
WALK_MIN = 0.002            # ... pinching at least this much of it,
WALK_STEP = 0.010           # ... pulling it this far out each time,
WALK_TARGET = 0.035         # ... until it is this far out
WALK_TRIES = 6
WEDGE_CLEAR = 0.002         # ... else wedge it in, if the gap at the spines is this much wider than it:
WEDGE_HUG = 0.001           # its face this far off the straighter neighbour's,
WEDGE_RUN_UP = 0.015        # slid in as usual until its back is this far from the turned one's face,
WEDGE_SPEED = 0.02          # then slowly,
WEDGE_STOP_FORCE = 60.0     # stopping only above this (more, and it slips back in the fingers),
WEDGE_PUSH_FORCE = 100.0    # and once let go, pushed home until the wrist feels this
CLOSE_SPEED = 0.05          # m/s per finger when closing on a book from a few mm away
STRONG_MARGIN = 0.02        # rad: joint margin the shove's arm configuration must keep
STRONG_SEEDS = 60           # random starts per round when choosing it

# Leaning scenario
LEAN_MIN_DEG = 5.0          # a neighbour leaning further than this into the slot gets stood back up
STAND_TRIES = 3             # lifts before giving up on standing it up
STAND_TO = (0.5, 0.2, -0.2)  # lift it to this fraction of its tipping angle short of upright, per try
LIFT_SPEED = 0.06           # m/s, peak, of the fingers lifting it
SIDE_GAP = 0.003            # a book going into a wide gap stands this far off the neighbour it leans on

# Scoring
ALIGN_TOL = 0.010           # in place: spine within this of the more protruding neighbour's spine, or
EDGE_IN = 0.010             # ... if the book is deeper than that neighbour, at least this behind the front edge
EDGE_AIM = 0.005            # push a deeper book this much further in than EDGE_IN needs
UPRIGHT_DEG = 5.0           # upright within 5 deg (project proposal)


# ---------------------------------------------------------------------------
# 1. Scene: the generated bookcase plus the target book
# ---------------------------------------------------------------------------
def build_model(cfg: ShelfConfig, row, panda_xml: str) -> mujoco.MjModel:
    """The scene from shelf_scene.build_scene, plus the target book parked on
    the floor beside the bookcase (out of the way while the row settles), a
    stronger gripper, and gravity compensation on the arm."""
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

    # Gravity compensation has to be set before compiling: MuJoCo skips it
    # when the compiled model has no compensated bodies.
    spec = mujoco.MjSpec.from_string(ET.tostring(root, encoding="unicode"))
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
        self.book_of = {}                   # page-section geom -> its book's geom: see book_children

    def sync(self, data, finger=None):
        """Copy the scene (books, fingers) from the simulation, for collision
        checks; with `finger`, plan with each finger that far open instead."""
        self.d.qpos[:] = data.qpos
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


class Runner:
    """Steps the simulation while moving the commanded hand pose along
    min-jerk paths. Each step re-solves IK from the last command."""

    def __init__(self, model, data, ik: ArmIK, q_cmd, grip, viewer=None, log=print):
        self.m, self.d, self.ik = model, data, ik
        self.q_cmd, self.grip = np.array(q_cmd), grip
        self.pos, self.rot = ik.fk(self.q_cmd)
        self.viewer, self.log = viewer, log
        self.t0 = data.time
        self.wall0, self.next_frame = time.perf_counter(), 0.0
        self.ft = int(model.sensor("panda/ft_force").adr[0])       # wrist force sensor (hand frame)

    def wrist_force(self):
        return self.d.sensordata[self.ft:self.ft + 3].copy()

    def halt(self):
        """Hold the hand where it actually is, so the arm stops pressing on
        whatever it ran into."""
        self.q_cmd = self.d.qpos[self.ik.qadr].copy()
        self.pos, self.rot = self.ik.fk(self.q_cmd)

    @property
    def t(self):
        return self.d.time - self.t0

    def step(self):
        self.d.ctrl[:7] = self.q_cmd
        self.d.ctrl[7] = self.grip
        mujoco.mj_step(self.m, self.d)
        if self.viewer is not None and self.t >= self.next_frame:     # draw at 60 fps, in real time
            self.next_frame += 1 / 60
            if not self.viewer.is_running():
                raise SystemExit
            self.viewer.sync()
            lag = self.t - (time.perf_counter() - self.wall0)
            if lag > 0:
                time.sleep(lag)

    def move(self, name, pos, rot=None, speed=TRANSIT_SPEED, grip=None, stop_force=None):
        """Move the hand frame to (pos, rot) in a straight line, with a slerp
        on its orientation. Peak speeds are `speed` and TURN_SPEED. With
        `stop_force`, stop and hold as soon as the wrist force along the hand's
        approach axis changes by more than that (N): what it pushes has stopped
        giving way. (The other components change as the board takes over the
        weight of a held book.) The force is smoothed over ~50 ms first, as a
        book chattering on the board's edge gives short spikes of 20-30 N.
        Returns False then."""
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
            self.q_cmd = self.ik.solve(self.q_cmd, self.pos, self.rot)
            self.step()
            smooth += blend * (self.wrist_force()[2] - smooth)
            if stop_force is not None and abs(smooth - f0) > stop_force:
                self.halt()
                self.log(f"  {self.t:5.2f} s    stopped: the wrist feels over {stop_force:.0f} N")
                return False
        return True

    def follow(self, name, poses, speed=TRANSIT_SPEED):
        """Move the hand through a list of (pos, rot) poses as one min-jerk
        move along the whole path (straight lines and slerps between them)."""
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
            self.q_cmd = self.ik.solve(self.q_cmd, self.pos, self.rot)
            self.step()

    def move_joints(self, name, q_goal, speed=JOINT_SPEED):
        """Move the arm in joint space to q_goal along a min-jerk path (peak
        joint speed `speed`, rad/s). Only for moves clear of the shelf: nothing
        here checks what the hand passes."""
        self.log(f"  {self.t:5.2f} s  {name}")
        q0, q_goal = self.q_cmd.copy(), np.asarray(q_goal, dtype=float)
        T = max(0.3, 1.875 * float(np.max(np.abs(q_goal - q0))) / speed)
        n = int(round(T / self.m.opt.timestep))
        for k in range(1, n + 1):
            u = k / n
            self.q_cmd = q0 + (q_goal - q0) * (u ** 3 * (10 - 15 * u + 6 * u * u))
            self.step()
        self.pos, self.rot = self.ik.fk(self.q_cmd)

    def wait(self, name, duration, grip=None):
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


class Episode:
    """One insertion attempt in a settled scene: geometry, hand poses, the
    steps of each sequence, and the score."""

    def __init__(self, model, data, row, viewer=None, log=print):
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
        self.info = {}
        self.intended = set()      # row books the sequence sets out to move (a leaner it stands up): not disturbance
        self.lean_on_neighbour = False   # put a book that could tip over in a wide gap against a neighbour
        self._perception = None

    # --- Geometry, measured from the simulation --------------------------------
    def obstacles(self):
        return obstacles(self.m, self.d, self.shelf, self.row)

    def slot(self):
        """The slot as it is now: x of the left and right neighbours' faces at
        their spines, and y of their spines (shelf frame)."""
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
        seem to take up more of the row than it does."""
        W, ti, books = self.row.shelf_width, self.row.target_index, self.row.books
        xl, xr, _, _ = self.slot()
        return (xl + W / 2 - sum(b.thickness for b in books[:ti]),
                W / 2 - xr - sum(b.thickness for b in books[ti:]))

    def estimate(self):
        """What a camera would report about the row now (ground truth)."""
        from perception import GroundTruthPerception           # perception imports this module
        if self._perception is None:
            self._perception = GroundTruthPerception(self.m, self.row)
        return self._perception.estimate(self.d)

    def gap_at(self, z, est=None):
        """(x_left, x_right) of the slot at height z above the board, at the
        spine plane: the right-most edge of anything left of it and the
        left-most edge of anything right of it (side panels included)."""
        from perception import slice_x
        est = est or self.estimate()
        zz = self.z0 + z
        left = [s[1] for b in est.books if b.slot_offset < 0 and (s := slice_x(b.spine_corners, zz))]
        right = [s[0] for b in est.books if b.slot_offset > 0 and (s := slice_x(b.spine_corners, zz))]
        return max(left + [-self.row.shelf_width / 2]), min(right + [self.row.shelf_width / 2])

    def seat_target(self):
        """Where the book's spine should end up (y), and which rule applies:
        'flush' with the spine of the neighbour that sticks out more, if the
        book gets there before its back reaches the back panel; otherwise
        'inside' the shelf's front edge by at least EDGE_IN (aiming EDGE_AIM
        further in, but never deeper than the back panel allows)."""
        _, _, fl, fr = self.slot()
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

    def choose_arm(self, candidates, allowed=frozenset(), finger=None):
        """Many arm configurations reach a start pose. Near a joint limit the IK
        swings the other joints round, and the arm can jump to another
        configuration mid-move and knock the row. So follow each candidate's
        whole planned path, drop those where the arm runs into anything (other
        than `allowed` geoms its fingers may press on), and keep the one that
        stays farthest from the joint limits. candidates: (start pose, path, label).
        `finger`: plan with the fingers this far open (default: as they are)."""
        ik, seed_rng, best = self.ik, np.random.default_rng(0), None
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
                    if ok and (best is None or worst > best[0]):
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
        self.run = Runner(m, d, self.ik, q, GRIP_OPEN, self.viewer, self.log)

    def begin(self, candidates, allowed=frozenset(), finger=None):
        """Get the hand to the start of a step, ready to follow one of the
        candidates (start pose, path, label), and return its label. At the
        start of the episode the arm is placed at the best configuration. Later
        the hand carries on from where it is, if some candidate's start and
        path can be reached from there with joint limits to spare and without
        hitting anything; otherwise the arm moves for real to the best
        configuration (reposition). Raises RuntimeError if none is possible."""
        run = self.run
        if run is not None:
            self.ik.sync(self.d, finger)
            self.ik.q_rest = run.q_cmd.copy()                       # carrying on keeps the arm's posture: plan with it
            for start, path, label in candidates:
                lead = segment((run.pos, run.rot), start, 30)
                _, worst, ok = self.ik.follow(run.q_cmd, lead + list(path), allowed=allowed)
                if ok and worst >= 2 * MIN_MARGIN:                  # a little spare: the real path drifts
                    run.move("move to the start of the next step", *start)
                    return label
        q, label = self.choose_arm(candidates, allowed, finger)
        if run is None:
            self.place_arm(q)
        else:
            self.reposition(q)                                       # q puts the hand at that start pose
        return label

    def reposition(self, q):
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
        home = np.array(PANDA_HOME[:7])

        def clear(q0, q1):
            return not any(ik.collides(q0 + s * (q1 - q0)) for s in np.linspace(0, 1, 25)[1:])

        if clear(run.q_cmd, q):
            run.move_joints("move the arm to its next configuration", q)
        elif clear(run.q_cmd, home) and clear(home, q):
            run.move_joints("tuck the arm", home)
            run.move_joints("move the arm to its next configuration", q)
        else:
            # Neither: by way of some other configuration, sampled around the two ends.
            rng = np.random.default_rng(0)
            for _ in range(WAYPOINT_TRIES):
                w = np.clip((run.q_cmd + q) / 2 + rng.normal(0, 0.6, 7), ik.lo, ik.hi)
                if not ik.collides(w) and clear(run.q_cmd, w) and clear(w, q):
                    run.move_joints("move the arm out of the way", w)
                    run.move_joints("move the arm to its next configuration", q)
                    break
            else:
                raise RuntimeError("No collision-free joint move to the next arm configuration")
        ik.q_rest = np.array(q, dtype=float)

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
            c = self.shelf.box(self.m, self.d, self.book_geom)
            if c[:, 1].mean() > self.D / 2 - CLEAR:                 # centre not over the board: can't let go
                raise RuntimeError("the book jammed at the slot's entrance")
        run.wait("open the gripper", 0.4, grip=GRIP_OPEN)
        t_release = run.t

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
                break
        else:
            raise RuntimeError("No way to turn the hand for the push without nearing a joint limit")
        run.move("back off", *back_off)
        run.move("turn the hand to push", *ready)
        run.move("reach the spine", *self.push_pose(sign, x_book, spine + 0.005))
        push_force = PUSH_STOP_FORCE if wedge_from is None else WEDGE_PUSH_FORCE
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

    def score(self, t_start, t_release, t_done, blocked, **info) -> dict:
        m, d = self.m, self.d
        c = self.shelf.box(m, d, self.book_geom)
        tilt = book_tilt_deg(d.xquat[self.book_body])
        back_gap = c[:, 1].min() + self.D / 2
        spine = c[:, 1].max()                                        # frontmost point of the book
        target, rule = self.seat_target()                            # from where the neighbours ended up
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
            "time_to_release_s": t_release - t_start, "time_to_seat_s": t_done - t_start,
            **self.info, **info,
        }
        res["success"] = res["upright"] and res["in_place"] and res["released"]
        return res

    # --- Clear scenario: the book starts grasped -------------------------------------
    def insert(self, method="none needed", x_slot=None, wedge_from=None):
        xl, xr, _, _ = self.slot()
        if x_slot is None:
            x_slot = (xl + xr) / 2
            if self.lean_on_neighbour and xr - xl > 2 * self.t:
                # Room for it to lean past its tipping angle, centred: stand it
                # against the taller upright neighbour instead.
                est = self.estimate()
                up = [s for s in (-1, 1) if est.book(s) and abs(est.book(s).lean) < math.radians(3)]
                if up:
                    s = max(up, key=lambda s: est.book(s).height)
                    x_slot = xl + self.t / 2 + SIDE_GAP if s < 0 else xr - self.t / 2 - SIDE_GAP
        pre = self.book_pre(x_slot)
        obs = self.obstacles()
        candidates = [(self.grasp_pose(gs, pre), self.insertion_plan(gs, ps, x_slot, pre[1], obs), (gs, ps))
                      for gs in (1, -1) for ps in (1, -1)]
        first = self.run is None
        gs, ps = self.begin(candidates)
        self.take_book(gs, pre, name="close the gripper on the book at the pre-insertion pose" if first else
                       "take the book (the hand-over is not modelled)")
        t_start = self.run.t if first else 0.0                       # else: from the start of the episode
        return self.score(t_start, *self.seat(x_slot, pre[1], ps, wedge_from), method=method)

    # --- Tight scenario: widen the gap, then insert ----------------------------------
    def insert_tight(self, strategy="one-sided", margin=GAP_MARGIN, wedge=False):
        """Widen the gap with the fingertips, then insert the book. wedge (after
        a shove whose neighbour turned): wedge the book in once widened, or at
        once if the fingertips can't widen it (wedge_slot)."""
        xl, xr, fl, fr = self.slot()
        need = self.t + margin - (xr - xl)
        self.info["widening_m"] = max(0.0, need)
        if need <= 0:
            self.log(f"  the {1000 * (xr - xl):.0f} mm gap already fits the book")
            return self.insert()
        plan = self.plan_fingertip(xl, xr, fl, fr, need, strategy)
        if plan is None:
            if wedge and xr - xl >= self.t + WEDGE_CLEAR:
                return self.insert("wedge", *self.wedge_slot())
            return self.skip(self.info["fingertip"])
        self.info["method"] = "fingertip, both sides" if plan[0] > 0 and plan[1] > 0 else "fingertip, one side"
        return self.fingertip_then_insert(xl, xr, *plan, gap_goal=self.t + margin, wedge=wedge)

    def plan_fingertip(self, xl, xr, fl, fr, need, strategy):
        """How far to push each neighbour with the fingertips: (left, right,
        fingertip depth), or None if they don't fit the gap, or can't reach a
        neighbour that has room behind it.

        'one-sided' pushes one side (more free space first), as far as the
        widening needs or its free space allows. If that leaves the gap short,
        it pushes the other side too, by what is missing (top_up). One side
        saves the trip across the gap and the second push, and leaves one
        neighbour's face straight for the book to slide along. 'split' tries
        splitting the widening evenly first (each neighbour moves half, as far
        as each side's free space allows). Each falls back on the other."""
        if xr - xl < 2 * (FINGER_HALF_W + FIT_CLEAR):
            self.info["fingertip"] = "gap narrower than the fingers"
            return None
        room_l, room_r = (r - ROOM_KEEP for r in self.free_space())
        obs = self.obstacles()
        x_mid = (xl + xr) / 2
        dl_even = min(max(need / 2, need - room_r), room_l)
        split = (dl_even, need - dl_even)
        sides = sorted((-1, 1), key=lambda s: -(room_l if s < 0 else room_r))  # more free space first
        one_sided = [(min(need, room_l), 0.0) if s < 0 else (0.0, min(need, room_r)) for s in sides]
        for dl, dr in ([split] + one_sided) if strategy == "split" else (one_sided + [split]):
            if min(dl, dr) < 0 or max(dl, dr) <= 0 or dl > room_l + 1e-9 or dr > room_r + 1e-9:
                continue
            tip_in = self.fingertip_depth([xl - dl + FINGER_HALF_W, x_mid, xr + dr - FINGER_HALF_W], obs)
            if (dl > 0 and fl - tip_in < MIN_REACH) or (dr > 0 and fr - tip_in < MIN_REACH):
                continue
            return dl, dr, tip_in
        self.info["fingertip"] = "fingers can't reach a neighbour with room behind it"
        return None

    def fingertip_depth(self, xs, obs):
        """Deepest the fingertips may go (smallest y) while the hand sweeps
        sideways over the hand positions xs: the palm stays clear of the books."""
        lo, hi = min(xs) - HAND_HALF_T, max(xs) + HAND_HALF_T
        return tip_limit(obs, (lo + hi) / 2, [((hi - lo) / 2, FINGERTIP_Z - PALM_Z)])

    def fingertip_then_insert(self, xl, xr, dl, dr, tip_in, gap_goal, wedge=False):
        obs = self.obstacles()
        x_mid = (xl + xr) / 2
        x_left = xl - dl + FINGER_HALF_W if dl > 0 else x_mid       # hand x when each push ends
        x_right = xr + dr - FINGER_HALF_W if dr > 0 else x_mid
        x_slot = (xl - dl + xr + dr) / 2                             # slot centre once widened
        lo, hi = min(x_left, x_mid) - HAND_HALF_T - CLEAR, max(x_right, x_mid) + HAND_HALF_T + CLEAR
        tip_out = max(front for a, b, front in obs if b > lo and a < hi) + 0.02
        pre = self.book_pre(x_slot)
        candidates = []
        for ps in (1, -1):
            gap = ([self.push_pose(ps, x_mid, y) for y in np.linspace(tip_out, tip_in, 15)[1:]]
                   + [self.push_pose(ps, x, tip_in) for x in np.linspace(x_mid, x_left, 10)[1:]]
                   + [self.push_pose(ps, x, tip_in) for x in np.linspace(x_left, x_right, 15)[1:]]
                   + [self.push_pose(ps, x, tip_in) for x in np.linspace(x_right, x_slot, 10)[1:]]
                   + [self.push_pose(ps, x_slot, y) for y in np.linspace(tip_in, tip_out, 15)[1:]])
            for gs in (1, -1):
                path = gap + segment(gap[-1], self.grasp_pose(gs, pre), 40) + \
                    self.insertion_plan(gs, ps, x_slot, pre[1], obs)
                candidates.append((self.push_pose(ps, x_mid, tip_out), path, (ps, gs)))
        ti = self.row.target_index
        first = self.run is None
        # The fingers may touch the neighbours, and the books beyond one they push: those go ahead of
        # them, but not in the planner's static scene, where a push longer than the neighbour is thick
        # would run them into the next book.
        allowed = ({self.row_geoms[ti - 1], self.row_geoms[ti]} | (set(self.row_geoms[:ti]) if dl > 0 else set())
                   | (set(self.row_geoms[ti:]) if dr > 0 else set()))
        try:
            ps, gs = self.begin(candidates, allowed=allowed, finger=0.04)
        except RuntimeError:
            if not wedge or xr - xl < self.t + WEDGE_CLEAR:
                raise
            return self.insert("wedge", *self.wedge_slot())          # (nothing has moved yet)
        run = self.run
        t_start = run.t if first else 0.0                            # else: from the start of the episode
        if run.grip != GRIP_OPEN:
            run.wait("open the gripper", 0.3, grip=GRIP_OPEN)
        run.move("put the fingers into the gap", *self.push_pose(ps, x_mid, tip_in))
        room_l, room_r = self.free_space()
        if dl > 0:
            self.push_neighbour(ps, -1, xl - dl, xl - room_l + ROOM_KEEP, tip_in)
        if dr > 0:
            self.push_neighbour(ps, +1, xr + dr, xr + room_r - ROOM_KEEP, tip_in)
        if (dl > 0) != (dr > 0):                                     # one side pushed: push the other if still short
            side = 1 if dl > 0 else -1
            more = self.top_up(ps, gs, side, tip_in, gap_goal,
                               xr + room_r - ROOM_KEEP if side > 0 else xl - room_l + ROOM_KEEP)
            if more is not None:
                self.info["method"] = "fingertip, one side, then the other"
                x_slot, tip_in, tip_out, d = more
                dl, dr = (dl, d) if side > 0 else (d, dr)
        run.move("back to the middle", *self.push_pose(ps, x_slot, tip_in))
        run.move("take the fingers out", *self.push_pose(ps, x_slot, tip_out))
        gap_time = run.t - t_start
        xl, xr, _, _ = self.slot()
        self.log(f"         the gap is now {1000 * (xr - xl):.1f} mm at the spines, for a {1000 * self.t:.1f} mm book")
        if xr - xl < self.t:                                         # the push fell short: it can't go in straight
            return self.skip("the push left the gap narrower than the book")
        x_slot, wedge_from = (xl + xr) / 2, None
        if wedge and self.channel()[0] < self.t + CHANNEL_CLEAR and xr - xl >= self.t + WEDGE_CLEAR:
            x_slot, wedge_from = self.wedge_slot()
            self.info["method"] += ", then wedge"
        pre = self.book_pre(x_slot)
        run.move("turn the hand to the pre-insertion pose", *self.grasp_pose(gs, pre))
        self.take_book(gs, pre, name="take the book (the hand-over is not modelled)")
        return self.score(t_start, *self.seat(x_slot, pre[1], ps, wedge_from), gap_time_s=gap_time,
                          push_left_m=dl, push_right_m=dr)

    def top_up(self, ps, gs, side, tip_in, gap_goal, limit):
        """After pushing one side only: if the gap is still short of gap_goal,
        push the other neighbour (side -1 left, +1 right) by what is missing,
        never commanding past x = limit. Planned from where the arm is: only
        if the fingertips reach that neighbour and the arm can carry on
        through the insertion. Returns (slot centre, fingertip depth, depth to
        take the fingers out at, distance pushed), or None if it doesn't push."""
        run, ti = self.run, self.row.target_index
        gl, gr, sl, sr = self.slot()
        short = gap_goal - (gr - gl)
        face, spine = (gr, sr) if side > 0 else (gl, sl)
        d = min(short, side * (limit - face))
        if short < 0.001 or d < 0.001:
            return None
        name = "right" if side > 0 else "left"
        x_from = self.shelf.to_shelf(run.pos + run.rot @ [0, 0, FINGERTIP_Z])[0]   # where the first push left the hand
        x_to = face + side * (d - FINGER_HALF_W)
        x_slot = (gl + gr + side * d) / 2
        obs = self.obstacles()
        tip = max(tip_in, self.fingertip_depth([x_from, x_to], obs))   # the palm must clear the books it now crosses
        if spine - tip < MIN_REACH:
            self.log(f"         the gap is {1000 * short:.1f} mm short, but the fingertips can't reach the {name} neighbour")
            return None
        lo, hi = min(x_from, x_to) - HAND_HALF_T - CLEAR, max(x_from, x_to) + HAND_HALF_T + CLEAR
        tip_out = max(front for a, b, front in obs if b > lo and a < hi) + 0.02
        back = (run.pos + self.ys * (tip - self.tip_y((run.pos, run.rot))), run.rot)
        pre = self.book_pre(x_slot)
        path = (segment((run.pos, run.rot), back, 5)
                + [self.push_pose(ps, x, tip) for x in np.linspace(x_from, x_to, 15)[1:]]
                + [self.push_pose(ps, x, tip) for x in np.linspace(x_to, x_slot, 10)[1:]]
                + [self.push_pose(ps, x_slot, y) for y in np.linspace(tip, tip_out, 15)[1:]])
        path += segment(path[-1], self.grasp_pose(gs, pre), 40) + self.insertion_plan(gs, ps, x_slot, pre[1], obs)
        self.ik.sync(self.d)
        pushed = set(self.row_geoms[ti:]) if side > 0 else set(self.row_geoms[:ti])   # (as in fingertip_then_insert)
        _, worst, ok = self.ik.follow(run.q_cmd, path, allowed={self.row_geoms[ti - 1], self.row_geoms[ti]} | pushed)
        if not ok or worst < MIN_MARGIN:
            self.log(f"         the gap is {1000 * short:.1f} mm short, but the arm can't push the {name} neighbour "
                     f"too without nearing a joint limit or running into the shelf")
            return None
        self.log(f"         the gap is {1000 * short:.1f} mm short: push the {name} neighbour too")
        if tip > tip_in + 1e-4:
            run.move("back off to cross the gap", *back)
        self.push_neighbour(ps, side, face + side * d, limit, tip)
        return x_slot, tip, tip_out, d

    def push_neighbour(self, push_sign, side, target, limit, tip_in, tries=3):
        """Push the left (side -1) or right (+1) neighbour until its face, at
        its spine, reaches x = target, never commanding past x = limit. Under
        load the arm gives way by up to ~2.5 cm (joint 6 tops out at 12 N m),
        and the neighbour swivels, so push again by what is missing."""
        name = "left" if side < 0 else "right"
        cmd = target
        for k in range(tries):
            self.run.move(f"push the {name} neighbour{' further' if k else ''} to its mark",
                          *self.push_pose(push_sign, cmd - side * FINGER_HALF_W, tip_in), speed=GAP_PUSH_SPEED)
            xl, xr, _, _ = self.slot()
            short = side * (target - (xl if side < 0 else xr))
            if short < 0.001 or cmd == limit:
                return
            cmd = cmd + side * short                                   # aim past the mark by what is missing
            if side * (cmd - limit) > 0:
                cmd = limit

    # --- Leaning scenario: stand the neighbour back up, then insert -----------------
    def insert_leaning(self, strategy="one-sided", margin=GAP_MARGIN):
        """Stand the neighbour that leans into the slot back up (stand_up, or
        clamp_up if the fingers can't get under it), then carry on as in a
        tight row (insert, widening first if the gap is short)."""
        est = self.estimate()
        lean = [s for s in (-1, 1) if est.book(s) is not None and -s * est.book(s).lean > math.radians(LEAN_MIN_DEG)]
        how = None
        if lean:
            side = lean[0]
            self.info["lean_deg"] = math.degrees(abs(est.book(side).lean))
            self.intended.add(self.row.target_index - 1 if side < 0 else self.row.target_index)
            up = self.stand_up(side)
            if up is None:                                           # no way under it: clamp it instead
                up = self.clamp_up(side)
            if not up:
                return self.skip(self.info.get("stand", "the neighbour fell back into the slot"))
            how = self.info.pop("method")                            # insert_tight labels its own step
            self.lean_on_neighbour = True

        def label(then):
            return f"{how}, then {then}" if how and then else how or then

        try:
            res = self.insert_tight(strategy, margin)
        except RuntimeError:
            self.info["method"] = label(self.info.get("method"))
            raise
        res["method"] = label(res.get("method"))
        return res

    def deepest_tip(self, xs, z_axis):
        """fingertip_depth, and, if the turned hand reaches below the board's
        top, no deeper than keeps its palm in front of the board's edge."""
        tip = self.fingertip_depth(xs, self.obstacles())
        if z_axis - HAND_HALF_L < self.z0 + CLEAR:
            tip = max(tip, self.D / 2 + CLEAR - (FINGERTIP_Z - PALM_Z))
        return tip

    def stand_up(self, side):
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
            self.info["method"] = "lift the leaner"
            run = self.run
            if run.grip != GRIP_CLOSED:
                run.wait("close the gripper", 0.3, grip=GRIP_CLOSED)
            run.move("put the closed fingers under the leaning book", *self.col_pose(sign, start[0], tip_in, start[1]))
            run.follow(f"lift it up, {1000 * r:.0f} mm up its face",
                       [self.col_pose(sign, x, tip_in, z) for x, z in arc], speed=LIFT_SPEED)
            run.move("take the fingers out", *self.col_pose(sign, arc[-1][0], tip_out, arc[-1][1]))
            run.wait("let it settle", 0.5)
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
        t, depth, h = bk.thickness, bk.depth, bk.height
        R = self.shelf.R.T @ self.d.xmat[body].reshape(3, 3)          # its axes, shelf frame
        c = self.shelf.to_shelf(self.d.xpos[body])
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
            self.info["method"] = "clamp the leaner"
            run = self.run
            run.wait("open the gripper to the book's thickness", 0.3, grip=GRIP_OPEN * opening / 0.04)
            run.move("put the fingers around the leaning book", *pose(0.0))
            run.wait("clamp it", 0.5, grip=GRIP_CLOSED)
            run.follow(f"swing it up about its bottom edge, held {1000 * r:.0f} mm up",
                       [pose(swing * s) for s in np.linspace(0, 1, 25)[1:]], speed=LIFT_SPEED)
            run.wait("let go", 0.4, grip=GRIP_OPEN * opening / 0.04)
            p_end, R_end = pose(swing)
            run.move("take the fingers out", p_end + R_end @ [0, 0, -0.07])
            run.wait("let it settle", 0.5)
            b = self.estimate().book(side)
            self.log(f"         it now leans {math.degrees(-side * b.lean):+.1f} deg into the slot")
            if -side * b.lean < math.radians(LEAN_MIN_DEG):
                return True
            self.info["stand"] = "it fell back into the slot after the clamp let go"
            return False
        self.info["stand"] = f"can't clamp the leaning book either ({why})"
        return False

    # --- Packed scenario: no gap at all ----------------------------------------------
    def insert_packed(self, margin=GAP_MARGIN, method="shove"):
        """Open a slot where the neighbours touch, then insert the book.
        'shove' (shove_then_insert): make the neighbour on the side with room
        stick out of the row, pinch it, and shove it and the books beyond it
        into the room, then insert as in a clear row. 'stagger'
        (stagger_open): open the slot just wide enough for the fingers, then
        carry on as in a tight row. Pushed at its front corner a book swivels,
        so the tight-row push from inside the gap does the rest of the
        widening better."""
        xl, xr, _, _ = self.slot()
        need = self.t + margin - (xr - xl)
        self.info["widening_m"] = max(0.0, need)
        if need <= 0:
            return self.insert()
        room = dict(zip((-1, 1), self.free_space()))
        sides = [s for s in (-1, 1) if room[s] - ROOM_KEEP >= need]
        if method == "shove" and sum(room[s] - ROOM_KEEP for s in (-1, 1)) >= need:
            return self.shove_then_insert(room, need, margin)
        if not sides:
            return self.skip("not enough free space in the row for the book")
        if not self.stagger_open(max(sides, key=lambda s: room[s]),
                                 min(need, 2 * (FINGER_HALF_W + FIT_CLEAR) + STAGGER_OPEN)):
            return self.skip(self.info["open"])
        res = self.insert_tight("one-sided", margin)
        then = res.get("method")
        res["method"] = "stagger" if then in (None, "stagger") else f"stagger, then {then}"
        return res

    def stagger_open(self, side, opening):
        """Push the slot's neighbour on the side away from `side` straight back
        into the shelf, so the face of the one on `side` shows in front of it;
        put the closed fingertips against that face and push it aside, away
        from the slot, until the gap at the spines is `opening` wide. Returns
        True once the hand is out again, or False with the reason in
        self.info["open"]."""
        ti = self.row.target_index
        w = FINGER_HALF_W
        xl, xr, fl, fr = self.slot()
        x_f, n_spine, o_spine = (xr, fr, fl) if side > 0 else (xl, fl, fr)
        o = self.shelf.box(self.m, self.d, self.row_geoms[ti - 1 if side > 0 else ti])
        room_back = o[:, 1].min() + self.D / 2 - BACK_GAP              # how far it can go back
        push_back = min(max(0.0, STAGGER - (n_spine - o_spine)), room_back)
        if n_spine - (o_spine - push_back) < MIN_REACH + 0.002:
            self.info["open"] = "no room behind the other neighbour to push it back"
            return False
        if o[:, 0].max() - o[:, 0].min() < 2 * w + 0.003:
            self.info["open"] = "the other neighbour is thinner than the fingers"
            return False
        x_o = x_f - side * (w + 0.0015)                              # fingers on the other book, beside the face
        z = self.hand_z
        o_new = o_spine - push_back
        tip_aside = o_new + 0.002                                    # just in front of its new spine
        tip_out = max(f for _, _, f in self.obstacles()) + 0.02
        x_end = x_f + side * (opening - w)
        cands = []
        for sign in (1, -1):
            path = ([self.col_pose(sign, x_o, y, z) for y in np.linspace(tip_out, tip_aside, 20)[1:]]
                    + [self.col_pose(sign, x, tip_aside, z) for x in np.linspace(x_o, x_end, 15)[1:]]
                    + [self.col_pose(sign, x_end, y, z) for y in np.linspace(tip_aside, tip_out, 10)[1:]])
            cands.append((self.col_pose(sign, x_o, tip_out, z), path, sign))
        try:
            sign = self.begin(cands, allowed={self.row_geoms[ti - 1], self.row_geoms[ti]}, finger=0.0)
        except RuntimeError as e:
            self.info["open"] = str(e)
            return False
        self.info["method"] = "stagger"
        run = self.run
        run.wait("close the gripper", 0.3, grip=GRIP_CLOSED)
        if push_back > 0:
            run.move("reach the other neighbour's spine", *self.col_pose(sign, x_o, o_spine + 0.005, z))
            run.move(f"push it {1000 * push_back:.0f} mm back into the shelf", *self.col_pose(sign, x_o, o_new, z),
                     speed=PUSH_SPEED, stop_force=PUSH_STOP_FORCE)
        run.move("put the fingertips beside the face that shows", *self.col_pose(sign, x_o, tip_aside, z))
        goal = x_end
        for k in range(3):                                          # the arm gives way under load: push on
            run.move(f"push it aside{' further' if k else ''}", *self.col_pose(sign, goal, tip_aside, z),
                     speed=GAP_PUSH_SPEED)
            xl, xr, _, _ = self.slot()
            short = opening - (xr - xl)
            if short < 0.001:
                break
            goal += side * short
        # Ease off its face before taking the fingers out, or friction drags
        # the book out of the shelf with them.
        run.move("ease off its face", *self.col_pose(sign, goal - side * EASE_OFF, tip_aside, z))
        run.move("take the fingers out", *self.col_pose(sign, goal - side * EASE_OFF, tip_out, z))
        run.wait("let it settle", 0.3)
        return True

    def shove_then_insert(self, room, need, margin):
        """Open the slot with shove_open on the side with more free space, or
        on the other if the books beside its neighbour can't go back far
        enough; if neither side alone has room for the book, on the one with
        more as far as its room allows. If the gap is still short, shove the
        far half of the run on its own and then the near half again. Then
        insert the book as in a clear row, along the straight way in, or, if
        that is too narrow, after widening what is missing as in a tight row
        (which takes it from the other side, if that is where the room is)."""
        ti = self.row.target_index
        order = sorted((-1, 1), key=lambda s: -room[s])
        if room[order[0]] - ROOM_KEEP >= need:
            options = [(side, need) for side in order if room[side] - ROOM_KEEP >= need]
        else:
            # Shove one side as far as its room allows; the tight-row widening below takes the
            # rest from the other side. (Shoving that side too would mean making its neighbour
            # stick out again after it was pushed back for this shove.)
            options = [(order[0], room[order[0]] - ROOM_KEEP)]
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
                return self.skip("; ".join(why))
        self.info["method"] = "walk out, shove" if self.info.get("walked_m", 0) > 0.005 else "shove"
        k = ti if side > 0 else ti - 1
        cannot = self.shove_open(side, k, gap, room[side] - ROOM_KEEP)
        if cannot and self.walked is None and self.walk_out(k):   # (too little to pinch once they went back)
            cannot = self.shove_open(side, k, gap, room[side] - ROOM_KEEP)
        if cannot:
            return self.skip(cannot)
        # A long run can push back too hard for the pinch to keep the neighbour from turning
        # (it pivots about its back corner instead of sliding the run). If the gap is still
        # short, shove the far half of the run on its own, then the near half into the gap
        # that leaves.
        run = list(range(k, len(self.row_geoms))) if side > 0 else list(range(k, -1, -1))
        if self.narrowest_gap() < gap - 0.001 and len(run) >= SEGMENT_MIN:
            for j in sorted(run[2:-1], key=lambda j: abs(abs(j - k) - len(run) // 2))[:3]:
                if self.shove_ready(side, j) is not None:
                    continue
                self.info["method"] = "shove in two"
                pair = (min(j, j - side), max(j, j - side))
                try:
                    self.shove_open(side, j, gap, self.free_space()[1 if side > 0 else 0] - ROOM_KEEP, pair)
                    self.shove_open(side, k, gap, max(self.narrowest_gap(pair), 0.0) + 0.005)
                except RuntimeError as e:
                    self.log(f"         can't shove the run in two: {e}")
                break
        # Still short, and the other side has room: the widening below pushes the other neighbour
        # into it with the fingertips. That neighbour went back for the shove, and the palm has to
        # stay in front of the books it sweeps past, so the fingertips would reach only its front:
        # first push those books back level with it.
        other = ti - 1 if side > 0 else ti
        room_other = self.free_space()[0 if side > 0 else 1] - ROOM_KEEP
        if self.narrowest_gap() < need - 0.001 and room_other > 0.003:
            level = self.book_box(other)[:, 1].max()
            xl, xr, _, _ = self.slot()
            far = xl - room_other if side > 0 else xr + room_other
            lo, hi = min(xl, xr, far) - HAND_HALF_T - CLEAR, max(xl, xr, far) + HAND_HALF_T + CLEAR
            proud = [i for i in range(len(self.row_geoms)) if i != other and front_x(self.book_box(i))[1] > lo
                     and front_x(self.book_box(i))[0] < hi and self.book_box(i)[:, 1].max() > level + 0.01]
            for i in sorted(proud, key=lambda i: abs(i - other)):
                try:
                    self.push_back(i, level, "shoved neighbour" if i == k else "book beside the gap")
                except RuntimeError as e:                           # (the widening may still reach)
                    self.log(f"         can't push it back level: {e}")
        xl, xr, _, _ = self.slot()
        narrowest = self.info["narrowest_gap_m"] = self.narrowest_gap()
        self.log(f"         the gap is now {1000 * (xr - xl):.1f} mm at the spines and {1000 * narrowest:.1f} mm at "
                 f"its narrowest, for a {1000 * self.t:.1f} mm book")
        width, x_mid = self.channel()
        if width >= self.t + CHANNEL_CLEAR:
            return self.insert(self.info["method"], x_mid)
        # The arm couldn't push the run far enough, and the neighbour turned in the fingers: the
        # gap is shut at the back. Widen it as in a tight row, aiming at the narrowest gap rather
        # than the gap at the spines (the fingertips push the other neighbour, into the room on
        # its side), then wedge the book in; or wedge it in at once if they can't.
        how = self.info["method"]
        res = self.insert_tight("one-sided", margin + (xr - xl) - narrowest, wedge=True)
        res["method"] = f"{how}, then {res.get('method') or 'insert'}"
        return res

    def wedge_slot(self):
        """Where to wedge the book in after a shove whose neighbour turned (its
        front went with the fingers, its back stayed): (x of the book's
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
        half = self.m.geom_size[self.row_geoms[k]][0] + PINCH_OPEN
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
            run.move(f"pull it {1000 * WALK_STEP:.0f} mm out", *at(sign, tip + WALK_STEP), speed=0.01)
            run.halt()
            run.wait("let go of it", 0.3, grip=GRIP_OPEN * (half + 0.002) / 0.04)
            run.move("take the fingers out", *at(sign, tip + WALK_STEP + 0.03))
        out_by = self.book_box(k)[:, 1].max() - start
        self.info["walked_m"] = out_by
        self.log(f"         walked the neighbour {1000 * out_by:.0f} mm out")
        if out_by > 0.005:
            self.info["method"] = "walk out, shove"
        return out_by > 0.005

    def shove_ready(self, side, k=None):
        """None if row book k (by default the slot's neighbour on `side`) can
        be pinched and shoved (it isn't too thick, and the books a finger goes
        in front of can go back far enough), else why not."""
        k = (self.row.target_index if side > 0 else self.row.target_index - 1) if k is None else k
        half = self.m.geom_size[self.row_geoms[k]][0] + PINCH_OPEN
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
        fronts = [(i, front_x(self.book_box(i))) for i in range(len(self.row_geoms)) if i != k]
        return [i for i, (a, b) in fronts if any(b > lo and a < hi for lo, hi in fingers)]

    def stick_out(self, k, flank):
        """How far row book k can be made to stick out of the books in flank,
        up to SHOVE_BACK, by pushing them back."""
        spine = self.book_box(k)[:, 1].max()
        return min([SHOVE_BACK] + [spine - self.book_box(i)[:, 1].max() + self.room_behind(i) for i in flank])

    def shove_open(self, side, k, need, limit, pair=None):
        """Shove row book k, the slot's neighbour on `side`, and every book
        beyond it, away from the slot until the gap is `need` wide where the
        book will go (moving it at most `limit`), then push it back in line.
        Each pinch: push the books a finger will be in front of straight back,
        so k sticks out of them, and shove_stage. pair: the row books (left,
        right) whose gap to open, if not the slot's. Returns None, or why it
        can't pinch it."""
        other = self.row.target_index - 1 if side > 0 else self.row.target_index
        half = self.m.geom_size[self.row_geoms[k]][0] + PINCH_OPEN
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
                cannot = self.shove_stage(side, k, half, clear, need - gap, room, stage, pair)
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
            if gap >= need - 0.001 or (stage and gap < before + 0.002):   # open, or it no longer gives
                break
        if self.book_box(k)[:, 1].max() > line + 0.003:
            self.push_back(k, line, "shoved neighbour")
        return None

    def shove_stage(self, side, k, half, clear, short, room, stage, pair=None):
        """One pinch of shove_open: pinch row book k with the fingertips `clear`
        in front of the books beside it and the palm clear of the rest, the
        hand turned to its yaw; shove it `short` further away from the slot
        (or as far as a strong arm configuration can), but at most `room`;
        let go. Returns None, or why it can't pinch it."""
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
        yaw, gap0 = self.book_yaw(k), self.narrowest_gap(pair)
        goal = x_in + side * push
        for i in range(2 if stage else 4):                           # the arm gives way under load: push on
            run.move(f"shove it and the books beyond{' further' if i else ''}",
                     *self.pinch_pose(sign, goal, tip_in, z, psi, pitch), speed=GAP_PUSH_SPEED)
            gained = self.narrowest_gap(pair) - gap0
            if gained >= push - 0.001:
                break
            goal = x_in + side * min(side * (goal - x_in) + push - gained, push + SHOVE_STEP if stage else room, room)
        self.info["shove_turn_deg"] = self.info.get("shove_turn_deg", 0.0) + math.degrees(self.book_yaw(k) - yaw)
        # Stop pressing where the arm has got to (it lags under the load, and would spring on
        # when let go), open, ease the fingers off its faces, then take them out.
        run.halt()
        run.wait("let go of it", 0.3, grip=GRIP_OPEN * (half + 0.002) / 0.04)
        run.move("ease off its faces", run.pos - side * 0.003 * self.xs)
        run.move("take the fingers out", run.pos + run.rot @ [0, 0, tip_in - tip_out])
        return None

    def push_back(self, i, goal, what):
        """Push row book i straight back along its own depth (fingers stacked,
        hand square to its spine) until the middle of its spine is at y = goal,
        or as deep as the hand can go."""
        c, psi = self.book_box(i), self.book_yaw(i)
        u = np.array([-math.sin(psi), math.cos(psi)])               # out of its spine
        mid = c[:, :2].mean(axis=0) + u * self.m.geom_size[self.row_geoms[i]][1]   # middle of its spine
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
        jacp = np.zeros((3, ik.m.nv))
        mujoco.mj_jac(ik.m, ik.d, jacp, None, point, ik.hand)
        torque = np.abs(jacp[:, ik.dadr].T @ direction)
        return float(np.min(self.m.actuator_forcerange[:7, 1] / np.maximum(torque, 1e-9)))

    def choose_strong(self, candidates, allowed, finger, direction, point_hand, moving=frozenset()):
        """choose_arm for a push that has to be strong: of the configurations
        that follow the whole path within STRONG_MARGIN of the joint limits and
        hit nothing, the one that can push hardest along `direction` at
        `point_hand` (hand frame) over the path's poses whose indices are each
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
                            force = min(force, self.push_capacity(q, pos + rot @ point_hand, direction))
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
        """Corners (8, 3; shelf frame) of row book i."""
        return self.shelf.box(self.m, self.d, self.row_geoms[i])

    def book_yaw(self, i):
        """How far row book i is turned about the vertical, in the shelf frame (rad)."""
        R = self.shelf.R.T @ self.d.xmat[self.row_bodies[i]].reshape(3, 3)
        return math.atan2(R[1, 0], R[0, 0])

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
        boxes = ([self.book_box(i) for i in range(len(self.row_geoms)) if i not in skip]
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
        it has turned more than atan(thickness / depth)."""
        g = self.row_geoms[i]
        R = self.shelf.R.T @ self.d.geom_xmat[g].reshape(3, 3)
        half_t, half_d, half_h = self.m.geom_size[g]
        face = (self.shelf.to_shelf(self.d.geom_xpos[g]) + side * math.copysign(half_t, R[0, 0]) * R[:, 0]
                - half_h * R[:, 2])
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
        return {"success": False, "skipped": reason, "neighbour_shift_m": float(shifts.max()),
                "total_shift_m": float(shifts.sum()), **self.info}


# ---------------------------------------------------------------------------
# 4. Episodes and CLI
# ---------------------------------------------------------------------------
def make_episode(cfg: ShelfConfig, scenario: str, rng, panda_xml: str, max_tries: int = 10):
    """Like shelf_scene.make_episode, on the model with the target book."""
    for _ in range(max_tries):
        row = sample_row(cfg, scenario, rng)
        model = build_model(cfg, row, panda_xml)
        data = mujoco.MjData(model)
        if settle(model, data, row)["stable"]:
            return model, data, row
    raise RuntimeError(f"No stable {scenario} row in {max_tries} tries")


def report(res: dict) -> str:
    verdict = "SUCCESS" if res["success"] else "FAIL"
    how = f" [{res['method']}]" if res.get("method") else ""
    if "skipped" in res:
        return f"  SKIPPED{how}: {res['skipped']} | other books: most moved {1000 * res['neighbour_shift_m']:.1f} mm"
    place = (f"spine {1000 * res['offset_m']:+.1f} mm from the neighbour's" if res["rule"] == "flush" else
             f"deeper than its neighbours, spine {1000 * res['offset_m']:.1f} mm inside the front edge")
    return (f"  {verdict}{how}: tilt {res['tilt_deg']:.1f} deg, {place}"
            f"{' (hand blocked)' if res['blocked'] else ''}, {'released' if res['released'] else 'STILL TOUCHING'}\n"
            f"  other books: most moved {1000 * res['neighbour_shift_m']:.1f} mm (all together "
            f"{1000 * res['total_shift_m']:.0f} mm), most tilted {res['neighbour_tilt_change_deg']:.1f} deg"
            f" | let go after {res['time_to_release_s']:.1f} s, in place after {res['time_to_seat_s']:.1f} s")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenario", choices=("clear", "tight", "leaning", "packed"), default="clear")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--trials", type=int, default=1, help="run seeds seed .. seed+trials-1 and summarise")
    ap.add_argument("--strategy", choices=("one-sided", "split"), default="one-sided",
                    help="tight: push one neighbour aside, or split the widening between both")
    ap.add_argument("--margin", type=float, default=GAP_MARGIN, help="tight: room left around the book, m")
    ap.add_argument("--packed", choices=("shove", "stagger"), default="shove",
                    help="packed: shove the neighbour and the books beyond it, or stagger the neighbours")
    ap.add_argument("--panda", default=DEFAULT_PANDA, help="Menagerie panda.xml")
    ap.add_argument("--book-meshes", action="store_true",
                    help="draw the converted repo book covers (run dae_to_obj.py first)")
    ap.add_argument("--book-pages", type=int, default=0, metavar="N",
                    help="split every book's pages into N sections hinged on the spine (0 = one rigid box)")
    ap.add_argument("--view", action="store_true", help="watch in the MuJoCo viewer (run with mjpython on macOS)")
    a = ap.parse_args()

    def attempt(ep):
        if a.scenario == "clear":
            return ep.insert(method="")
        if a.scenario == "packed":
            return ep.insert_packed(a.margin, a.packed)
        if a.scenario == "leaning":
            return ep.insert_leaning(a.strategy, a.margin)
        return ep.insert_tight(a.strategy, a.margin)

    results = []
    cfg = ShelfConfig(book_meshes=a.book_meshes, book_pages=a.book_pages)
    for seed in range(a.seed, a.seed + a.trials):
        model, data, row = make_episode(cfg, a.scenario, np.random.default_rng(seed), a.panda)
        b = row.target_book
        print(f"seed {seed}: target level {row.target_level} (board top {row.target_board_top:.2f} m), "
              f"gap {1000 * (row.gap_right - row.gap_left):.0f} mm for a {1000 * b.thickness:.0f} mm book "
              f"({1000 * b.depth:.0f} mm deep, {b.mass:.2f} kg)")
        log = print if a.trials == 1 else (lambda *_: None)
        if a.view:
            import mujoco.viewer
            with mujoco.viewer.launch_passive(model, data) as viewer:
                # The viewer only aims its camera with the scene's defaults (at
                # the gap) when it opens a file, so do it here.
                with viewer.lock():
                    mujoco.mjv_defaultFreeCamera(model, viewer.cam)
                res = attempt(Episode(model, data, row, viewer, log))
                print(report(res))
                print("Close the viewer window to exit.")
                while viewer.is_running():
                    mujoco.mj_step(model, data)
                    viewer.sync()
                    time.sleep(model.opt.timestep)
        else:
            ep = Episode(model, data, row, None, log)
            try:
                res = attempt(ep)
                print(report(res))
            except RuntimeError as e:                 # no safe way to do it in this scene, or it jammed
                res = {"success": False, "error": str(e), **ep.info}
                print(f"  FAIL{' [' + ep.info['method'] + ']' if 'method' in ep.info else ''}: {e}")
        results.append(res)

    if a.trials > 1:
        n = len(results)
        ok = sum(r["success"] for r in results)
        print(f"\n{ok}/{n} succeeded. Of the rest: "
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
