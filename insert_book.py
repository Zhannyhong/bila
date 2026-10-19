"""
Scripted insertion of the target book, for the clear and tight scenarios.

    uv run python insert_book.py --scenario clear --seed 0              # one episode, headless
    uv run python insert_book.py --scenario tight --seed 0 --trials 50  # success rate over seeds 0..49
    uv run mjpython insert_book.py --scenario tight --seed 0 --view     # watch it (macOS needs mjpython)

Insertion (shelf frame: x along the row, y out of the shelf, z up)
  1. Hold the book at the pre-insertion pose: lined up with the gap, its back
     1 cm over the board's front edge and 3 mm above it. The fingers pinch its
     thickness near the spine, 11.5 cm above its bottom.
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

from shelf_scene import (FINGER_HALF_W, FINGER_LEN, FINGER_THICK, PANDA_HOME, ShelfConfig, _f, book_tilt_deg,
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
PAD_INSET = 0.029           # the pads hold the book this far behind its spine (palm 8 mm off the spine)
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
IK_SEEDS = 20               # random starts per round when choosing the arm configuration
IK_ROUNDS = 3               # rounds tried before giving up (most scenes need only the first)
MIN_MARGIN = 0.05           # rad: the chosen path must keep every joint this far from its limits

# Gap creation (tight scenario)
GAP_MARGIN = 0.006          # widen the gap to the book's thickness plus this
FIT_CLEAR = 0.001           # fingers need this much room each side to enter the gap
MIN_REACH = 0.010           # fingertips must get this far behind a neighbour's spine to push it
ROOM_KEEP = 0.002           # leave this much of a side's free space unused
GAP_PUSH_SPEED = 0.03

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
    ET.SubElement(body, "geom", name=b.name + "_geom", **{"class": "book"},
                  size=_f(b.thickness / 2, b.depth / 2, b.height / 2), mass=f"{b.mass:.5g}",
                  friction=_f(b.friction, 0.005, 0.0001), rgba=_f(*b.rgba))
    key = root.find("keyframe/key")
    key.set("qpos", key.get("qpos") + " " + _f(*park, 1, 0, 0, 0))   # its joint comes last in qpos

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
        self.ignore = set()                 # geoms the arm may touch anywhere (the book it handles)

    def sync(self, data):
        """Copy the scene (books, fingers) from the simulation, for collision checks."""
        self.d.qpos[:] = data.qpos

    def collides(self, q, allowed=()):
        """Whether the arm at q is more than 1 mm into anything, other than
        the geoms in `allowed`, which its fingers may press on."""
        self.d.qpos[self.qadr] = q
        mujoco.mj_fwdPosition(self.m, self.d)
        for c in self.d.contact[:self.d.ncon]:
            if (c.geom1 in self.robot) == (c.geom2 in self.robot) or c.dist > -0.001:
                continue
            arm, other = (c.geom1, c.geom2) if c.geom1 in self.robot else (c.geom2, c.geom1)
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

    def solve(self, q, pos, rot, iters=2, max_step=0.2):
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
                dq = Jinv @ e + null @ (0.05 * (self.q_rest - q) * free)
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
        `allowed` is given, once the arm runs into something else."""
        worst = self.margin(q)
        if allowed is not None and self.collides(q, allowed):
            return q, worst, False
        for k, (pos, rot) in enumerate(poses):
            q = self.solve(q, pos, rot, iters=4)
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


def hand_pose_for_point(point_hand, point_world, rot):
    """Hand position that puts `point_hand` (hand frame) at `point_world`."""
    return point_world - rot @ point_hand


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
        self.run = None
        self.info = {}

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
        """Room each side of the slot: the spaces between books plus the row's end."""
        W, ti = self.row.shelf_width, self.row.target_index
        xs = [self.shelf.box(self.m, self.d, g)[:, 0] for g in self.row_geoms]
        spaces = [xs[i].min() - xs[i - 1].max() for i in range(1, len(xs))]
        return xs[0].min() + W / 2 + sum(spaces[:ti - 1]), W / 2 - xs[-1].max() + sum(spaces[ti:])

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

    def choose_arm(self, candidates, allowed=frozenset()):
        """Many arm configurations reach a start pose. Near a joint limit the IK
        swings the other joints round, and the arm can jump to another
        configuration mid-move and knock the row. So follow each candidate's
        whole planned path, drop those where the arm runs into anything (other
        than `allowed` geoms its fingers may press on), and keep the one that
        stays farthest from the joint limits. candidates: (start pose, path, label)."""
        ik, seed_rng, best = self.ik, np.random.default_rng(0), None
        ik.sync(self.d)
        for rnd in range(IK_ROUNDS):                   # more random starts only if the first ones fall short
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

    def take_book(self, grasp_sign, centre, yaw=0.0, name="close the gripper on the book"):
        """Put the book between the open fingers, held as at (centre, yaw), and
        close them. Where the book comes from is not modelled."""
        m, d, ik = self.m, self.d, self.ik
        R_h = d.xmat[ik.hand].reshape(3, 3)
        R_b = R_h @ self.grasp_pose(grasp_sign, centre, yaw)[1].T @ self.shelf.rot(yaw)
        for j in ("panda/finger_joint1", "panda/finger_joint2"):
            d.qpos[m.jnt_qposadr[m.joint(j).id]] = self.t / 2 + 0.0005
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

    def seat(self, x_slot, y_from, push_sign):
        """Holding the book straight in front of (or partly in) the slot: rest
        it on the board, slide it in, let go, and push it into place.
        Returns (time let go, time in place, whether the hand was blocked)."""
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
        if not run.move("slide it in along the board", *self.hand_for_book([x_slot, y_hold, self.z_rest]),
                        speed=INSERT_SPEED, stop_force=SLIDE_STOP_FORCE):
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
        pushed = run.move("push it into place" if not blocked else
                          f"push it in (nearby spines stop the hand {1000 * (tip_goal - target):.0f} mm short)",
                          *self.push_pose(sign, x_book, tip_goal), speed=PUSH_SPEED, stop_force=PUSH_STOP_FORCE)
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
                              *self.push_pose(sign, x_book, tip_goal), speed=PUSH_SPEED, stop_force=PUSH_STOP_FORCE)
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
        robot = self.ik.robot
        touching = any((con.geom1 == self.book_geom and con.geom2 in robot) or
                       (con.geom2 == self.book_geom and con.geom1 in robot) for con in d.contact[:d.ncon])
        shifts = np.linalg.norm(d.xpos[self.row_bodies] - self.start_pos, axis=1)
        dtilt = np.abs(np.array([book_tilt_deg(d.xquat[i]) for i in self.row_bodies]) - self.start_tilt)
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
    def insert(self, method="none needed"):
        xl, xr, _, _ = self.slot()
        x_slot = (xl + xr) / 2
        pre = self.book_pre(x_slot)
        obs = self.obstacles()
        candidates = [(self.grasp_pose(gs, pre), self.insertion_plan(gs, ps, x_slot, pre[1], obs), (gs, ps))
                      for gs in (1, -1) for ps in (1, -1)]
        q, (gs, ps) = self.choose_arm(candidates)
        self.place_arm(q)
        self.take_book(gs, pre, name="close the gripper on the book at the pre-insertion pose")
        t_start = self.run.t
        return self.score(t_start, *self.seat(x_slot, pre[1], ps), method=method)

    # --- Tight scenario: widen the gap, then insert ----------------------------------
    def insert_tight(self, strategy="one-sided", margin=GAP_MARGIN):
        xl, xr, fl, fr = self.slot()
        need = self.t + margin - (xr - xl)
        self.info["widening_m"] = max(0.0, need)
        if need <= 0:
            self.log(f"  the {1000 * (xr - xl):.0f} mm gap already fits the book")
            return self.insert()
        plan = self.plan_fingertip(xl, xr, fl, fr, need, strategy)
        if plan is None:
            return self.skip(self.info["fingertip"])
        self.info["method"] = "fingertip, both sides" if plan[0] > 0 and plan[1] > 0 else "fingertip, one side"
        return self.fingertip_then_insert(xl, xr, *plan, gap_goal=self.t + margin)

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

    def fingertip_then_insert(self, xl, xr, dl, dr, tip_in, gap_goal):
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
        q, (ps, gs) = self.choose_arm(candidates, allowed={self.row_geoms[ti - 1], self.row_geoms[ti]})
        self.place_arm(q)
        run = self.run
        t_start = run.t
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
        x_slot = (xl + xr) / 2
        pre = self.book_pre(x_slot)
        run.move("turn the hand to the pre-insertion pose", *self.grasp_pose(gs, pre))
        self.take_book(gs, pre, name="take the book (the hand-over is not modelled)")
        return self.score(t_start, *self.seat(x_slot, pre[1], ps), gap_time_s=gap_time,
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
        _, worst, ok = self.ik.follow(run.q_cmd, path, allowed={self.row_geoms[ti - 1], self.row_geoms[ti]})
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

    def skip(self, reason) -> dict:
        """Leave the book out: a failure, with how far the other books moved
        (only a push that fell short moves them)."""
        self.log(f"  skipped: {reason}")
        shifts = np.linalg.norm(self.d.xpos[self.row_bodies] - self.start_pos, axis=1)
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
    ap.add_argument("--scenario", choices=("clear", "tight"), default="clear")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--trials", type=int, default=1, help="run seeds seed .. seed+trials-1 and summarise")
    ap.add_argument("--strategy", choices=("one-sided", "split"), default="one-sided",
                    help="tight: push one neighbour aside, or split the widening between both")
    ap.add_argument("--margin", type=float, default=GAP_MARGIN, help="tight: room left around the book, m")
    ap.add_argument("--panda", default=DEFAULT_PANDA, help="Menagerie panda.xml")
    ap.add_argument("--view", action="store_true", help="watch in the MuJoCo viewer (run with mjpython on macOS)")
    a = ap.parse_args()

    def attempt(ep):
        return ep.insert(method="") if a.scenario == "clear" else ep.insert_tight(a.strategy, a.margin)

    results = []
    for seed in range(a.seed, a.seed + a.trials):
        model, data, row = make_episode(ShelfConfig(), a.scenario, np.random.default_rng(seed), a.panda)
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
