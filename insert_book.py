"""
Scripted insertion of the target book for the clear scenario.

    uv run python insert_book.py --seed 0               # one episode, headless
    uv run python insert_book.py --seed 0 --trials 50   # success rate over seeds 0..49
    uv run mjpython insert_book.py --seed 0 --view      # watch it (macOS needs mjpython)

The sequence (shelf frame: x along the row, y out of the shelf, z up)
  1. Start already grasped at the pre-insertion pose: the book is lined up with
     the gap, its back 1 cm over the board's front edge and 3 mm above it. The
     fingers pinch its thickness near the spine, 11.5 cm above its bottom.
  2. Rest its back on the board's edge, then slide it in along the board while
     holding it, until the hand would come within 5 mm of another book's spine.
     Each finger reaches 2.6 cm out from its pad, more than the clearance
     beside the book, so the fingers can't follow it in.
  3. Open the gripper.
  4. Back off, turn the hand 90 deg so the fingers are stacked vertically, and
     push the spine low down with the lower fingertip, until the book's back is
     5 mm off the back panel like every other book. If deeper neighbours block
     the hand first, it stops there and the episode counts as not seated.
  5. Back out and score it: upright within 5 deg, seated, and released.

Control: Menagerie's position actuators, fed joint targets by differential IK
along a min-jerk Cartesian path, with gravity compensation on the arm. The
gripper is made as strong as the real Franka Hand (70 N continuous): Menagerie's
gripper squeezes a book with only 1-2 N, too little to hold one.
"""

import argparse
import math
import os
import time
import xml.etree.ElementTree as ET

import numpy as np
import mujoco

from shelf_scene import PANDA_HOME, ShelfConfig, _f, book_tilt_deg, build_scene, sample_row, settle

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_PANDA = os.path.join(HERE, "mujoco_menagerie", "franka_emika_panda", "panda.xml")

# Panda hand geometry, measured from Menagerie's collision meshes.
# Hand frame: z = approach (along the fingers), y = finger travel.
FINGERTIP_Z = 0.1123        # end of the fingers
PAD_Z = 0.1029              # centre of the finger pads
PALM_Z = 0.066              # face of the hand between the fingers
FINGER_OPEN_SPAN = 0.0663   # outer face of a fully open finger, from the hand axis
FINGER_OPEN_MID = 0.0544    # centre of a fully open finger, from the hand axis
FINGER_HALF_W = 0.0105      # finger half-width, across its travel
HAND_HALF_T = 0.0316        # hand half-thickness, across finger travel
HAND_HALF_L = 0.105         # hand half-length, along finger travel

# Grasp and sequence settings
GRIP_FORCE = 70.0           # N on each side of the book (the real Franka Hand's continuous force)
PAD_INSET = 0.029           # pad centre this far behind the spine (palm 8 mm off the spine)
PAD_HEIGHT = 0.115          # ... and this high above the book's bottom. Low, so sliding the book on
                            # the board can't tip it: board friction at its back-bottom corner tips it
                            # about the pads if friction x PAD_HEIGHT > (pads to book back). Not lower,
                            # because the arm can only hold its hand level >= ~0.5 m above its base.
PRE_OVERLAP = 0.01          # pre-insertion: the book's back is this far over the board's front edge
HOVER = 0.003               # ... and this high above the board
CLEAR = 0.005               # gap kept between the hand and any other book or panel
BACK_GAP = 0.005            # seated: back face this far off the back panel (as the generator places books)
PUSH_LIFT = 0.01            # pushing: bottom of the turned hand this far above the board
TRANSIT_SPEED = 0.10        # m/s, peak of each min-jerk move
INSERT_SPEED = 0.06
PUSH_SPEED = 0.05
TURN_SPEED = 1.5            # rad/s
IK_SEEDS = 20               # random starts per round when choosing the arm configuration
IK_ROUNDS = 3               # rounds tried before giving up (most scenes need only the first)
MIN_MARGIN = 0.05           # rad: the chosen path must keep every joint this far from its limits

# Scoring
SEAT_TOL = 0.005            # seated if the back is within BACK_GAP + SEAT_TOL of the back panel
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

    def follow(self, q, poses):
        """Track a list of (pos, rot) hand poses from q, warm-started like the
        runner does. Returns (end q, worst joint-limit margin on the way, reached the end)."""
        worst = self.margin(q)
        for pos, rot in poses:
            q = self.solve(q, pos, rot, iters=4)
            worst = min(worst, self.margin(q))
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

    def move(self, name, pos, rot=None, speed=TRANSIT_SPEED, grip=None):
        """Move the hand frame to (pos, rot) in a straight line, with a slerp
        on its orientation. Peak speeds are `speed` and TURN_SPEED."""
        self.log(f"  {self.t:5.2f} s  {name}")
        if grip is not None:
            self.grip = grip
        rot = self.rot if rot is None else rot
        p0, R0 = self.pos.copy(), self.rot.copy()
        dist, ang = np.linalg.norm(pos - p0), np.linalg.norm(rotation_between(R0, rot))
        T = max(0.3, 1.875 * max(dist / speed, ang / TURN_SPEED))   # min-jerk peak = 1.875 x mean
        n = int(round(T / self.m.opt.timestep))
        for k in range(1, n + 1):
            u = k / n
            self.pos, self.rot = interpolate(p0, R0, pos, rot, u ** 3 * (10 - 15 * u + 6 * u * u))
            self.q_cmd = self.ik.solve(self.q_cmd, self.pos, self.rot)
            self.step()

    def wait(self, name, duration, grip=None):
        self.log(f"  {self.t:5.2f} s  {name}")
        if grip is not None:
            self.grip = grip
        for _ in range(int(round(duration / self.m.opt.timestep))):
            self.step()


# ---------------------------------------------------------------------------
# 3. The scripted sequence
# ---------------------------------------------------------------------------
GRIP_CLOSED, GRIP_OPEN = 0.0, 255.0


def hand_pose_for_point(point_hand, point_world, rot):
    """Hand position that puts `point_hand` (hand frame) at `point_world`."""
    return point_world - rot @ point_hand


def insert_book(model, data, row, viewer=None, log=print) -> dict:
    """Run the scripted insertion from the settled scene. Returns the score."""
    shelf = ShelfFrame(row)
    xs, ys, zs = shelf.R.T                                    # shelf axes in world
    b = row.target_book
    t, d, h = b.thickness, b.depth, b.height
    D, H, z0 = row.shelf_depth, row.clear_height, row.target_board_top
    x_gap = (row.gap_left + row.gap_right) / 2
    spine_seated = -D / 2 + BACK_GAP + d                      # spine y once seated
    if 2 * HAND_HALF_L + PUSH_LIFT + CLEAR > H:
        raise ValueError(f"Level is {H:.2f} m tall: too low to turn the hand for the push")

    book_geom = model.geom(b.name + "_geom").id
    book_body = model.body(b.name).id
    book_q = model.jnt_qposadr[model.joint(b.name + "_joint").id]
    others = [model.body(bk.name).id for bk in row.books]
    start_pos = data.xpos[others].copy()
    start_tilt = np.array([book_tilt_deg(data.xquat[i]) for i in others])
    obs = obstacles(model, data, shelf, row)

    # Hand poses. Holding: the fingers close across the book's thickness (along
    # the row), approaching along -y. Pushing: the hand is turned 90 deg about
    # the approach axis so the fingers are stacked vertically, and pushes the
    # spine low down with the lower finger, so the push can't tip the book
    # over backwards. Both work with the hand either way up (sign).
    pad_in_book = np.array([0, d / 2 - PAD_INSET, PAD_HEIGHT - h / 2])
    hand_z = z0 + HAND_HALF_L + PUSH_LIFT                     # hand axis height while pushing

    def grasp_pose(sign, book_shelf):
        rot = np.column_stack([np.cross(sign * xs, -ys), sign * xs, -ys])
        return hand_pose_for_point([0, 0, PAD_Z], shelf.to_world(book_shelf + pad_in_book), rot), rot

    def push_pose(sign, x, tip_y):
        rot = np.column_stack([np.cross(sign * zs, -ys), sign * zs, -ys])
        low_tip = [0, -sign * FINGER_OPEN_MID, FINGERTIP_Z]
        return hand_pose_for_point(low_tip, shelf.to_world([x, tip_y, hand_z - FINGER_OPEN_MID]), rot), rot

    def push_path(start, sign, x, spine_now, tip_goal):
        """From `start` (just let go): back off, turn, push to tip_goal.
        Returns the back-off pose, the turned pose and the whole path."""
        tip_back = max(D / 2 + 0.015, spine_now + 0.02)
        tip_now = shelf.to_shelf(start[0] + start[1] @ [0, 0, FINGERTIP_Z])[1]
        back_off = (start[0] + ys * (tip_back - tip_now), start[1])
        ready = push_pose(sign, x, tip_back)
        path = ([interpolate(*start, *back_off, s) for s in np.linspace(0, 1, 15)[1:]]
                + [interpolate(*back_off, *ready, s) for s in np.linspace(0, 1, 50)[1:]]
                + [push_pose(sign, x, y) for y in np.linspace(tip_back, tip_goal, 30)])
        return back_off, ready, path

    def push_limit(x):
        return tip_limit(obs, x, [(FINGER_HALF_W, 0.0), (HAND_HALF_T, FINGERTIP_Z - PALM_Z)])

    hold_limit = tip_limit(obs, x_gap, [(FINGER_OPEN_SPAN, 0.0), (HAND_HALF_L, FINGERTIP_Z - PALM_Z)])

    # --- 1. Choose the arm configuration -------------------------------------
    # Many arm configurations reach the pre-insertion pose. Near a joint limit
    # the IK swings the other joints round, and the arm can jump to another
    # configuration mid-move and knock the row. So follow the whole planned
    # sequence (slide in, back off, turn, push) from each candidate, and keep
    # the one that stays farthest from the joint limits.
    ik = ArmIK(model, PANDA_HOME[:7])
    book_pre = np.array([x_gap, D / 2 - PRE_OVERLAP + d / 2, z0 + h / 2 + HOVER])
    y_hold = max(spine_seated, hold_limit + PAD_INSET + FINGERTIP_Z - PAD_Z) - d / 2   # refined once grasped
    tip_goal = max(spine_seated, push_limit(x_gap))
    seed_rng = np.random.default_rng(0)
    best = None
    for rnd in range(IK_ROUNDS):                  # more random starts only if the first ones fall short
        seeds = [seed_rng.uniform(ik.lo, ik.hi) for _ in range(IK_SEEDS)]
        if rnd == 0:
            seeds.insert(0, np.array(PANDA_HOME[:7]))
        for grasp_sign in (1, -1):
            slide = [grasp_pose(grasp_sign, [x_gap, y, z0 + h / 2]) for y in np.linspace(book_pre[1], y_hold, 40)]
            for s in seeds:
                q, ok = ik.solve_far(s, *grasp_pose(grasp_sign, book_pre))
                if not ok:
                    continue
                ik.q_rest = q
                q_slid, worst_slide, ok = ik.follow(q, slide)
                if not ok or (best is not None and worst_slide <= best[0]):
                    continue
                for push_sign in (1, -1):
                    path = push_path(slide[-1], push_sign, x_gap, y_hold + d / 2, tip_goal)[2]
                    _, worst_push, ok = ik.follow(q_slid, path)
                    score = min(worst_slide, worst_push)
                    if ok and (best is None or score > best[0]):
                        best = (score, q, push_sign)
        if best is not None and best[0] >= MIN_MARGIN:
            break
    if best is None or best[0] < MIN_MARGIN:
        raise RuntimeError("No arm configuration does the whole sequence without nearing a joint limit")
    _, q_pre, push_sign = best
    ik.q_rest = q_pre

    # Put the arm there with the book between the fingers, then close them.
    data.qpos[ik.qadr] = q_pre
    for j in ("panda/finger_joint1", "panda/finger_joint2"):
        data.qpos[model.jnt_qposadr[model.joint(j).id]] = t / 2 + 0.0005
    p_hand, R_hand = ik.fk(q_pre)
    data.qpos[book_q:book_q + 3] = p_hand + R_hand @ [0, 0, PAD_Z] - shelf.R @ pad_in_book
    data.qpos[book_q + 3:book_q + 7] = mat2quat(shelf.R)
    data.qvel[:] = 0
    mujoco.mj_forward(model, data)
    run = Runner(model, data, ik, q_pre, GRIP_CLOSED, viewer, log)
    run.wait("close the gripper on the book at the pre-insertion pose", 0.5)

    # From here on, move the book: keep its pose in the hand as measured now.
    R_h = data.xmat[ik.hand].reshape(3, 3)
    R_b = data.xmat[book_body].reshape(3, 3)
    book_in_hand = R_h.T @ (data.xpos[book_body] - data.xpos[ik.hand])
    rot_held = shelf.R @ (R_h.T @ R_b).T                       # hand rotation that keeps the book upright

    def hand_for_book(book_shelf):
        return shelf.to_world(book_shelf) - rot_held @ book_in_hand, rot_held

    # While held, the spine stays a fixed distance in front of the fingertips.
    tip = shelf.to_shelf(data.xpos[ik.hand] + R_h @ [0, 0, FINGERTIP_Z])
    spine = shelf.box(model, data, book_geom)[:, 1].max()
    spine_ahead = spine - tip[1]
    t_start = run.t

    # --- 2. Rest it on the board's edge, then slide it in along the board ---
    # A two-finger pinch can't stop a heavy book from pitching nose-down (its
    # weight acts several cm behind the pads). Resting its back on the board
    # first lets the board carry that, as a person would do it.
    y_hold = max(spine_seated, hold_limit + spine_ahead) - d / 2
    if y_hold > D / 2 - CLEAR:                                  # its centre must be over the board
        raise RuntimeError("Hand can't get the book far enough onto the board to let go")
    z_rest = z0 + h / 2 + 0.0005
    run.move("rest it on the board's edge", *hand_for_book([book_pre[0], book_pre[1], z_rest]), speed=0.01)
    run.move("slide it in along the board", *hand_for_book([x_gap, y_hold, z_rest]), speed=INSERT_SPEED)

    # --- 3. Let go -----------------------------------------------------------
    run.wait("open the gripper", 0.4, grip=GRIP_OPEN)
    t_release = run.t

    # --- 4. Back off, turn the hand, push the spine with the lower fingertip -
    # Replan from where the book really is. If the planned turn direction now
    # comes too close to a joint limit, try the other one.
    c = shelf.box(model, data, book_geom)
    x_book, spine = c[:, 0].mean(), c[:, 1].max()
    tip_goal = max(spine_seated, push_limit(x_book))
    blocked = tip_goal > spine_seated
    for sign in (push_sign, -push_sign):
        back_off, ready, path = push_path((run.pos, run.rot), sign, x_book, spine, tip_goal)
        _, worst, ok = ik.follow(run.q_cmd, path)
        if ok and worst >= MIN_MARGIN:
            break
    else:
        raise RuntimeError("No way to turn the hand for the push without nearing a joint limit")

    run.move("back off", *back_off)
    run.move("turn the hand to push", *ready)
    run.move("reach the spine", *push_pose(sign, x_book, spine + 0.005))
    run.move("push it home" if not blocked else
             f"push it in (deeper neighbours stop the hand {1000 * (tip_goal - spine_seated):.0f} mm short)",
             *push_pose(sign, x_book, tip_goal), speed=PUSH_SPEED)
    t_done = run.t

    # --- 5. Back out and score ----------------------------------------------
    run.move("back out", *push_pose(sign, x_book, max(D / 2, tip_goal) + 0.05))
    run.wait("let it settle", 1.0)

    c = shelf.box(model, data, book_geom)
    tilt = book_tilt_deg(data.xquat[book_body])
    back_gap = c[:, 1].min() + D / 2
    robot_geoms = {g for g in range(model.ngeom) if model.body(model.geom_bodyid[g]).name.startswith("panda/")}
    touching = any((con.geom1 == book_geom and con.geom2 in robot_geoms) or
                   (con.geom2 == book_geom and con.geom1 in robot_geoms) for con in data.contact[:data.ncon])
    shift = np.linalg.norm(data.xpos[others] - start_pos, axis=1).max()
    dtilt = np.abs(np.array([book_tilt_deg(data.xquat[i]) for i in others]) - start_tilt).max()
    res = {
        "upright": tilt <= UPRIGHT_DEG, "seated": back_gap <= BACK_GAP + SEAT_TOL, "released": not touching,
        "tilt_deg": tilt, "back_gap_m": back_gap, "blocked": blocked,
        "neighbour_shift_m": float(shift), "neighbour_tilt_change_deg": float(dtilt),
        "time_to_release_s": t_release - t_start, "time_to_seat_s": t_done - t_start,
    }
    res["success"] = res["upright"] and res["seated"] and res["released"]
    return res


# ---------------------------------------------------------------------------
# 4. Episodes and CLI
# ---------------------------------------------------------------------------
def make_clear_episode(cfg: ShelfConfig, rng, panda_xml: str, max_tries: int = 10):
    """Like shelf_scene.make_episode, on the model with the target book."""
    for _ in range(max_tries):
        row = sample_row(cfg, "clear", rng)
        model = build_model(cfg, row, panda_xml)
        data = mujoco.MjData(model)
        if settle(model, data, row)["stable"]:
            return model, data, row
    raise RuntimeError(f"No stable clear row in {max_tries} tries")


def report(res: dict) -> str:
    verdict = "SUCCESS" if res["success"] else "FAIL"
    return (f"  {verdict}: tilt {res['tilt_deg']:.1f} deg, back {1000 * res['back_gap_m']:.1f} mm off the panel"
            f"{' (hand blocked)' if res['blocked'] else ''}, {'released' if res['released'] else 'STILL TOUCHING'}\n"
            f"  neighbours: moved {1000 * res['neighbour_shift_m']:.1f} mm, tilted {res['neighbour_tilt_change_deg']:.1f} deg"
            f" | released after {res['time_to_release_s']:.1f} s, seated after {res['time_to_seat_s']:.1f} s")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--trials", type=int, default=1, help="run seeds seed .. seed+trials-1 and summarise")
    ap.add_argument("--panda", default=DEFAULT_PANDA, help="Menagerie panda.xml")
    ap.add_argument("--view", action="store_true", help="watch in the MuJoCo viewer (run with mjpython on macOS)")
    a = ap.parse_args()

    results = []
    for seed in range(a.seed, a.seed + a.trials):
        model, data, row = make_clear_episode(ShelfConfig(), np.random.default_rng(seed), a.panda)
        b = row.target_book
        print(f"seed {seed}: target level {row.target_level} (board top {row.target_board_top:.2f} m), "
              f"gap {1000 * (row.gap_right - row.gap_left):.0f} mm for a {1000 * b.thickness:.0f} mm book "
              f"({1000 * b.depth:.0f} mm deep, {b.mass:.2f} kg)")
        log = print if a.trials == 1 else (lambda *_: None)
        if a.view:
            import mujoco.viewer
            with mujoco.viewer.launch_passive(model, data) as viewer:
                res = insert_book(model, data, row, viewer, log)
                print(report(res))
                print("Close the viewer window to exit.")
                while viewer.is_running():
                    mujoco.mj_step(model, data)
                    viewer.sync()
                    time.sleep(model.opt.timestep)
        else:
            try:
                res = insert_book(model, data, row, None, log)
                print(report(res))
            except RuntimeError as e:                 # no safe way to do it in this scene
                res = {"success": False, "error": str(e)}
                print(f"  FAIL: {e}")
        results.append(res)

    if a.trials > 1:
        n = len(results)
        ok = sum(r["success"] for r in results)
        print(f"\n{ok}/{n} succeeded. Of the rest: "
              f"{sum(r.get('blocked', False) and not r['success'] for r in results)} blocked by deeper neighbours, "
              f"{sum('error' in r for r in results)} with no safe arm path, "
              f"{sum(not r.get('upright', True) for r in results)} not upright, "
              f"{sum(not r.get('released', True) for r in results)} not released")


if __name__ == "__main__":
    main()
