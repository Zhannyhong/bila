"""
Scripted controller for the LEANING scenario: push the leaning neighbour back
upright so the slot for the target book opens up again.

    uv run python scripted_controller.py --seed 0                 # one episode, prints metrics
    uv run python scripted_controller.py --seeds 20               # batch evaluation
    uv run mjpython scripted_controller.py --seed 0 --view        # watch it (macOS needs mjpython)

How it works (all in the shelf frame: x along the row, y out of the shelf, z up)

  1. Perception (privileged): read the book poses from MjData and find the
     wedge between the leaning book and the book it rests on. The wedge is
     widest at the bottom and closes where the leaner's top touches the
     other book.
  2. Plan: pick the highest point in the wedge that the tool still fits into
     (more leverage), and how deep the tool can go before the wide hand would
     touch a book.
  3. Execute: joint-space move to a staging pose 10 cm out, then straight
     Cartesian lines. Each control tick turns the Cartesian setpoint into
     joint targets with damped least-squares IK and sends them to the Panda's
     position actuators. Gravity is compensated, like the real Franka does.
  4. Closed-loop pushes: push away from the slot until the leaner is upright
     (no further: the books behind it aren't packed and would topple like
     dominoes). If the slot is still narrower than the new book, push low to
     slide the books into the spare row space until the slot is wide enough.

A thin pusher rod is added to the hand (default 8 cm, --tool-length 0 to
use the bare fingers), pointing into the shelf and pitched 15 deg down.
Books sit up to ~15 cm behind the front of the board, so without the rod
the 21 cm-wide hand has to enter the shelf and hits the books next to the slot.

Unsupported cases are raised as exceptions: NoWedge (the leaner lies flat
against the far book, so there's nothing to push from) and RuntimeError
(pose out of reach).

This is the baseline that a residual PPO policy would add corrections to:
`ScriptedLeaningController.step()` produces a Cartesian setpoint each tick,
and the residual can be added to that setpoint before IK.
"""

from __future__ import annotations

import argparse
import math
import os
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
import mujoco

from shelf_scene import ShelfConfig, Row, book_tilt_deg, build_scene, sample_row, settle

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_PANDA = os.path.join(HERE, "mujoco_menagerie", "franka_emika_panda", "panda.xml")

FINGERTIP = 0.1034        # hand origin -> closed fingertip, along the hand's z axis
FINGER_HALF = 0.012       # half-width of the closed fingertips
TOOL_RADIUS = 0.006


# ---------------------------------------------------------------------------
# 1. Scene: shelf_scene's episode + an optional pusher rod on the hand
# ---------------------------------------------------------------------------
def make_leaning_episode(cfg: ShelfConfig, rng: np.random.Generator, panda_xml: str,
                         tool_length: float, max_tries: int = 10, drift_tol: float = 2e-3):
    """Like shelf_scene.make_episode("leaning"), but edits the scene with
    MjSpec to add a pusher rod to the hand before compiling. (At the tucked
    home pose the rod is >10 cm from the shelf, so settling is unaffected.)"""
    for _ in range(max_tries):
        row = sample_row(cfg, "leaning", rng) # build a row of books
        spec = mujoco.MjSpec.from_string(build_scene(cfg, row, panda_xml)) # create the mujoco model with the row of books
        if tool_length > 0:
            hand = spec.body("panda/hand")
            hand.add_geom(name="pusher", type=mujoco.mjtGeom.mjGEOM_CAPSULE,
                          size=[TOOL_RADIUS, tool_length / 2, 0],
                          pos=[0, 0, FINGERTIP + tool_length / 2],
                          rgba=[0.9, 0.2, 0.2, 1], mass=0.02,
                          condim=3, friction=[0.3, 0.005, 0.0001])
        model = spec.compile()
        data = mujoco.MjData(model)
        rep = settle(model, data, row)
        if rep["max_drift_m"] < drift_tol and rep["max_tilt_deg"] < 60:   # = rep["stable"] at the default
            return model, data, row
    raise RuntimeError("no stable leaning row")


# ---------------------------------------------------------------------------
# 2. Geometry helpers (privileged state -> shelf frame)
# ---------------------------------------------------------------------------
class ShelfFrame:
    def __init__(self, row: Row):
        c, s = math.cos(row.shelf_yaw), math.sin(row.shelf_yaw)
        self.R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])   # shelf -> world rotation
        self.p = np.array(row.shelf_pos, dtype=float)

    def to_shelf(self, w):
        return (np.asarray(w) - self.p) @ self.R          # works for (3,) and (N,3)

    def to_world(self, s):
        return self.p + self.R @ np.asarray(s)


def book_corners(model, data, name, frame: ShelfFrame):
    """8 corners of a book's box, in the shelf frame."""
    g = model.geom(name + "_geom")
    half = model.geom_size[g.id]
    c, R = data.geom_xpos[g.id], data.geom_xmat[g.id].reshape(3, 3)
    signs = np.array([[i, j, k] for i in (-1, 1) for j in (-1, 1) for k in (-1, 1)])
    return frame.to_shelf(c + (signs * half) @ R.T)


def book_edge_points(model, data, name, frame, n=40):
    """Points along the 12 edges of a book box (shelf frame), for gap measurements."""
    C = book_corners(model, data, name, frame)
    pts = []
    for a in range(8):
        for b in range(a + 1, 8):
            if bin(a ^ b).count("1") == 1:                 # corners that share an edge
                t = np.linspace(0, 1, n)[:, None]
                pts.append(C[a] * (1 - t) + C[b] * t)
    return np.vstack(pts)


def slot_width(model, data, row: Row, frame: ShelfFrame, band_height: float) -> float:
    """Free width between the two books around the target slot, measured over
    the bottom `band_height` of the level (the part the new book needs)."""
    z0 = row.target_board_top
    left = book_edge_points(model, data, row.books[row.target_index - 1].name, frame)
    right = book_edge_points(model, data, row.books[row.target_index].name, frame)
    inband = lambda P: P[(P[:, 2] >= z0 + 0.003) & (P[:, 2] <= z0 + band_height)]
    L, Rr = inband(left), inband(right)
    if len(L) == 0 or len(Rr) == 0:
        return float("nan")
    return float(Rr[:, 0].min() - L[:, 0].max())


def face_x_at(model, data, name, frame, toward, z):
    """x of the book's side face that points `toward` (+1/-1 along x), at height z."""
    g = model.geom(name + "_geom")
    half = model.geom_size[g.id]
    R = frame.R.T @ data.geom_xmat[g.id].reshape(3, 3)     # book axes in shelf frame
    c = frame.to_shelf(data.geom_xpos[g.id])
    n = R[:, 0] * (1 if R[0, 0] * toward > 0 else -1)     # outward normal of that face
    pc = c + n * half[0]
    return float(pc[0] - n[2] * (z - pc[2]) / n[0])


# ---------------------------------------------------------------------------
# 3. Differential IK (damped least squares) on a scratch MjData
# ---------------------------------------------------------------------------
class TaskSpaceIK:
    def __init__(self, model, tcp_offset: float, prefix="panda/"):
        self.m, self.d = model, mujoco.MjData(model)
        self.hand = model.body(prefix + "hand").id
        self.off = tcp_offset
        jids = [model.joint(f"{prefix}joint{i}").id for i in range(1, 8)]
        self.qadr = np.array([model.jnt_qposadr[j] for j in jids])
        self.dadr = np.array([model.jnt_dofadr[j] for j in jids])
        self.lo, self.hi = model.jnt_range[jids, 0], model.jnt_range[jids, 1]

    def tcp(self, d):
        R = d.xmat[self.hand].reshape(3, 3)
        return d.xpos[self.hand] + R[:, 2] * self.off, d.xquat[self.hand].copy()

    def solve(self, q0, pos, quat, iters=10, damping=1e-2, max_step=0.1, rot_weight=0.5, roll_weight=0.1):
        """Damped least squares from q0 toward (pos, quat). Rotation about the
        tool axis (roll) is weighted by `roll_weight`: the pusher rod is round,
        so roll barely matters, and freeing it makes far poses reachable."""
        d = self.d
        q = np.array(q0, dtype=float)
        jacp, jacr = np.zeros((3, self.m.nv)), np.zeros((3, self.m.nv))
        res = np.zeros(3)
        for _ in range(iters):
            d.qpos[self.qadr] = q
            mujoco.mj_kinematics(self.m, d)
            mujoco.mj_comPos(self.m, d)
            p, qc = self.tcp(d)
            R = d.xmat[self.hand].reshape(3, 3)
            mujoco.mju_subQuat(res, quat, qc)                       # in the current hand frame
            z = R[:, 2]
            W = rot_weight * (np.eye(3) - (1 - roll_weight) * np.outer(z, z))   # down-weight roll
            err = np.concatenate([pos - p, W @ (R @ res)])
            if np.linalg.norm(err) < 1e-6:
                break
            mujoco.mj_jac(self.m, d, jacp, jacr, p, self.hand)
            J = np.vstack([jacp[:, self.dadr], W @ jacr[:, self.dadr]])
            dq = J.T @ np.linalg.solve(J @ J.T + damping ** 2 * np.eye(6), err)
            nrm = np.linalg.norm(dq)
            if nrm > max_step:
                dq *= max_step / nrm
            q = np.clip(q + dq, self.lo, self.hi)
        return q, self.pose_error(q, pos, quat)

    def pose_error(self, q, pos, quat):
        """(position error in m, tool-axis error in rad) at configuration q."""
        d = self.d
        d.qpos[self.qadr] = q
        mujoco.mj_kinematics(self.m, d)
        p, _ = self.tcp(d)
        R = d.xmat[self.hand].reshape(3, 3)
        Rd = np.zeros(9)
        mujoco.mju_quat2Mat(Rd, quat)
        axis = math.acos(np.clip(R[:, 2] @ Rd.reshape(3, 3)[:, 2], -1, 1))
        return float(np.linalg.norm(pos - p)), float(axis)

    def solve_global(self, q0, pos, quat, restarts=20, seed=0):
        """IK with restarts (from q0, then random perturbations of it).
        Returns the solution with the smallest error, preferring ones near q0."""
        rng = np.random.default_rng(seed)
        best = None
        for k in range(restarts):
            start = q0 if k == 0 else np.clip(q0 + rng.normal(0, 0.6, 7), self.lo, self.hi)
            q, (ep, ea) = self.solve(start, pos, quat, iters=300)
            score = ep + 0.05 * ea + 0.001 * np.linalg.norm(q - q0)
            if best is None or score < best[0]:
                best = (score, q, ep, ea)
            if ep < 1e-4 and ea < 0.02:
                break
        return best[1], best[2], best[3]


# ---------------------------------------------------------------------------
# 4. The plan: where to put the tool and which way to push
# ---------------------------------------------------------------------------
@dataclass
class LeanPlan:
    leaner: str
    lean_dir: int              # +1: leaner's top moved toward +x (into the slot)
    z: float                   # contact height (shelf frame)
    x_entry: float             # tool x when it goes into the wedge
    x_limit: float             # never push the tool past this x (leaner upright + 1 cm)
    y_out: float               # tool tip y in front of everything
    y_in: float                # tool tip y while pushing
    y_deep: float              # tool tip y for the low sliding push (near mid-depth)
    wedge_at_z: float


class NoWedge(RuntimeError):
    """The leaner lies flat against the book across the slot: there is no
    opening to push from, so this needs a different primitive."""


HAND_HALF_WIDTH = 0.11    # the Panda hand is ~21 cm wide across the fingers


def plan_unlean(model, data, row: Row, frame: ShelfFrame, contact_half: float,
                standoff=0.03, tip_to_hand=0.12) -> LeanPlan:
    """`tip_to_hand`: how far the hand's front sits behind the tool tip along y
    (rod + finger length, times cos(pitch)). The hand is wide, so the tool may
    only go in as far as the hand stays in front of every book it spans."""
    nd = row.nudges[0]
    leaner = row.books[nd["book"]]
    f = frame.R.T @ np.array(nd["force"])
    lean_dir = 1 if f[0] > 0 else -1
    far = row.books[row.target_index] if nd["book"] == row.target_index - 1 else row.books[row.target_index - 1]
    z0 = row.target_board_top

    C = book_corners(model, data, leaner.name, frame)
    top_z = C[:, 2].max()
    bottom = C[np.argsort(C[:, 2])[:4]]
    pivot_x = bottom[:, 0][np.argmax(lean_dir * bottom[:, 0])]  # gap-side bottom edge

    # Highest point of the wedge (below 60% of the leaner's height) the tool
    # fits in: higher = more leverage, but the wedge narrows toward the top.
    need = 2 * contact_half + 0.012
    best = None
    for z in np.arange(z0 + 0.035, z0 + 0.6 * (top_z - z0), 0.004):
        xl = face_x_at(model, data, leaner.name, frame, lean_dir, z)
        xf = face_x_at(model, data, far.name, frame, -lean_dir, z)
        w = (xf - xl) * lean_dir
        if w >= need:
            best = (z, xl, w)
    if best is None:
        raise NoWedge("leaner lies flat against the far book: no wedge to push from")
    z, xl, w = best

    # Book fronts (y of front face) near the slot. The tool goes 3 cm past the
    # leaner's front (deeper, near mid-depth, for the sliding push), but never
    # so deep that the hand would touch a book under it. The pre-approach
    # point is in front of all of them.
    front = lambda b: book_corners(model, data, b.name, frame)[:, 1].max()
    x_tool = xl + lean_dir * (contact_half + 0.004)
    near = [b for b in row.books if abs(b.x - x_tool) < HAND_HALF_WIDTH + b.thickness / 2 + 0.03]
    y_out = max(front(b) for b in near) + standoff
    hand_ok = max(front(b) for b in near) + 0.01 - tip_to_hand     # shallowest tip y the hand allows
    y_in = max(front(leaner) - 0.03, hand_ok)
    y_deep = max(front(leaner) - leaner.depth / 2, hand_ok)
    if y_in > front(leaner) - 0.008:
        raise NoWedge("a neighbour sticks out so far that the hand can't get the tool onto the leaner")

    return LeanPlan(leaner.name, lean_dir, float(z),
                    x_entry=float(x_tool),
                    x_limit=float(pivot_x + lean_dir * contact_half - lean_dir * 0.01),
                    y_out=float(y_out), y_in=float(y_in), y_deep=float(y_deep), wedge_at_z=float(w))


# ---------------------------------------------------------------------------
# 5. The scripted controller (a small state machine)
# ---------------------------------------------------------------------------
class ScriptedLeaningController:
    """Call step(data) every physics step; it writes data.ctrl (and the
    gravity compensation in data.qfrc_applied).

    Phases
      to_stage joint-space move to a staging pose 10 cm in front of the slot
      approach straight line to the shelf front
      insert   tool straight into the wedge under the leaning book
      unlean   push away from the slot until the leaner is upright (closed loop
               on its tilt: an upright book is stable, pushing further would
               topple the books behind it like dominoes)
      settle   short pause
      lower, deepen, widen, settle2   only if the slot is still narrower than
               the new book: push low on the now-upright book, sliding it and
               the books behind it into the spare row space (closed loop on slot
               width). If the book spins instead of sliding, retry once nearer
               its mid-depth.
      backoff, retract        let go and pull the tool out
      done
    """

    def __init__(self, model, data, row: Row, tool_length: float,
                 speed=0.10, push_speed=0.03, widen_speed=0.02,
                 clearance=0.008, max_widen=0.10, upright_deg=1.5, leash=0.015, pitch_deg=15.0,
                 spin_limit_deg=20.0):
        self.m, self.row = model, row
        self.frame = ShelfFrame(row)
        self.ik = TaskSpaceIK(model, FINGERTIP + tool_length)
        self.contact_half = TOOL_RADIUS if tool_length > 0 else FINGER_HALF
        self.speed, self.push_speed, self.widen_speed = speed, push_speed, widen_speed
        self.need, self.clearance, self.max_widen = row.target_book.thickness, clearance, max_widen
        self.upright_deg, self.leash = upright_deg, leash
        self.spin_limit_deg = spin_limit_deg
        self.widen_tries = 0
        self.z_low = row.target_board_top + 0.05      # low push for sliding books (hand clears the board)
        self.dt = model.opt.timestep

        # Tool orientation: pointing into the shelf (-y) and pitched down by pitch_deg,
        # finger opening along the row (x). Pointing straight forward this close
        # to the base is awkward for the Panda; 15 deg down makes every slot reachable.
        c, s_ = math.cos(math.radians(pitch_deg)), math.sin(math.radians(pitch_deg))
        Rx = np.array([[1, 0, 0], [0, c, -s_], [0, s_, c]])
        Rs = Rx @ np.array([[0, 1, 0], [0, 0, -1], [-1, 0, 0.0]])  # columns: hand x, y, z in shelf frame
        self.quat = np.zeros(4)
        mujoco.mju_mat2Quat(self.quat, (self.frame.R @ Rs).flatten())

        # Plan, then find reachable joint configurations for a staging point
        # 10 cm in front of the slot and the pre-approach point at the shelf
        # front. Only the move to the staging point is in joint space (it can
        # swing the tool around); everything after it is a straight line.
        self.q_home = data.qpos[self.ik.qadr].copy()
        for standoff in (0.03, 0.015):
            self.plan = plan_unlean(model, data, row, self.frame, self.contact_half, standoff,
                                    tip_to_hand=(tool_length + 0.045) * math.cos(math.radians(pitch_deg)))
            p = self.plan
            self.stage = np.array([p.x_entry, p.y_out + 0.10, p.z + 0.02])
            self.target = np.array([p.x_entry, p.y_out, p.z])
            self.q_stage, err, ang = self.ik.solve_global(self.q_home, self.frame.to_world(self.stage), self.quat)
            if err < 0.005 and ang < 0.1:
                _, err, ang = self.ik.solve_global(self.q_stage, self.frame.to_world(self.target), self.quat)
                if err < 0.005 and ang < 0.1:
                    break
        else:
            raise RuntimeError(f"pre-approach pose out of reach (IK error {err*1000:.0f} mm, "
                               f"{math.degrees(ang):.0f} deg)")

        self.q_des = self.q_home.copy()
        self.setpoint = None
        self.phase, self.t_phase = "to_stage", 0.0
        self.T_joint = 2.5
        self.push_start_x = None
        self.widen_stop = None                    # why the sliding push ended (for logging)
        self._gap, self._tick = None, 0

    # -- measurements (privileged: read from the simulator) ------------------
    def slot_width(self, data):
        return slot_width(self.m, data, self.row, self.frame, self.row.target_book.height)

    def leaner_tilt(self, data):
        return book_tilt_deg(data.xquat[self.m.body(self.plan.leaner).id])

    def leaner_yaw(self, data):
        """Rotation of the leaner about the vertical, in degrees (0 = square to the shelf)."""
        R = self.frame.R.T @ data.xmat[self.m.body(self.plan.leaner).id].reshape(3, 3)
        return abs(math.degrees(math.atan2(R[1, 0], R[0, 0])))

    def tcp_shelf(self, data):
        R = data.xmat[self.ik.hand].reshape(3, 3)
        return self.frame.to_shelf(data.xpos[self.ik.hand] + R[:, 2] * self.ik.off)

    # -- motion helpers -------------------------------------------------------
    def _track(self):
        """Joint targets for the current Cartesian setpoint (3 IK iterations,
        warm-started from the previous targets)."""
        self.q_des, _ = self.ik.solve(self.q_des, self.frame.to_world(self.setpoint), self.quat, iters=3)

    def _goto(self, target, speed):
        """Move the setpoint toward target at `speed`; True once there."""
        delta = target - self.setpoint
        dist = np.linalg.norm(delta)
        step = speed * self.dt
        self.setpoint = target.copy() if dist <= step else self.setpoint + delta / dist * step
        self._track()
        return dist <= step

    def _push(self, data, direction, speed):
        """Advance the setpoint along x by `direction`, but never more than
        `leash` ahead of the actual tool. This caps the force on the books:
        a blocked push stops instead of winding up the joint controllers."""
        lag = (self.setpoint[0] - self.tcp_shelf(data)[0]) * direction
        if lag < self.leash:
            self.setpoint[0] += direction * speed * self.dt
        self._track()

    def _next(self, phase):
        self.phase, self.t_phase = phase, 0.0

    def _start_backoff(self, data):
        tcp = self.tcp_shelf(data)
        self.backoff_to = np.array([tcp[0] + self.plan.lean_dir * 0.006, tcp[1], tcp[2]])
        self._next("backoff")

    # -- main entry -----------------------------------------------------------
    def step(self, data):
        self.t_phase += self.dt
        p, ph = self.plan, self.phase
        away = -p.lean_dir                       # push direction: away from the slot

        if ph == "to_stage":
            s = min(1.0, self.t_phase / self.T_joint)
            s = s * s * (3 - 2 * s)                                     # smoothstep
            self.q_des = (1 - s) * self.q_home + s * self.q_stage
            if self.t_phase >= self.T_joint + 0.5:
                self.setpoint = self.stage.copy()
                self._next("approach")

        elif ph == "approach":
            if self._goto(self.target, self.speed):
                self._next("insert")

        elif ph == "insert":
            if self._goto(np.array([p.x_entry, p.y_in, p.z]), self.speed):
                self._next("unlean")

        elif ph == "unlean":
            self._push(data, away, self.push_speed)
            upright = self.leaner_tilt(data) < self.upright_deg
            at_limit = (self.setpoint[0] - p.x_limit) * away >= 0
            if upright or at_limit:
                self.setpoint = self.tcp_shelf(data)                    # stop where the tool is
                self._next("settle")

        elif ph == "settle":
            self._track()
            if self.t_phase > 0.3:
                if self.need + self.clearance - self.slot_width(data) > 0.002:
                    # Ease off, then go low. First try near the book's front; if the
                    # book spins, retry once near its mid-depth (spins it less, but
                    # brings the wide hand closer to the neighbours).
                    x = self.tcp_shelf(data)[0] - away * 0.003
                    self.lower_to = np.array([x, p.y_in, self.z_low])
                    self.deep_to = np.array([x, p.y_in, self.z_low])
                    self.yaw0 = self.leaner_yaw(data)
                    self._next("lower")
                else:
                    self._start_backoff(data)

        elif ph == "lower":
            if self._goto(self.lower_to, self.push_speed):
                self._next("deepen")

        elif ph == "deepen":
            if self._goto(self.deep_to, self.push_speed):
                self.push_start_x = self.setpoint[0]
                self._next("widen")

        elif ph == "widen":
            self._push(data, away, self.widen_speed)
            self._tick += 1
            if self._tick % 25 == 0:                                    # gap check at 20 Hz
                self._gap = self.slot_width(data)
            wide = self._gap is not None and self._gap >= self.need + self.clearance
            too_far = (self.setpoint[0] - self.push_start_x) * away >= self.max_widen
            spinning = self.leaner_yaw(data) - self.yaw0 > self.spin_limit_deg
            if spinning and not wide and self.widen_tries == 0 and p.y_deep < p.y_in - 0.01:
                self.widen_tries = 1                                    # retry once, deeper
                tcp = self.tcp_shelf(data)
                x = tcp[0] - away * 0.005
                self.lower_to = np.array([x, p.y_in, self.z_low])       # ease off first
                self.deep_to = np.array([x, p.y_deep, self.z_low])
                self.yaw0 = self.leaner_yaw(data)
                self._next("lower")
            elif wide or too_far or spinning:
                self.widen_stop = "wide enough" if wide else ("max travel" if too_far else "book started to spin")
                self.setpoint = self.tcp_shelf(data)
                self._next("settle2")

        elif ph == "settle2":
            self._track()
            if self.t_phase > 0.3:
                self._start_backoff(data)

        elif ph == "backoff":                                           # release the contact first
            if self._goto(self.backoff_to, self.push_speed):
                self._next("retract")

        elif ph == "retract":
            if self._goto(np.array([self.backoff_to[0], p.y_out, self.backoff_to[2]]), self.speed):
                self._next("done")
        # "done": hold the last joint targets

        data.ctrl[:7] = self.q_des
        data.ctrl[7] = 0.0                                              # gripper closed
        # Gravity/Coriolis compensation on the arm (the real Franka does this internally).
        data.qfrc_applied[self.ik.dadr] = data.qfrc_bias[self.ik.dadr]


# ---------------------------------------------------------------------------
# 6. Episode runner + metrics
# ---------------------------------------------------------------------------
def robot_contacts(model, data, robot_root):
    """Number of contacts between the robot and anything else."""
    n = 0
    for c in data.contact[:data.ncon]:
        r1 = model.body_rootid[model.geom_bodyid[c.geom1]] == robot_root
        r2 = model.body_rootid[model.geom_bodyid[c.geom2]] == robot_root
        n += r1 != r2
    return n


def run_episode(seed: int, panda_xml=DEFAULT_PANDA, tool_length=0.08, view=False,
                max_time=20.0, cfg: Optional[ShelfConfig] = None, drift_tol=2e-3, **ctl_kwargs) -> dict:
    cfg = cfg or ShelfConfig()
    rng = np.random.default_rng(seed)
    model, data, row = make_leaning_episode(cfg, rng, panda_xml, tool_length, drift_tol=drift_tol)
    frame = ShelfFrame(row)
    band = row.target_book.height
    need = row.target_book.thickness

    ids = [model.body(b.name).id for b in row.books]
    pos0, tilt0 = data.xpos[ids].copy(), [book_tilt_deg(data.xquat[i]) for i in ids]
    gap0 = slot_width(model, data, row, frame, band)

    ctl = ScriptedLeaningController(model, data, row, tool_length, **ctl_kwargs)
    robot_root = model.body_rootid[ctl.ik.hand]
    leaner_i = [b.name for b in row.books].index(ctl.plan.leaner)
    tilt_leaner0 = tilt0[leaner_i]
    approach_hits, phase_log = 0, []

    viewer = None
    if view:
        from mujoco import viewer as mj_viewer
        viewer = mj_viewer.launch_passive(model, data)
        viewer.cam.fixedcamid, viewer.cam.type = model.camera("scene_cam").id, mujoco.mjtCamera.mjCAMERA_FIXED

    t_end_extra = None
    t0 = data.time                       # settling already advanced the clock
    while data.time - t0 < max_time:
        ctl.step(data)
        mujoco.mj_step(model, data)
        if ctl.phase in ("to_stage", "approach"):
            approach_hits = max(approach_hits, robot_contacts(model, data, robot_root))
        if not phase_log or phase_log[-1][0] != ctl.phase:
            phase_log.append((ctl.phase, round(data.time, 2)))
        if ctl.phase == "done":
            t_end_extra = t_end_extra or data.time + 1.0                # let things settle 1 s
            if data.time > t_end_extra:
                break
        if viewer is not None:
            if not viewer.is_running():
                break
            viewer.sync()
            time.sleep(model.opt.timestep)

    gap1 = slot_width(model, data, row, frame, band)
    tilt1 = [book_tilt_deg(data.xquat[i]) for i in ids]
    moved = np.linalg.norm(data.xpos[ids] - pos0, axis=1)
    others = [i for i in range(len(ids)) if i != leaner_i]
    result = {
        "seed": seed,
        "finished": ctl.phase == "done",
        "phases": phase_log,
        "need_mm": 1000 * need,
        "gap_before_mm": 1000 * gap0,
        "gap_after_mm": 1000 * gap1,
        "leaner_tilt_before": tilt_leaner0,
        "leaner_tilt_after": tilt1[leaner_i],
        "max_other_tilt_change": max(abs(tilt1[i] - tilt0[i]) for i in others),
        "max_other_move_mm": 1000 * max(moved[i] for i in others),
        "contact_z_above_board_mm": 1000 * (ctl.plan.z - row.target_board_top),
        "robot_hit_during_approach": approach_hits > 0,
        "widen_stop": ctl.widen_stop,
    }
    result["success"] = bool(result["finished"] and gap1 >= need + 0.004
                             and result["leaner_tilt_after"] < 8
                             and result["max_other_tilt_change"] < 10)
    if viewer is not None:
        while viewer.is_running():                                      # keep the window open
            viewer.sync()
            time.sleep(0.02)
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seeds", type=int, help="evaluate seeds 0..N-1 instead of one episode")
    ap.add_argument("--panda", default=DEFAULT_PANDA)
    ap.add_argument("--tool-length", type=float, default=0.08, help="pusher rod length (0 = bare fingers)")
    ap.add_argument("--drift-tol", type=float, default=2e-3,
                    help="max book drift (m) allowed when accepting a settled row")
    ap.add_argument("--spin-limit", type=float, default=20.0,
                    help="stop (or retry deeper) the sliding push if the book yaws this much (deg)")
    ap.add_argument("--view", action="store_true", help="open the viewer (macOS: run with mjpython)")
    a = ap.parse_args()

    seeds = range(a.seeds) if a.seeds else [a.seed]
    results = []
    for s in seeds:
        try:
            r = run_episode(s, a.panda, a.tool_length, view=a.view and not a.seeds,
                            drift_tol=a.drift_tol, spin_limit_deg=a.spin_limit)
        except RuntimeError as e:
            r = {"seed": s, "success": False, "error": str(e)}     # NoWedge, unreachable, unstable row
        results.append(r)
        if "error" in r:
            print(f"seed {s:3d}  ERROR {r['error']}")
        else:
            print(f"seed {s:3d}  {'OK  ' if r['success'] else 'FAIL'}  gap {r['gap_before_mm']:5.1f} -> "
                  f"{r['gap_after_mm']:5.1f} mm (need {r['need_mm']:4.1f})  leaner tilt "
                  f"{r['leaner_tilt_before']:4.1f} -> {r['leaner_tilt_after']:4.1f} deg  "
                  f"others: tilt {r['max_other_tilt_change']:4.1f} deg, move {r['max_other_move_mm']:4.1f} mm"
                  f"{'  [hit on approach]' if r['robot_hit_during_approach'] else ''}"
                  f"{'  [widen: ' + r['widen_stop'] + ']' if r['widen_stop'] else ''}")
    if len(results) > 1:
        ok = sum(r["success"] for r in results)
        print(f"\nsuccess {ok}/{len(results)}")


if __name__ == "__main__":
    main()
