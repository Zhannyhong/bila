"""
What the gap-creation policy sees: fixed-size vectors built from a
ShelfEstimate (what the robot knows about the row) and the robot's own readings.

    from bila.rl.observation import ObservationBuilder, robot_state
    obs = ObservationBuilder(est, target=(thickness, height))      # once, after the look
    x = obs.actor(est, robot_state(model, data), now=data.time)    # every control step
    v = obs.critic(model, data, truth)                             # privileged extras, simulation only
    obs.actor_layout["gap_narrowest"]                              # where a feature sits in x

Everything is in the slot frame, fixed when the robot looks: origin at the slot
(along the row, from the estimate), the board's front edge (out of the shelf)
and the board's top (up), with the shelf's axes. So the same situation gives
the same numbers wherever the shelf, its level or the robot base is.

Actor: what a real robot has, so the policy can run on camera estimates.
  - The slot's neighbours and the books next to them (offsets -3..-1, +1..+3):
    spine centre, thickness, height, lean, swivel and visible, and whether the
    book was seen at all (if not, its numbers are 0).
  - The gap at each measured height (unseen means wider than the camera saw:
    GAP_CAP), whether each height is within the book to insert, the narrowest
    gap up to its height, and how much wider it must get (book + GAP_MARGIN).
    The per-height widths jump where a book's top is near a height; the
    narrowest gap doesn't, so lean on it.
  - Free space behind each neighbour (perception.free_space): books before
    it, its width, and whether the camera saw far enough to tell.
  - The book to insert (thickness, height), and the estimate's age: the robot
    looks once, and the row changes while it pushes.
  - Robot: arm joint angles and speeds, gripper opening, the tool tip's
    position, orientation (two columns of its rotation) and velocity in the
    slot frame, and the wrist force/torque in the hand frame.
The environment appends what only it knows: the base controller's phase and
action, time left, the last residual action.

Critic (simulation only, for training): the same book and gap features from
the ground truth now, not from the look, plus each nearby book's mass,
friction and velocity.

All values in SI units (m, rad, N, s); normalise them in the RL wrapper.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import mujoco
import numpy as np

from bila.control.insert_book import GAP_MARGIN
from bila.vision.perception import FREE_SPACE_CAP, ShelfEstimate, free_space

BOOK_OFFSETS = (-3, -2, -1, 1, 2, 3)      # the books the policy sees, by place relative to the slot
GAP_CAP = 0.10                            # m: gap widths are capped here; unseen counts as this wide
TOOL_OFFSET = 0.1034                      # m: tool tip along the hand's z axis from the hand frame (fingertips)


# ---------------------------------------------------------------------------
# Building vectors with named parts
# ---------------------------------------------------------------------------
class Parts:
    """Collects named arrays into one vector and remembers where each sits."""

    def __init__(self):
        self.values: List[np.ndarray] = []
        self.layout: Dict[str, slice] = {}
        self.size = 0

    def add(self, name: str, value) -> None:
        v = np.atleast_1d(np.asarray(value, dtype=np.float64)).ravel()
        self.values.append(v)
        self.layout[name] = slice(self.size, self.size + len(v))
        self.size += len(v)

    def vector(self) -> np.ndarray:
        return np.concatenate(self.values).astype(np.float32)


# ---------------------------------------------------------------------------
# Robot readings
# ---------------------------------------------------------------------------
@dataclass
class RobotState:
    """The robot's own readings, in the world frame (a real robot fills this
    from its drivers)."""
    joint_pos: np.ndarray      # (7,) rad
    joint_vel: np.ndarray      # (7,) rad/s
    gripper: float             # m, opening between the fingers
    tool_pos: np.ndarray       # (3,) tool tip position
    tool_rot: np.ndarray       # (3, 3) tool (hand) orientation
    tool_vel: np.ndarray       # (6,) tool tip linear (3) and angular (3) velocity
    wrench: np.ndarray         # (6,) wrist force (3) and torque (3), in the hand frame


def robot_state(model: mujoco.MjModel, data: mujoco.MjData, tool_offset: float = TOOL_OFFSET,
                prefix: str = "panda/") -> RobotState:
    """RobotState from the simulation. The tool tip is tool_offset along the
    hand's z axis (the fingertips by default; set it for a pusher rod)."""
    joints = [model.joint(f"{prefix}joint{i}").id for i in range(1, 8)]
    fingers = [model.joint(f"{prefix}finger_joint{i}").id for i in (1, 2)]
    hand = model.body(f"{prefix}hand").id
    R = data.xmat[hand].reshape(3, 3)
    tip = data.xpos[hand] + R[:, 2] * tool_offset
    jacp, jacr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
    mujoco.mj_jac(model, data, jacp, jacr, tip, hand)
    force = data.sensordata[model.sensor(f"{prefix}ft_force").adr[0]:][:3]
    torque = data.sensordata[model.sensor(f"{prefix}ft_torque").adr[0]:][:3]
    return RobotState(
        joint_pos=data.qpos[model.jnt_qposadr[joints]].copy(),
        joint_vel=data.qvel[model.jnt_dofadr[joints]].copy(),
        gripper=float(data.qpos[model.jnt_qposadr[fingers]].sum()),
        tool_pos=tip.copy(), tool_rot=R.copy(),
        tool_vel=np.concatenate([jacp @ data.qvel, jacr @ data.qvel]),
        wrench=np.concatenate([force, torque]))


# ---------------------------------------------------------------------------
# The observation
# ---------------------------------------------------------------------------
class ObservationBuilder:
    """Builds the actor's and the critic's vectors for one episode. Make it
    once, from the estimate taken when the robot looks: that fixes the slot
    frame for the episode."""

    def __init__(self, est: ShelfEstimate, target: Tuple[float, float], x_slot_map: Optional[float] = None):
        x_slot = est.slot_x()
        if math.isnan(x_slot):                    # a neighbour unseen: fall back on the shelf map
            x_slot = x_slot_map if x_slot_map is not None else 0.0
        self.origin = np.array([x_slot, est.front_y, est.board_top])   # slot frame origin, in the shelf frame
        self.R_shelf = est.shelf_pose[:3, :3]                           # world <- shelf (= slot frame axes)
        self.p_shelf = est.shelf_pose[:3, 3]
        self.target = target
        self.actor_layout: Dict[str, slice] = {}
        self.critic_layout: Dict[str, slice] = {}

    # --- frames -----------------------------------------------------------------
    def to_slot(self, p_world: np.ndarray) -> np.ndarray:
        """World point(s) -> slot frame."""
        return (np.asarray(p_world) - self.p_shelf) @ self.R_shelf - self.origin

    def rot_to_slot(self, R_world: np.ndarray) -> np.ndarray:
        return self.R_shelf.T @ R_world

    # --- shelf features ---------------------------------------------------------------
    def add_shelf(self, parts: Parts, est: ShelfEstimate, prefix: str = "") -> None:
        """Book, gap and free-space features of an estimate (shelf frame -> slot frame)."""
        for k in BOOK_OFFSETS:
            b = est.book(k)
            name = f"{prefix}book{k:+d}"
            if b is None:
                parts.add(name + "_seen", 0.0)
                parts.add(name, np.zeros(8))
                continue
            centre = b.spine_corners.mean(axis=0) - self.origin
            parts.add(name + "_seen", 1.0)
            parts.add(name, [*centre, b.thickness, b.height, b.lean, b.yaw, b.visible])
        widths = np.where(np.isnan(est.gap_widths), GAP_CAP, np.minimum(est.gap_widths, GAP_CAP))
        narrowest = est.min_gap_width(self.target[1])
        narrowest = GAP_CAP if math.isnan(narrowest) else min(narrowest, GAP_CAP)
        parts.add(prefix + "gap_widths", widths)
        parts.add(prefix + "gap_relevant", (est.gap_heights <= self.target[1]).astype(float))
        parts.add(prefix + "gap_narrowest", narrowest)
        parts.add(prefix + "gap_widening_needed", max(0.0, self.target[0] + GAP_MARGIN - narrowest))
        for side, name in ((-1, "left"), (1, "right")):
            fs = free_space(est, side)
            parts.add(f"{prefix}free_space_{name}", [fs.books / FREE_SPACE_CAP, fs.width, float(fs.resolved)])

    # --- the two vectors ----------------------------------------------------------------
    def actor(self, est: ShelfEstimate, robot: RobotState, now: float) -> np.ndarray:
        """What the policy sees: the estimate (taken at est.stamp) and the robot now."""
        parts = Parts()
        self.add_shelf(parts, est)
        parts.add("target_book", self.target)
        parts.add("estimate_age", now - est.stamp)
        R = self.rot_to_slot(robot.tool_rot)
        parts.add("joint_pos", robot.joint_pos)
        parts.add("joint_vel", robot.joint_vel)
        parts.add("gripper", robot.gripper)
        parts.add("tool_pos", self.to_slot(robot.tool_pos))
        parts.add("tool_rot", R[:, :2].T)                    # first two columns: a smooth 6-number rotation
        parts.add("tool_vel", np.concatenate([self.R_shelf.T @ robot.tool_vel[:3], self.R_shelf.T @ robot.tool_vel[3:]]))
        parts.add("wrench", robot.wrench)
        self.actor_layout = parts.layout
        return parts.vector()

    def critic(self, model: mujoco.MjModel, data: mujoco.MjData, truth: ShelfEstimate,
               book_names: Dict[int, str]) -> np.ndarray:
        """Privileged extras for the critic (simulation only): the ground truth's
        book and gap features now, and each nearby book's mass, friction and
        velocity (linear, angular; slot axes). book_names maps a slot offset to
        the book's body name in the model."""
        parts = Parts()
        self.add_shelf(parts, truth, prefix="true_")
        vel = np.zeros(6)
        for k in BOOK_OFFSETS:
            name = book_names.get(k)
            if name is None:
                parts.add(f"true_book{k:+d}_physics", np.zeros(8))
                continue
            body = model.body(name).id
            geom = model.geom(name + "_geom").id
            mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, body, vel, 0)   # (angular, linear), world axes
            parts.add(f"true_book{k:+d}_physics", [model.body_mass[body], model.geom_friction[geom, 0],
                                                   *(self.R_shelf.T @ vel[3:]), *(self.R_shelf.T @ vel[:3])])
        self.critic_layout = parts.layout
        return parts.vector()


def book_names_by_offset(row) -> Dict[int, str]:
    """Slot offset -> body name of each book in the target row (simulation)."""
    ti = row.target_index
    return {(i - ti if i < ti else i - ti + 1): b.name for i, b in enumerate(row.books)}
