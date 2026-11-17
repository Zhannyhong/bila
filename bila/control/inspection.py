"""
Where the robot looks from before it touches the row: the inspection pose.

The wrist camera looks straight into the shelf from CAM_HEIGHT above the target
board, centred on the target slot, or moved along the row by `dx` for a look
further to one side. It goes as far back as the arm allows, to see as much of
the row as it can: about 50 cm from the spines if it can, else 45 or 40 cm
(STANDOFFS). The robot's base stands about 80 cm from the shelf, so from 50 cm
the arm has to fold back toward it, and it can't always: in 48 test scenes it
reached 50 cm in 43 and 45 cm in the other 5.

Further back sees more of the row. On the same 32 scenes, from 50 cm the camera
saw 93% of the row's spines in full on average and the whole row in 41% of
scenes; from 30 cm, 69% and 3%. The price is resolution: about 1.5 mm per
pixel at 640 px wide, against 0.9 mm from 30 cm.

The camera is rolled 180 deg, so its image comes out upside down: rotate it
before use. Held the right way up, the camera needs a hand pose the Panda often
can't reach in front of a shelf (its wrist bends one way only): from 30 cm it
reached the centre view in only half the scenes. Rolled, it reached every view
tried, from 30 cm left to 30 cm right of the slot.

Getting the arm there is not modelled: `look_from` puts it at the pose
directly, as if a planner or the mobile base had brought it there.

    from bila.control.inspection import solve_inspection, look_from
    q, standoff = solve_inspection(model, data, row)   # None if no safe arm pose reaches any of them
    look_from(model, data, q)                          # wrist_cam now looks at the slot
"""

from __future__ import annotations

from typing import Optional, Tuple

import mujoco
import numpy as np

from bila.vision.camera import CAMERA
from bila.control.insert_book import MIN_MARGIN, ArmIK, ShelfFrame
from bila.sim.shelf_scene import PANDA_HOME, Row

STANDOFFS = (0.48, 0.43, 0.38)  # camera this far in front of the board's front edge, furthest tried first
                                # (spines are 1-3 cm behind the edge: about 50, 45, 40 cm from them)
CAM_HEIGHT = 0.15    # camera this high above the target board (about the middle of the books)
UPSIDE_DOWN = True   # camera rolled 180 deg: reachable far more often (see above)
IK_SEEDS = 48        # random arm configurations tried, besides the ready pose (12 found the 50 cm
                     # view in only 2 of 12 scenes, 48 in 10)


def slot_x(row: Row) -> float:
    """x of the target slot along the row, from the shelf map."""
    return (row.gap_left + row.gap_right) / 2


def camera_target(row: Row, dx: float = 0.0, standoff: float = STANDOFFS[0], height: float = CAM_HEIGHT,
                  upside_down: bool = UPSIDE_DOWN) -> Tuple[np.ndarray, np.ndarray]:
    """World position and rotation for the wrist camera: in front of the slot
    (moved `dx` along the row), looking straight into the shelf. MuJoCo
    camera axes: x right, y up, z back toward the robot (it looks along -z)."""
    shelf = ShelfFrame(row)
    pos = shelf.to_world(np.array([slot_x(row) + dx, row.shelf_depth / 2 + standoff, row.target_board_top + height]))
    x, y = ((1, 0, 0), (0, 0, -1)) if upside_down else ((-1, 0, 0), (0, 0, 1))
    return pos, shelf.R @ np.column_stack([x, y, (0, 1, 0)]).astype(float)


def camera_in_hand(model: mujoco.MjModel) -> Tuple[np.ndarray, np.ndarray]:
    """The wrist camera's rotation and position in the hand frame (fixed: the
    camera is a child of the hand body)."""
    cam = model.camera(CAMERA).id
    assert model.cam_bodyid[cam] == model.body("panda/hand").id, "wrist_cam must be attached to the hand"
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, model.cam_quat[cam])
    return R.reshape(3, 3), model.cam_pos[cam].copy()


def solve_view(model: mujoco.MjModel, data: mujoco.MjData, row: Row, dx: float = 0.0,
               ik: Optional[ArmIK] = None, **target) -> Optional[np.ndarray]:
    """Arm joint angles (7) that put the wrist camera at camera_target(row, dx,
    **target), without touching anything in the scene as it is in `data`, and
    with every joint at least MIN_MARGIN from its limits. Of the configurations
    found from several starting guesses, keeps the one farthest from the
    limits. None if there is none."""
    ik = ik or ArmIK(model, PANDA_HOME[:7])
    ik.sync(data)
    cam_pos, cam_rot = camera_target(row, dx, **target)
    R_hc, p_hc = camera_in_hand(model)
    hand_rot = cam_rot @ R_hc.T
    hand_pos = cam_pos - hand_rot @ p_hc
    rng = np.random.default_rng(0)
    seeds = [np.array(PANDA_HOME[:7])] + [rng.uniform(ik.lo, ik.hi) for _ in range(IK_SEEDS)]
    best = None
    for seed in seeds:
        q, converged = ik.solve_far(seed, hand_pos, hand_rot)
        if converged and not ik.collides(q) and (best is None or ik.margin(q) > ik.margin(best)):
            best = q
    return best if best is not None and ik.margin(best) >= MIN_MARGIN else None


def solve_inspection(model: mujoco.MjModel, data: mujoco.MjData, row: Row, dx: float = 0.0,
                     ik: Optional[ArmIK] = None) -> Optional[Tuple[np.ndarray, float]]:
    """The furthest-back view the arm reaches: (joint angles, standoff) for the
    first of STANDOFFS that solve_view finds, or None if it finds none."""
    ik = ik or ArmIK(model, PANDA_HOME[:7])
    for standoff in STANDOFFS:
        q = solve_view(model, data, row, dx, ik, standoff=standoff)
        if q is not None:
            return q, standoff
    return None


def solve_camera(model: mujoco.MjModel, data: mujoco.MjData, ik: ArmIK, cam_pos, cam_rot, rng,
                 seeds: int = IK_SEEDS) -> Optional[np.ndarray]:
    """Arm joint angles that put the wrist camera at cam_pos with MuJoCo
    camera axes cam_rot (columns x right, y up, z back), collision-free and
    clear of the joint limits; None if none is found. solve_view for any
    camera pose, from the ready pose and `seeds` random starts."""
    R_hc, p_hc = camera_in_hand(model)
    hand_rot = cam_rot @ R_hc.T
    hand_pos = cam_pos - hand_rot @ p_hc
    ik.sync(data)
    best = None
    for seed in [np.array(PANDA_HOME[:7])] + [rng.uniform(ik.lo, ik.hi) for _ in range(seeds)]:
        q, converged = ik.solve_far(seed, hand_pos, hand_rot)
        if converged and not ik.collides(q) and (best is None or ik.margin(q) > ik.margin(best)):
            best = q
    return best if best is not None and ik.margin(best) >= MIN_MARGIN else None


def looking(forward, ref) -> np.ndarray:
    """MuJoCo camera axes (columns x right, y up, z back) looking along
    `forward`, with its image's right along ref projected square to forward."""
    f = np.asarray(forward, float) / np.linalg.norm(forward)
    x = np.asarray(ref, float) - np.dot(ref, f) * f
    x /= np.linalg.norm(x)
    z = -f
    return np.column_stack([x, np.cross(z, x), z])


def look_from(model: mujoco.MjModel, data: mujoco.MjData, q) -> None:
    """Put the arm at joint angles q, at rest, and command it to hold there."""
    jids = [model.joint(f"panda/joint{i}").id for i in range(1, 8)]
    data.qpos[model.jnt_qposadr[jids]] = q
    data.qvel[model.jnt_dofadr[jids]] = 0
    data.ctrl[[model.actuator(f"panda/actuator{i}").id for i in range(1, 8)]] = q   # (the cart's lift comes first)
    mujoco.mj_forward(model, data)
