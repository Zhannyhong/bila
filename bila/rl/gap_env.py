"""
A Gymnasium environment for the reshelving residual: the scripted controller
(insert_book.Episode.reshelve) makes room for the book and inserts it, and the
policy corrects its hand during the contact moves (Runner.contact): making
room (widen's pushes, the lift and the clamp's swing, the shove) and the
insertion (resting the book on the board and sliding it in, then pushing it
home).

    uv run python -m bila.rl.gap_env pool --scenario tight --seeds 1000 --workers 12   # build the training scenes once
    uv run python -m bila.rl.gap_env pool --scenario tight --seeds 40 --first-seed 20000 --name tight_val   # validation
    uv run python -m bila.rl.gap_env pool --scenario tight --seeds 100 --perception camera \
        --segmenter bila_weights/segmenter/segmenter.pt --name tight_camera      # what the robot sees
    from bila.rl.gap_env import GapEnv
    env = GapEnv(scenarios=("clear", "tight"))                       # loads scenes/gap_pool_<scenario>.pkl
    env = GapEnv(scenarios=("tight",), first_phases=("widen",), insert=False)   # gap creation only
    env = GapEnv(scenarios=("clear", "tight"), control_phases=("slide", "push"), learn_stop=False)
    # Scripted preparation, then five continuous insertion corrections. No learned stopping.

One env step is one residual decision (every RESIDUAL_EVERY physics steps,
25 Hz). The scripted episode runs in a worker thread: when its Runner asks
the residual for a correction (decide), the thread hands the decision to the
env and waits; env.step(action) passes the action back and runs the thread
on to its next decision. The script keeps everything else (which step, the
plans, the looks, every move outside a contact), so between two decisions it
may do a lot: a look, a move to the next step.

An episode is the whole reshelve, from the settled row to the script's score
(Episode.score), or ends on a failure. With insert=False it covers gap
creation only: it ends when the script would insert the book (Episode.insert
is replaced with a stop).

Errors the policy has to make up for:
  - perception: the script and the policy both see the row through the same
    eyes (perception=):
      noisy          NoisyEyes, a NoisyPerception (the wrist camera's
                     measured errors, and one rigid shift of everything it
                     sees per episode, as a hand-eye calibration error gives;
                     no rendering) behind the interface Episode uses for its
                     camera. The arm still moves to the views;
      noisy-learned  the same, imitating the learned masks' pipeline;
      camera         the rendered wrist camera itself (rgbd_perception.make_eyes:
                     a D405's stereo errors and a calibration error drawn per
                     episode), with the simulation's book masks or, with
                     segmenter=WEIGHTS, the learned ones (learned_segmenter.py,
                     loaded once per process). What a robot would see: the
                     noise models fail the script as often but in other ways
                     (eval_reshelve.py), so a policy should train on this.
    A pool records how its first runs saw the rows (PoolEntry.perception),
    and an env only takes a pool that saw them as it does;
  - the grasp (GraspError): the book sits a little off in the fingers from
    where the script believes it took it (further out or deeper in, higher
    or lower, turned about the fingers' axis), which it doesn't know of.
Both are drawn per scene/error variant, the same every time it is used.
Pool generation can make several independently seeded variants of each layout
(--error-variants and --error-seed). reset also accepts explicit error seeds.

Observation (a Dict):
  actor   observation.ObservationBuilder's actor vector from the last look
          (what a real robot has), then (actor_layout): the contact phase
          (one-hot over PHASES), time in it, the script's tool target
          (position and two columns of its rotation, slot frame), the
          correction so far (over its limits), the last action, whether the
          move may be ended now, the wrist force/torque's change since the
          move started, and while sliding the book in, where the robot
          believes the book is.
  critic  privileged extras only (the ground truth now, the books' physics),
          then the grasp error, and once inserting: the book's true position
          and how far it is from the believed one, its spine's distance to
          acceptable placement, and the contact forces on it. A critic built for it reads
          both.
Action: (6,) in [-1, 1], all times action_scale (set_action_scale ramps it
up), then the end action: above RESIDUAL_END it ends the move where the hand
is (the slide only once the book is over the board; Runner.contact). The first
five parts (x, y, z in the shelf frame, yaw, pitch):
  making room   the increments of the correction at this decision: times
                RESIDUAL_STEP and RESIDUAL_TURN_STEP;
  insertion     the correction itself (Contact.target): 0 the script's pose,
                +-1 the move's bounds on that side (insert_book.SLIDE_LO...,
                PUSH_LO...), reached at RESIDUAL_STEPS a decision at most. A
                steady small action stays a small correction instead of adding
                up to the bounds, and the action cost is on the correction.
With learn_stop=False the action has only the five correction parts. The
observation layout stays the same, with may_end and the previous end action 0.

Reward per decision: the change in a potential of the true state, less costs.
  potential   +GAP_REWARD per m of the true narrowest gap (Episode.narrowest_gap
              on the simulation), counted up to the book + GAP_ENOUGH, until
              the script turns to insert (then it stays as it was);
              -TILT_PENALTY per deg a book (not a leaner being stood up)
              has tilted beyond TILT_FREE since the start; -LIFT_PENALTY per
              m a book's lowest corner has risen beyond LIFT_FREE;
              once inserting, +PROGRESS_REWARD per m closer to acceptable
              placement (Episode.placement), negative when moving away.
              No further depth reward within the accepted region.
  costs       FORCE_PENALTY per N of peak wrist force above FORCE_FREE during
              the contact moves, ACTION_PENALTY * |increment / step|^2 (the
              five correction parts), TIME_PENALTY; once inserting,
              DISTURB_PENALTY per m the other books move.
  once        +GAP_BONUS as the script turns to insert, if the true gap is
              the book + GAP_MARGIN (the script's own test, Episode.widen_gap).
  at the end  +SUCCESS_BONUS if Episode.score is a success (upright, in
              place, let go of), -MISSED_PENALTY if it isn't; -FAILURE_PENALTY
              if a book tilts TOPPLE_DEG more than at the start, a leaner
              falls TOPPLE_DEG the other way, the arm hits the shelf
              (Runner.watch_contacts), or the script gives up or fails (it
              jammed the book at the entrance, found no plan).
              (insert=False: +GAP_BONUS if the gap is wide enough at insert,
              nothing if narrower.)
x/y displacement of the books while making room is not penalised: widening
and shoving move books along and into the row on purpose (evaluation still
reports it).
Every step reports these unnormalised reward terms, and the terminal info
contains their episode totals. In insertion-only control, reward counting
starts after scripted preparation, at the first controlled contact.

Training scenes (a pool per scenario, seeds from POOL_FIRST_SEED, never the
test seeds 0-49): each settled scene's state, its noise seeds, and what a run
of the script up to its first contact move found: the inspection views and
the first arm plan (choose_arm; planning them is most of a tight episode's
time), whether it makes any contact at all, and the phase of its first
contact ("slide" for a row wide enough already). The residual hasn't acted
before them, so they are exact; each is used only if its inputs hash the same.
first_phases picks scenes by their first contact: a tight row the script
shoves (its neighbours touch) takes ~10x as long as one it widens, and belongs
with the packed rows. (Pools built for gap creation only marked the rows wide
enough already as making no contact: rebuild them to train on those too.)
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import math
import os
import pickle
import queue
import threading
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass
from functools import partial
from multiprocessing import Pool
from typing import Dict, List, Optional, Sequence, Union

import gymnasium as gym
import mujoco
import numpy as np

from bila.control.insert_book import (DEFAULT_PANDA, GAP_MARGIN, LEAN_MIN_DEG, PAD_Z, RESIDUAL_LIMITS, RESIDUAL_STEPS, Episode,
                         axis_rot, build_model, make_episode, mat2quat)
from bila.vision.rgbd_perception import make_eyes
from bila.rl.observation import TOOL_OFFSET, ObservationBuilder, Parts, book_names_by_offset, robot_state
from bila.vision.perception import GroundTruthPerception, NoisyEyes  # (NoisyEyes: perception without rendering)
from bila.sim.shelf_scene import ShelfConfig, book_tilt_deg

from bila import ROOT as HERE
POOL_DIR = os.path.join(HERE, "scenes")
POOL_FIRST_SEED = 10_000     # training scene seeds start here (test seeds are 0-49)
STATE = mujoco.mjtState.mjSTATE_INTEGRATION
PHASES = ("widen", "lift", "clamp", "shove", "slide", "push")
SCENARIOS = ("clear", "tight", "leaning", "packed")
MAX_DECISIONS = 3000         # truncate an episode after this many

# Reward (per decision; see the module docstring)
GAP_REWARD = 1000.0          # per m of true narrowest gap (1 per mm) ...
GAP_ENOUGH = GAP_MARGIN      # ... counted up to the book plus this
GAP_BONUS = 10.0             # once, as the script turns to insert with the gap wide enough
TILT_PENALTY = 0.1           # per deg of tilt change ...
TILT_FREE = 2.0              # ... beyond this
LIFT_PENALTY = 500.0         # per m a book's lowest corner rises (0.5 per mm) ...
LIFT_FREE = 0.001            # ... beyond this (contacts settle a little)
PROGRESS_REWARD = 100.0      # per m closer to acceptable placement (1 per cm)
DISTURB_PENALTY = 50.0       # per m the other books move while it goes in (0.5 per cm, summed over them)
FORCE_PENALTY = 0.01         # per N of peak wrist force ...
FORCE_FREE = 60.0            # ... beyond this
ACTION_PENALTY = 0.1         # times |increment / step|^2
TIME_PENALTY = 0.01
SUCCESS_BONUS = 10.0         # the book placed (Episode.score)
MISSED_PENALTY = 5.0         # inserted, but not placed
FAILURE_PENALTY = 10.0
TOPPLE_DEG = 15.0            # a book tilting this much more than at the start has toppled

# The actor's and the critic's extras (GapEnv.extras, GapEnv.critic_extras): name, size
ACTOR_EXTRAS = (("phase", len(PHASES)), ("phase_time", 1), ("target", 3), ("target_rot", 6), ("offset", 5),
                ("last_action", 6), ("may_end", 1), ("wrench_change", 6), ("holding", 1), ("held_book", 3))
CRITIC_EXTRAS = (("grasp_error", 3), ("inserting", 1), ("true_book", 3), ("book_error", 3), ("to_go", 1),
                 ("book_forces", 3))
SCORE_KEYS = ("success", "safe_success", "upright", "in_place", "released", "offset_m", "tilt_deg",
              "neighbour_shift_m", "total_shift_m", "time_to_seat_s")
REWARD_KEYS = ("gap", "tilt", "lift", "force", "action", "time", "gap_bonus", "insertion_progress",
               "disturbance", "terminal")


# ---------------------------------------------------------------------------
# The grasp error
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class GraspError:
    """How far the book sits off where the script believes it took it:
    standard deviations at scale 1, each draw clipped at `clip` of them.
    (The take-over isn't modelled: the script puts the book in the fingers
    exactly where it means to, which a real hand-over wouldn't.)"""
    out: float = 0.0015                     # m: further out of the fingers (negative: deeper in; the spine sits
    #                                         PALM_GAP off the palm, so either way has room)
    up: float = 0.003                       # m: held higher up the book, or lower
    pitch: float = math.radians(1.0)        # rad: turned about the fingers' axis
    clip: float = 2.5

    def draws(self, rng: np.random.Generator, scale: float, n: int = 10) -> List[np.ndarray]:
        """n errors (out, up, pitch) to try in turn (shift_in_hand): the first
        that doesn't put the book into anything is the episode's."""
        if scale <= 0:
            return []
        g = lambda std: float(np.clip(rng.normal(), -self.clip, self.clip)) * std * scale
        return [np.array([g(self.out), g(self.up), g(self.pitch)]) for _ in range(n)]


def deepest(ep: Episode) -> float:
    """Deepest the hand or the held book is into anything else (m, 0 if nothing)."""
    mine = lambda g: g in ep.ik.robot or ep.ik.book_of.get(g, g) == ep.book_geom
    worst = 0.0
    for c in ep.d.contact[:ep.d.ncon]:
        if mine(c.geom1) != mine(c.geom2):
            worst = max(worst, -c.dist)
    return worst


def shift_in_hand(ep: Episode, err) -> bool:
    """Move the book in the closed fingers by err = (out, up, pitch): out of
    the fingers (toward their tips), up the book, and turned about the
    fingers' closing axis through the pads, so its faces stay on the pads.
    The script still believes it holds the book as it took it (ep.in_hand_*).
    Returns False, and leaves the book as it was, if it would then be 0.5 mm
    further into anything than it is."""
    m, d = ep.m, ep.d
    qa, va = m.jnt_qposadr[ep.book_joint], m.jnt_dofadr[ep.book_joint]
    before, q0 = deepest(ep), d.qpos[qa:qa + 7].copy()
    out, up, pitch = err
    R_h = d.xmat[ep.ik.hand].reshape(3, 3)
    pads = d.xpos[ep.ik.hand] + R_h @ [0.0, 0.0, PAD_Z]
    bp, bR = q0[:3].copy(), quat_to_mat(q0[3:])
    turn = axis_rot(bR[:, 0], pitch) if pitch else np.eye(3)
    d.qpos[qa:qa + 3] = pads + turn @ (bp - pads) + bR @ [0.0, -out, up]   # -y of the book: toward the fingertips
    d.qpos[qa + 3:qa + 7] = mat2quat(turn @ bR)
    d.qvel[va:va + 6] = 0
    mujoco.mj_forward(m, d)
    if deepest(ep) <= before + 0.0005:
        return True
    d.qpos[qa:qa + 7] = q0
    mujoco.mj_forward(m, d)
    return False


def quat_to_mat(q) -> np.ndarray:
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, np.asarray(q, float))
    return R.reshape(3, 3)


# ---------------------------------------------------------------------------
# The script, with its first plan cached
# ---------------------------------------------------------------------------
class ReachedInsert(BaseException):
    """The script would now insert the book: a gap-creation-only episode is over.
    (BaseException, like Abandon, so no `except` in the script catches it.)"""


class Abandon(BaseException):
    """The env stops the episode (a failure, truncation, reset or close)."""


@dataclass
class PoolEntry:
    """One training scene: the settled row and what the script's first run found."""
    scenario: str
    seed: int
    row: object                             # shelf_scene.Row
    state: np.ndarray                       # mj_getState(STATE) of the settled scene
    views: Optional[tuple] = None           # (inputs' digest, Episode.inspection_views)
    first_plan: Optional[tuple] = None      # (inputs' digest, arm angles, label) of the first choose_arm
    contact: Optional[bool] = None          # whether the script makes any contact move (None: not run yet)
    why: str = ""                           # if not, why not
    phase: str = ""                         # the first contact move's phase (PHASES), if any
    variant: int = 0                         # independent error variant of this geometry
    perception_seed: Optional[Union[int, Sequence[int]]] = None  # None preserves the old [seed, 1] stream
    grasp_seed: Optional[Union[int, Sequence[int]]] = None       # None preserves the old [seed, 2] stream
    perception: str = "noisy"               # how its first run saw the row (sight); older pools: noisy

    def identity(self) -> dict:
        """A replayable case, including the legacy seed streams of old pools."""
        return {"scenario": self.scenario, "geometry_seed": self.seed, "variant": self.variant,
                "perception_seed": self.perception_seed if self.perception_seed is not None else [self.seed, 1],
                "grasp_seed": self.grasp_seed if self.grasp_seed is not None else [self.seed, 2]}

    @property
    def noise_rng(self) -> np.random.Generator:
        """The camera noise of this scene: the same every time it is used."""
        return np.random.default_rng(self.identity()["perception_seed"])

    @property
    def grasp_rng(self) -> np.random.Generator:
        """The grasp error of this scene: the same every time it is used."""
        return np.random.default_rng(self.identity()["grasp_seed"])


class Packed:
    """A pool entry kept pickled, with its seed and contact flag at hand:
    unpickled, a pool of 1000 takes ~270 MB in every worker."""
    phase = ""                              # (pools packed before phases were recorded)
    variant = 0                             # (pools made before independent error variants)
    perception = "noisy"                    # (pools made before the camera could be chosen)

    def __init__(self, entry: PoolEntry):
        self.seed, self.contact, self.phase, self.blob = entry.seed, entry.contact, entry.phase, pickle.dumps(entry)
        self.variant, self.perception = entry.variant, entry.perception

    def unpack(self) -> PoolEntry:
        return pickle.loads(self.blob)


def make_entry(cfg: ShelfConfig, scenario: str, seed: int) -> PoolEntry:
    _, data, row = make_episode(cfg, scenario, np.random.default_rng(seed), DEFAULT_PANDA)
    model = build_model(cfg, row, DEFAULT_PANDA)
    state = np.empty(mujoco.mj_stateSize(model, STATE))
    mujoco.mj_getState(model, data, state, STATE)
    return PoolEntry(scenario, seed, row, state)


PERCEPTIONS = ("noisy", "noisy-learned", "camera")
_segmenters: Dict[tuple, object] = {}


def sight(perception: str = "noisy", segmenter: Optional[str] = None) -> str:
    """How eyes see the row, as pools record it: the perception, and for the
    camera with learned masks, the weights' file name."""
    if perception not in PERCEPTIONS:
        raise ValueError(f"perception must be one of {PERCEPTIONS}, not {perception!r}")
    if segmenter and perception != "camera":
        raise ValueError("a segmenter goes with perception='camera'")
    return f"camera+{os.path.basename(segmenter)}" if segmenter else perception


def learned_masks(weights: str, device: Optional[str] = None):
    """The learned segmenter (learned_segmenter.LearnedSegmenter), loaded
    once per process: it is ~350 MB, and an env or pool worker makes a
    camera every episode."""
    key = (os.path.abspath(os.path.expanduser(weights)), device)
    if key not in _segmenters:
        from bila.vision.learned_segmenter import LearnedSegmenter   # (it imports torch's dynamo, which must precede rendering)
        import torch
        torch.set_num_threads(1)              # (one per worker process: a thread pool in each overloads the machine)
        _segmenters[key] = LearnedSegmenter(key[0], device=device)
    return _segmenters[key]


def make_eyes_for(entry: PoolEntry, model, perception: str = "noisy", noise_scale: float = 1.0,
                  segmenter: Optional[str] = None, segmenter_device: Optional[str] = None):
    """An episode's eyes on a pool entry's row (see the module docstring),
    drawn from its perception seed: the same every time it is used.
    noise_scale: the noise models' spread (the camera has none to scale)."""
    sight(perception, segmenter)
    if perception == "camera":
        if noise_scale != 1.0:
            raise ValueError("noise_scale applies to the noise models, not the camera")
        import cv2
        cv2.setNumThreads(1)                  # (one per worker process, as above)
        masks = learned_masks(segmenter, segmenter_device) if segmenter else None
        return make_eyes(model, entry.row, entry.identity()["perception_seed"], "realistic", masks)
    return NoisyEyes(model, entry.row, entry.noise_rng, noise_scale, learned=perception == "noisy-learned")


def load_scene(cfg: ShelfConfig, entry: PoolEntry):
    """(model, data) of a pool entry, settled. (The model is rebuilt from the
    row: a compiled one is ~36 MB.)"""
    model = build_model(cfg, entry.row, DEFAULT_PANDA)
    data = mujoco.MjData(model)
    mujoco.mj_setState(model, data, entry.state, STATE)
    mujoco.mj_forward(model, data)
    return model, data


def digest(*parts) -> str:
    """A hash of nested tuples/lists/sets of arrays and numbers (the inputs of a plan)."""
    h = hashlib.sha1()

    def add(x):
        if isinstance(x, (list, tuple)):
            h.update(b"(")
            for y in x:
                add(y)
            h.update(b")")
        elif isinstance(x, (set, frozenset)):
            add(sorted(x))
        elif x is None or isinstance(x, (str, bool, int, float, np.integer, np.floating)):
            h.update(repr(x).encode())
        else:
            h.update(np.ascontiguousarray(x, dtype=float).tobytes())

    for p in parts:
        add(p)
    return h.hexdigest()


class GapEpisode(Episode):
    """Episode for training: reuses the pool entry's inspection views and
    first arm plan, puts the book in the fingers off by the first of `grasp`
    (GraspError.draws) that fits, notes the true gap as it turns to insert
    (at_insert), and records the peak wrist force during the contact moves
    (peak_force, reset by the env). through_insert=False: stops where the
    script would insert the book instead (ReachedInsert)."""

    def __init__(self, model, data, entry: PoolEntry, eyes, through_insert: bool = True, grasp: Sequence = ()):
        super().__init__(model, data, entry.row, None, lambda *_: None, eyes)
        self.entry, self.planned, self.peak_force = entry, False, 0.0
        self.through_insert, self.grasp_draws = through_insert, list(grasp)
        self.grasp_error = np.zeros(3)      # the one applied (none if no draw fitted)
        self.at_insert = None               # {"gap": true narrowest gap (m), "wide": wide enough} as it turned to insert

    def insert(self):
        if not self.through_insert:
            raise ReachedInsert
        with truth(self):
            self.at_insert = {"gap": float(self.narrowest_gap()),
                              "wide": bool(self.widen_gap() >= self.t + GAP_MARGIN - 0.001)}
        return super().insert()

    def take_book(self, grasp_sign, centre, yaw=0.0, name="close the gripper on the book"):
        super().take_book(grasp_sign, centre, yaw, name)
        for err in self.grasp_draws:
            if shift_in_hand(self, err):
                self.grasp_error = np.asarray(err, dtype=float)
                break

    def place_arm(self, q):
        super().place_arm(q)
        run, step = self.run, self.run.step

        def watched():
            step()
            if run.touching is not None:
                self.peak_force = max(self.peak_force, float(np.linalg.norm(run.wrist_force())))

        run.step = watched

    def inspection_views(self):
        key = digest(self.d.qpos)
        cached = self.entry.views
        if cached is not None and cached[0] == key:
            return copy.deepcopy(cached[1])
        views = super().inspection_views()
        self.entry.views = (key, copy.deepcopy(views))
        return views

    def choose_arm(self, candidates, allowed=frozenset(), finger=None, reach_from=None):
        if self.planned:
            return super().choose_arm(candidates, allowed, finger, reach_from)
        self.planned = True
        plans = [(start, path) for start, path, _ in candidates]
        key = digest(plans, allowed, finger, reach_from, self.d.qpos, self.ik.belief)
        cached = self.entry.first_plan
        if cached is not None and cached[0] == key:
            _, q, label = cached
            self.ik.q_rest = q.copy()
            self.ik.sync(self.d, finger)    # leave the planner's scene as planning would
            return q.copy(), copy.deepcopy(label)
        q, label = super().choose_arm(candidates, allowed, finger, reach_from)
        try:                                # (clamp_up's label holds a closure: a pool can't pickle it, so plan again)
            pickle.dumps(label)
            self.entry.first_plan = (key, q.copy(), copy.deepcopy(label))
        except (AttributeError, pickle.PicklingError, TypeError):
            pass
        return q, label


@contextmanager
def truth(ep: Episode):
    """Ask the episode about the row as it really is: its methods read the
    simulation when it has no eyes. (Called only while its thread waits, or
    by that thread itself.)"""
    eyes, ep.eyes = ep.eyes, None
    try:
        yield ep
    finally:
        ep.eyes = eyes


def to_go(ep: Episode) -> float:
    """Distance to the scorer's accepted spine positions (m, ground truth).
    Flush penalises undershoot and overshoot beyond ALIGN_TOL. Inside needs
    only EDGE_IN behind the shelf edge, not the controller's deeper aim."""
    spine = ep.shelf.box(ep.m, ep.d, ep.book_geom)[:, 1].max()
    return ep.placement(spine)[1]


def book_contacts(ep: Episode) -> np.ndarray:
    """Normal force on the book (N) from the other books, the robot, and anything else (the board)."""
    m, d, out, f = ep.m, ep.d, np.zeros(3), np.zeros(6)
    book = lambda g: ep.ik.book_of.get(g, g)
    rows, robot = set(ep.row_geoms), ep.ik.robot
    for i in range(d.ncon):
        c = d.contact[i]
        if ep.book_geom not in (book(c.geom1), book(c.geom2)):
            continue
        other = c.geom2 if book(c.geom1) == ep.book_geom else c.geom1
        mujoco.mj_contactForce(m, d, i, f)
        out[0 if book(other) in rows else 1 if other in robot else 2] += abs(f[0])
    return out


# ---------------------------------------------------------------------------
# The script's thread and the residual that hands decisions to the env
# ---------------------------------------------------------------------------
class Bridge:
    """The residual the episode's Runner calls (decide): it hands the decision
    (the Contact) to the env and blocks until the env sends the action."""

    def __init__(self, control_phases=PHASES):
        self.control_phases = frozenset(control_phases)
        self.episode: Optional[GapEpisode] = None   # the one the thread made (run_script)
        self.decisions: queue.Queue = queue.Queue()
        self.actions: queue.Queue = queue.Queue()

    def decide(self, run, contact):
        if contact.phase not in self.control_phases:
            return np.zeros(6)          # the script prepares the row without any policy decisions
        self.decisions.put(("decide", contact))
        action = self.actions.get()
        if action is None:
            raise Abandon
        return action


def run_script(make_episode, bridge: Bridge):
    """The worker thread: make the episode (make_episode(), into
    bridge.episode), run the script to its score (or to insert), let go of
    its eyes, then report how it ended. The eyes are made, used and let go of
    in this thread: a rendered camera's OpenGL context (macOS's CGL) is
    locked to the thread that made it, and another one rendering with it
    waits for ever."""
    eyes, end = None, None
    try:
        ep = bridge.episode = make_episode()
        eyes, ep.residual = ep.eyes, bridge
        res = ep.reshelve()
        end = ("gave_up", res["skipped"]) if "skipped" in res else ("scored", res)
    except ReachedInsert:
        end = ("insert", "")
    except Abandon:
        pass
    except RuntimeError as e:                # the script found no way to go on (no plan, unseen books, a jam)
        end = ("error", str(e))
    except Exception as e:                   # a bug: still end the episode, and say what it was
        end = ("crash", f"{type(e).__name__}: {e}")
    finally:
        if eyes is not None:
            eyes.close()
    if end is not None:
        bridge.decisions.put(("done", end))


class PreparationFailed(RuntimeError):
    """This exact case never reached a phase the policy controls."""

    def __init__(self, case: dict, reason: str):
        self.record = {**case, "seed": case["geometry_seed"], "outcome": "preparation_failed",
                       "failure": reason, "decisions": 0, "success": False, "safe_success": False}
        super().__init__(f"{case}: {reason}")


# ---------------------------------------------------------------------------
# The environment
# ---------------------------------------------------------------------------
class GapEnv(gym.Env):
    """See the module docstring. scenarios: drawn from uniformly per episode.
    pool: {scenario: [PoolEntry or Packed]} (default: scenes/gap_pool_<scenario>.pkl).
    first_phases: only scenes whose first contact is one of these (None: all).
    control_phases: contacts where PPO acts. Other contacts run with zero
    corrections before/between decisions. ("slide", "push") trains insertion
    after scripted preparation. learn_stop=False uses five continuous actions.
    perception, segmenter (weights), segmenter_device: the eyes (module
    docstring, make_eyes_for); the pool must have seen its rows the same way.
    noise_scale: NoisyPerception's scale. grasp_scale: GraspError's.
    action_scale: of the correction's steps. insert: run through the
    insertion to the score (False: gap creation only).
    reset(options={"scenario": s, "geometry_seed": seed, "variant": v})
    replays an exact case, with optional perception_seed/grasp_seed overrides.
    Explicit cases (or index in the filtered pool) raise PreparationFailed
    instead of substituting another case. Random training resets try the
    pool in random order and report unsuccessful preparation attempts."""

    metadata = {"render_modes": []}

    def __init__(self, scenarios: Sequence[str] = ("tight",), pool: Optional[Dict[str, list]] = None,
                 first_phases: Optional[Sequence[str]] = None, noise_scale: float = 1.0, action_scale: float = 1.0,
                 max_decisions: int = MAX_DECISIONS, cfg: Optional[ShelfConfig] = None, insert: bool = True,
                 grasp: GraspError = GraspError(), grasp_scale: float = 1.0,
                 control_phases: Sequence[str] = PHASES, learn_stop: bool = True, perception: str = "noisy",
                 segmenter: Optional[str] = None, segmenter_device: Optional[str] = None):
        if not control_phases or set(control_phases) - set(PHASES):
            raise ValueError(f"control_phases must be a nonempty subset of {PHASES}")
        self.sight = sight(perception, segmenter)
        if perception == "camera" and noise_scale != 1.0:
            raise ValueError("noise_scale applies to the noise models, not the camera")
        if first_phases is not None and set(first_phases) - set(PHASES):
            raise ValueError(f"unknown first contact phase: {first_phases}")
        if not insert and not (set(control_phases) - {"slide", "push"}):
            raise ValueError("insertion contact phases require insert=True")
        self.cfg = cfg or ShelfConfig()
        self.scenarios = tuple(scenarios)
        pool = pool if pool is not None else {s: load_pool(s) for s in self.scenarios}
        packed = {s: [e if isinstance(e, Packed) else Packed(e) for e in pool[s]] for s in self.scenarios}
        for s in self.scenarios:
            seen = sorted({p.perception for p in packed[s]} - {self.sight})
            if seen:
                raise ValueError(f"the {s} pool saw its rows as {', '.join(seen)}, this env as {self.sight}: "
                                 f"build one with the same perception (gap_env.py pool --perception ...)")
        self.pool = {s: [p for p in packed[s] if first_phases is None or p.phase in first_phases]
                     for s in self.scenarios}
        for s in self.scenarios:
            if not self.pool[s]:
                raise ValueError(f"no {s} scene matches the first-contact filter")
        self.perception, self.segmenter, self.segmenter_device = perception, segmenter, segmenter_device
        self.noise_scale, self.action_scale, self.max_decisions = noise_scale, action_scale, max_decisions
        self.insert, self.grasp, self.grasp_scale = insert, grasp, grasp_scale
        self.control_phases, self.learn_stop = tuple(control_phases), learn_stop
        self.ep: Optional[GapEpisode] = None
        self.thread: Optional[threading.Thread] = None
        self.bridge: Optional[Bridge] = None
        n_actor, n_critic = self.sizes()
        box = lambda n: gym.spaces.Box(-np.inf, np.inf, (n,), np.float32)
        self.observation_space = gym.spaces.Dict({"actor": box(n_actor), "critic": box(n_critic)})
        self.action_space = gym.spaces.Box(-1.0, 1.0, (6 if learn_stop else 5,), np.float32)
        start = n_actor - sum(n for _, n in ACTOR_EXTRAS)
        self.actor_layout = layout(ACTOR_EXTRAS, start)          # where the extras sit in obs["actor"]
        self.critic_layout = layout(CRITIC_EXTRAS, n_critic - sum(n for _, n in CRITIC_EXTRAS))

    def set_action_scale(self, scale: float) -> None:
        self.action_scale = float(scale)

    # --- episode lifecycle ---------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.stop()
        options = options or {}
        index = options.get("index")
        exact = index is not None or "geometry_seed" in options
        scenario = options.get("scenario") or self.scenarios[
            0 if exact else self.np_random.integers(len(self.scenarios))]
        entries = self.pool[scenario]
        if "geometry_seed" in options:
            matches = [i for i, p in enumerate(entries)
                       if p.seed == options["geometry_seed"] and p.variant == options.get("variant", 0)]
            if len(matches) != 1:
                raise ValueError(f"expected one case for {options}, found {len(matches)}")
            indices = matches
        elif index is not None:
            if not 0 <= index < len(entries):
                raise IndexError(f"case index {index} outside {scenario}'s pool")
            indices = [index]
        else:
            indices = self.np_random.permutation(len(entries))
        failed = []
        for i in indices:
            packed = entries[i]
            entry = packed.unpack()
            for key in ("perception_seed", "grasp_seed"):
                if key in options:
                    setattr(entry, key, options[key])
            if entry.row is None:
                reason = entry.why or "no settled scene"
            else:
                msg = self.start(entry)
                if msg[0] == "decide" and not self.ep.run.collision["steps"]:
                    break
                reason = ("the arm collided during preparation" if msg[0] == "decide" else
                          f"before the first controlled contact: {msg[1][0]}: {msg[1][1]}")
                self.stop()
            error = PreparationFailed(entry.identity(), reason)
            if exact:
                raise error
            failed.append(error.record)
        else:
            raise PreparationFailed({"scenario": scenario, "geometry_seed": -1},
                                    f"no case reached a controlled contact ({len(failed)} preparation failures)")
        self.reset_info = {"seed": entry.seed, **entry.identity(), "preparation_failures": failed}
        self.reward_totals = dict.fromkeys(REWARD_KEYS, 0.0)
        ep = self.ep
        self.builder = ObservationBuilder(ep.est, target=(ep.t, ep.h))
        self.ground_truth = GroundTruthPerception(ep.m, ep.row)
        self.book_names = book_names_by_offset(ep.row)
        truth_now = self.ground_truth.estimate(ep.d)
        # Neighbours leaning into the slot at the start are meant to move (stood up).
        self.leaners = {i for i in (ep.row.target_index - 1, ep.row.target_index)
                        if -np.sign(ep.offset(i)) * truth_now.book(ep.offset(i)).lean > math.radians(LEAN_MIN_DEG)}
        self.leaners.update(ep.intended)       # scripted preparation may already have stood these books up
        self.steady = [i for i in range(len(ep.row_bodies)) if i not in self.leaners]
        self.start_lift = self.lifts()
        self.collisions0 = ep.run.collision["steps"]
        self.phi_terms, self.gap0, _ = self.potential_terms()
        self.phi = sum(self.phi_terms.values())
        self.inserting = None         # once the script has turned to insert: the insertion's reward state
        if ep.at_insert is not None:  # (a row wide enough already: inserting from the start, no bonus to earn)
            self.start_insertion()
        self.last_action, self.decisions = np.zeros(6), 0
        self.see(msg[1])
        return self.observe(truth_now), self.reset_info.copy()

    def start(self, entry: PoolEntry):
        """Start the script on a pool entry; return its first message."""
        model, data = load_scene(self.cfg, entry)
        grasp = self.grasp.draws(entry.grasp_rng, self.grasp_scale)
        make = lambda: GapEpisode(model, data, entry, make_eyes_for(entry, model, self.perception, self.noise_scale,
                                                                      self.segmenter, self.segmenter_device),
                                  through_insert=self.insert, grasp=grasp)
        self.bridge = Bridge(self.control_phases)
        self.thread = threading.Thread(target=run_script, args=(make, self.bridge), daemon=True)
        self.thread.start()
        msg = self.bridge.decisions.get()
        self.ep = self.bridge.episode          # (None if making it failed: msg says why)
        return msg

    def see(self, contact):
        """The contact the script now asks about (None: the episode is over)."""
        if contact is not None and contact is not getattr(self, "contact", None):
            self.wrench0 = robot_state(self.ep.m, self.ep.d).wrench    # its move has just started
        self.contact = contact

    def step(self, action):
        a = np.zeros(6)
        a[:self.action_space.shape[0]] = np.clip(np.asarray(action, dtype=float).reshape(self.action_space.shape), -1, 1)
        self.ep.peak_force = 0.0
        self.bridge.actions.put(np.concatenate([self.correction(a[:5] * self.action_scale), a[5:]]))
        kind, what = self.bridge.decisions.get()
        self.decisions += 1
        ep = self.ep
        truth_now = self.ground_truth.estimate(ep.d)
        potential, gap, tilt = self.potential_terms()
        terms = dict.fromkeys(REWARD_KEYS, 0.0)
        terms.update({k: potential[k] - self.phi_terms[k] for k in potential})
        terms.update(force=-FORCE_PENALTY * max(0.0, ep.peak_force - FORCE_FREE),
                     action=-ACTION_PENALTY * float(np.sum((a[:5] * self.action_scale) ** 2)), time=-TIME_PENALTY)
        terms.update(self.insertion_rewards())
        self.phi_terms, self.phi, self.last_action = potential, sum(potential.values()), a
        failure = self.failure(truth_now, tilt)
        if kind == "done" and failure is None and what[0] not in ("insert", "scored"):
            failure = f"the script {'gave up' if what[0] == 'gave_up' else 'failed'}: {what[1]}"
        info = {"gap_m": gap, "gap_gain_m": gap - self.gap0, "peak_force_n": ep.peak_force}
        terminated = failure is not None or kind == "done"
        truncated = not terminated and self.decisions >= self.max_decisions
        if failure is not None:
            terms["terminal"] = -FAILURE_PENALTY
            info["outcome"], info["failure"] = "failure", failure
        elif kind == "done" and what[0] == "insert":                  # (gap creation only)
            with truth(ep):
                wide = ep.widen_gap() >= ep.t + GAP_MARGIN - 0.001
            terms["gap_bonus"] += GAP_BONUS if wide else 0.0
            info["outcome"] = "success" if wide else "narrow"
        elif kind == "done":
            res = what[1]
            terms["terminal"] = SUCCESS_BONUS if res.get("success") else -MISSED_PENALTY
            info["outcome"] = "success" if res.get("success") else "missed"
            info.update({k: (v.item() if isinstance(v, np.generic) else v) for k, v in res.items() if k in SCORE_KEYS})
        elif truncated:
            info["outcome"] = "truncated"
        reward = float(sum(terms.values()))
        info["reward_terms"] = terms
        for k, v in terms.items():
            self.reward_totals[k] += v
        self.see(what if kind == "decide" else None)
        obs = self.observe(truth_now)
        if terminated or truncated:
            info.update(self.reset_info)
            info.update(decisions=self.decisions, method=ep.info.get("method", ""),
                        collision_during=sorted(ep.run.collision["during"]), ended=dict(Counter(ep.run.ended)),
                        grasp_error=ep.grasp_error.tolist(), reward_totals=self.reward_totals.copy())
            if ep.at_insert is not None:
                info.update(wide_at_insert=ep.at_insert["wide"], gap_at_insert_m=ep.at_insert["gap"])
            self.stop()
        return obs, float(reward), terminated, truncated, info

    def correction(self, a) -> np.ndarray:
        """The five correction parts of an action (in [-1, 1], times the action
        scale) as the Runner takes them for the contact deciding now: in a
        target contact (the insertion) the correction itself, 0 the script's
        pose and +-1 the move's bounds on that side; otherwise its change,
        +-1 a full RESIDUAL_STEPS."""
        c = self.contact
        if c is not None and c.target:
            return np.where(a >= 0, a * c.hi, -a * c.lo)
        return a * RESIDUAL_STEPS

    def stop(self):
        """End the script's thread, if it is waiting for an action (it lets
        go of the episode's eyes as it ends: run_script)."""
        if self.thread is not None and self.thread.is_alive():
            self.bridge.actions.put(None)
            self.thread.join()
        self.thread = None

    def close(self):
        self.stop()

    # --- reward terms -------------------------------------------------------------
    def lifts(self) -> np.ndarray:
        """Height of each row book's lowest corner above the board (m)."""
        ep = self.ep
        return np.array([ep.shelf.box(ep.m, ep.d, g)[:, 2].min() - ep.z0 for g in ep.row_geoms])

    def potential(self):
        """(potential, true narrowest gap, largest tilt change of a steady
        book). Once the script has turned to insert, the gap is the one it
        had then."""
        terms, gap, tilt = self.potential_terms()
        return float(sum(terms.values())), gap, tilt

    def potential_terms(self):
        """The existing shaping potential split into named, unscaled terms."""
        ep = self.ep
        if ep.at_insert is not None:
            gap = ep.at_insert["gap"]
        else:
            with truth(ep):
                gap = ep.narrowest_gap()
        tilt = np.array([book_tilt_deg(ep.d.xquat[ep.row_bodies[i]]) for i in self.steady])
        dtilt = np.abs(tilt - ep.start_tilt[self.steady])
        lift = self.lifts() - self.start_lift
        terms = {"gap": GAP_REWARD * min(gap, ep.t + GAP_ENOUGH),
                 "tilt": -TILT_PENALTY * np.maximum(0.0, dtilt - TILT_FREE).sum(),
                 "lift": -LIFT_PENALTY * np.maximum(0.0, lift - LIFT_FREE).sum()}
        return {k: float(v) for k, v in terms.items()}, float(gap), float(dtilt.max(initial=0.0))

    def start_insertion(self) -> float:
        """Where the insertion's reward counts from (the book is in the hand
        by the first decision after the script turned to insert); returns
        GAP_BONUS if the gap was wide enough then."""
        ep = self.ep
        bodies = np.asarray(ep.row_bodies)[[i for i in range(len(ep.row_bodies)) if i not in ep.intended]]
        self.inserting = {"bodies": bodies, "pos": ep.d.xpos[bodies].copy(), "go": to_go(ep), "moved": 0.0}
        return GAP_BONUS if ep.at_insert["wide"] else 0.0

    def insertion_reward(self) -> float:
        """The insertion's part of this decision's reward: GAP_BONUS at the
        first decision after the script turned to insert, and from there the
        book's progress, less the other books moving."""
        return float(sum(self.insertion_rewards().values()))

    def insertion_rewards(self) -> dict:
        """Insertion shaping and the gap bonus, before reward normalisation."""
        ep = self.ep
        if ep.at_insert is None:
            return {}
        if self.inserting is None:
            return {"gap_bonus": self.start_insertion()}
        s = self.inserting
        go, moved = to_go(ep), float(np.linalg.norm(ep.d.xpos[s["bodies"]] - s["pos"], axis=1).sum())
        terms = {"insertion_progress": PROGRESS_REWARD * (s["go"] - go),
                 "disturbance": -DISTURB_PENALTY * (moved - s["moved"])}
        s["go"], s["moved"] = go, moved
        return terms

    def failure(self, truth_now, tilt) -> Optional[str]:
        """Why the episode has failed, or None."""
        ep = self.ep
        if tilt > TOPPLE_DEG:
            return f"a book tilted {tilt:.0f} deg"
        for i in self.leaners:
            k = ep.offset(i)
            if np.sign(k) * truth_now.book(k).lean > math.radians(TOPPLE_DEG):
                return "the leaner fell the other way"
        if ep.run.collision["steps"] > self.collisions0:
            return "the arm hit " + ", ".join(sorted(ep.run.collision["parts"]))
        return None

    # --- observation ---------------------------------------------------------------
    def extras(self, rs) -> np.ndarray:
        """What only the env knows (ACTOR_EXTRAS): the contact phase, time in
        it, the script's tool target, the correction so far, the last action,
        whether the move may be ended, the wrist's change since it started,
        and while sliding the book in, where the robot believes it is."""
        ep, run, c, b = self.ep, self.ep.run, self.contact, self.builder
        p = Parts()
        p.add("phase", [c is not None and c.phase == ph for ph in PHASES])
        p.add("phase_time", c.steps * ep.m.opt.timestep if c is not None else 0.0)
        p.add("target", b.to_slot(run.pos + run.rot[:, 2] * TOOL_OFFSET))
        p.add("target_rot", b.rot_to_slot(run.rot)[:, :2].T)
        p.add("offset", run.offset / RESIDUAL_LIMITS)
        p.add("last_action", self.last_action)
        p.add("may_end", self.learn_stop and c is not None and (c.may_end is None or c.may_end()))
        p.add("wrench_change", rs.wrench - self.wrench0 if c is not None else np.zeros(6))
        holding = c is not None and c.phase == "slide"
        p.add("holding", holding)
        p.add("held_book", b.to_slot(ep.held_pose()[0]) if holding else np.zeros(3))
        return p.vector()

    def critic_extras(self) -> np.ndarray:
        """The critic's privileged extras (CRITIC_EXTRAS): the grasp error, and
        once inserting, the book's true position, how far the robot's belief
        of it is off (while it holds it), its distance to acceptable placement, and
        the forces on it."""
        ep, b = self.ep, self.builder
        p = Parts()
        p.add("grasp_error", ep.grasp_error)
        inserting = ep.at_insert is not None
        p.add("inserting", inserting)
        true_book = ep.d.xpos[ep.book_body]
        p.add("true_book", b.to_slot(true_book) if inserting else np.zeros(3))
        holding = self.contact is not None and self.contact.phase == "slide"
        p.add("book_error", b.R_shelf.T @ (true_book - ep.held_pose()[0]) if holding else np.zeros(3))
        p.add("to_go", to_go(ep) if inserting else 0.0)
        p.add("book_forces", book_contacts(ep) if inserting else np.zeros(3))
        return p.vector()

    def observe(self, truth_now):
        ep = self.ep
        rs = robot_state(ep.m, ep.d)
        actor = self.builder.actor(ep.est, rs, now=ep.d.time)
        critic = self.builder.critic(ep.m, ep.d, truth_now, self.book_names)
        return {"actor": np.concatenate([actor, self.extras(rs)]).astype(np.float32),
                "critic": np.concatenate([critic, self.critic_extras()]).astype(np.float32)}

    def sizes(self):
        """Lengths of the actor and critic vectors (from the first scene's
        ground truth: they don't depend on the row)."""
        # A validation scenario can have no successful preparations. Its rows
        # still define the same observation shape and must be evaluable as failures.
        entry = next((e for pool in self.pool.values() for p in pool for e in [p.unpack()]
                      if e.row is not None), None)
        if entry is None:
            raise ValueError("the pools contain no settled geometry to initialise the environment")
        model, data = load_scene(self.cfg, entry)
        est = GroundTruthPerception(model, entry.row).estimate(data)
        b = ObservationBuilder(est, target=(entry.row.target_book.thickness, entry.row.target_book.height))
        n_actor = len(b.actor(est, robot_state(model, data), now=data.time))
        n_critic = len(b.critic(model, data, est, book_names_by_offset(entry.row)))
        return n_actor + sum(n for _, n in ACTOR_EXTRAS), n_critic + sum(n for _, n in CRITIC_EXTRAS)


def layout(parts, start: int) -> Dict[str, slice]:
    """{name: slice} of named parts laid out one after another from `start`."""
    out = {}
    for name, n in parts:
        out[name] = slice(start, start + n)
        start += n
    return out


# ---------------------------------------------------------------------------
# Building the pool
# ---------------------------------------------------------------------------
def pool_path(name: str, root: str = POOL_DIR) -> str:
    """A pool's file; its name is the scenario's unless given (a validation pool's)."""
    return os.path.join(root, f"gap_pool_{name}.pkl")


def load_pool(name: str, root: str = POOL_DIR) -> List[Packed]:
    path = pool_path(name, root)
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} is missing: build it with `python -m bila.rl.gap_env pool --scenario ...`")
    with open(path, "rb") as f:
        return pickle.load(f)


class FirstContact:
    """A residual that stops the script at its first contact move, noting its phase."""
    phase = ""

    def decide(self, run, contact):
        self.phase = contact.phase
        raise Abandon


def warm(args, perception: str = "noisy", segmenter: Optional[str] = None,
         segmenter_device: Optional[str] = None) -> PoolEntry:
    """Make a pool entry and run the script on it up to its first contact
    move (the slide, for a row wide enough already), so the entry holds its
    inspection views and first plan. perception, segmenter: the eyes it sees
    the row with (make_eyes_for), recorded in the entry."""
    scenario, seed, *errors = args
    variant, perception_seed, grasp_seed = errors or (0, None, None)
    cfg, seen = ShelfConfig(), sight(perception, segmenter)
    try:
        entry = make_entry(cfg, scenario, seed)
    except RuntimeError as e:                # no stable row for this seed
        return PoolEntry(scenario, seed, None, np.zeros(0), contact=False, why=str(e),
                         variant=variant, perception_seed=perception_seed, grasp_seed=grasp_seed, perception=seen)
    entry.variant, entry.perception_seed, entry.grasp_seed = variant, perception_seed, grasp_seed
    entry.perception = seen
    model, data = load_scene(cfg, entry)
    eyes = make_eyes_for(entry, model, perception, 1.0, segmenter, segmenter_device)
    ep = GapEpisode(model, data, entry, eyes)
    ep.residual = first = FirstContact()
    try:
        res = ep.reshelve()
        entry.contact, entry.why = False, "the script gave up: " + res.get("skipped", "")
    except Abandon:
        entry.contact, entry.phase = True, first.phase
    except RuntimeError as e:
        entry.contact, entry.why = False, str(e)
    except Exception as e:                   # a bug: leave the scene out rather than lose the whole pool
        entry.contact, entry.why = False, f"crash: {type(e).__name__}: {e}"
    finally:
        eyes.close()
    return entry


def pool_jobs(scenario: str, seeds: int, first_seed: int, error_variants: int = 1,
              error_seed: Optional[int] = None) -> list:
    """Geometry is unchanged across variants. Error streams are independent,
    reproducible, and independent of worker scheduling. Default: legacy seeds."""
    if seeds < 1 or error_variants < 1 or first_seed < 0 or (error_seed is not None and error_seed < 0):
        raise ValueError("seeds and error_variants must be positive, and seed values nonnegative")
    jobs = []
    for seed in range(first_seed, first_seed + seeds):
        for variant in range(error_variants):
            if error_variants == 1 and error_seed is None:
                jobs.append((scenario, seed))
            else:
                draw = lambda stream: int(np.random.SeedSequence(
                    [0 if error_seed is None else error_seed, seed, variant, stream]).generate_state(1)[0])
                jobs.append((scenario, seed, variant, draw(1), draw(2)))
    return jobs


def build_pool(scenario: str, seeds: int, first_seed: int = POOL_FIRST_SEED, workers: int = 1,
               root: str = POOL_DIR, name: Optional[str] = None, error_variants: int = 1,
               error_seed: Optional[int] = None, perception: str = "noisy", segmenter: Optional[str] = None,
               segmenter_device: Optional[str] = None) -> List[PoolEntry]:
    """Build and save a pool (as `name`, default the scenario), seeing the
    rows with `perception` (and `segmenter`: make_eyes_for). Refuses to
    overwrite one: they take hours."""
    path = pool_path(name or scenario, root)
    if os.path.exists(path):
        raise FileExistsError(f"{path} exists: move it away first")
    sight(perception, segmenter)
    jobs = pool_jobs(scenario, seeds, first_seed, error_variants, error_seed)
    see = partial(warm, perception=perception, segmenter=segmenter, segmenter_device=segmenter_device)
    if workers > 1:
        with Pool(workers) as p:
            entries = []
            for i, e in enumerate(p.imap(see, jobs)):
                entries.append(e)
                print(f"  {i + 1}/{len(jobs)}  seed {e.seed}, variant {e.variant}: "
                      f"{e.phase if e.contact else e.why}", flush=True)
    else:
        entries = [see(j) for j in jobs]
    os.makedirs(root, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump([Packed(e) for e in entries], f)
    phases = {p: sum(e.phase == p for e in entries) for p in PHASES}
    print(f"{scenario} ({sight(perception, segmenter)}): {sum(e.contact for e in entries)} of {len(entries)} "
          f"scenes make contact (first contact: {phases}) -> {path}")
    return entries


def first_phase(packed: Packed) -> Packed:
    """The scene again, up to its first contact (fast: its views and plan are
    cached), to record that contact's phase."""
    entry = packed.unpack()
    if entry.contact:
        model, data = load_scene(ShelfConfig(), entry)
        ep = GapEpisode(model, data, entry, NoisyEyes(model, entry.row, entry.noise_rng))
        ep.residual = first = FirstContact()
        try:
            ep.reshelve()
        except (Abandon, ReachedInsert, RuntimeError):
            pass
        entry.phase = first.phase
    return Packed(entry)


def add_phases(name: str, workers: int = 1, root: str = POOL_DIR) -> None:
    """Record the first contact phase of every scene of a pool built before
    phases were. Writes a new file and swaps it in only once it is complete."""
    pool, path = load_pool(name, root), pool_path(name, root)
    with Pool(workers) as p:
        done = list(p.imap(first_phase, pool))
    with open(path + ".new", "wb") as f:
        pickle.dump(done, f)
    os.replace(path + ".new", path)
    counts = {ph: sum(d.phase == ph for d in done) for ph in PHASES}
    print(f"{name}: first contact {counts}, {sum(1 for d in done if d.contact and not d.phase)} unknown -> {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pool", help="build the training scenes for a scenario")
    p.add_argument("--scenario", default="tight", choices=SCENARIOS)
    p.add_argument("--seeds", type=int, default=1000)
    p.add_argument("--first-seed", type=int, default=POOL_FIRST_SEED)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--name", help="save as gap_pool_<name>.pkl (default: the scenario)")
    p.add_argument("--error-variants", type=int, default=1, help="independent error variants per shelf layout")
    p.add_argument("--error-seed", type=int, help="seed for generating independent perception and grasp seeds")
    p.add_argument("--perception", default="noisy", choices=PERCEPTIONS, help="how the rows are seen")
    p.add_argument("--segmenter", metavar="WEIGHTS", help="camera: learned book masks (learned_segmenter.py)")
    p.add_argument("--segmenter-device", choices=("cuda", "mps", "cpu"),
                   help="where the segmenter runs (default: the GPU if there is one)")
    p = sub.add_parser("phases", help="record each scene's first contact phase in an older pool")
    p.add_argument("name")
    p.add_argument("--workers", type=int, default=1)
    args = ap.parse_args()
    if args.cmd == "pool":
        build_pool(args.scenario, args.seeds, args.first_seed, args.workers, name=args.name,
                   error_variants=args.error_variants, error_seed=args.error_seed, perception=args.perception,
                   segmenter=args.segmenter, segmenter_device=args.segmenter_device)
    elif args.cmd == "phases":
        add_phases(args.name, args.workers)


if __name__ == "__main__":
    from bila.rl import gap_env                          # so the pool pickles gap_env.PoolEntry, not __main__'s (unloadable)
    gap_env.main()
