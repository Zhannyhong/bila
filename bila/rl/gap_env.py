"""
A Gymnasium environment for the gap-creation residual: the scripted
controller (insert_book.Episode) makes room for the book, and the policy
corrects its hand during the contact moves (Runner.contact: widen's pushes,
the lift and the clamp's swing, the shove).

    uv run python -m bila.rl.gap_env pool --scenario tight --seeds 1000 --workers 12   # build the training scenes once
    uv run python -m bila.rl.gap_env pool --scenario tight --seeds 40 --first-seed 20000 --name tight_val   # validation
    from bila.rl.gap_env import GapEnv
    env = GapEnv(scenarios=("tight",), first_phases=("widen",))                 # loads scenes/gap_pool_tight.pkl

One env step is one residual decision (every RESIDUAL_EVERY physics steps,
25 Hz). The scripted episode runs in a worker thread: when its Runner asks
the residual for a correction (decide), the thread hands the decision to the
env and waits; env.step(action) passes the action back and runs the thread
on to its next decision. The script keeps everything else (which step, the
plans, the looks, every move outside a contact), so between two decisions it
may do a lot: a look, a move to the next step.

An episode covers gap creation only: it ends when the script would insert
the book (Episode.insert is replaced with a stop), or on a failure. The full
reshelve, with the residual attached, is for evaluation.

Perception: the script and the policy both see the row through NoisyEyes, a
NoisyPerception (the wrist camera's measured errors, no rendering) behind the
interface Episode uses for its camera. The arm still moves to the views.

Observation (a Dict):
  actor   observation.ObservationBuilder's actor vector from the last look
          (what a real robot has), then: the contact phase (one-hot over
          PHASES), time in it, the script's tool target (position and two
          columns of its rotation, slot frame), the correction so far (over
          RESIDUAL_MAX) and the last action.
  critic  privileged extras only (the ground truth now, the books' physics):
          a critic built for it reads both.
Action: (3,) in [-1, 1], times RESIDUAL_STEP * action_scale: the increment of
the correction (shelf frame) at this decision. set_action_scale ramps it up.

Reward per decision: the change in a potential of the true state, less costs.
  potential   +GAP_REWARD per m of the true narrowest gap (Episode.narrowest_gap
              on the simulation), counted up to the book + GAP_ENOUGH;
              -TILT_PENALTY per deg a book (not a leaner being stood up)
              has tilted beyond TILT_FREE since the start; -LIFT_PENALTY per
              m a book's lowest corner has risen beyond LIFT_FREE.
  costs       FORCE_PENALTY per N of peak wrist force above FORCE_FREE during
              the contact moves, ACTION_PENALTY * |increment / RESIDUAL_STEP|^2,
              TIME_PENALTY.
  at the end  +SUCCESS_BONUS if the script reaches insert with the true gap at
              the book + GAP_MARGIN (the script's own test, Episode.widen_gap);
              -FAILURE_PENALTY if a book tilts TOPPLE_DEG more than at the
              start, a leaner falls TOPPLE_DEG the other way, the arm hits the
              shelf (Runner.watch_contacts), or the script gives up or fails.
              Reaching insert narrower (the book will be wedged) gets neither.
x/y displacement of the books is not penalised: widening and shoving move
books along and into the row on purpose (evaluation still reports it).

Training scenes (a pool per scenario, seeds from POOL_FIRST_SEED, never the
test seeds 0-49): each settled scene's state, its camera-noise seed, and what
a run of the script up to its first contact move found: the inspection views
and the first arm plan (choose_arm; planning them is most of a tight
episode's time), whether it makes any contact at all (rows wide enough
already are left out), and the phase of its first contact. The residual
hasn't acted before them, so they are exact; each is used only if its inputs
hash the same. first_phases picks scenes by their first contact: a tight row
the script shoves (its neighbours touch) takes ~10x as long as one it widens,
and belongs with the packed rows.
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
from contextlib import contextmanager
from dataclasses import dataclass
from multiprocessing import Pool
from typing import Dict, List, Optional, Sequence

import gymnasium as gym
import mujoco
import numpy as np

from bila.control.insert_book import (DEFAULT_PANDA, GAP_MARGIN, LEAN_MIN_DEG, RESIDUAL_MAX, RESIDUAL_STEP, Episode, build_model,
                         make_episode)
from bila.rl.observation import TOOL_OFFSET, ObservationBuilder, book_names_by_offset, robot_state
from bila.vision.perception import GroundTruthPerception, NoisyPerception
from bila.sim.shelf_scene import ShelfConfig, book_tilt_deg

from bila import ROOT as HERE
POOL_DIR = os.path.join(HERE, "scenes")
POOL_FIRST_SEED = 10_000     # training scene seeds start here (test seeds are 0-49)
STATE = mujoco.mjtState.mjSTATE_INTEGRATION
PHASES = ("widen", "lift", "clamp", "shove")
MAX_DECISIONS = 1500         # truncate an episode after this many (packed ones take ~400)

# Reward (per decision; see the module docstring)
GAP_REWARD = 1000.0          # per m of true narrowest gap (1 per mm) ...
GAP_ENOUGH = GAP_MARGIN      # ... counted up to the book plus this
TILT_PENALTY = 0.1           # per deg of tilt change ...
TILT_FREE = 2.0              # ... beyond this
LIFT_PENALTY = 500.0         # per m a book's lowest corner rises (0.5 per mm) ...
LIFT_FREE = 0.001            # ... beyond this (contacts settle a little)
FORCE_PENALTY = 0.01         # per N of peak wrist force ...
FORCE_FREE = 60.0            # ... beyond this
ACTION_PENALTY = 0.1         # times |increment / RESIDUAL_STEP|^2
TIME_PENALTY = 0.01
SUCCESS_BONUS = 10.0
FAILURE_PENALTY = 10.0
TOPPLE_DEG = 15.0            # a book tilting this much more than at the start has toppled


# ---------------------------------------------------------------------------
# Perception without rendering
# ---------------------------------------------------------------------------
class NoisyEyes:
    """Episode's eyes (the camera interface its look() uses) through a
    NoisyPerception: one estimate per look, with the camera's measured
    errors (calibrated on the first look's fused sweep), drawn from rng."""

    def __init__(self, model, row, rng: np.random.Generator, scale: float = 1.0):
        self.perception = NoisyPerception(GroundTruthPerception(model, row), rng, scale)
        self.data = None

    def capture(self, data):
        self.data = data                    # the arm holds still for the "image": estimate it at the end of the look
        return None

    def estimate_from_frames(self, frames, x_slot=None, placed_x=None):
        return self.perception.estimate(self.data)

    def close(self):
        pass


# ---------------------------------------------------------------------------
# The script, stopped at insert, with its first plan cached
# ---------------------------------------------------------------------------
class ReachedInsert(BaseException):
    """The script would now insert the book: the gap-creation episode is over.
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

    @property
    def noise_rng(self) -> np.random.Generator:
        """The camera noise of this scene: the same every time it is used."""
        return np.random.default_rng([self.seed, 1])


class Packed:
    """A pool entry kept pickled, with its seed and contact flag at hand:
    unpickled, a pool of 1000 takes ~270 MB in every worker."""
    phase = ""                              # (pools packed before phases were recorded)

    def __init__(self, entry: PoolEntry):
        self.seed, self.contact, self.phase, self.blob = entry.seed, entry.contact, entry.phase, pickle.dumps(entry)

    def unpack(self) -> PoolEntry:
        return pickle.loads(self.blob)


def make_entry(cfg: ShelfConfig, scenario: str, seed: int) -> PoolEntry:
    _, data, row = make_episode(cfg, scenario, np.random.default_rng(seed), DEFAULT_PANDA)
    model = build_model(cfg, row, DEFAULT_PANDA)
    state = np.empty(mujoco.mj_stateSize(model, STATE))
    mujoco.mj_getState(model, data, state, STATE)
    return PoolEntry(scenario, seed, row, state)


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
    """Episode for training: stops where the script would insert the book,
    reuses the pool entry's inspection views and first arm plan, and records
    the peak wrist force during the contact moves (peak_force, reset by the
    env)."""

    def __init__(self, model, data, entry: PoolEntry, eyes):
        super().__init__(model, data, entry.row, None, lambda *_: None, eyes)
        self.entry, self.planned, self.peak_force = entry, False, 0.0

    def insert(self):
        raise ReachedInsert

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
        self.entry.first_plan = (key, q.copy(), copy.deepcopy(label))
        return q, label


@contextmanager
def truth(ep: Episode):
    """Ask the episode about the row as it really is: its methods read the
    simulation when it has no eyes. (Called only while its thread waits.)"""
    eyes, ep.eyes = ep.eyes, None
    try:
        yield ep
    finally:
        ep.eyes = eyes


# ---------------------------------------------------------------------------
# The script's thread and the residual that hands decisions to the env
# ---------------------------------------------------------------------------
class Bridge:
    """The residual the episode's Runner calls (decide): it hands the decision
    (the Contact) to the env and blocks until the env sends the increment."""

    def __init__(self):
        self.decisions: queue.Queue = queue.Queue()
        self.actions: queue.Queue = queue.Queue()

    def decide(self, run, contact):
        self.decisions.put(("decide", contact))
        inc = self.actions.get()
        if inc is None:
            raise Abandon
        return inc


def run_script(ep: GapEpisode, bridge: Bridge):
    """The worker thread: the script up to insert, then report how it ended."""
    try:
        res = ep.reshelve()
        end = ("gave_up", res.get("skipped", ""))
    except ReachedInsert:
        end = ("insert", "")
    except Abandon:
        return
    except RuntimeError as e:                # the script found no way to go on (no plan, unseen books)
        end = ("error", str(e))
    except Exception as e:                   # a bug: still end the episode, and say what it was
        end = ("crash", f"{type(e).__name__}: {e}")
    bridge.decisions.put(("done", end))


# ---------------------------------------------------------------------------
# The environment
# ---------------------------------------------------------------------------
class GapEnv(gym.Env):
    """See the module docstring. scenarios: drawn from uniformly per episode.
    pool: {scenario: [PoolEntry or Packed]} (default: scenes/gap_pool_<scenario>.pkl).
    first_phases: only scenes whose first contact is one of these (None: all).
    noise_scale: NoisyPerception's scale. action_scale: of RESIDUAL_STEP.
    reset(options={"index": i}) starts on scene i of the first scenario's
    (filtered) list instead of a random one: for validation."""

    metadata = {"render_modes": []}

    def __init__(self, scenarios: Sequence[str] = ("tight",), pool: Optional[Dict[str, list]] = None,
                 first_phases: Optional[Sequence[str]] = None, noise_scale: float = 1.0, action_scale: float = 1.0,
                 max_decisions: int = MAX_DECISIONS, cfg: Optional[ShelfConfig] = None):
        self.cfg = cfg or ShelfConfig()
        self.scenarios = tuple(scenarios)
        pool = pool if pool is not None else {s: load_pool(s) for s in self.scenarios}
        packed = {s: [e if isinstance(e, Packed) else Packed(e) for e in pool[s]] for s in self.scenarios}
        self.pool = {s: [p for p in packed[s] if first_phases is None or p.phase in first_phases]
                     for s in self.scenarios}
        for s in self.scenarios:
            if not any(e.contact is not False for e in self.pool[s]):
                raise ValueError(f"no {s} scene in the pool makes any contact move")
        self.noise_scale, self.action_scale, self.max_decisions = noise_scale, action_scale, max_decisions
        self.ep: Optional[GapEpisode] = None
        self.thread: Optional[threading.Thread] = None
        self.bridge: Optional[Bridge] = None
        n_actor, n_critic = self.sizes()
        box = lambda n: gym.spaces.Box(-np.inf, np.inf, (n,), np.float32)
        self.observation_space = gym.spaces.Dict({"actor": box(n_actor), "critic": box(n_critic)})
        self.action_space = gym.spaces.Box(-1.0, 1.0, (3,), np.float32)

    def set_action_scale(self, scale: float) -> None:
        self.action_scale = float(scale)

    # --- episode lifecycle ---------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.stop()
        index = (options or {}).get("index")
        while True:
            scenario = self.scenarios[0 if index is not None else self.np_random.integers(len(self.scenarios))]
            entries = [e for e in self.pool[scenario] if e.contact is not False]
            if not entries:
                raise RuntimeError("no scene left in the pool that makes a contact move")
            packed = entries[index % len(entries) if index is not None else self.np_random.integers(len(entries))]
            entry = packed.unpack()
            msg = self.start(entry)
            if msg[0] == "decide":
                break
            packed.contact = False                                      # ended before any contact: never again
            self.ep = None
        ep = self.ep
        self.builder = ObservationBuilder(ep.est, target=(ep.t, ep.h))
        self.ground_truth = GroundTruthPerception(ep.m, ep.row)
        self.book_names = book_names_by_offset(ep.row)
        truth_now = self.ground_truth.estimate(ep.d)
        # Neighbours leaning into the slot at the start are meant to move (stood up).
        self.leaners = {i for i in (ep.row.target_index - 1, ep.row.target_index)
                        if -np.sign(ep.offset(i)) * truth_now.book(ep.offset(i)).lean > math.radians(LEAN_MIN_DEG)}
        self.steady = [i for i in range(len(ep.row_bodies)) if i not in self.leaners]
        self.start_lift = self.lifts()
        self.collisions0 = ep.run.collision["steps"]
        self.phi, self.gap0, _ = self.potential()
        self.contact, self.last_action, self.decisions = msg[1], np.zeros(3), 0
        return self.observe(truth_now), {"scenario": entry.scenario, "seed": entry.seed}

    def start(self, entry: PoolEntry):
        """Start the script on a pool entry; return its first message."""
        model, data = load_scene(self.cfg, entry)
        self.ep = GapEpisode(model, data, entry, NoisyEyes(model, entry.row, entry.noise_rng, self.noise_scale))
        self.bridge = Bridge()
        self.ep.residual = self.bridge
        self.thread = threading.Thread(target=run_script, args=(self.ep, self.bridge), daemon=True)
        self.thread.start()
        return self.bridge.decisions.get()

    def step(self, action):
        a = np.clip(np.asarray(action, dtype=float), -1.0, 1.0)
        inc = a * RESIDUAL_STEP * self.action_scale
        self.ep.peak_force = 0.0
        self.bridge.actions.put(inc)
        kind, what = self.bridge.decisions.get()
        self.decisions += 1
        ep = self.ep
        truth_now = self.ground_truth.estimate(ep.d)
        phi, gap, tilt = self.potential()
        reward = (phi - self.phi - FORCE_PENALTY * max(0.0, ep.peak_force - FORCE_FREE)
                  - ACTION_PENALTY * float(np.sum((inc / RESIDUAL_STEP) ** 2)) - TIME_PENALTY)
        self.phi, self.last_action = phi, a
        failure = self.failure(truth_now, tilt)
        if kind == "done" and failure is None and what[0] != "insert":
            failure = f"the script {'gave up' if what[0] == 'gave_up' else 'failed'}: {what[1]}"
        info = {"gap_m": gap, "gap_gain_m": gap - self.gap0, "peak_force_n": ep.peak_force}
        terminated = failure is not None or kind == "done"
        truncated = not terminated and self.decisions >= self.max_decisions
        if failure is not None:
            reward -= FAILURE_PENALTY
            info["outcome"], info["failure"] = "failure", failure
        elif kind == "done":
            with truth(ep):
                wide = ep.widen_gap() >= ep.t + GAP_MARGIN - 0.001
            reward += SUCCESS_BONUS if wide else 0.0
            info["outcome"] = "success" if wide else "narrow"
        elif truncated:
            info["outcome"] = "truncated"
        self.contact = what if kind == "decide" else None
        obs = self.observe(truth_now)
        if terminated or truncated:
            info.update(decisions=self.decisions, method=ep.info.get("method", ""),
                        collision_during=sorted(ep.run.collision["during"]))
            self.stop()
        return obs, float(reward), terminated, truncated, info

    def stop(self):
        """End the script's thread, if it is waiting for an action."""
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
        """(potential, true narrowest gap, largest tilt change of a steady book)."""
        ep = self.ep
        with truth(ep):
            gap = ep.narrowest_gap()
        tilt = np.array([book_tilt_deg(ep.d.xquat[ep.row_bodies[i]]) for i in self.steady])
        dtilt = np.abs(tilt - ep.start_tilt[self.steady])
        lift = self.lifts() - self.start_lift
        phi = (GAP_REWARD * min(gap, ep.t + GAP_ENOUGH) - TILT_PENALTY * np.maximum(0.0, dtilt - TILT_FREE).sum()
               - LIFT_PENALTY * np.maximum(0.0, lift - LIFT_FREE).sum())
        return float(phi), float(gap), float(dtilt.max(initial=0.0))

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
    def extras(self) -> np.ndarray:
        """What only the env knows: the contact phase, time in it, the script's
        tool target, the correction so far and the last action."""
        run, c, b = self.ep.run, self.contact, self.builder
        phase = np.array([c is not None and c.phase == p for p in PHASES], dtype=float)
        in_phase = c.steps * self.ep.m.opt.timestep if c is not None else 0.0
        target = b.to_slot(run.pos + run.rot[:, 2] * TOOL_OFFSET)
        rot = b.rot_to_slot(run.rot)[:, :2].T.ravel()
        return np.concatenate([phase, [in_phase], target, rot, run.offset / RESIDUAL_MAX, self.last_action])

    def observe(self, truth_now):
        ep = self.ep
        actor = self.builder.actor(ep.est, robot_state(ep.m, ep.d), now=ep.d.time)
        critic = self.builder.critic(ep.m, ep.d, truth_now, self.book_names)
        return {"actor": np.concatenate([actor, self.extras()]).astype(np.float32), "critic": critic}

    def sizes(self):
        """Lengths of the actor and critic vectors (from the first scene's
        ground truth: they don't depend on the row)."""
        entry = self.pool[self.scenarios[0]][0].unpack()
        model, data = load_scene(self.cfg, entry)
        est = GroundTruthPerception(model, entry.row).estimate(data)
        b = ObservationBuilder(est, target=(entry.row.target_book.thickness, entry.row.target_book.height))
        n_actor = len(b.actor(est, robot_state(model, data), now=data.time))
        n_critic = len(b.critic(model, data, est, book_names_by_offset(entry.row)))
        return n_actor + len(PHASES) + 1 + 3 + 6 + 3 + 3, n_critic


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


def warm(args) -> PoolEntry:
    """Make a pool entry and run the script on it up to its first contact
    move, so the entry holds its inspection views and first plan."""
    scenario, seed = args
    cfg = ShelfConfig()
    try:
        entry = make_entry(cfg, scenario, seed)
    except RuntimeError as e:                # no stable row for this seed
        return PoolEntry(scenario, seed, None, np.zeros(0), contact=False, why=str(e))
    model, data = load_scene(cfg, entry)
    ep = GapEpisode(model, data, entry, NoisyEyes(model, entry.row, entry.noise_rng))
    ep.residual = first = FirstContact()
    try:
        res = ep.reshelve()
        entry.contact, entry.why = False, "the script gave up: " + res.get("skipped", "")
    except Abandon:
        entry.contact, entry.phase = True, first.phase
    except ReachedInsert:
        entry.contact, entry.why = False, "wide enough already"
    except RuntimeError as e:
        entry.contact, entry.why = False, str(e)
    except Exception as e:                   # a bug: leave the scene out rather than lose the whole pool
        entry.contact, entry.why = False, f"crash: {type(e).__name__}: {e}"
    return entry


def build_pool(scenario: str, seeds: int, first_seed: int = POOL_FIRST_SEED, workers: int = 1,
               root: str = POOL_DIR, name: Optional[str] = None) -> List[PoolEntry]:
    """Build and save a pool (as `name`, default the scenario). Refuses to
    overwrite one: they take hours."""
    path = pool_path(name or scenario, root)
    if os.path.exists(path):
        raise FileExistsError(f"{path} exists: move it away first")
    jobs = [(scenario, s) for s in range(first_seed, first_seed + seeds)]
    if workers > 1:
        with Pool(workers) as p:
            entries = []
            for i, e in enumerate(p.imap(warm, jobs)):
                entries.append(e)
                print(f"  {i + 1}/{len(jobs)}  seed {e.seed}: {e.phase if e.contact else e.why}", flush=True)
    else:
        entries = [warm(j) for j in jobs]
    os.makedirs(root, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump([Packed(e) for e in entries], f)
    phases = {p: sum(e.phase == p for e in entries) for p in PHASES}
    print(f"{scenario}: {sum(e.contact for e in entries)} of {len(entries)} scenes make contact "
          f"(first contact: {phases}) -> {path}")
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
    p.add_argument("--scenario", default="tight", choices=("tight", "leaning", "packed"))
    p.add_argument("--seeds", type=int, default=1000)
    p.add_argument("--first-seed", type=int, default=POOL_FIRST_SEED)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--name", help="save as gap_pool_<name>.pkl (default: the scenario)")
    p = sub.add_parser("phases", help="record each scene's first contact phase in an older pool")
    p.add_argument("name")
    p.add_argument("--workers", type=int, default=1)
    args = ap.parse_args()
    if args.cmd == "pool":
        build_pool(args.scenario, args.seeds, args.first_seed, args.workers, name=args.name)
    elif args.cmd == "phases":
        add_phases(args.name, args.workers)


if __name__ == "__main__":
    from bila.rl import gap_env                          # so the pool pickles gap_env.PoolEntry, not __main__'s (unloadable)
    gap_env.main()
