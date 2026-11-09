"""
PPO for the reshelving residual (gap_env.GapEnv: making room and inserting),
with each worker collecting its rollout on its own.

    uv run --group rl python train_residual.py --decisions 4000000                        # all four scenarios
    uv run --group rl python train_residual.py --name tight --scenarios tight --phases widen --no-insert   # gap only
    uv run --group rl python train_residual.py --name insertion --scenarios clear tight \
        --control-phases slide push --no-learn-stop                    # script prepares, PPO inserts
    uv run --group rl python train_residual.py --resume runs/reshelve/last.pt              # carry on after a stop
    uv run --group rl tensorboard --logdir runs

It needs a training pool per scenario (scenes/gap_pool_<scenario>.pkl) and a
validation pool (gap_pool_<scenario>_val.pkl, seeds 20000+): gap_env.py pool.

Why not stable-baselines3: its vectorised envs step every worker in
lockstep, and the script now and then re-plans for 10-45 s (choose_strong, a
later choose_arm), which stalls all the others: 12 workers made 5.6
decisions/s. Here each worker process holds a copy of the actor and collects
`rollout` decisions by itself; the learner waits for all of them, updates the
policy, and sends the new weights. So a stall holds up only its own worker,
until the end of the rollout.

This module stays free of torch: the worker processes import it (spawn), and
torch would cost each ~700 MB. The learner (the policy, the PPO update) is
residual_ppo.py, imported only by train(). The workers act with NumpyActor,
a numpy copy of the policy's actor.

action_scale ramps from ramp_start to 1 over the first ramp_decisions: early
corrections are small, as random full ones topple books. (Not the end
action: its bias, end_bias, keeps it rare at first instead.)
Every eval_every updates: the deterministic policy (the actor's mean) on
val_scenes layouts and their error variants, at fixed eval_action_scale,
against the script alone (a zero correction, baseline.json): success rate (the book placed;
--no-insert: the gap wide enough), overall and per scenario, gap gained,
failures, moves ended, each scene's in val_<update>.json. Each eval saves
last.pt, and best.pt when the success rate (then the gap) is the best yet.
Logs: TensorBoard, log.jsonl and per-episode seeds/reward totals in
episodes.jsonl in runs/<name>/. The baseline is regenerated on resume.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import time
import traceback
from dataclasses import asdict, dataclass, fields
from typing import List, Tuple

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, "runs")


@dataclass
class Config:
    name: str = "reshelve"
    scenarios: Tuple[str, ...] = ("clear", "tight", "leaning", "packed")
    phases: Tuple[str, ...] = ()                 # optional FIRST-contact scene filter, empty means no filter
    control_phases: Tuple[str, ...] = ("widen", "lift", "clamp", "shove", "slide", "push")
    learn_stop: bool = True                      # False: five continuous corrections, scripted stopping
    insert: bool = True                          # through the insertion (False: gap creation only)
    train_suffix: str = ""                       # training pools: gap_pool_<scenario><train_suffix>.pkl
    val_suffix: str = "_val"                     # validation pools: gap_pool_<scenario><val_suffix>.pkl
    val_scenes: int = 10                         # distinct geometries per scenario
    val_variants: int = 0                        # variants per geometry, 0 means all in the validation pool
    eval_action_scale: float = 1.0               # fixed across checkpoints, independent of the training ramp
    workers: int = 12
    decisions: int = 2_000_000                   # stop after this many (all workers)
    hours: float = 0.0                           # ... or after this long (0: no limit)
    rollout: int = 512                           # decisions per worker per update
    gamma: float = 0.99                          # 25 Hz: a ~4 s horizon
    lam: float = 0.95
    clip: float = 0.2
    epochs: int = 10
    minibatch: int = 1024
    lr: float = 3e-4
    max_grad_norm: float = 0.5
    vf_coef: float = 0.5
    ent_coef: float = 0.0
    hidden: int = 256
    log_std: float = -1.0                        # initial: std 0.37 of a full increment
    end_log_std: float = -0.5                    # initial spread of the end action ...
    end_bias: float = -1.8                       # ... and its mean: above 0.5 in 1 decision of ~10,000
    ramp_start: float = 0.25
    ramp_decisions: int = 300_000
    noise_scale: float = 1.0                     # NoisyPerception's
    grasp_scale: float = 1.0                     # gap_env.GraspError's
    eval_every: int = 10                         # updates
    seed: int = 0
    resume: str = ""


class RunningStats:
    """Running mean and variance (parallel Welford), for normalising."""

    def __init__(self, shape=()):
        self.mean, self.var, self.count = np.zeros(shape), np.ones(shape), 1e-4

    def update(self, x: np.ndarray) -> None:
        x = np.asarray(x, dtype=np.float64).reshape(-1, *self.mean.shape)
        mean, var, n = x.mean(axis=0), x.var(axis=0), len(x)
        delta, total = mean - self.mean, self.count + n
        self.mean = self.mean + delta * n / total
        self.var = (self.var * self.count + var * n + delta ** 2 * self.count * n / total) / total
        self.count = total

    def normalize(self, x, clip: float = 10.0) -> np.ndarray:
        return np.clip((x - self.mean) / np.sqrt(self.var + 1e-8), -clip, clip).astype(np.float32)

    def state(self) -> dict:
        return {"mean": self.mean, "var": self.var, "count": self.count}

    def load(self, state: dict) -> "RunningStats":
        self.mean, self.var, self.count = state["mean"], state["var"], state["count"]
        return self


def ramp(cfg: Config, decisions: int) -> float:
    return cfg.ramp_start + (1.0 - cfg.ramp_start) * min(1.0, decisions / max(1, cfg.ramp_decisions))


# ---------------------------------------------------------------------------
# Workers
# ---------------------------------------------------------------------------
class NumpyActor:
    """residual_ppo.Policy's actor in numpy (Policy.actor_weights), with the
    observation normalisation it was trained with: choosing actions is all a
    worker does with it."""

    def __init__(self, msg: dict, rng: np.random.Generator):
        self.layers, self.std = msg["actor"]["layers"], np.exp(msg["actor"]["log_std"])
        self.norm, self.rng = RunningStats().load(msg["actor_stats"]), rng

    def mean(self, actor_obs) -> np.ndarray:
        h = self.norm.normalize(actor_obs)
        for i, (W, b) in enumerate(self.layers):
            h = h @ W.T + b
            if i < len(self.layers) - 1:
                h = np.tanh(h)
        return h

    def __call__(self, obs, sample: bool) -> np.ndarray:
        m = self.mean(obs["actor"])
        return (m + self.std * self.rng.standard_normal(m.shape) if sample else m).astype(np.float32)


def make_env(cfg: Config, validation: bool = False):
    from gap_env import GapEnv, load_pool
    suffix = cfg.val_suffix if validation else cfg.train_suffix
    pool = {s: load_pool(s + suffix) for s in cfg.scenarios}
    return GapEnv(scenarios=cfg.scenarios, pool=pool, first_phases=cfg.phases or None, noise_scale=cfg.noise_scale,
                  insert=cfg.insert, grasp_scale=cfg.grasp_scale, control_phases=cfg.control_phases,
                  learn_stop=cfg.learn_stop)


def episode_record(info: dict, ret: float, scenario: str = "") -> dict:
    keep = ("outcome", "failure", "decisions", "gap_gain_m", "method", "wide_at_insert", "ended", "offset_m",
            "tilt_deg", "total_shift_m", "safe_success", "seed", "geometry_seed", "variant", "perception_seed",
            "grasp_seed", "grasp_error", "preparation_failures", "reward_totals")
    return {"return": ret, "scenario": scenario, **{k: info[k] for k in keep if k in info}}


def collect(env, actor: NumpyActor, n: int, state: dict) -> dict:
    """n decisions on env, carrying on the episode in progress (state)."""
    if state.get("obs") is None:
        state["obs"], info0 = env.reset(seed=state["seed"])
        state["ret"], state["scenario"] = 0.0, info0["scenario"]
    obs = state["obs"]
    out = {k: [] for k in ("actor", "critic", "action", "reward", "terminated", "truncated")}
    finals, episodes, t0 = {}, [], time.perf_counter()
    for t in range(n):
        a = actor(obs, sample=True)
        nxt, r, term, trunc, info = env.step(a)
        out["actor"].append(obs["actor"])
        out["critic"].append(obs["critic"])
        out["action"].append(a)
        out["reward"].append(r)
        out["terminated"].append(term)
        out["truncated"].append(trunc)
        state["ret"] += r
        if term or trunc:
            episodes.append(episode_record(info, state["ret"], state["scenario"]))
            if trunc and not term:
                finals[t] = (nxt["actor"], nxt["critic"])
            nxt, info0 = env.reset()
            state["ret"], state["scenario"] = 0.0, info0["scenario"]
        obs = nxt
    state["obs"] = obs
    arrays = {k: np.asarray(v, dtype=bool if k in ("terminated", "truncated") else np.float32) for k, v in out.items()}
    return {**arrays, "finals": finals, "last": (obs["actor"], obs["critic"]), "episodes": episodes,
            "wall_s": time.perf_counter() - t0}


def evaluate(env, actor: NumpyActor, scenes, zero: bool) -> List[dict]:
    """The deterministic policy (or a zero correction) on validation scenes,
    each exact geometry/error case. Preparation failures stay in the denominator."""
    from gap_env import PreparationFailed
    out = []
    for case in scenes:
        try:
            obs, info0 = env.reset(options=case)
        except PreparationFailed as e:
            out.append({**e.record, "return": 0.0, "action_scale": env.action_scale})
            continue
        ret = 0.0
        while True:
            a = np.zeros(env.action_space.shape, np.float32) if zero else actor(obs, sample=False)
            obs, r, term, trunc, info = env.step(a)
            ret += r
            if term or trunc:
                out.append({**info0, **episode_record(info, ret, case["scenario"]),
                            "action_scale": env.action_scale})
                break
    return out


def worker(conn, cfg: Config, index: int):
    """A worker process: an env, and commands from the learner."""
    env = val = None
    try:
        env = make_env(cfg)
        rng = np.random.default_rng([cfg.seed, index])
        state = {"seed": cfg.seed * 1000 + index}
        while True:
            cmd, msg = conn.recv()
            if cmd == "close":
                break
            actor = NumpyActor(msg, rng)
            if cmd == "collect":
                env.set_action_scale(msg["scale"])
                conn.send(("ok", collect(env, actor, msg["n"], state)))
            elif cmd == "evaluate":
                val = val or make_env(cfg, validation=True)
                val.set_action_scale(msg["scale"])
                conn.send(("ok", evaluate(val, actor, msg["scenes"], msg["zero"])))
    except Exception:
        conn.send(("error", traceback.format_exc()))
    finally:
        for e in (env, val):
            if e is not None:
                e.close()


class Workers:
    """The worker processes, and sending them all a command at once."""

    def __init__(self, cfg: Config):
        ctx = mp.get_context("spawn")             # (fork after torch and threads is unsafe)
        self.conns, self.procs = [], []
        for i in range(cfg.workers):
            a, b = ctx.Pipe()
            p = ctx.Process(target=worker, args=(b, cfg, i), daemon=True)
            p.start()
            self.conns.append(a)
            self.procs.append(p)

    def ask(self, msgs: List[Tuple[str, dict]]) -> list:
        for c, m in zip(self.conns, msgs):
            c.send(m)
        out = []
        for i, c in enumerate(self.conns):
            try:
                status, res = c.recv()
            except EOFError:
                raise RuntimeError(f"worker {i} died (exit code {self.procs[i].exitcode}; -9: out of memory?)")
            if status == "error":
                raise RuntimeError(f"worker {i} failed:\n" + res)
            out.append(res)
        return out

    def close(self):
        for c in self.conns:
            try:
                c.send(("close", None))
            except (BrokenPipeError, OSError):
                pass
        for p in self.procs:
            p.join(timeout=30)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def summary(episodes: List[dict]) -> dict:
    """Success rate (overall and per scenario), the other outcomes, how often
    the gap was wide enough at insert, moves ended per episode, gap gained
    and return over episodes."""
    if not episodes:
        return {}
    o = [e.get("outcome") for e in episodes]
    out = {"episodes": len(episodes), **{k: o.count(k) / len(o) for k in ("success", "missed", "failure", "narrow",
                                                                           "truncated", "preparation_failed")},
           "gap_gain_mm": 1000 * float(np.mean([e.get("gap_gain_m", 0.0) for e in episodes])),
           "ended_per_episode": float(np.mean([sum(e.get("ended", {}).values()) for e in episodes])),
           "return": float(np.mean([e["return"] for e in episodes]))}
    controlled = [e for e in episodes if e.get("outcome") != "preparation_failed"]
    out["controlled_episodes"] = len(controlled)
    out["preparation_attempts_failed"] = sum(len(e.get("preparation_failures", [])) for e in episodes)
    if controlled:
        out["success_given_control"] = sum(e["outcome"] == "success" for e in controlled) / len(controlled)
        for k in sorted({k for e in controlled for k in e.get("reward_totals", {})}):
            out[f"reward_{k}"] = float(np.mean([e.get("reward_totals", {}).get(k, 0.0) for e in controlled]))
        for kind in sorted({e["outcome"] for e in controlled}):
            out[f"return_{kind}"] = float(np.mean([e["return"] for e in controlled if e["outcome"] == kind]))
    wide = [e["wide_at_insert"] for e in episodes if "wide_at_insert" in e]
    if wide:
        out["wide_at_insert"] = float(np.mean(wide))
    for s in sorted({e.get("scenario", "") for e in episodes} - {""}):
        mine = [e.get("outcome") for e in episodes if e.get("scenario") == s]
        out[f"success_{s}"] = mine.count("success") / len(mine)
    return out


def validation_scenes(cfg: Config) -> List[dict]:
    """Fixed geometry/error identities, sorted independently of pool ordering.
    val_scenes counts layouts, not variants. Failed preparations are retained
    unless an explicit first-contact filter excludes their unknown phase."""
    from gap_env import load_pool
    if cfg.val_scenes < 1 or cfg.val_variants < 0:
        raise ValueError("val_scenes must be positive and val_variants nonnegative")
    out = []
    for s in cfg.scenarios:
        pool = [p for p in load_pool(s + cfg.val_suffix) if not cfg.phases or p.phase in cfg.phases]
        seeds = sorted({p.seed for p in pool})[:cfg.val_scenes]
        for seed in seeds:
            variants = sorted((p for p in pool if p.seed == seed), key=lambda p: p.variant)
            out.extend(p.unpack().identity() for p in (variants[:cfg.val_variants] if cfg.val_variants else variants))
    if not out:
        raise ValueError("no validation cases match the scene filter")
    return out


def case_key(case: dict) -> str:
    return json.dumps({k: case[k] for k in ("scenario", "geometry_seed", "variant", "perception_seed", "grasp_seed")},
                      sort_keys=True)


def paired_summary(baseline: List[dict], policy: List[dict]) -> dict:
    """Recoveries and regressions on exactly matching geometry/error draws."""
    base, learned = ({case_key(e): e["outcome"] == "success" for e in rows} for rows in (baseline, policy))
    if base.keys() != learned.keys() or len(base) != len(baseline) or len(learned) != len(policy):
        raise ValueError("baseline and policy must contain the same unique validation cases")
    recovered = sum(learned[k] and not base[k] for k in base)
    lost = sum(base[k] and not learned[k] for k in base)
    return {"recovered": recovered, "lost": lost, "success_delta": (recovered - lost) / len(base)}


def run_eval(workers: Workers, learner, cfg: Config, scenes: List[dict], zero: bool) -> List[dict]:
    split = [scenes[i::cfg.workers] for i in range(cfg.workers)]
    msg = learner.message(zero=zero, scale=cfg.eval_action_scale)
    res = workers.ask([("evaluate", {**msg, "scenes": s}) for s in split])
    return sorted((e for r in res for e in r), key=case_key)


def train(cfg: Config) -> None:
    if not 0 < cfg.eval_action_scale <= 1:
        raise ValueError("eval_action_scale must be in (0, 1]")
    import torch
    from torch.utils.tensorboard import SummaryWriter
    from residual_ppo import Learner
    torch.set_num_threads(4)
    out = os.path.join(RUNS, cfg.name)
    os.makedirs(out, exist_ok=True)
    probe = make_env(cfg)
    sizes = (probe.observation_space["actor"].shape[0], probe.observation_space["critic"].shape[0],
             probe.action_space.shape[0])
    probe.close()
    del probe
    learner = Learner(cfg, sizes)
    if cfg.resume:
        learner.load(cfg.resume)
    with open(os.path.join(out, "config.json"), "w") as f:
        json.dump(asdict(cfg), f, indent=1)
    writer = SummaryWriter(out)
    log = open(os.path.join(out, "log.jsonl"), "a")
    episode_log = open(os.path.join(out, "episodes.jsonl"), "a")
    val_scenes = validation_scenes(cfg)
    workers = Workers(cfg)
    t_start = time.perf_counter()
    try:
        base_path = os.path.join(out, "baseline.json")
        # Re-evaluate on resume too: the code, pools or phase configuration may
        # have changed, so a baseline from the directory is not evidence for this run.
        baseline = run_eval(workers, learner, cfg, val_scenes, zero=True)
        with open(base_path, "w") as f:
            json.dump(baseline, f, indent=1)
        print(f"script alone on {len(val_scenes)} validation scenes: {summary(baseline)}", flush=True)
        while learner.decisions < cfg.decisions:
            if cfg.hours and time.perf_counter() - t_start > cfg.hours * 3600:
                break
            t0, scale = time.perf_counter(), ramp(cfg, learner.decisions)
            rollouts = workers.ask([("collect", learner.message(scale=scale, n=cfg.rollout))] * cfg.workers)
            t_collect = time.perf_counter() - t0
            stats = learner.update(rollouts)
            episodes = [e for r in rollouts for e in r["episodes"]]
            for episode in episodes:
                episode_log.write(json.dumps({"update": learner.updates, **episode}) + "\n")
            episode_log.flush()
            row = {"update": learner.updates, "decisions": learner.decisions, "scale": scale,
                   "decisions_per_s": cfg.workers * cfg.rollout / t_collect, "collect_s": t_collect,
                   "update_s": time.perf_counter() - t0 - t_collect, **stats,
                   **{f"train_{k}": v for k, v in summary(episodes).items()}}
            if learner.updates % cfg.eval_every == 0:
                scenes = run_eval(workers, learner, cfg, val_scenes, zero=False)
                json.dump(scenes, open(os.path.join(out, f"val_{learner.updates:05d}.json"), "w"), indent=1)
                val = summary(scenes)
                val.update(paired_summary(baseline, scenes), action_scale=cfg.eval_action_scale)
                row.update({f"val_{k}": v for k, v in val.items()})
                key = (val["success"], val["gap_gain_mm"])
                learner.save(os.path.join(out, "last.pt"))
                if learner.best is None or key > tuple(learner.best):
                    learner.best = key
                    learner.save(os.path.join(out, "best.pt"))
            for k, v in row.items():
                if isinstance(v, (int, float)) and k not in ("update", "decisions"):
                    writer.add_scalar(k, v, learner.decisions)
            log.write(json.dumps(row) + "\n")
            log.flush()
            print(" ".join(f"{k}={v:.3g}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()), flush=True)
    except KeyboardInterrupt:
        print("stopped: saving last.pt", flush=True)
    finally:
        learner.save(os.path.join(out, "last.pt"))
        workers.close()
        writer.close()
        log.close()
        episode_log.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    for f in fields(Config):
        default = f.default
        if isinstance(default, tuple):
            ap.add_argument("--" + f.name.replace("_", "-"), nargs="+", default=list(default),
                            help=("filter scenes by their first contact" if f.name == "phases" else
                                  "contacts controlled by PPO" if f.name == "control_phases" else None))
        elif isinstance(default, bool):                  # --insert / --no-insert
            ap.add_argument("--" + f.name.replace("_", "-"), action=argparse.BooleanOptionalAction, default=default)
        else:
            ap.add_argument("--" + f.name.replace("_", "-"), type=type(default), default=default)
    args = vars(ap.parse_args())
    train(Config(**{k: tuple(v) if isinstance(v, list) else v for k, v in args.items()}))


if __name__ == "__main__":
    main()
