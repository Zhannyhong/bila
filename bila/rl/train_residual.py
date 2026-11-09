"""
PPO for the gap-creation residual (gap_env.GapEnv), with each worker
collecting its rollout on its own.

    uv run --group rl python -m bila.rl.train_residual --name tight --decisions 2000000          # tight rows the script widens
    uv run --group rl python -m bila.rl.train_residual --name tight --resume runs/tight/last.pt  # carry on after a stop
    uv run --group rl tensorboard --logdir runs

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
corrections are small, as random full ones topple books.
Every eval_every updates: the deterministic policy (the actor's mean) on the
validation scenes (a separate pool, seeds 20000+), against the script alone
(a zero correction, run once: baseline.json): success rate, gap gained,
failures, each scene's in val_<update>.json. Each eval saves last.pt, and
best.pt when the success rate (then the gap) is the best yet. Logs:
TensorBoard and log.jsonl in runs/<name>/.
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

from bila import ROOT as HERE
RUNS = os.path.join(HERE, "runs")


@dataclass
class Config:
    name: str = "tight"
    scenarios: Tuple[str, ...] = ("tight",)
    phases: Tuple[str, ...] = ("widen",)          # train on scenes whose first contact is one of these
    val_pool: str = "tight_val"
    val_scenes: int = 30
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
    ramp_start: float = 0.25
    ramp_decisions: int = 300_000
    noise_scale: float = 1.0                     # NoisyPerception's
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
    from bila.rl.gap_env import GapEnv, load_pool
    if validation:
        return GapEnv(scenarios=cfg.scenarios[:1], pool={cfg.scenarios[0]: load_pool(cfg.val_pool)},
                      first_phases=cfg.phases, noise_scale=cfg.noise_scale)
    return GapEnv(scenarios=cfg.scenarios, first_phases=cfg.phases, noise_scale=cfg.noise_scale)


def episode_record(info: dict, ret: float) -> dict:
    keep = ("outcome", "failure", "decisions", "gap_gain_m", "method")
    return {"return": ret, **{k: info[k] for k in keep if k in info}}


def collect(env, actor: NumpyActor, n: int, state: dict) -> dict:
    """n decisions on env, carrying on the episode in progress (state)."""
    if state.get("obs") is None:
        state["obs"], _ = env.reset(seed=state["seed"])
        state["ret"] = 0.0
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
            episodes.append(episode_record(info, state["ret"]))
            if trunc and not term:
                finals[t] = (nxt["actor"], nxt["critic"])
            nxt, _ = env.reset()
            state["ret"] = 0.0
        obs = nxt
    state["obs"] = obs
    arrays = {k: np.asarray(v, dtype=bool if k in ("terminated", "truncated") else np.float32) for k, v in out.items()}
    return {**arrays, "finals": finals, "last": (obs["actor"], obs["critic"]), "episodes": episodes,
            "wall_s": time.perf_counter() - t0}


def evaluate(env, actor: NumpyActor, indices, zero: bool) -> List[dict]:
    """The deterministic policy (or a zero correction) on validation scenes."""
    out = []
    for i in indices:
        obs, info0 = env.reset(options={"index": i})
        ret = 0.0
        while True:
            a = np.zeros(3, np.float32) if zero else actor(obs, sample=False)
            obs, r, term, trunc, info = env.step(a)
            ret += r
            if term or trunc:
                out.append({"index": i, "seed": info0["seed"], **episode_record(info, ret)})
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
                conn.send(("ok", evaluate(val, actor, msg["indices"], msg["zero"])))
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
    """Success rate, failures, gap gained and return over episodes."""
    if not episodes:
        return {}
    o = [e.get("outcome") for e in episodes]
    return {"episodes": len(episodes), "success": o.count("success") / len(o), "failure": o.count("failure") / len(o),
            "narrow": o.count("narrow") / len(o),
            "gap_gain_mm": 1000 * float(np.mean([e.get("gap_gain_m", 0.0) for e in episodes])),
            "return": float(np.mean([e["return"] for e in episodes]))}


def validation_count(cfg: Config) -> int:
    from bila.rl.gap_env import load_pool
    usable = [p for p in load_pool(cfg.val_pool) if p.contact is not False and p.phase in cfg.phases]
    return min(cfg.val_scenes, len(usable))


def run_eval(workers: Workers, learner, cfg: Config, n: int, zero: bool) -> List[dict]:
    split = [list(range(i, n, cfg.workers)) for i in range(cfg.workers)]
    msg = learner.message(zero=zero, scale=ramp(cfg, learner.decisions))
    res = workers.ask([("evaluate", {**msg, "indices": s}) for s in split])
    return sorted((e for r in res for e in r), key=lambda e: e["index"])


def train(cfg: Config) -> None:
    import torch
    from torch.utils.tensorboard import SummaryWriter
    from bila.rl.residual_ppo import Learner
    torch.set_num_threads(4)
    out = os.path.join(RUNS, cfg.name)
    os.makedirs(out, exist_ok=True)
    probe = make_env(cfg)
    sizes = (probe.observation_space["actor"].shape[0], probe.observation_space["critic"].shape[0])
    probe.close()
    del probe
    learner = Learner(cfg, sizes)
    if cfg.resume:
        learner.load(cfg.resume)
    with open(os.path.join(out, "config.json"), "w") as f:
        json.dump(asdict(cfg), f, indent=1)
    writer = SummaryWriter(out)
    log = open(os.path.join(out, "log.jsonl"), "a")
    n_val = validation_count(cfg)
    workers = Workers(cfg)
    t_start = time.perf_counter()
    try:
        base_path = os.path.join(out, "baseline.json")
        if os.path.exists(base_path):
            baseline = json.load(open(base_path))
        else:
            baseline = run_eval(workers, learner, cfg, n_val, zero=True)
            json.dump(baseline, open(base_path, "w"), indent=1)
        print(f"script alone on {n_val} validation scenes: {summary(baseline)}", flush=True)
        while learner.decisions < cfg.decisions:
            if cfg.hours and time.perf_counter() - t_start > cfg.hours * 3600:
                break
            t0, scale = time.perf_counter(), ramp(cfg, learner.decisions)
            rollouts = workers.ask([("collect", learner.message(scale=scale, n=cfg.rollout))] * cfg.workers)
            t_collect = time.perf_counter() - t0
            stats = learner.update(rollouts)
            episodes = [e for r in rollouts for e in r["episodes"]]
            row = {"update": learner.updates, "decisions": learner.decisions, "scale": scale,
                   "decisions_per_s": cfg.workers * cfg.rollout / t_collect, "collect_s": t_collect,
                   "update_s": time.perf_counter() - t0 - t_collect, **stats,
                   **{f"train_{k}": v for k, v in summary(episodes).items()}}
            if learner.updates % cfg.eval_every == 0:
                scenes = run_eval(workers, learner, cfg, n_val, zero=False)
                json.dump(scenes, open(os.path.join(out, f"val_{learner.updates:05d}.json"), "w"), indent=1)
                val = summary(scenes)
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


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    for f in fields(Config):
        default = f.default
        if isinstance(default, tuple):
            ap.add_argument("--" + f.name.replace("_", "-"), nargs="+", default=list(default))
        else:
            ap.add_argument("--" + f.name.replace("_", "-"), type=type(default), default=default)
    args = vars(ap.parse_args())
    train(Config(**{k: tuple(v) if isinstance(v, list) else v for k, v in args.items()}))


if __name__ == "__main__":
    main()
