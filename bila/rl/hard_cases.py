"""
Pools where the script fails as often as it succeeds, for the insertion
residual: label every case of some pools with how the script alone ends on it
(under GapEnv's own conditions: its grasp error and its eyes, a zero
correction), then keep every failure the residual could act on and as many
successes, split by layout into training and validation pools.

    uv run --group rl python -m bila.rl.hard_cases label --pools clear_cand tight_cand --perception camera \
        --segmenter bila_weights/segmenter/segmenter.pt --segmenter-device cpu --workers 10 --out results/labels.jsonl
    uv run --group rl python -m bila.rl.hard_cases mix --labels results/labels.jsonl --suffix _mix
    # -> scenes/gap_pool_<scenario>_mix.pkl and _mix_val.pkl, for train_residual --train-suffix _mix --val-suffix _mix_val
    # a test set: other layouts (seeds 30000 and up), labelled the same way, all in one pool
    uv run --group rl python -m bila.rl.hard_cases mix --labels results/labels_test.jsonl --pools clear=clear_test ... \
        --suffix _test50 --val-fraction 0
    uv run --group rl python -m bila.rl.hard_cases test --run runs/insertion_mix --out results/residual_test.json

label: one JSON line per case, written as each ends (resumable: cases already
in --out are skipped). The outcome is GapEnv's: success, missed (the book
went in but isn't placed), failure (it jammed, toppled a book, the arm hit
something), truncated, or preparation_failed (the script never reached a
controlled contact: no residual can change it).

mix: per scenario, every missed/failure/truncated case and as many successes
drawn at random; a scenario without failures is left out. Each layout goes
whole to training or validation (--val-fraction of each kind, at least one),
so validation layouts are new to the policy.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
from multiprocessing import Pool

import numpy as np

FAILED = ("missed", "failure", "truncated")
_env = {}


def _init(pools, perception, segmenter, device, control_phases):
    from bila.rl.gap_env import GapEnv, load_pool
    _env["env"] = GapEnv(scenarios=tuple(pools), pool={s: load_pool(n) for s, n in pools.items()},
                         control_phases=tuple(control_phases), learn_stop=False, perception=perception,
                         segmenter=segmenter, segmenter_device=device)


def _label(case):
    from bila.rl.train_residual import evaluate
    (out,) = evaluate(_env["env"], None, [case], zero=True)
    return {**case, "outcome": out["outcome"], "failure": out.get("failure", ""),
            "offset_m": out.get("offset_m"), "method": out.get("method", "")}


def key(case) -> str:
    return json.dumps([case["scenario"], case["geometry_seed"], case["variant"]])


def label(a) -> None:
    from bila.rl.gap_env import load_pool
    pools = dict(p.split("=") if "=" in p else (p.split("_")[0], p) for p in a.pools)   # scenario=name or <scenario>_...
    done = {key(json.loads(line)) for line in open(a.out)} if os.path.exists(a.out) else set()
    cases = [p.unpack().identity() for s, n in pools.items() for p in load_pool(n) if p.contact]
    cases = [c for c in cases if key(c) not in done]
    print(f"{len(cases)} cases to label ({len(done)} already in {a.out})", flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with Pool(a.workers, _init, (pools, a.perception, a.segmenter, a.segmenter_device, a.control_phases)) as p, \
            open(a.out, "a") as f:
        for i, r in enumerate(p.imap_unordered(_label, cases)):
            f.write(json.dumps(r) + "\n")
            f.flush()
            print(f"  {i + 1}/{len(cases)} {r['scenario']} {r['geometry_seed']}: {r['outcome']} {r['failure']}",
                  flush=True)


def mix(a) -> None:
    from bila.rl.gap_env import load_pool, pool_path
    rows = {key(r): r for r in map(json.loads, open(a.labels))}
    rng = np.random.default_rng(a.seed)
    pools = dict(p.split("=") if "=" in p else (p.split("_")[0], p) for p in a.pools) if a.pools else None
    for scenario in sorted({r["scenario"] for r in rows.values()}):
        mine = [r for r in rows.values() if r["scenario"] == scenario]
        bad = sorted((r for r in mine if r["outcome"] in FAILED), key=key)
        good = sorted((r for r in mine if r["outcome"] == "success"), key=key)
        if not bad:
            print(f"{scenario}: the script never fails on its {len(good)} controlled cases: left out")
            continue
        n = min(len(bad), len(good))
        bad = [bad[i] for i in rng.permutation(len(bad))[:n]]
        good = [good[i] for i in rng.permutation(len(good))[:n]]
        k = max(1, round(a.val_fraction * n)) if a.val_fraction > 0 else 0   # 0: one pool, all of it (a test set)
        val = {key(r) for r in bad[:k] + good[:k]}
        chosen = {key(r) for r in bad + good}
        source = pools[scenario] if pools else f"{scenario}_cand"
        split = {"": [], "_val": []}
        for packed in load_pool(source):
            c = packed.unpack().identity()
            if key(c) in chosen:
                split["_val" if key(c) in val else ""].append(packed)
        for tail, entries in split.items():
            if tail and not entries:
                continue
            path = pool_path(scenario + a.suffix + tail)
            if os.path.exists(path):
                raise FileExistsError(f"{path} exists: move it away first")
            with open(path, "wb") as f:
                pickle.dump(entries, f)
            print(f"{scenario}{a.suffix}{tail}: {len(entries)} cases "
                  f"({sum(key(p.unpack().identity()) in {key(r) for r in bad} for p in entries)} the script fails) "
                  f"-> {path}")
        kinds = {o: sum(r["outcome"] == o for r in bad) for o in FAILED}
        print(f"  {scenario}: {n} failures {kinds} + {n} successes, of {len(mine)} labelled "
              f"({sum(r['outcome'] == 'preparation_failed' for r in mine)} never reached the slide)")


def _test(job):
    cfg, msg, cases = job
    from bila.rl.train_residual import NumpyActor, evaluate, make_env
    env = make_env(cfg, validation=True)
    try:
        actor = NumpyActor(msg, np.random.default_rng(0))
        env.set_action_scale(cfg.eval_action_scale)
        return evaluate(env, actor, cases, zero=True), evaluate(env, actor, cases, zero=False)
    finally:
        env.close()


def test(a) -> None:
    """A trained residual and the script alone on every case of the test pools,
    once: paired, case by case. Nothing is chosen on these cases."""
    from dataclasses import fields
    from bila.rl.residual_ppo import Learner
    from bila.rl.train_residual import Config, case_key, make_env, paired_summary, summary, validation_scenes
    raw = json.load(open(os.path.join(a.run, "config.json")))
    cfg = Config(**{f.name: tuple(raw[f.name]) if isinstance(f.default, tuple) else raw[f.name]
                    for f in fields(Config) if f.name in raw})
    cfg.val_suffix, cfg.val_scenes, cfg.val_variants = a.suffix, 10_000, 0
    probe = make_env(cfg, validation=True)
    sizes = (probe.observation_space["actor"].shape[0], probe.observation_space["critic"].shape[0],
             probe.action_space.shape[0])
    probe.close()
    learner = Learner(cfg, sizes)
    learner.load(os.path.join(a.run, a.checkpoint))
    msg = learner.message(scale=cfg.eval_action_scale)
    cases = validation_scenes(cfg)
    with Pool(a.workers) as p:
        res = p.map(_test, [(cfg, msg, cases[i::a.workers]) for i in range(a.workers)])
    script = sorted((e for r in res for e in r[0]), key=case_key)
    policy = sorted((e for r in res for e in r[1]), key=case_key)
    out = {"checkpoint": os.path.join(a.run, a.checkpoint), "suffix": a.suffix,
           "script": summary(script), "policy": summary(policy), **paired_summary(script, policy),
           "cases": [{"case": json.loads(case_key(s)), "script": s["outcome"], "policy": q["outcome"],
                      "script_offset_m": s.get("offset_m"), "policy_offset_m": q.get("offset_m")}
                     for s, q in zip(script, policy)]}
    json.dump(out, open(a.out, "w"), indent=1)
    print(f"{len(cases)} test cases: script {out['script']['success']:.2f}, policy {out['policy']['success']:.2f}; "
          f"recovered {out['recovered']}, lost {out['lost']} -> {a.out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("label")
    p.add_argument("--pools", nargs="+", required=True, help="pool names (<scenario>_..., or scenario=name)")
    p.add_argument("--perception", default="camera")
    p.add_argument("--segmenter")
    p.add_argument("--segmenter-device")
    p.add_argument("--control-phases", nargs="+", default=["slide", "push"])
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    p.add_argument("--out", required=True)
    p = sub.add_parser("mix")
    p.add_argument("--labels", required=True)
    p.add_argument("--pools", nargs="+", help="the labelled pools (default: <scenario>_cand)")
    p.add_argument("--suffix", default="_mix")
    p.add_argument("--val-fraction", type=float, default=0.3)
    p.add_argument("--seed", type=int, default=0)
    p = sub.add_parser("test")
    p.add_argument("--run", required=True, help="runs/<name>: its config.json and checkpoints")
    p.add_argument("--checkpoint", default="best.pt")
    p.add_argument("--suffix", default="_test50", help="test pools: gap_pool_<scenario><suffix>.pkl")
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    p.add_argument("--out", required=True)
    a = ap.parse_args()
    {"label": label, "mix": mix, "test": test}[a.cmd](a)


if __name__ == "__main__":
    main()
