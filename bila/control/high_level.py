"""
A learned high-level policy: which step the controller takes next.

Episode.reshelve picks its steps by hand-written rules (next_step): stand a
leaning neighbour up, widen, shove, insert, with fixed retry counts, and a
jammed book ends the episode. Here a policy picks them instead, from a few
numbers the camera's estimate gives, and one more step lets it recover:

    insert          take the book, slide it in, let go, push it home (ends the episode,
                    unless it jams at the entrance: then the book is still in the hand)
    widen           push a neighbour aside to the book + GAP_MARGIN (as the rules do)
    widen_more      ... to the book + GAP_MARGIN + EXTRA: a margin for what the camera got wrong
    shove           shove the touching neighbour and the books beyond it into the row's room
    stand_up_left   stand the left / right neighbour up
    stand_up_right
    set_aside       after a jam (the only step then, but give_up): back the book out, put it back where
                    it came from, look again; then the policy chooses again (insert again, widen more...)
    give_up         leave the slot for the librarian

The skills are insert_book's; only the choice is learned. The policy is a
Q-function over (features, action), fitted offline (fitted Q iteration) on
episodes of the rules with random deviations (EpsilonRules). Reward: 1 if the
book ends up placed, less STEP_COST per step taken.

    uv run python -m bila.control.high_level collect --perception noisy-learned --seeds 5000:5100 --eps 0.3 \
        --out results/hl_train.jsonl --workers 12
    uv run --group rl python -m bila.control.high_level fit --data results/hl_train.jsonl --out runs/hl/q.npz
    uv run python -m bila.control.high_level collect --perception noisy-learned --seeds 0:60 --policy runs/hl/q.npz \
        --out results/hl_eval_q.jsonl                  # and with --eps 0 for the rules on the same rows
    uv run python -m bila.control.cart_demo --scenario tight --policy runs/hl/q.npz

run(ep, policy) replaces ep.reshelve() and returns the same score dict.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from multiprocessing import Pool

import mujoco
import numpy as np

from bila.control import insert_book as ib

ACTIONS = ("insert", "widen", "widen_more", "shove", "stand_up_left", "stand_up_right", "set_aside", "give_up")
EXTRA = 0.004               # widen_more's margin over GAP_MARGIN
MAX_STEPS = 10              # steps per episode at most (then it gives up)
MAX_JAMS = 2                # set-asides at most
STEP_COST = 0.02            # per step: the time it takes
BACK_OUT = 0.08             # m the hand backs a jammed book out of the slot
FEATURES = ("lean_left_deg", "lean_right_deg", "gap_spare_mm", "widen_spare_mm", "fingers_spare_mm",
            "room_left_mm", "room_right_mm", "book_mm", "stand_ups", "widens", "shoves", "jams", "steps",
            "last_did", "holding")


class Stuck(Exception):
    """The book jammed (or can't get onto the board): it is still in the hand."""


def features(ep, taken, jams, holding) -> np.ndarray:
    est = ep.estimate()
    lean = [(-s * est.book(s).lean if est.book(s) is not None else 0.0) for s in (-1, 1)]
    xl, xr, _, _ = ep.slot()
    gl, gr = ep.finger_gap()
    room = ep.free_space()
    count = lambda n: sum(s.startswith(n) for s, _ in taken)
    return np.array([math.degrees(lean[0]), math.degrees(lean[1]), 1000 * (xr - xl - ep.t),
                     1000 * (ep.widen_gap() - ep.t - ib.GAP_MARGIN), 1000 * (gr - gl - 2 * (ib.FINGER_HALF_W + ib.FIT_CLEAR)),
                     1000 * room[0], 1000 * room[1], 1000 * ep.t, count("stand_up"), count("widen"), count("shove"),
                     jams, len(taken), float(taken[-1][1]) if taken else 1.0, float(holding)])


def valid(f: np.ndarray, taken, jams, holding) -> np.ndarray:
    """Which actions make sense now: the skills' own preconditions, and the
    rules' retry caps (so no policy loops)."""
    x = dict(zip(FEATURES, f))
    stuck = lambda n: any(s == n and not did for s, did in taken)
    ok = {a: False for a in ACTIONS}
    ok["give_up"] = True
    if holding:                                   # a jammed book in the hand: put it aside (then decide again)
        ok["set_aside"] = jams <= MAX_JAMS
    else:
        ok["insert"] = x["gap_spare_mm"] >= 0
        fingers = x["fingers_spare_mm"] >= 0
        ok["widen"] = fingers and x["widens"] < ib.WIDEN_STEPS and not stuck("widen") and x["widen_spare_mm"] < 0
        ok["widen_more"] = fingers and x["widens"] < ib.WIDEN_STEPS and not stuck("widen_more")
        ok["shove"] = not fingers and x["shoves"] == 0
        for side, name in ((0, "stand_up_left"), (1, "stand_up_right")):
            ok[name] = x[("lean_left_deg", "lean_right_deg")[side]] > ib.LEAN_MIN_DEG / 2 and x["stand_ups"] < ib.STAND_STEPS
    if len(taken) >= MAX_STEPS:
        ok = {a: a == "give_up" for a in ACTIONS}
    return np.array([ok[a] for a in ACTIONS])


class Rules:
    """insert_book's next_step, as an action (and set_aside after a jam, which the rules can't do)."""

    def __call__(self, ep, f, mask, taken):
        if f[FEATURES.index("holding")]:
            return ACTIONS.index("set_aside") if mask[ACTIONS.index("set_aside")] else ACTIONS.index("give_up")
        step = ep.next_step(ib.GAP_MARGIN, [(("widen" if s == "widen_more" else s.rsplit("_", 1)[0]
                                              if s.startswith("stand_up") else s), did) for s, did in taken])
        if step is None:
            return ACTIONS.index("give_up")
        name = step[0] if step[0] != "stand_up" else ("stand_up_left" if step[1] < 0 else "stand_up_right")
        return ACTIONS.index(name) if mask[ACTIONS.index(name)] else ACTIONS.index("give_up")


class EpsilonRules(Rules):
    """The rules, with a random valid action instead with probability eps (data for fitting)."""

    def __init__(self, eps, rng):
        self.eps, self.rng = eps, rng

    def __call__(self, ep, f, mask, taken):
        explore = np.flatnonzero(mask & (np.arange(len(ACTIONS)) != ACTIONS.index("give_up")))
        if len(explore) and self.rng.random() < self.eps:          # (giving up at random teaches nothing)
            return int(self.rng.choice(explore))
        return super().__call__(ep, f, mask, taken)


class QPolicy:
    """argmax over the valid actions of a fitted Q (a small MLP, numpy only)."""

    def __init__(self, path):
        w = np.load(path)
        self.mean, self.std = w["mean"], w["std"]
        self.layers = [(w[f"W{i}"], w[f"b{i}"]) for i in range(int(w["n"]))]

    def q(self, f):
        h = (np.atleast_2d(f) - self.mean) / self.std
        for i, (W, b) in enumerate(self.layers):
            h = h @ W + b
            if i < len(self.layers) - 1:
                h = np.maximum(h, 0)
        return h

    def __call__(self, ep, f, mask, taken):
        return int(np.argmax(np.where(mask, self.q(f)[0], -np.inf)))


def set_aside(ep, home):
    """Back the jammed book out of the slot, let go, put it back where it was
    at the start (home: its free joint's qpos), and look again."""
    run, m, d = ep.run, ep.m, ep.d
    run.move("back the jammed book out", run.pos + ep.ys * BACK_OUT, run.rot, speed=0.02)
    run.wait("open the gripper", 0.4, grip=ib.GRIP_OPEN)
    # ponytail: the book goes back by magic (as the hand-over in insert does): the tray in the cart, the
    # parking spot otherwise. A robot would carry it back, or hold it while widening with the other hand.
    qa, va = m.jnt_qposadr[ep.book_joint], m.jnt_dofadr[ep.book_joint]
    d.qpos[qa:qa + 7], d.qvel[va:va + 6] = home, 0
    mujoco.mj_forward(m, d)
    run.move("move the hand out of the shelf", run.pos + ep.ys * 0.05, run.rot)
    ep.looked()


def run(ep, policy, record=None) -> dict:
    """ep.reshelve() with `policy` choosing the steps. record: a list to get
    (features, valid mask, action) per decision."""
    qa = ep.m.jnt_qposadr[ep.book_joint]
    home = ep.d.qpos[qa:qa + 7].copy()
    taken, jams, holding = [], 0, False
    while True:
        f = features(ep, taken, jams, holding)
        mask = valid(f, taken, jams, holding)
        a = policy(ep, f, mask, taken)
        name = ACTIONS[a]
        if record is not None:
            record.append((f.tolist(), mask.tolist(), a))
        ep.info["method"] = ", ".join([s for s, _ in taken] + [name])
        ep.log(f"  -- {name.replace('_', ' ')} (high-level policy)")
        if name == "give_up":
            if holding:                                      # (the book can't stay jammed in the slot)
                set_aside(ep, home)
            return ep.skip(ep.info.get("why", "the high-level policy gave up"))
        if name == "insert":
            try:
                return insert(ep)
            except Stuck as e:
                jams, holding = jams + 1, True
                ep.info["jams"] = jams
                taken.append((name, False))
                ep.log(f"  stuck: {e}")
                continue
        if name == "set_aside":
            set_aside(ep, home)
            holding = False
            taken.append((name, True))
            continue
        if name.startswith("stand_up"):
            did = ep.stand_up(-1 if name.endswith("left") else 1)
        elif name == "widen_more":
            did = ep.widen(ib.GAP_MARGIN + EXTRA)
        else:
            did = getattr(ep, name)(ib.GAP_MARGIN)
        taken.append((name, bool(did)))


def insert(ep):
    """ep.insert(), with a jam reported as Stuck (the book still held) rather
    than the end of the episode."""
    try:
        return ep.insert()
    except RuntimeError as e:
        if "jammed" in str(e) or "far enough onto the board" in str(e):
            raise Stuck(str(e)) from None
        raise


# --- collect and fit -------------------------------------------------------------------------------
def episode(job) -> dict:
    scenario, seed, perception, eps, policy_path = job
    from bila.sim.shelf_scene import ShelfConfig
    try:
        model, data, row = ib.make_episode(ShelfConfig(), scenario, np.random.default_rng(seed), ib.DEFAULT_PANDA)
    except RuntimeError as e:
        return {"scenario": scenario, "seed": seed, "error": str(e), "steps": []}
    eyes = ib.eyes_for(None if perception == "truth" else perception, model, row, seed)
    ep = ib.Episode(model, data, row, log=lambda *_: None, eyes=eyes)
    policy = QPolicy(policy_path) if policy_path else EpsilonRules(eps, np.random.default_rng([seed, 7]))
    steps = []
    try:
        res = run(ep, policy, steps)
    except RuntimeError as e:
        res = {"success": False, "error": str(e)}
    finally:
        if eyes is not None and hasattr(eyes, "close"):
            eyes.close()
    keep = ("success", "safe_success", "skipped", "error", "method", "jams", "offset_m")
    return {"scenario": scenario, "seed": seed, "perception": perception, "eps": eps, "policy": policy_path,
            **{k: res[k] for k in keep if k in res}, "steps": steps}


def collect(a) -> None:
    lo, hi = (int(v) for v in a.seeds.split(":"))
    done = set()
    if os.path.exists(a.out):
        done = {(r["scenario"], r["seed"]) for r in map(json.loads, open(a.out))}
    jobs = [(s, seed, a.perception, a.eps, a.policy) for s in a.scenarios for seed in range(lo, hi)
            if (s, seed) not in done]
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with Pool(a.workers) as p, open(a.out, "a") as f:
        for i, r in enumerate(p.imap_unordered(episode, jobs)):
            f.write(json.dumps(r) + "\n")
            f.flush()
            print(f"  {i + 1}/{len(jobs)} {r['scenario']} {r['seed']}: "
                  f"{'placed' if r.get('success') else r.get('skipped') or r.get('error') or 'missed'} "
                  f"[{', '.join(ACTIONS[s[2]] for s in r['steps'])}]", flush=True)


def transitions(rows):
    """(s, a, r, s', valid', done) over every decision of every episode."""
    S, A, R, S2, M2, D = [], [], [], [], [], []
    for r in rows:
        steps = r["steps"]
        for i, (f, _, a) in enumerate(steps):
            last = i == len(steps) - 1
            S.append(f)
            A.append(a)
            R.append((1.0 if (last and r.get("success")) else 0.0) - STEP_COST)
            S2.append(steps[i + 1][0] if not last else f)
            M2.append(steps[i + 1][1] if not last else [False] * len(ACTIONS))
            D.append(last)
    return tuple(np.array(v, dtype=bool if k in (4, 5) else np.float32) for k, v in enumerate((S, A, R, S2, M2, D)))


def fit(a) -> None:
    """Fitted Q iteration: Q(s, a) <- r + max over valid a' of Q(s', a'), a small MLP refit each round."""
    import torch
    torch.manual_seed(0)
    rows = [r for path in a.data for r in map(json.loads, open(path)) if r["steps"]]
    S, A, R, S2, M2, D = transitions(rows)
    mean, std = S.mean(0), S.std(0) + 1e-6
    norm = lambda x: torch.as_tensor((x - mean) / std)
    s, s2, act = norm(S), norm(S2), torch.as_tensor(A).long()
    r, done, m2 = torch.as_tensor(R), torch.as_tensor(D), torch.as_tensor(M2)
    target = r.clone()
    for it in range(a.iterations):
        net = torch.nn.Sequential(torch.nn.Linear(len(FEATURES), a.hidden), torch.nn.ReLU(),
                                  torch.nn.Linear(a.hidden, a.hidden), torch.nn.ReLU(),
                                  torch.nn.Linear(a.hidden, len(ACTIONS)))
        opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
        for _ in range(a.epochs):
            perm = torch.randperm(len(s))
            for i in range(0, len(s), 256):
                b = perm[i:i + 256]
                loss = ((net(s[b]).gather(1, act[b, None])[:, 0] - target[b]) ** 2).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
        with torch.no_grad():
            q2 = torch.where(m2, net(s2), torch.tensor(-1e9)).max(1).values
            target = r + torch.where(done, torch.zeros_like(q2), q2.clamp(-1, 1.5))
        print(f"iteration {it + 1}: loss {float(loss):.4f}, mean target {float(target.mean()):.3f}", flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    lin = [l for l in net if isinstance(l, torch.nn.Linear)]
    np.savez(a.out, mean=mean, std=std, n=len(lin),
             **{f"W{i}": l.weight.detach().numpy().T for i, l in enumerate(lin)},
             **{f"b{i}": l.bias.detach().numpy() for i, l in enumerate(lin)})
    print(f"{len(rows)} episodes, {len(S)} decisions -> {a.out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect")
    c.add_argument("--scenarios", nargs="+", default=["clear", "tight", "leaning", "packed"])
    c.add_argument("--seeds", default="5000:5100", help="lo:hi per scenario")
    c.add_argument("--perception", default="noisy-learned", choices=("truth", "noisy", "noisy-learned", "camera"))
    c.add_argument("--eps", type=float, default=0.3, help="random deviations from the rules (0: the rules)")
    c.add_argument("--policy", help="a fitted Q (fit's --out) instead of the rules")
    c.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    c.add_argument("--out", required=True)
    f = sub.add_parser("fit")
    f.add_argument("--data", nargs="+", required=True)
    f.add_argument("--out", required=True)
    f.add_argument("--iterations", type=int, default=8)
    f.add_argument("--epochs", type=int, default=60)
    f.add_argument("--hidden", type=int, default=64)
    a = ap.parse_args()
    collect(a) if a.cmd == "collect" else fit(a)


if __name__ == "__main__":
    main()
