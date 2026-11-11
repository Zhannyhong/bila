"""
The controller alone (insert_book.Episode.reshelve, no residual) over many
rows with a chosen source of perception, in parallel, and the results of
several sources side by side. It answers whether a noise model the residual
trains on (perception.NoisyPerception, gap_env.py) fails the controller as
often, and in the same ways, as the camera pipeline it stands in for.

    uv run python eval_reshelve.py run --perception noisy-learned --out results/noisy-learned.jsonl
    uv run python eval_reshelve.py run --perception noisy --out results/noisy.jsonl
    MUJOCO_GL=egl uv run python eval_reshelve.py run --perception camera --out results/camera.jsonl
    MUJOCO_GL=egl uv run --group rl python eval_reshelve.py run --perception camera \
        --segmenter ~/bila_data/segmenter/segmenter.pt --out results/camera-learned.jsonl
    uv run python eval_reshelve.py compare results/camera-learned.jsonl results/noisy-learned.jsonl ...

run: every scenario's seeds (--seeds, default 0:200, the rows insert_book.py's
results use) on --workers processes, one JSON line per row in --out, written
as each row ends. Rows already in --out are skipped, so a run can be resumed,
and --shard i/n takes every n-th row from the i-th (0-based): one job per
node of a cluster, each with its own --out, concatenated afterwards. With
--segmenter, each worker loads the segmenter once, on the GPU if there is one
(LearnedSegmenter's choice: CUDA, else Apple's Metal); on a node with
several, CUDA_VISIBLE_DEVICES per job picks one, and --segmenter-device
overrides the choice. --noise-scale widens the noisy sources' errors, as
gap_env's noise_scale does.

compare: per file (its perception and segmenter, as each line records them),
how many rows the controller placed per scenario, how the rest ended
(outcome()), and for the first file against each other one, the rows one
placed and the other didn't. Noise draws differ between sources, so rows
needn't agree one by one: the rates and the kinds of failure are the test.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import time
import traceback
from collections import Counter, defaultdict

import numpy as np

SCENARIOS = ("clear", "tight", "leaning", "packed")
KEEP = ("success", "safe_success", "method", "skipped", "error", "upright", "in_place", "released", "blocked",
        "rule", "offset_m", "tilt_deg", "neighbour_shift_m", "total_shift_m", "time_to_seat_s", "looks", "look_time_s",
        "collision_s")

_worker = {}


def outcome(r: dict) -> str:
    """How a row ended, in a few kinds."""
    if r.get("crash"):
        return "crash"
    if r.get("success"):
        return "placed"
    if "skipped" in r:
        return f"gave up: {r['skipped']}"
    if "error" in r:
        return f"stopped: {r['error']}"
    if not r.get("upright", True):
        return "not upright"
    if not r.get("released", True):
        return "not let go of"
    if not r.get("in_place", True):
        # (offset_m: past the neighbour's spine for the "flush" rule, inside the front edge for "inside")
        return "pushed too deep" if r.get("rule") == "flush" and r.get("offset_m", 0.0) < 0 else "left proud"
    return "other"


def parse_seeds(text: str) -> list:
    lo, hi = (int(v) for v in text.split(":"))
    return list(range(lo, hi))


def _init(perception, segmenter, camera_noise, noise_scale=1.0, segmenter_device=None, scene=None):
    import cv2
    cv2.setNumThreads(1)                     # (its thread pool per worker overloads the machine)
    seg = None
    if segmenter:
        from learned_segmenter import LearnedSegmenter    # (imports torch's dynamo first: rendering needs that order)
        import torch
        torch.set_num_threads(1)
        seg = LearnedSegmenter(os.path.expanduser(segmenter), device=segmenter_device)
    _worker.update(perception=perception, segmenter=seg, camera_noise=camera_noise, noise_scale=noise_scale,
                   scene=scene or {},
                   label={"perception": perception, "segmenter": segmenter or "", "camera_noise": camera_noise,
                          "noise_scale": noise_scale, **(scene or {})})


def run_row(job) -> dict:
    """One row, the controller alone, with the worker's perception."""
    scenario, seed = job
    from insert_book import DEFAULT_PANDA, Episode, eyes_for, make_episode
    from shelf_scene import ShelfConfig
    t0, eyes, ep = time.time(), None, None
    out = {"scenario": scenario, "seed": seed, **_worker["label"]}
    try:
        model, data, row = make_episode(ShelfConfig(**_worker["scene"]), scenario, np.random.default_rng(seed),
                                        DEFAULT_PANDA)
        eyes = eyes_for(_worker["perception"], model, row, seed, _worker["camera_noise"], _worker["segmenter"],
                        _worker["noise_scale"])
        ep = Episode(model, data, row, None, lambda *_: None, eyes)
        try:
            res = ep.reshelve()
        except RuntimeError as e:              # no safe way to do it in this scene, or it jammed (as insert_book.main)
            res = {"success": False, "safe_success": False, "error": str(e), **ep.info, **ep.collisions()}
        out.update({k: (v.item() if isinstance(v, np.generic) else v) for k, v in res.items() if k in KEEP})
    except Exception:
        out.update(success=False, crash=traceback.format_exc().strip().splitlines()[-1])
    finally:
        if eyes is not None:
            eyes.close()
    out["outcome"], out["wall_s"] = outcome(out), round(time.time() - t0, 1)
    return out


def run(a) -> None:
    jobs = [(s, k) for s in a.scenarios for k in parse_seeds(a.seeds)]
    i, n = (int(v) for v in a.shard.split("/"))
    jobs = jobs[i::n]
    done = set()
    if os.path.exists(a.out):
        with open(a.out) as f:
            done = {(r["scenario"], r["seed"]) for r in map(json.loads, f) if r}
    todo = [j for j in jobs if j not in done]
    print(f"{len(todo)} rows to run ({len(jobs) - len(todo)} already in {a.out}), {a.workers} workers", flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    ctx = mp.get_context("spawn")
    scene = {k: v for k, v in (("catalogue", a.catalogue), ("book_pages", a.book_pages),
                               ("book_meshes", a.book_meshes)) if v}       # (the default scene: none, as before)
    with ctx.Pool(a.workers, _init, (a.perception, a.segmenter, a.camera_noise, a.noise_scale, a.segmenter_device,
                                     scene),
                  maxtasksperchild=50) as pool, \
            open(a.out, "a") as f:
        for k, r in enumerate(pool.imap_unordered(run_row, todo), 1):
            f.write(json.dumps(r) + "\n")
            f.flush()
            print(f"  {k}/{len(todo)}  {r['scenario']} {r['seed']}: {r['outcome']} ({r['wall_s']:.0f} s)", flush=True)


def load(path) -> list:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def name_of(rows, path) -> str:
    r = rows[0] if rows else {}
    seg = f" + {os.path.basename(r['segmenter'])}" if r.get("segmenter") else ""
    scale = f" x{r['noise_scale']:g}" if r.get("noise_scale", 1.0) != 1.0 else ""
    books = (" real books" if r.get("catalogue") else "") + (f" {r['book_pages']} pages" if r.get("book_pages") else "")
    return f"{r.get('perception', '?')}{scale}{seg}{books} ({os.path.basename(path)})"


def compare(a) -> None:
    files = [(p, load(p)) for p in a.files]
    print("placed / rows, per scenario")
    print(f"  {'source':<46}" + "".join(f"{s:>12}" for s in SCENARIOS) + f"{'all':>12}")
    for p, rows in files:
        by = defaultdict(list)
        for r in rows:
            by[r["scenario"]].append(r)
        cells = [f"{sum(r['success'] for r in by[s])}/{len(by[s])}" if by[s] else "-" for s in SCENARIOS]
        print(f"  {name_of(rows, p):<46}" + "".join(f"{c:>12}" for c in cells)
              + f"{sum(r['success'] for r in rows)}/{len(rows)}".rjust(12))
    print("\nhow the rest ended")
    for p, rows in files:
        print(f"  {name_of(rows, p)}")
        for kind, n in Counter(r["outcome"] for r in rows if not r["success"]).most_common():
            per = Counter(r["scenario"] for r in rows if r["outcome"] == kind)
            print(f"    {n:4d}  {kind}  ({', '.join(f'{s} {per[s]}' for s in SCENARIOS if per[s])})")
    if len(files) > 1:
        (p0, rows0), key = files[0], lambda r: (r["scenario"], r["seed"])
        first = {key(r): r["success"] for r in rows0}
        print(f"\nrow by row, against {name_of(rows0, p0)}")
        for p, rows in files[1:]:
            both = [(first[key(r)], r["success"]) for r in rows if key(r) in first]
            print(f"  {name_of(rows, p)}: {len(both)} rows in both; both placed {sum(x and y for x, y in both)}, "
                  f"only the first {sum(x and not y for x, y in both)}, only this {sum(y and not x for x, y in both)}, "
                  f"neither {sum(not x and not y for x, y in both)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="the controller alone on many rows")
    r.add_argument("--perception", choices=("truth", "camera", "noisy", "noisy-learned"), required=True)
    r.add_argument("--segmenter", metavar="WEIGHTS", help="camera: learned book masks (learned_segmenter.py)")
    r.add_argument("--segmenter-device", choices=("cuda", "mps", "cpu"),
                   help="where the segmenter runs (default: the GPU if there is one)")
    r.add_argument("--camera-noise", choices=("realistic", "basic"), default="realistic")
    r.add_argument("--noise-scale", type=float, default=1.0,
                   help="noisy, noisy-learned: the errors' spread (gap_env's noise_scale)")
    r.add_argument("--catalogue", action="store_true", help="the repo's real books (ShelfConfig.catalogue)")
    r.add_argument("--book-pages", type=int, default=0, metavar="N",
                   help="cut every book into N page sections hinged on the spine (with --catalogue: every real "
                        "book without its own sections)")
    r.add_argument("--book-meshes", action="store_true", help="draw the books' covers (what the camera sees)")
    r.add_argument("--scenarios", nargs="+", choices=SCENARIOS, default=list(SCENARIOS))
    r.add_argument("--seeds", default="0:200", help="lo:hi, per scenario")
    r.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    r.add_argument("--shard", default="0/1", help="i/n: every n-th row from the i-th (one cluster job each)")
    r.add_argument("--out", required=True, help="JSON lines, appended to (rows already in it are skipped)")
    c = sub.add_parser("compare", help="sources side by side (the first is the reference)")
    c.add_argument("files", nargs="+")
    a = ap.parse_args()
    if a.cmd == "run":
        if a.segmenter and a.perception != "camera":
            ap.error("--segmenter goes with --perception camera")
        if a.noise_scale != 1.0 and not a.perception.startswith("noisy"):
            ap.error("--noise-scale goes with --perception noisy or noisy-learned")
        run(a)
    else:
        compare(a)


if __name__ == "__main__":
    main()
