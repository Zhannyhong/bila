"""
How well the camera-based ShelfEstimate matches the ground truth, over many
scenes, with and without depth noise.

    MUJOCO_GL=egl uv run python eval_perception.py collect --seeds 12   # slow (~20 min): stores scenes/perception_eval.pkl
    uv run python eval_perception.py report                             # tables from the stored estimates
    uv run python eval_perception.py calibrate                          # noise for NoisyPerception, from them
    uv run python eval_perception.py validate                           # does NoisyPerception err like the camera?

`collect` puts the arm at the inspection pose in each scene (12 seeds of each
scenario by default) and stores three estimates side by side: the ground truth,
the camera's with perfect depth, and the camera's with D405-like depth noise.
The slow part (solving the arm pose) then happens once; `report`, and the
calibration of NoisyPerception, read the stored estimates.

Every camera book is matched to its true book by the simulation's object id,
so errors are book for book. Errors are reported by how far the book stands
from the slot (|slot offset| 1, 2-3, 4-6), since the policy needs the
neighbours most.
"""

from __future__ import annotations

import argparse
import math
import os
import pickle
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from perception import BookEstimate, ShelfEstimate

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_PATH = os.path.join(HERE, "scenes", "perception_eval.pkl")
OFFSET_BINS = {"neighbours (1)": (1, 1), "near (2-3)": (2, 3), "further (4-6)": (4, 6)}


@dataclass
class SceneRecord:
    scenario: str
    seed: int
    standoff: float                         # camera distance used (from the board's front edge)
    target: tuple                           # (thickness, height) of the book to insert
    truth: ShelfEstimate
    camera: Dict[str, ShelfEstimate]        # "perfect", "noisy"
    truth_for: Dict[str, List[Optional[int]]]   # per camera book: index of its true book in truth.books


# ---------------------------------------------------------------------------
# Collect
# ---------------------------------------------------------------------------
def collect(seeds: int, path: str):
    from camera import DepthNoise, WristCamera
    from inspection import look_from, solve_inspection
    from perception import GroundTruthPerception
    from rgbd_perception import RGBDPerception
    from shelf_scene import SCENARIOS, ShelfConfig, make_episode

    panda = os.path.join(HERE, "mujoco_menagerie", "franka_emika_panda", "panda.xml")
    records = []
    for scenario in SCENARIOS:
        for seed in range(seeds):
            model, data, row, _ = make_episode(ShelfConfig(), scenario, np.random.default_rng(seed), panda_xml=panda)
            view = solve_inspection(model, data, row)
            if view is None:
                print(f"{scenario} {seed}: no inspection pose, skipped")
                continue
            look_from(model, data, view[0])
            truth = GroundTruthPerception(model, row).estimate(data)
            index = {model.geom(b.name + "_geom").id: i for i, b in enumerate(row.books)}
            camera, truth_for = {}, {}
            for kind, noise in (("perfect", None), ("noisy", DepthNoise())):
                cam = WristCamera(model, noise=noise, rng=np.random.default_rng(seed))
                with RGBDPerception(model, row, camera=cam) as perception:
                    camera[kind] = perception.estimate(data)
                    truth_for[kind] = [index.get(g) for g in perception.last_labels]
            records.append(SceneRecord(scenario, seed, view[1], (row.target_book.thickness, row.target_book.height),
                                       truth, camera, truth_for))
            print(f"{scenario} {seed}: {len(camera['perfect'].books)} of {len(truth.books)} books found", flush=True)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as fh:
        pickle.dump(records, fh)
    print(f"stored {len(records)} scenes in {path}")


class _Unpickler(pickle.Unpickler):
    """Run as a script, collect stores SceneRecord as __main__.SceneRecord:
    find it here wherever the file is loaded from."""
    def find_class(self, module, name):
        return SceneRecord if name == "SceneRecord" else super().find_class(module, name)


def load(path: str = DEFAULT_PATH) -> List[SceneRecord]:
    with open(path, "rb") as fh:
        return _Unpickler(fh).load()


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------
def pairs(rec: SceneRecord, kind: str):
    """(camera book, true book) for every camera book matched to a true one."""
    return [(b, rec.truth.books[i]) for b, i in zip(rec.camera[kind].books, rec.truth_for[kind]) if i is not None]


def book_errors(records: List[SceneRecord], kind: str, lo: int, hi: int) -> Dict[str, np.ndarray]:
    """Errors of matched books with lo <= |true slot offset| <= hi, fully in view."""
    out = {k: [] for k in ("corner_x", "corner_y", "corner_z", "corner", "thickness", "height",
                           "lean", "yaw", "spine_y")}
    for rec in records:
        for b, t in pairs(rec, kind):
            if not lo <= abs(t.slot_offset) <= hi or b.visible < 1:
                continue
            r = b.spine_corners - t.spine_corners
            out["corner_x"] += list(r[:, 0])
            out["corner_y"] += list(r[:, 1])
            out["corner_z"] += list(r[:, 2])
            out["corner"].append(np.linalg.norm(r, axis=1).max())
            out["thickness"].append(b.thickness - t.thickness)
            out["height"].append(b.height - t.height)
            out["lean"].append(math.degrees(b.lean - t.lean))
            out["yaw"].append(math.degrees(b.yaw - t.yaw))
            out["spine_y"].append(b.spine_y - t.spine_y)
    return {k: np.array(v) for k, v in out.items()}


def naming(records: List[SceneRecord], kind: str):
    """(books named as in the truth, books matched)."""
    ok = n = 0
    for rec in records:
        for b, t in pairs(rec, kind):
            n += 1
            ok += b.slot_offset == t.slot_offset
    return ok, n


def seen_per_side(rec: SceneRecord, kind: str):
    """Books found in an unbroken run out from the slot, per side, and how many
    the row has on that side."""
    found = {t.slot_offset for _, t in pairs(rec, kind)}
    out = []
    for side in (-1, 1):
        total = sum(1 for t in rec.truth.books if np.sign(t.slot_offset) == side)
        k = 0
        while k < total and side * (k + 1) in found:
            k += 1
        out.append((k, total))
    return out


def gap_errors(records: List[SceneRecord], kind: str):
    """(per-height errors where both are known, errors of the narrowest gap up
    to the target book's height, share of heights up to that height not seen)."""
    per_height, narrowest, unseen = [], [], []
    for rec in records:
        est, truth = rec.camera[kind], rec.truth
        both = ~np.isnan(est.gap_widths) & ~np.isnan(truth.gap_widths)
        per_height += list(est.gap_widths[both] - truth.gap_widths[both])
        up_to = est.gap_heights <= rec.target[1]
        unseen.append(np.isnan(est.gap_widths[up_to]).mean())
        narrowest.append(est.min_gap_width(rec.target[1]) - truth.min_gap_width(rec.target[1]))
    return np.array(per_height), np.array(narrowest), np.array(unseen)


# ---------------------------------------------------------------------------
# Calibration of NoisyPerception
# ---------------------------------------------------------------------------
NOISE_PARAMS = ("x", "z", "spine_y", "thickness", "height", "lean", "yaw")
NOISE_BANDS = ((1, 1), (2, 3), (4, 99))     # |slot offset| bands NoisyPerception draws its noise for


def book_params(b: BookEstimate) -> Dict[str, float]:
    """The per-book quantities NoisyPerception perturbs: spine centre along the
    row (x) and up (z), spine depth, size and angles."""
    centre = b.spine_corners.mean(axis=0)
    return {"x": centre[0], "z": centre[2], "spine_y": b.spine_y, "thickness": b.thickness,
            "height": b.height, "lean": b.lean, "yaw": b.yaw}


def robust(v) -> tuple:
    """(median, std) for a Gaussian whose central 95% matches the errors': std
    = 95th percentile of |error - median| / 1.96. The camera's errors have
    heavier tails than a Gaussian, so a std from the bulk would make the noise
    model too clean, and a plain std would let a few wild errors (e.g. a far
    fallen book half hidden by others) blow up every book's noise."""
    v = np.asarray(v)
    med = float(np.median(v))
    return med, float(np.percentile(np.abs(v - med), 95) / 1.96)


def noise_stats(records: List[SceneRecord], kind: str = "noisy"):
    """Per |slot offset| band, the camera's error in each NOISE_PARAMS quantity
    (m or rad), over matched books fully in view, as robust (median, std):
    {band: {param: (median, std)}}; and per band the number of books and the
    share of them with any error beyond 3 of those stds (outliers)."""
    stats, counts, outliers = {}, {}, {}
    for band in NOISE_BANDS:
        errs = {p: [] for p in NOISE_PARAMS}
        for rec in records:
            for b, t in pairs(rec, kind):
                if band[0] <= abs(t.slot_offset) <= band[1] and b.visible == 1:
                    est, true = book_params(b), book_params(t)
                    for p in NOISE_PARAMS:
                        errs[p].append(est[p] - true[p])
        stats[band] = {p: robust(v) for p, v in errs.items()}
        counts[band] = len(errs["x"])
        wild = np.zeros(counts[band], dtype=bool)
        for p, v in errs.items():
            med, sd = stats[band][p]
            wild |= np.abs(np.asarray(v) - med) > 3 * max(sd, 1e-9)
        outliers[band] = float(wild.mean())
    return stats, counts, outliers


def view_reach(records: List[SceneRecord], kind: str = "perfect") -> float:
    """How far along the row from the slot the camera sees books: the distance
    (to a book's outer bottom corner) that best separates the books it found
    from those it didn't."""
    seen, dist = [], []
    for rec in records:
        found = {i for i in rec.truth_for[kind] if i is not None}
        x_slot = rec.truth.slot_x()
        for i, t in enumerate(rec.truth.books):
            outer = t.spine_corners[0, 0] if t.slot_offset < 0 else t.spine_corners[1, 0]
            seen.append(i in found)
            dist.append(abs(outer - x_slot))
    seen, dist = np.array(seen), np.array(dist)
    candidates = np.sort(dist)
    agreement = [np.mean((dist <= L) == seen) for L in candidates]
    return float(candidates[int(np.argmax(agreement))])


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def summary(v: np.ndarray, scale: float = 1.0) -> str:
    v = v[~np.isnan(v)] * scale
    if len(v) == 0:
        return "-"
    return f"{np.mean(v):+.2f} / {np.std(v):.2f} / {np.percentile(np.abs(v), 95):.2f}"


def report(records: List[SceneRecord]):
    scenes = sorted({(r.scenario) for r in records})
    print(f"{len(records)} scenes ({', '.join(f'{s}: {sum(r.scenario == s for r in records)}' for s in scenes)}); "
          f"camera at {dict(zip(*np.unique([r.standoff for r in records], return_counts=True)))} m from the board edge\n")
    for kind in ("perfect", "noisy"):
        ok, n = naming(records, kind)
        print(f"== {kind} depth: books named as in the truth {ok}/{n}")
        print("   mean / std / 95% of |error|   (mm or deg)")
        rows = [("corner x", "corner_x", 1000), ("corner y (depth)", "corner_y", 1000), ("corner z", "corner_z", 1000),
                ("worst corner", "corner", 1000), ("thickness", "thickness", 1000), ("height", "height", 1000),
                ("spine depth", "spine_y", 1000), ("lean", "lean", 1), ("swivel", "yaw", 1)]
        errs = {name: book_errors(records, kind, *rng) for name, rng in OFFSET_BINS.items()}
        print(f"   {'':18s}" + "".join(f"{name:>28s}" for name in OFFSET_BINS))
        for label, key, scale in rows:
            print(f"   {label:18s}" + "".join(f"{summary(errs[name][key], scale):>28s}" for name in OFFSET_BINS))
        print(f"   {'books':18s}" + "".join(f"{len(errs[name]['corner']):>28d}" for name in OFFSET_BINS))
        per_height, narrowest, unseen = gap_errors(records, kind)
        print(f"   gap width, per height:            {summary(per_height, 1000)} mm")
        print(f"   narrowest gap up to the book:     {summary(narrowest, 1000)} mm; "
              f"heights not seen {100 * unseen.mean():.1f}%")
        sides = [s for rec in records for s in seen_per_side(rec, kind)]
        seen = np.array([k for k, _ in sides])
        whole = np.mean([k == total for k, total in sides])
        print(f"   books seen out from the slot per side: min {seen.min()}, median {np.median(seen):.0f}, "
              f"whole side {100 * whole:.0f}% of sides\n")


def print_calibration(records: List[SceneRecord]):
    """NoisyPerception's settings, measured: the camera's errors with D405-like
    depth noise, and how far along the row it sees."""
    stats, counts, outliers = noise_stats(records, "noisy")
    print("CAMERA_ERRORS = {   # |slot offset| band: {quantity: (median, std)}, m or rad")
    for band, by_param in stats.items():
        inner = ", ".join(f'"{p}": ({m:.5f}, {sd:.5f})' for p, (m, sd) in by_param.items())
        print(f"    {band}: {{{inner}}},   # {counts[band]} books, {100 * outliers[band]:.0f}% with an outlier")
    print("}")
    print(f"VIEW_REACH = {view_reach(records, 'noisy'):.3f}   # m along the row from the slot")


class _Fixed:
    """A perception source that returns a stored estimate."""
    def __init__(self, est: ShelfEstimate):
        self.est = est

    def estimate(self, data=None) -> ShelfEstimate:
        return self.est


def with_model(records: List[SceneRecord], seed: int = 0) -> List[SceneRecord]:
    """The records with a "model" estimate added: NoisyPerception (as
    calibrated in perception.py) on each scene's ground truth."""
    from dataclasses import replace
    from perception import NoisyPerception
    rng = np.random.default_rng(seed)
    out = []
    for rec in records:
        est = NoisyPerception(_Fixed(rec.truth), rng).estimate(None)
        index = {t.slot_offset: i for i, t in enumerate(rec.truth.books)}
        out.append(replace(rec, camera={**rec.camera, "model": est},
                           truth_for={**rec.truth_for, "model": [index[b.slot_offset] for b in est.books]}))
    return out


def print_validation(records: List[SceneRecord]):
    """The camera's errors (with depth noise) next to NoisyPerception's, per
    quantity and band, and for the gap and the books seen."""
    records = with_model(records)
    cam, _, _ = noise_stats(records, "noisy")
    mod, _, _ = noise_stats(records, "model")
    print("median / std of the error: camera (noisy depth)  vs  NoisyPerception   (mm, or deg for lean and swivel)")
    for band in cam:
        print(f"  |slot offset| {band[0]}-{band[1] if band[1] < 99 else 'more'}")
        for q in cam[band]:
            k = 180 / math.pi if q in ("lean", "yaw") else 1000
            (cm, cs), (mm, ms) = cam[band][q], mod[band][q]
            print(f"    {q:10s} {k * cm:+7.2f} / {k * cs:5.2f}   vs  {k * mm:+7.2f} / {k * ms:5.2f}")
    for kind, name in (("noisy", "camera"), ("model", "NoisyPerception")):
        _, narrowest, unseen = gap_errors(records, kind)
        sides = [k for rec in records for k, _ in seen_per_side(rec, kind)]
        known = narrowest[~np.isnan(narrowest)]
        print(f"  {name:16s} narrowest gap error: median {1000 * np.median(known):+.2f}, 95% |.| "
              f"{1000 * np.percentile(np.abs(known), 95):.2f} mm (unknown in {len(narrowest) - len(known)} of "
              f"{len(narrowest)} scenes); books seen per side: median {np.median(sides):.0f}, min {min(sides)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=("collect", "report", "calibrate", "validate"))
    ap.add_argument("--seeds", type=int, default=12, help="collect: seeds per scenario")
    ap.add_argument("--path", default=DEFAULT_PATH)
    a = ap.parse_args()
    if a.command == "collect":
        collect(a.seeds, a.path)
    elif a.command == "report":
        report(load(a.path))
    elif a.command == "calibrate":
        print_calibration(load(a.path))
    else:
        print_validation(load(a.path))


if __name__ == "__main__":
    main()
