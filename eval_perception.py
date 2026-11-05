"""
How well the camera-based ShelfEstimate matches the ground truth, over many
scenes, for cameras with no error, basic depth noise, and a realistic D405.

    # slow (~1.5 h on 8 workers with software rendering): stores scenes/perception_eval.pkl
    MUJOCO_GL=egl OMP_NUM_THREADS=1 uv run python eval_perception.py collect --workers 8
    uv run python eval_perception.py report                             # tables from the stored estimates
    uv run python eval_perception.py calibrate                          # noise for NoisyPerception, from them
    uv run python eval_perception.py validate                           # does NoisyPerception err like the camera?

`collect` takes, in each scene (seeds 1000-1019 of clear, tight, leaning and
packed by default: away from the seeds the controllers are tested on), the
images the controller's first look takes: one from each view of the sweep
(insert_book.SWEEP) the arm reaches, fused as Episode.look fuses them. It
stores the ground truth and three camera estimates side by side:
  perfect     exact depth, exact calibration;
  basic       independent depth noise per pixel (DepthNoise());
  realistic   a D405's stereo errors and depth offset (DepthNoise.d405()) and a
              hand-eye / intrinsics calibration error (CalibrationError()),
              as insert_book.py --perception camera uses.
With `--segmenter segmenter.pt` it also stores
  learned     the realistic camera's own images, with the book masks from the
              learned segmenter (learned_segmenter.py) instead of the
              simulation's object ids: the difference to `realistic` is the
              segmenter's alone.
The slow part (solving the arm poses, rendering) then happens once; `report`,
and the calibration of NoisyPerception, read the stored estimates.

Every camera book is matched to its true book by the simulation's object id,
so errors are book for book. (A learned mask has no id: it takes the id of
the true book whose mask it overlaps most, learned_segmenter.OracleLabelled.) Errors are reported by how far the book stands
from the slot (|slot offset| 1, 2-3, 4-6), since the policy needs the
neighbours most.

The realistic camera's error has two parts, calibrated separately: a rigid
motion of the whole row, the same for every book in a scene (its calibration
and depth offset: perception.CAMERA_SYSTEMATIC), fitted per scene; and what
is left of each book's error once that is taken out (perception.CAMERA_ERRORS).
"""

from __future__ import annotations

import argparse
import math
import os
import pickle
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from perception import (SYSTEMATIC_PARAMS, BookEstimate, ShelfEstimate, apply_shift, book_from_spine, fit_rigid,
                        rotvec_from_matrix, slot_pivot)

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_PATH = os.path.join(HERE, "scenes", "perception_eval.pkl")
OFFSET_BINS = {"neighbours (1)": (1, 1), "near (2-3)": (2, 3), "further (4-6)": (4, 6)}
KINDS = ("perfect", "basic", "realistic", "learned")   # learned: only with collect --segmenter
CALIBRATE_ON = "realistic"            # the camera NoisyPerception imitates
COLLECT_SCENARIOS = ("clear", "tight", "leaning", "packed")   # collapsed is out of the project's scope
FIRST_SEED = 1000                     # clear of the controllers' test seeds (0-49) and their training seeds


@dataclass
class SceneRecord:
    scenario: str
    seed: int
    standoff: float                         # camera distance of the middle view (from the board's front edge)
    target: tuple                           # (thickness, height) of the book to insert
    truth: ShelfEstimate
    camera: Dict[str, ShelfEstimate]        # KINDS
    truth_for: Dict[str, List[Optional[int]]]   # per camera book: index of its true book in truth.books


# ---------------------------------------------------------------------------
# Collect
# ---------------------------------------------------------------------------
_segmenters = {}                      # learned segmenters loaded in this process, by weights file


def learned_segmenter(weights: str):
    if weights not in _segmenters:
        from learned_segmenter import LearnedSegmenter
        _segmenters[weights] = LearnedSegmenter(weights)
    return _segmenters[weights]


def collect_scene(task) -> Optional[SceneRecord]:
    """One scene's SceneRecord, or None if the arm reaches no view of it. Each
    camera kind gets its own camera, seeded with the scene's seed, as
    insert_book.make_eyes makes one. task: (scenario, seed) or (scenario,
    seed, segmenter weights) to add the learned kind."""
    scenario, seed, *weights = task
    if weights:
        import learned_segmenter as _  # noqa: F401  (imports torch._dynamo, needed before the first render)
    from camera import CalibrationError, DepthNoise, WristCamera
    from insert_book import SWEEP
    from inspection import look_from, solve_inspection
    from perception import GroundTruthPerception
    from rgbd_perception import RGBDPerception
    from shelf_scene import ShelfConfig, make_episode

    panda = os.path.join(HERE, "mujoco_menagerie", "franka_emika_panda", "panda.xml")
    model, data, row, _ = make_episode(ShelfConfig(), scenario, np.random.default_rng(seed), panda_xml=panda)
    views = [(dx, v) for dx in SWEEP if (v := solve_inspection(model, data, row, dx))]
    if not views:
        print(f"{scenario} {seed}: no inspection pose, skipped", flush=True)
        return None
    truth = GroundTruthPerception(model, row).estimate(data)
    index = {model.geom(b.name + "_geom").id: i for i, b in enumerate(row.books)}
    cameras = {"perfect": dict(noise=None),
               "basic": dict(noise=DepthNoise()),
               "realistic": dict(noise=DepthNoise.d405(), calibration=CalibrationError())}
    camera, truth_for = {}, {}
    for kind, errors in cameras.items():
        cam = WristCamera(model, rng=np.random.default_rng(seed), **errors)
        with RGBDPerception(model, row, camera=cam) as perception:
            frames = []
            for _, (q, _) in views:
                look_from(model, data, q)
                frames.append(perception.capture(data))
            camera[kind] = perception.estimate_from_frames(frames)
            truth_for[kind] = [index.get(g) for g in perception.last_labels]
        if kind == "realistic" and weights:
            from learned_segmenter import OracleLabelled
            learned = RGBDPerception(model, row, camera=cam,        # cam is closed: these frames only
                                     segmenter=OracleLabelled(learned_segmenter(weights[0]), model))
            camera["learned"] = learned.estimate_from_frames(frames)
            truth_for["learned"] = [index.get(g) for g in learned.last_labels]
    standoff = min(views, key=lambda v: abs(v[0]))[1][1]
    print(f"{scenario} {seed}: {len(views)} views, {len(camera['realistic'].books)} of {len(truth.books)} books found",
          flush=True)
    return SceneRecord(scenario, seed, standoff, (row.target_book.thickness, row.target_book.height),
                       truth, camera, truth_for)


def collect(scenarios, first_seed: int, seeds: int, path: str, workers: int = 1, segmenter: Optional[str] = None):
    tasks = [(s, seed) + ((segmenter,) if segmenter else ()) for s in scenarios
             for seed in range(first_seed, first_seed + seeds)]
    if workers > 1:
        from multiprocessing import Pool
        with Pool(workers) as pool:
            records = pool.map(collect_scene, tasks, chunksize=1)
    else:
        records = [collect_scene(t) for t in tasks]
    records = [r for r in records if r is not None]
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


def systematic_fit(rec: SceneRecord, kind: str):
    """The rigid motion (R, t) about the slot (perception.slot_pivot) that best
    maps the scene's true spine corners onto the camera's, over the matched
    books fully in view: the part of the camera's error all its books share.
    None if fewer than two books were matched."""
    matched = [(b, t) for b, t in pairs(rec, kind) if b.visible == 1]
    if len(matched) < 2:
        return None
    src = np.concatenate([t.spine_corners for _, t in matched])
    dst = np.concatenate([b.spine_corners for b, _ in matched])
    return fit_rigid(src, dst, slot_pivot(rec.truth))


def systematic_params(R, t) -> Dict[str, float]:
    return dict(zip(SYSTEMATIC_PARAMS, (*t, *rotvec_from_matrix(R))))


def systematic_stats(records: List[SceneRecord], kind: str = CALIBRATE_ON):
    """Per SYSTEMATIC_PARAMS, the robust (median, std) of the per-scene rigid
    motion over the scenes (m, rad), and how many scenes it was fitted in."""
    fits = [systematic_params(*f) for rec in records if (f := systematic_fit(rec, kind)) is not None]
    return {p: robust([f[p] for f in fits]) for p in SYSTEMATIC_PARAMS}, len(fits)


def without_systematic(records: List[SceneRecord], kind: str = CALIBRATE_ON) -> List[SceneRecord]:
    """The records with each scene's systematic_fit taken back out of the
    `kind` estimate's spine corners: what is left is each book's own error."""
    from dataclasses import replace
    out = []
    for rec in records:
        fit = systematic_fit(rec, kind)
        if fit is None:
            out.append(rec)
            continue
        R, t = fit
        pivot = slot_pivot(rec.truth)
        # The inverse of p -> R (p - pivot) + pivot + t is p -> R^T (p - pivot - t) + pivot.
        books = [book_from_spine(apply_shift(b.spine_corners - t, R.T, np.zeros(3), pivot), b.slot_offset,
                                 b.depth, b.visible) for b in rec.camera[kind].books]
        out.append(replace(rec, camera={**rec.camera, kind: replace(rec.camera[kind], books=books)}))
    return out


def noise_stats(records: List[SceneRecord], kind: str = CALIBRATE_ON):
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


def view_reach(records: List[SceneRecord], kind: str = CALIBRATE_ON) -> float:
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
    for kind in KINDS:
        if any(kind not in r.camera for r in records):
            continue
        ok, n = naming(records, kind)
        print(f"== {kind} camera: books named as in the truth {ok}/{n}")
        if kind == "learned":
            est = sum(len(r.camera[kind].books) for r in records)
            true = sum(len(r.truth.books) for r in records)
            print(f"   books found {n} of {true} in the row; {est - n} estimates match no true book")
        shift, n_fit = systematic_stats(records, kind)
        print(f"   whole-row shift (median / std, over {n_fit} scenes): "
              + ", ".join(f"{p} {1000 * m:+.2f}/{1000 * s:.2f} mm" for p, (m, s) in shift.items() if p in ("x", "y", "z"))
              + ", " + ", ".join(f"{p} {math.degrees(m):+.2f}/{math.degrees(s):.2f} deg"
                                 for p, (m, s) in shift.items() if p not in ("x", "y", "z")))
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
    """NoisyPerception's settings, measured on the realistic camera: the
    whole-row shift it shares across a scene's books, each book's error once
    that is taken out, and how far along the row it sees."""
    shift, n_fit = systematic_stats(records)
    inner = ", ".join(f'"{p}": ({m:.5f}, {sd:.5f})' for p, (m, sd) in shift.items())
    print(f"CAMERA_SYSTEMATIC = {{{inner}}}   # {n_fit} scenes")
    stats, counts, outliers = noise_stats(without_systematic(records))
    print("CAMERA_ERRORS = {   # |slot offset| band: {quantity: (median, std)}, m or rad")
    for band, by_param in stats.items():
        inner = ", ".join(f'"{p}": ({m:.5f}, {sd:.5f})' for p, (m, sd) in by_param.items())
        print(f"    {band}: {{{inner}}},   # {counts[band]} books, {100 * outliers[band]:.0f}% with an outlier")
    print("}")
    print(f"VIEW_REACH = {view_reach(records):.3f}   # m along the row from the slot")


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
    """The realistic camera's errors next to NoisyPerception's: the whole-row
    shift, each book's error with and without it, and the gap and the books
    seen."""
    records = with_model(records)
    kinds = (CALIBRATE_ON, "model")
    unit = lambda q: (180 / math.pi, "deg") if q in ("lean", "yaw", "rx", "ry", "rz") else (1000, "mm")
    print(f"median / std of the error: camera ({CALIBRATE_ON})  vs  NoisyPerception")
    (cam, n_cam), (mod, n_mod) = (systematic_stats(records, k) for k in kinds)
    print(f"  whole-row shift ({n_cam} vs {n_mod} scenes)")
    for q in SYSTEMATIC_PARAMS:
        k, u = unit(q)
        (cm, cs), (mm, ms) = cam[q], mod[q]
        print(f"    {q:10s} {k * cm:+7.2f} / {k * cs:5.2f}   vs  {k * mm:+7.2f} / {k * ms:5.2f}  {u}")
    for label, recs in (("each book, shift included", records),
                        ("each book, shift taken out", without_systematic(without_systematic(records), "model"))):
        cam, _, _ = noise_stats(recs, CALIBRATE_ON)
        mod, _, _ = noise_stats(recs, "model")
        print(f"  {label}")
        for band in cam:
            print(f"   |slot offset| {band[0]}-{band[1] if band[1] < 99 else 'more'}")
            for q in cam[band]:
                k, u = unit(q)
                (cm, cs), (mm, ms) = cam[band][q], mod[band][q]
                print(f"    {q:10s} {k * cm:+7.2f} / {k * cs:5.2f}   vs  {k * mm:+7.2f} / {k * ms:5.2f}  {u}")
    for kind, name in ((CALIBRATE_ON, "camera"), ("model", "NoisyPerception")):
        _, narrowest, unseen = gap_errors(records, kind)
        sides = [k for rec in records for k, _ in seen_per_side(rec, kind)]
        known = narrowest[~np.isnan(narrowest)]
        print(f"  {name:16s} narrowest gap error: median {1000 * np.median(known):+.2f}, 95% |.| "
              f"{1000 * np.percentile(np.abs(known), 95):.2f} mm (unknown in {len(narrowest) - len(known)} of "
              f"{len(narrowest)} scenes); books seen per side: median {np.median(sides):.0f}, min {min(sides)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=("collect", "report", "calibrate", "validate"))
    ap.add_argument("--seeds", type=int, default=20, help="collect: seeds per scenario")
    ap.add_argument("--first-seed", type=int, default=FIRST_SEED, help="collect: seeds first-seed .. first-seed+seeds-1")
    ap.add_argument("--scenarios", nargs="+", default=list(COLLECT_SCENARIOS), help="collect: scenarios")
    ap.add_argument("--workers", type=int, default=1, help="collect: scenes rendered in parallel")
    ap.add_argument("--segmenter", help="collect: learned segmenter weights, to add the learned kind")
    ap.add_argument("--path", default=DEFAULT_PATH)
    a = ap.parse_args()
    if a.command == "collect":
        collect(a.scenarios, a.first_seed, a.seeds, a.path, a.workers,
                os.path.expanduser(a.segmenter) if a.segmenter else None)
    elif a.command == "report":
        report(load(a.path))
    elif a.command == "calibrate":
        print_calibration(load(a.path))
    else:
        print_validation(load(a.path))


if __name__ == "__main__":
    main()
