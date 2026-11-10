"""
Training images for the learned perception: RGB-D images of random scenes with
a mask for every book and a box and text for every call-number sticker, all
read off the simulation.

    # in WSL; software rendering is fine (~0.3 s an image, the arm poses cost more)
    MUJOCO_GL=egl OMP_NUM_THREADS=1 uv run python -m bila.vision.make_vision_dataset \\
        --out ~/bila_data/train --first-seed 2000 --scenes 600 --workers 6
    ... --out ~/bila_data/val --first-seed 9000 --scenes 60

Scenes (one per seed; seeds 0-49 are the controllers' tests and 1000-1999
eval_perception's, so start at 2000):
  shelf   shelf_scene's bookcase in any of the five scenarios, as the wrist
          camera sees it: from the three views of the first look
          (insert_book.SWEEP) and one random view (along the row, 15-48 cm
          back, at any height in the level). Half the shelf scenes are plain
          coloured boxes, the rest wear the repo's book covers (sampled sizes,
          the repo's own books, or books split into page sections).
  cart    cart_demo's scene (the repo's books, with call-number stickers):
          the wrist camera's first look, and close-ups for reading stickers:
          two of the row near the slot (10-25 cm from the spines) and one
          looking down into the tray (15-30 cm above the spines).
CART_SHARE of the scenes are cart scenes. Every scene also gets its own
lighting and bookcase colour, and the depth carries a D405's errors
(DepthNoise.d405()). No calibration error: it changes only the pose the
robot believes, and labels here use the true pose.

Every wrist-camera view is a real arm pose (as inspection.solve_view finds
them, collision-free), so the gripper is in the images as it would be. Views
the arm can't reach are skipped.

Layout of --out, one file of each per image `<seed>_<view>`:
  rgb/    PNG, 8 bit
  depth/  PNG, 16 bit, in units of DEPTH_UNIT (0.1 mm); 0 = no reading
  masks/  PNG, 16 bit: 0 = no book, k = the k-th entry of the image's `books`
  labels/ JSON: camera (K, pose world <- camera, OpenCV axes), books (name,
          where it stands, pixels, box) and stickers (text, corners, pixels)
  scenes/<seed>.json  what the scene was, how long it took (written last, so
          a run that stops can be restarted and skips scenes that are done)
  index.jsonl  one line per image, appended as scenes finish

One mask per book (rgbd_perception.book_of_geom): its box, the face meshes
that draw its cover, its page sections and its sticker are all one book.
Sticker corners are the sticker's printed face projected into the image, in
reading order (top-left, top-right, bottom-right, bottom-left of the text),
so cv2.getPerspectiveTransform onto a rectangle gives the label upright.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from multiprocessing import Pool
from typing import List, Optional

import cv2
import mujoco
import numpy as np

from bila import ROOT as HERE
PANDA = os.path.join(HERE, "mujoco_menagerie", "franka_emika_panda", "panda.xml")
DEPTH_UNIT = 1e-4             # m per count in the depth PNGs: 1.5 m (the camera's range) is 15000
CART_SHARE = 0.4              # share of scenes that are cart scenes (with stickers)
IK_SEEDS = 24                 # random arm configurations tried per view, besides the ready pose
MIN_BOOK_PX = 1               # a book with fewer pixels than this is left out of an image's list
SHELF_VIEWS = 1               # random wrist views per shelf scene, besides the sweep
ROW_CLOSEUPS = 2              # close-ups of the row's stickers per cart scene
TRAY_CLOSEUPS = 1             # close-ups looking down into the tray per cart scene
LIFT_TRIES = (0.45, 0.35, 0.55, 0.25, 0.65)   # lift heights tried for a view of the tray (as cart_demo)


# ---------------------------------------------------------------------------
# Scenes
# ---------------------------------------------------------------------------
def scene_kind(seed: int) -> str:
    return "cart" if np.random.default_rng([seed, 1]).random() < CART_SHARE else "shelf"


def shelf_appearance(rng) -> dict:
    """How a shelf scene's books look: plain boxes half the time, else the
    repo's covers on sampled books, on the repo's own books, or on books split
    into page sections (only the Voynich mesh has sections, so those all wear it)."""
    u = rng.random()
    if u < 0.5:
        return {}
    if u < 0.75:
        return dict(book_meshes=True)
    if u < 0.9:
        return dict(book_meshes=True, catalogue=True)
    return dict(book_meshes=True, book_pages=4)


def vary_look(model: mujoco.MjModel, rng) -> dict:
    """Random lighting and bookcase/floor colour, in place. Only what is drawn
    changes, not the physics."""
    ambient, diffuse = rng.uniform(0.15, 0.55), rng.uniform(0.3, 0.8)
    model.vis.headlight.ambient[:] = ambient
    model.vis.headlight.diffuse[:] = diffuse
    for i in range(model.nlight):
        model.light_diffuse[i] = rng.uniform(0.3, 0.9) * rng.uniform(0.85, 1.0, 3)   # slightly tinted
        model.light_pos[i] += rng.uniform(-0.8, 0.8, 3)
    shelf = model.body("shelf").id
    if rng.random() < 0.3:
        wood = np.full(3, rng.uniform(0.25, 0.95))                   # painted: white to dark grey
    else:
        wood = np.clip(np.array([0.55, 0.40, 0.25]) * rng.uniform(0.5, 1.5), 0, 1)
    for g in range(model.ngeom):
        if model.geom_bodyid[g] == shelf and not model.geom(g).name.startswith("fiducial"):
            model.geom_rgba[g, :3] = wood
    floor = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_MATERIAL, "floor")
    if floor >= 0:
        model.mat_rgba[floor, :3] = rng.uniform(0.5, 1.0, 3)
    return {"headlight": [float(ambient), float(diffuse)], "wood": wood.round(3).tolist()}


def cart_numbers(row, tray, seed):
    """Call numbers as cart_demo.main gives them: the shelf's from the map,
    and the returned books not for the target row get spare numbers of other rows."""
    from bila.control.cart_demo import STEP, call_numbers
    numbers, shelf_map = call_numbers(row)
    spare = [n for level, _, first, last in shelf_map if level != row.target_level
             for n in range(first + STEP // 2, last, STEP)]
    others = [o for o in tray if o is not row.target_book]
    for o, n in zip(others, np.random.default_rng(seed).choice(spare, len(others), replace=False)):
        numbers[o.name] = int(n)
    return numbers


# ---------------------------------------------------------------------------
# Camera poses
# ---------------------------------------------------------------------------
def solve_camera(model, data, ik, cam_pos, cam_rot, rng) -> Optional[np.ndarray]:
    """inspection.solve_camera with fewer random starts (IK_SEEDS)."""
    from bila.control.inspection import solve_camera as solve
    return solve(model, data, ik, cam_pos, cam_rot, rng, seeds=IK_SEEDS)


def looking(forward, ref) -> np.ndarray:
    from bila.control.inspection import looking as axes
    return axes(forward, ref)


# ---------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------
def book_labels(model, frame, oracle, where: dict):
    """(instance mask (H, W) uint16, [book dicts]) for one image."""
    labels = oracle.labels(frame)
    ids, counts = np.unique(labels, return_counts=True)
    mask = np.zeros(labels.shape, np.uint16)
    books = []
    for g, n in zip(ids, counts):
        if g < 0 or n < MIN_BOOK_PX:
            continue
        name = model.geom(int(g)).name[:-len("_geom")]
        v, u = np.nonzero(labels == g)
        books.append({"name": name, "where": where.get(name, "other"), "pixels": int(n),
                      "box": [int(u.min()), int(v.min()), int(u.max()) + 1, int(v.max()) + 1]})
        mask[labels == g] = len(books)
    return mask, books


def sticker_corners(model, data, g) -> np.ndarray:
    """World corners (4, 3) of a sticker's printed face, in reading order:
    top-left, top-right, bottom-right, bottom-left of its text. The sticker is
    a thin box on the spine (cart_demo.build): its face is at local +y, the
    text's lines run along local x, stacked along local z (up the book). Seen
    from outside (+y), local +x is on the reader's left."""
    hx, hy, hz = model.geom_size[g]
    R, p = data.geom_xmat[g].reshape(3, 3), data.geom_xpos[g]
    local = [(hx, hy, hz), (-hx, hy, hz), (-hx, hy, -hz), (hx, hy, -hz)]
    return np.array([p + R @ c for c in local])


def sticker_labels(model, data, frame, texts: dict) -> List[dict]:
    """Every sticker at least partly in the image: its text, corners (px, in
    reading order), pixels seen, and the share of its face that is seen (less
    than 1 when something covers it or it is cut off)."""
    from bila.vision.camera import project
    out = []
    for book, text in texts.items():
        g = model.geom(book + "_label").id
        n = int((frame.seg == g).sum())
        if n == 0:
            continue
        u, v, z = project(frame, sticker_corners(model, data, g))
        corners = np.column_stack([u, v])
        area = abs(cv2.contourArea(corners.astype(np.float32)))
        out.append({"book": book, "text": text, "corners": corners.round(2).tolist(), "pixels": n,
                    "seen": round(min(1.0, n / max(area, 1.0)), 3)})
    return out


# ---------------------------------------------------------------------------
# One scene
# ---------------------------------------------------------------------------
class Writer:
    """Writes one scene's images and their labels."""

    def __init__(self, out: str, seed: int, model, data, where, texts):
        self.out, self.seed, self.m, self.d = out, seed, model, data
        self.where, self.texts = where, texts
        from bila.vision.rgbd_perception import OracleSegmenter
        self.oracle = OracleSegmenter(model)
        self.records, self.render_s = [], 0.0

    def __call__(self, cam, view: str, extra: Optional[dict] = None):
        t = time.time()
        frame = cam.capture(self.d)
        self.render_s += time.time() - t
        mask, books = book_labels(self.m, frame, self.oracle, self.where)
        if not books:
            return
        stem = f"{self.seed:05d}_{view}"
        cv2.imwrite(os.path.join(self.out, "rgb", stem + ".png"), cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2BGR))
        depth = np.clip(np.round(frame.depth / DEPTH_UNIT), 0, 65535).astype(np.uint16)
        cv2.imwrite(os.path.join(self.out, "depth", stem + ".png"), depth)
        cv2.imwrite(os.path.join(self.out, "masks", stem + ".png"), mask)
        label = {"image": stem, "seed": self.seed, "view": view, "camera": cam.name,
                 "K": frame.K.round(4).tolist(), "pose": frame.pose.round(6).tolist(),
                 "depth_unit": DEPTH_UNIT, "books": books,
                 "stickers": sticker_labels(self.m, self.d, frame, self.texts) if self.texts else [],
                 **(extra or {})}
        with open(os.path.join(self.out, "labels", stem + ".json"), "w") as fh:
            json.dump(label, fh)
        self.records.append({"image": stem, "seed": self.seed, "view": view, "camera": cam.name,
                             "books": len(books), "stickers": len(label["stickers"])})


def make_shelf_scene(seed: int, rng):
    from bila.sim.shelf_scene import SCENARIOS, ShelfConfig, make_episode
    scenario = SCENARIOS[int(rng.integers(len(SCENARIOS)))]
    look = shelf_appearance(rng)
    model, data, row, _ = make_episode(ShelfConfig(**look), scenario, rng, panda_xml=PANDA)
    where = {b.name: "target_row" for b in row.books} | {b.name: f"level{fl.level}" for fl in row.filler_levels
                                                         for b in fl.books}
    return model, data, row, scenario, look, where


def shelf_views(model, data, row, rng, write, cam):
    """The sweep of the first look, then random views."""
    from bila.control.insert_book import SWEEP, ArmIK
    from bila.control.inspection import STANDOFFS, look_from, solve_inspection, solve_view
    from bila.sim.shelf_scene import PANDA_HOME
    ik = ArmIK(model, PANDA_HOME[:7])
    for dx in SWEEP:
        v = solve_inspection(model, data, row, dx, ik)
        if v is not None:
            look_from(model, data, v[0])
            write(cam, f"sweep{dx:+.1f}", {"dx": dx, "standoff": v[1]})
    for k in range(SHELF_VIEWS):
        dx, standoff = rng.uniform(-0.35, 0.35), rng.uniform(0.15, STANDOFFS[0])
        height = rng.uniform(0.05, max(0.06, row.clear_height - 0.05))
        q = solve_view(model, data, row, dx, ik, standoff=standoff, height=height)
        if q is not None:
            look_from(model, data, q)
            write(cam, f"random{k}", {"dx": dx, "standoff": standoff, "height": height})


def cart_views(model, data, row, tray, rng, write, cams):
    """The wrist camera's first look, then close-ups of the stickers in the
    row and in the tray."""
    from bila.control.insert_book import ArmIK
    from bila.control.inspection import look_from, slot_x, solve_inspection, solve_view
    from bila.sim.shelf_scene import PANDA_HOME
    ik = ArmIK(model, PANDA_HOME[:7])
    v = solve_inspection(model, data, row, 0.0, ik)
    if v is not None:
        look_from(model, data, v[0])
        write(cams["wrist"], "sweep+0.0", {"dx": 0.0, "standoff": v[1]})
    ti = row.target_index
    near = [b for b in row.books[max(0, ti - 2):ti + 2]]            # the slot's neighbours, two each side
    for k in range(ROW_CLOSEUPS):
        b = near[int(rng.integers(len(near)))]
        dx = b.x - slot_x(row) + rng.uniform(-0.02, 0.02)
        standoff, height = rng.uniform(0.08, 0.23), rng.uniform(0.03, 0.10)   # the sticker is 4.5 cm up the book
        q = solve_view(model, data, row, dx, ik, standoff=standoff, height=height)
        if q is not None:
            look_from(model, data, q)
            write(cams["wrist"], f"close_row{k}", {"dx": dx, "standoff": standoff, "height": height, "aim": b.name})

    lift = model.jnt_qposadr[model.joint("lift_z").id]
    lift_row = float(data.qpos[lift])
    for k in range(TRAY_CLOSEUPS):
        b = tray[int(rng.integers(len(tray)))]
        spot = data.geom_xpos[model.geom(b.name + "_label").id].copy()
        tilt = math.radians(rng.uniform(0, 20))
        heading = rng.uniform(0, 2 * math.pi)
        forward = np.array([math.sin(tilt) * math.cos(heading), math.sin(tilt) * math.sin(heading), -math.cos(tilt)])
        yaw = rng.uniform(0, 2 * math.pi)
        rot = looking(forward, [math.cos(yaw), math.sin(yaw), 0.0])
        pos = spot - forward * rng.uniform(0.15, 0.30)
        for h in LIFT_TRIES:
            data.qpos[lift] = h
            mujoco.mj_forward(model, data)
            q = solve_camera(model, data, ik, pos, rot, rng)
            if q is not None:
                look_from(model, data, q)
                write(cams["wrist"], f"close_tray{k}", {"lift": h, "aim": b.name})
                break
        data.qpos[lift] = lift_row
        mujoco.mj_forward(model, data)


def make_scene(task) -> List[dict]:
    """Every image of one scene; writes them and scenes/<seed>.json."""
    out, seed = task
    from bila.vision.camera import DepthNoise, WristCamera
    t0 = time.time()
    rng = np.random.default_rng(seed)
    kind = scene_kind(seed)
    info = {"seed": seed, "kind": kind}
    try:
        if kind == "shelf":
            model, data, row, scenario, look, where = make_shelf_scene(seed, rng)
            texts, tray = {}, []
        else:
            from bila.control import cart_demo
            from bila.sim.shelf_scene import settle
            scenario = ("clear", "tight", "leaning", "packed")[int(rng.integers(4))]
            cfg, row, tray, _ = cart_demo.sample(scenario, seed)
            numbers = cart_numbers(row, tray, seed)
            model, tray = cart_demo.build(cfg, row, tray, numbers)
            data = mujoco.MjData(model)
            if not settle(model, data, row)["stable"]:
                raise RuntimeError("the row did not settle")
            look = dict(book_meshes=True, catalogue=True)
            where = ({b.name: "target_row" for b in row.books}
                     | {b.name: f"level{fl.level}" for fl in row.filler_levels for b in fl.books}
                     | {b.name: "tray" for b in tray})
            books = {b.name: b for b in row.books + tray + [b for fl in row.filler_levels for b in fl.books]}
            texts = {n: f"{cart_demo.dewey(numbers[n])}\n{cart_demo.cutter(books[n])}".strip()
                     for n in numbers if n in books and mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, n + "_label") >= 0}
    except RuntimeError as e:
        info.update(error=str(e), seconds=round(time.time() - t0, 1))
        with open(os.path.join(out, "scenes", f"{seed:05d}.json"), "w") as fh:
            json.dump(info, fh)
        return []
    info.update(scenario=scenario, appearance=look, look=vary_look(model, rng), setup_s=round(time.time() - t0, 1))
    write = Writer(out, seed, model, data, where, texts)
    noise = DepthNoise.d405()
    cams = {"wrist": WristCamera(model, name="panda/wrist_cam", noise=noise, rng=np.random.default_rng([seed, 0]))}
    try:
        if kind == "shelf":
            shelf_views(model, data, row, rng, write, cams["wrist"])
        else:
            cart_views(model, data, row, tray, rng, write, cams)
    finally:
        for c in cams.values():
            c.close()
    info.update(images=len(write.records), render_s=round(write.render_s, 1), seconds=round(time.time() - t0, 1))
    with open(os.path.join(out, "scenes", f"{seed:05d}.json"), "w") as fh:
        json.dump(info, fh)
    return write.records


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--first-seed", type=int, default=2000)
    ap.add_argument("--scenes", type=int, default=10)
    ap.add_argument("--workers", type=int, default=1)
    a = ap.parse_args()
    if a.first_seed < 2000:
        ap.error("seeds below 2000 belong to the controllers' tests (0-49) and eval_perception (1000-1999)")
    out = os.path.expanduser(a.out)
    for sub in ("rgb", "depth", "masks", "labels", "scenes"):
        os.makedirs(os.path.join(out, sub), exist_ok=True)
    seeds = [s for s in range(a.first_seed, a.first_seed + a.scenes)
             if not os.path.exists(os.path.join(out, "scenes", f"{s:05d}.json"))]
    print(f"{len(seeds)} scenes to make ({a.scenes - len(seeds)} already done) -> {out}", flush=True)
    t0, images = time.time(), 0
    with Pool(a.workers) as pool, open(os.path.join(out, "index.jsonl"), "a") as index:
        for k, records in enumerate(pool.imap_unordered(make_scene, [(out, s) for s in seeds]), 1):
            for r in records:
                index.write(json.dumps(r) + "\n")
            index.flush()
            images += len(records)
            el = time.time() - t0
            print(f"{k}/{len(seeds)} scenes, {images} images, {el / 60:.1f} min "
                  f"({el / max(images, 1):.1f} s an image)", flush=True)


if __name__ == "__main__":
    main()
