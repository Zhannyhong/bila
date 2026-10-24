"""Render perception debug outputs for a settled shelf episode.

This script helps inspect:

- what the fixed scene camera sees,
- what the wrist camera sees,
- what the current geometric perception estimate reports,
- whether alternative wrist-camera placements would make debugging easier.

Examples:
  ./.venv/bin/python debug_perception.py --scenario leaning --seed 0
  ./.venv/bin/python debug_perception.py --scenario packed --seed 2 --camera wrist_cam
  ./.venv/bin/python debug_perception.py --scenario leaning --seed 0 \
      --wrist-pos 0.05 0 0.12 --wrist-xyaxes 0 1 0 1 0 0
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import asdict, is_dataclass
from typing import Any

import mujoco
import numpy as np
from PIL import Image

from perception import GroundTruthPerception, NoisyPerception
from shelf_scene import SCENARIOS, ShelfConfig, build_scene, make_panda_with_sensors, settle, sample_row

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_PANDA = os.path.join(HERE, "mujoco_menagerie", "franka_emika_panda", "panda.xml")


def _jsonable(x: Any) -> Any:
    if is_dataclass(x):
        return _jsonable(asdict(x))
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, dict):
        return {k: _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    return x


def _save_rgb(path: str, rgb: np.ndarray) -> None:
    Image.fromarray(rgb).save(path)


def _save_depth_preview(path: str, depth: np.ndarray) -> None:
    finite = np.isfinite(depth)
    if not np.any(finite):
        preview = np.zeros(depth.shape, dtype=np.uint8)
    else:
        lo = float(np.min(depth[finite]))
        hi = float(np.max(depth[finite]))
        if hi - lo < 1e-9:
            scaled = np.zeros(depth.shape, dtype=float)
        else:
            scaled = (depth - lo) / (hi - lo)
        # Nearer points appear brighter in the preview.
        preview = np.where(finite, (255 * (1.0 - np.clip(scaled, 0.0, 1.0))), 0).astype(np.uint8)
    Image.fromarray(preview).save(path)


def _camera_pose(model: mujoco.MjModel, data: mujoco.MjData, name: str) -> dict[str, Any]:
    cam = model.camera(name)
    cid = cam.id
    return {
        "name": name,
        "id": cid,
        "fovy_deg": float(model.cam_fovy[cid]),
        "world_pos": data.cam_xpos[cid].tolist(),
        "world_xmat": data.cam_xmat[cid].reshape(3, 3).tolist(),
    }


def _resolve_camera_name(model: mujoco.MjModel, requested: str) -> str:
    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, i) for i in range(model.ncam)]
    if requested in names:
        return requested
    suffixed = [n for n in names if n.endswith("/" + requested)]
    if len(suffixed) == 1:
        return suffixed[0]
    raise KeyError(requested)


def _render_camera(renderer: mujoco.Renderer, data: mujoco.MjData, name: str) -> tuple[np.ndarray, np.ndarray]:
    renderer.disable_depth_rendering()
    renderer.update_scene(data, camera=name)
    rgb = renderer.render().copy()

    renderer.enable_depth_rendering()
    renderer.update_scene(data, camera=name)
    depth = renderer.render().copy()
    renderer.disable_depth_rendering()
    return rgb, depth


def _maybe_override_panda_camera(panda_xml: str | None, pos: list[float] | None, xyaxes: list[float] | None) -> str | None:
    if panda_xml is None or (pos is None and xyaxes is None):
        return None
    base = make_panda_with_sensors(panda_xml)
    tree = ET.parse(base)
    root = tree.getroot()
    cam = root.find(".//camera[@name='wrist_cam']")
    if cam is None:
        raise RuntimeError("No camera named wrist_cam found in the Panda model XML")
    if pos is not None:
        cam.set("pos", " ".join(f"{v:.6g}" for v in pos))
    if xyaxes is not None:
        cam.set("xyaxes", " ".join(f"{v:.6g}" for v in xyaxes))
    ET.indent(root)
    fd, out = tempfile.mkstemp(prefix="panda_wrist_override_", suffix=".xml", dir=os.path.dirname(base))
    os.close(fd)
    tree.write(out, encoding="unicode")
    return out


def _maybe_override_attached_panda(xml: str, override_panda_path: str | None) -> str:
    if override_panda_path is None:
        return xml
    root = ET.fromstring(xml)
    model_ref = root.find(".//asset/model[@name='panda']")
    if model_ref is None:
        raise RuntimeError("No attached Panda model reference found in the scene XML")
    model_ref.set("file", override_panda_path)
    ET.indent(root)
    return ET.tostring(root, encoding="unicode")


def make_episode_for_debug(cfg: ShelfConfig, scenario: str, rng: np.random.Generator,
                           panda_xml: str | None, wrist_pos: list[float] | None,
                           wrist_xyaxes: list[float] | None, max_tries: int = 10):
    override_panda = _maybe_override_panda_camera(panda_xml, wrist_pos, wrist_xyaxes)
    for _ in range(max_tries):
        row = sample_row(cfg, scenario, rng)
        xml = build_scene(cfg, row, panda_xml)
        xml = _maybe_override_attached_panda(xml, override_panda)
        model = mujoco.MjModel.from_xml_string(xml)
        data = mujoco.MjData(model)
        rep = settle(model, data, row)
        if rep["stable"]:
            return model, data, row, rep
    raise RuntimeError(f"No stable {scenario} row in {max_tries} tries")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenario", choices=SCENARIOS, default="leaning")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--camera", choices=("all", "scene_cam", "wrist_cam"), default="all")
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--perception", choices=("ground_truth", "noisy"), default="ground_truth")
    ap.add_argument("--corner-std", type=float, default=0.002, help="noisy perception corner std, m")
    ap.add_argument("--min-gap", type=float, default=0.004, help="noisy perception min resolvable gap, m")
    ap.add_argument("--panda", default=DEFAULT_PANDA)
    ap.add_argument("--no-robot", action="store_true", help="omit the robot; wrist_cam will then be unavailable")
    ap.add_argument("--out", default=os.path.join(HERE, "debug_outputs", "perception"))
    ap.add_argument("--wrist-pos", nargs=3, type=float, metavar=("X", "Y", "Z"),
                    help="override wrist_cam position in the hand frame")
    ap.add_argument("--wrist-xyaxes", nargs=6, type=float, metavar=("XX", "XY", "XZ", "YX", "YY", "YZ"),
                    help="override wrist_cam xyaxes in the hand frame")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    model, data, row, rep = make_episode_for_debug(
        ShelfConfig(),
        args.scenario,
        rng,
        None if args.no_robot else args.panda,
        args.wrist_pos,
        args.wrist_xyaxes,
    )
    mujoco.mj_forward(model, data)

    out_dir = os.path.join(args.out, f"{args.scenario}_seed{args.seed}")
    os.makedirs(out_dir, exist_ok=True)

    gt = GroundTruthPerception(model, row).estimate(data)
    est = gt if args.perception == "ground_truth" else NoisyPerception(
        GroundTruthPerception(model, row), np.random.default_rng(args.seed), args.corner_std, args.min_gap
    ).estimate(data)

    estimate_payload = {
        "scenario": args.scenario,
        "seed": args.seed,
        "settle_report": _jsonable(rep),
        "estimate": _jsonable(est),
    }
    with open(os.path.join(out_dir, "estimate.json"), "w", encoding="utf-8") as fh:
        json.dump(estimate_payload, fh, indent=2)

    cameras = ["scene_cam", "wrist_cam"] if args.camera == "all" else [args.camera]
    renderer = mujoco.Renderer(model, height=args.height, width=args.width)
    saved = []
    try:
        for name in cameras:
            try:
                resolved = _resolve_camera_name(model, name)
                pose = _camera_pose(model, data, resolved)
            except KeyError:
                print(f"skip {name}: camera not present in this scene")
                continue

            rgb, depth = _render_camera(renderer, data, resolved)
            rgb_path = os.path.join(out_dir, f"{name}_rgb.png")
            depth_png_path = os.path.join(out_dir, f"{name}_depth_preview.png")
            depth_npy_path = os.path.join(out_dir, f"{name}_depth.npy")
            meta_path = os.path.join(out_dir, f"{name}_meta.json")

            _save_rgb(rgb_path, rgb)
            _save_depth_preview(depth_png_path, depth)
            np.save(depth_npy_path, depth)
            with open(meta_path, "w", encoding="utf-8") as fh:
                json.dump(pose, fh, indent=2)
            saved.append((name, rgb_path, depth_png_path, depth_npy_path, meta_path))
    finally:
        renderer.close()

    print(f"saved perception debug outputs in {out_dir}")
    print(f"scenario={args.scenario} seed={args.seed} perception={args.perception} stable={rep['stable']}")
    print(f"target level={row.target_level} board_top={row.target_board_top:.3f} m books={len(row.books)}")
    print(f"slot gap as generated={(row.gap_right - row.gap_left):.4f} m")
    print(f"estimated min gap up to target height={est.min_gap_width(row.target_book.height):.4f} m")
    if args.wrist_pos is not None or args.wrist_xyaxes is not None:
        print("used a wrist camera override for this run")
    for name, rgb_path, depth_png_path, depth_npy_path, meta_path in saved:
        print(f"- {name}")
        print(f"  rgb:   {rgb_path}")
        print(f"  depth: {depth_png_path}")
        print(f"  raw depth array: {depth_npy_path}")
        print(f"  camera metadata: {meta_path}")
    print(f"- estimate json: {os.path.join(out_dir, 'estimate.json')}")


if __name__ == "__main__":
    main()
