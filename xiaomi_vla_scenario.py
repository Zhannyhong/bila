"""Experimental Xiaomi Robotics VLA scenario definitions for this repo.

This module does not bundle Xiaomi's model weights. Instead it defines a
concrete bookshelf task package that a VLA-style policy can consume:

    images + short task instruction + structured context -> choose a macro-skill

For this repository the recommended role for a VLA is high-level skill
selection, not direct low-level control. The controllers in `insert_book.py`
and `scripted_controller.py` remain the execution backend.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import asdict, dataclass
from typing import Any

import mujoco
import numpy as np

from insert_book import DEFAULT_PANDA, make_episode
from robot_runtime import RobotDecision, observe_and_route
from shelf_scene import ShelfConfig, book_catalog_label


@dataclass(frozen=True)
class MacroSkill:
    name: str
    description: str
    backend: str


@dataclass(frozen=True)
class XiaomiVLAScenario:
    scenario_id: str
    model_id: str
    task_instruction: str
    initial_route: str
    observation_sources: list[str]
    state_summary: dict[str, Any]
    allowed_skills: list[MacroSkill]
    success_criteria: list[str]
    controller_bindings: dict[str, str]
    notes: list[str]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["allowed_skills"] = [asdict(s) for s in self.allowed_skills]
        return payload


def build_xiaomi_vla_leaning_insert_scenario(model: mujoco.MjModel, data: mujoco.MjData, row, *,
                                             seed: int | None = None,
                                             model_id: str = "XiaomiRobotics/Xiaomi-Robotics-1-5B",
                                             instruction: str | None = None,
                                             perception_mode: str = "ground_truth",
                                             rng: np.random.Generator | None = None) -> XiaomiVLAScenario:
    """Create one VLA-oriented task package for a leaning-row insertion episode."""
    decision: RobotDecision = observe_and_route(model, data, row,
                                                mode=perception_mode,
                                                rng=rng or np.random.default_rng(seed or 0))
    if decision.route != "leaning":
        raise ValueError(f"expected a leaning row, got route {decision.route!r}")

    lean_into_slot_deg = math.degrees(decision.leaning.lean_into_slot)
    target = row.target_book
    state_summary = {
        "seed": seed,
        "target_level": row.target_level,
        "left_neighbor_label": book_catalog_label(row.books[row.target_index - 1]),
        "right_neighbor_label": book_catalog_label(row.books[row.target_index]),
        "target_book_label": book_catalog_label(target),
        "target_book_thickness_mm": round(1000 * target.thickness, 1),
        "target_book_depth_mm": round(1000 * target.depth, 1),
        "target_book_height_mm": round(1000 * target.height, 1),
        "observed_min_gap_mm": round(1000 * decision.insertion.slot.min_gap_width, 1),
        "seat_rule": decision.insertion.seat_rule,
        "observed_lean_into_slot_deg": round(lean_into_slot_deg, 1),
        "needs_unlean_first": True,
        "recommended_follow_up": "reobserve_then_insert",
    }

    skills = [
        MacroSkill(
            name="recover_leaning_slot",
            description="Use the leaning-book controller to push the blocking neighbour upright and widen the slot if needed.",
            backend="insert_book.run_leaning_prep",
        ),
        MacroSkill(
            name="reobserve_gap",
            description="Call perception again after contact changes the row so the next insertion skill uses the updated slot estimate.",
            backend="robot_runtime.observe_and_route",
        ),
        MacroSkill(
            name="insert_if_clear",
            description="Insert the target book when the re-observed slot is already wide enough.",
            backend="insert_book.Episode.insert",
        ),
        MacroSkill(
            name="insert_if_tight",
            description="Widen the re-observed slot with the fingertip strategy, then insert the target book.",
            backend="insert_book.Episode.insert_tight",
        ),
        MacroSkill(
            name="abort_and_request_help",
            description="Stop when the row remains unreachable, unstable, or unsafe after recovery attempts.",
            backend="operator_escalation",
        ),
    ]

    return XiaomiVLAScenario(
        scenario_id=f"xiaomi_vla_leaning_insert_seed_{seed if seed is not None else 'unknown'}",
        model_id=model_id,
        task_instruction=instruction or (
            f"Shelve the target book {book_catalog_label(target)} between "
            f"{book_catalog_label(row.books[row.target_index - 1])} and {book_catalog_label(row.books[row.target_index])}. "
            "First recover the leaning neighbour that intrudes into the slot, then re-observe the row, and finally "
            "choose the correct insertion skill without toppling other books."
        ),
        initial_route=decision.route,
        observation_sources=["scene_cam", "wrist_cam", "joint_state", "wrist_force_torque", "ShelfEstimate"],
        state_summary=state_summary,
        allowed_skills=skills,
        success_criteria=[
            "The leaning neighbour no longer blocks the slot after recovery.",
            "The target book is inserted upright and released.",
            "Other books stay within the repository's disturbance tolerance.",
            "The policy re-observes after recovery before selecting the insertion skill.",
        ],
        controller_bindings={
            "recover_leaning_slot": "insert_book.run_leaning_prep",
            "reobserve_gap": "robot_runtime.observe_and_route",
            "insert_if_clear": "insert_book.Episode.insert",
            "insert_if_tight": "insert_book.Episode.insert_tight",
        },
        notes=[
            "This scenario uses the Xiaomi VLA as a high-level skill selector, not as a direct torque or joint controller.",
            "The scenario is intentionally narrow so VLA decisions can be compared against scripted routing on the same perception interface.",
        ],
    )


def build_demo_scenario(seed: int = 0, panda_xml: str = DEFAULT_PANDA,
                        perception_mode: str = "ground_truth",
                        model_id: str = "XiaomiRobotics/Xiaomi-Robotics-1-5B",
                        instruction: str | None = None) -> XiaomiVLAScenario:
    cfg = ShelfConfig()
    model, data, row = make_episode(cfg, "leaning", np.random.default_rng(seed), panda_xml)
    return build_xiaomi_vla_leaning_insert_scenario(model, data, row,
                                                    seed=seed,
                                                    model_id=model_id,
                                                    instruction=instruction,
                                                    perception_mode=perception_mode)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--panda", default=DEFAULT_PANDA)
    ap.add_argument("--perception", choices=("ground_truth", "noisy"), default="ground_truth")
    ap.add_argument("--model-id", default="XiaomiRobotics/Xiaomi-Robotics-1-5B",
                    help="Hugging Face model identifier; override this if you want to target a different XR-1 checkpoint such as a future 10B release")
    ap.add_argument("--instruction", help="optional natural-language instruction override")
    ap.add_argument("--out", help="optional JSON output path")
    args = ap.parse_args()

    scenario = build_demo_scenario(args.seed, args.panda, args.perception, args.model_id, args.instruction)
    payload = json.dumps(scenario.to_dict(), indent=2, sort_keys=True)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(payload + "\n")
    else:
        print(payload)


if __name__ == "__main__":
    main()
