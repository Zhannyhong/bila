"""Checks for the Xiaomi VLA bookshelf scenario package."""

import json

import numpy as np

import xiaomi_vla_scenario
from insert_book import make_episode
from shelf_scene import ShelfConfig


def test_build_xiaomi_vla_scenario_for_leaning_row():
    model, data, row = make_episode(ShelfConfig(), "leaning", np.random.default_rng(0),
                                    xiaomi_vla_scenario.DEFAULT_PANDA)

    scenario = xiaomi_vla_scenario.build_xiaomi_vla_leaning_insert_scenario(
        model, data, row, seed=0, model_id="XiaomiRobotics/Xiaomi-Robotics-1-10B"
    )

    assert scenario.initial_route == "leaning"
    assert scenario.model_id == "XiaomiRobotics/Xiaomi-Robotics-1-10B"
    assert scenario.state_summary["needs_unlean_first"] is True
    assert scenario.state_summary["observed_min_gap_mm"] > 0
    assert "between" in scenario.task_instruction
    assert scenario.state_summary["left_neighbor_label"] in scenario.task_instruction
    assert [skill.name for skill in scenario.allowed_skills][:2] == [
        "recover_leaning_slot",
        "reobserve_gap",
    ]


def test_xiaomi_vla_scenario_is_json_serializable():
    scenario = xiaomi_vla_scenario.build_demo_scenario(seed=1)
    payload = scenario.to_dict()

    assert payload["initial_route"] == "leaning"
    assert "insert_if_tight" in payload["controller_bindings"]
    assert "scene_cam" in payload["observation_sources"]
    json.dumps(payload)
