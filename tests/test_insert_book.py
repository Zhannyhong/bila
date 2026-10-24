"""Checks for the insertion pipeline's leaning handoff."""

import os

import mujoco
import numpy as np

import insert_book
import shelf_scene
from robot_runtime import observe_and_route
from shelf_scene import ShelfConfig

PANDA = os.path.join(os.path.dirname(os.path.abspath(shelf_scene.__file__)),
                     "mujoco_menagerie", "franka_emika_panda", "panda.xml")


def _geom_names(model: mujoco.MjModel) -> set[str]:
    return {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) for i in range(model.ngeom)}


def test_rebuild_insertion_scene_removes_pusher_and_preserves_state():
    cfg = ShelfConfig()
    model, data, row = insert_book.make_leaning_insert_episode(cfg, np.random.default_rng(0), PANDA)

    assert "pusher" in _geom_names(model)

    rebuilt_model, rebuilt_data = insert_book.rebuild_insertion_scene(cfg, row, PANDA, data)

    assert "pusher" not in _geom_names(rebuilt_model)
    assert rebuilt_data.time == data.time
    assert np.allclose(rebuilt_data.qpos, data.qpos)
    assert np.allclose(rebuilt_data.qvel, data.qvel)


def test_leaning_prep_hands_off_to_insertable_route():
    cfg = ShelfConfig()
    model, data, row = insert_book.make_leaning_insert_episode(cfg, np.random.default_rng(0), PANDA)

    prep = insert_book.run_leaning_prep(model, data, row)

    assert prep["finished"]

    rebuilt_model, rebuilt_data = insert_book.rebuild_insertion_scene(cfg, row, PANDA, data)
    decision = observe_and_route(rebuilt_model, rebuilt_data, row, mode="ground_truth", rng=np.random.default_rng(1))

    assert decision.route in {"clear", "tight"}
