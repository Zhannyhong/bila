# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

MuJoCo simulation for a library-robot "make room on the shelf" project: a randomized bookcase, a Franka Panda, and scripted baselines that insert a book into the target row. README.md is long and authoritative; read the relevant section before changing behaviour.

## Commands

```bash
git submodule update --init          # mujoco_menagerie (Panda model) is a submodule
uv sync                              # deps from pyproject.toml / uv.lock

uv run pytest                        # all tests (pythonpath = repo root, testpaths = tests/)
uv run pytest tests/test_scenes.py::test_name -x

uv run python -m bila.sim.shelf_scene --seed 0 --panda mujoco_menagerie/franka_emika_panda/panda.xml --out scenes
uv run python -m bila.viz.view_scene --scenario leaning --seed 0
uv run python -m bila.control.cart_demo --scenario tight                          # cart: read tray + row labels, make the gap, pick from the tray, insert
uv run python -m bila.control.insert_book --scenario tight --seed 0 [--trials 50]   # clear|tight|leaning|packed
```

- On macOS anything with `--view` must run under `mjpython` (`uv run mjpython -m bila.control.insert_book ... --view`).
- `--book-meshes` needs `generated_assets/books/`, produced once by `bila/sim/dae_to_obj.py --src <PBBM tams_books_world/meshes>` (see README "Book covers"). Missing assets silently fall back to plain boxes.
- No linter/formatter is configured.

## Architecture

Code lives in the `bila` package, grouped by job; run scripts as modules from the repo root (`uv run python -m bila.<group>.<module>`). Tests (`tests/`, flat) and notebooks import `bila.*` with the repo root on the path. `bila.ROOT` is the repo root; `scenes/`, `runs/`, `generated_assets/` and `mujoco_menagerie/` are resolved from it.

- `bila/sim/`: the scene (`shelf_scene`) and the book-mesh converter (`dae_to_obj`).
- `bila/vision/`: the camera and everything that turns images into a `ShelfEstimate` (`camera`, `perception`, `rgbd_perception`, `learned_segmenter`, `label_reader`, `call_numbers_seen`), plus the dataset maker and perception evaluation.
- `bila/control/`: the scripted controllers (`insert_book`, `cart_demo`) and the inspection pose (`inspection`).
- `bila/rl/`: the residual-PPO gap environment and training (`gap_env`, `observation`, `residual_ppo`, `train_residual`).
- `bila/viz/`: viewers and figures (`visualize`, `view_scene`, `live_view`, `record_cart`).

- **`bila/sim/shelf_scene.py`** is the core. `ShelfConfig` (randomization ranges) → `sample_row(cfg, scenario, rng)` → `Row` (levels, books, target slot, `nudges`) → `build_scene(cfg, row, panda_xml)` emits MJCF XML → compile → `settle()` applies the nudges and runs physics so leaning/collapsed poses are real resting poses. `make_episode` wraps this and resamples rows that don't settle. Scenarios: `SCENARIOS = ("clear", "tight", "leaning", "collapsed", "packed")`.
- **`bila/control/insert_book.py`** (largest file) has its own `make_episode`/`build_model`, `ArmIK`, `Runner`, and an `Episode` class holding the staged scripted insertion and gap-widening strategies (`--strategy`, `--packed shove|stagger`). Its module docstring describes the stages in detail.
- **`bila/vision/perception.py`**: `ShelfEstimate` = only what a wrist RGB-D camera could see (spine faces, gap profile). `GroundTruthPerception` / `NoisyPerception`. Decision code should consume a `ShelfEstimate`, not `MjData`.
- `PBBM-robot-librarian/` is an external (ROS) reference repo for book assets, not part of this codebase. Its `librarian_common/librarian_resources/books.csv` (`depth` there = our `thickness`) and `spines/id_N.png` are used as spine labels in `notebooks/01_mujoco_librarian_pomdp.ipynb`, which also holds the prototype POMDP env (`LibrarianEnv`); `notebooks/02_how_pbbm_shelves_books.ipynb` explains PBBM's pipeline.

## Invariants worth knowing

- Shelf frame: x along the row, y out of the shelf (back panel at −y), z up, origin on the floor at the bookcase centre; board front edge at y = depth/2.
- `qpos` layout: 9 robot values (7 arm + 2 fingers), then 7 per target-row book; with `book_pages=N`, each book adds `N−1` hinge angles right after its 7. Filler-level books are static (no joint) unless `static_other_levels=False`.
- Controllers reconstruct books from `<book>_geom` box `geom_size`. Covers and page sections must not change that: covers are massless non-colliding visuals; with pages the box stays but becomes non-colliding and page sections carry collision/mass. Use `shelf_scene.book_children` when attributing contacts to a book.
- Score "toppled" as tilt *change* from the settled pose, not absolute tilt (some settled books lean >45°).
- Measure gaps from the books (as `bila/vision/perception.py` does); `row.gap_right - row.gap_left` can be up to 3 mm short.
- `scenes/` is git-ignored (XML contains absolute paths). `bila/sim/shelf_scene.py` writes `panda_shelf.xml` into `mujoco_menagerie/franka_emika_panda/` so mesh paths resolve; the submodule ignores untracked files.
- Robot base height tracks the target board (board top − 0.40 m); don't lower `target_band` below 0.40 m without raising the base.
