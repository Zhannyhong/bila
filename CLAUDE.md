# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

MuJoCo simulation for a library-robot "make room on the shelf" project: a randomized bookcase, a Franka Panda, and scripted baselines that insert a book into the target row. README.md is long and authoritative; read the relevant section before changing behaviour.

## Commands

```bash
git submodule update --init          # mujoco_menagerie (Panda model) is a submodule
uv sync                              # deps from pyproject.toml / uv.lock

uv run pytest                        # all tests (pythonpath = repo root, testpaths = tests/)
uv run pytest tests/test_scenes.py::test_name -x

uv run python shelf_scene.py --seed 0 --panda mujoco_menagerie/franka_emika_panda/panda.xml --out scenes
uv run python view_scene.py --scenario leaning --seed 0
uv run python insert_book.py --scenario tight --seed 0 [--trials 50]   # clear|tight|leaning|packed
uv run python scripted_controller.py --seed 0 | --seeds 50             # leaning baseline (pusher rod)
```

- On macOS anything with `--view` must run under `mjpython` (`uv run mjpython insert_book.py ... --view`).
- `--book-meshes` needs `generated_assets/books/`, produced once by `dae_to_obj.py --src <PBBM tams_books_world/meshes>` (see README "Book covers"). Missing assets silently fall back to plain boxes.
- No linter/formatter is configured.

## Architecture

All modules are flat at the repo root; tests import them directly.

- **`shelf_scene.py`** is the core. `ShelfConfig` (randomization ranges) → `sample_row(cfg, scenario, rng)` → `Row` (levels, books, target slot, `nudges`) → `build_scene(cfg, row, panda_xml)` emits MJCF XML → compile → `settle()` applies the nudges and runs physics so leaning/collapsed poses are real resting poses. `make_episode` wraps this and resamples rows that don't settle. Scenarios: `SCENARIOS = ("clear", "tight", "leaning", "collapsed", "packed")`.
- **`insert_book.py`** (largest file) has its own `make_episode`/`build_model`, `ArmIK`, `Runner`, and an `Episode` class holding the staged scripted insertion and gap-widening strategies (`--strategy`, `--packed shove|stagger`). Its module docstring describes the stages in detail.
- **`scripted_controller.py`**: leaning baseline. Modifies the scene via `mujoco.MjSpec.from_string(...)` to add a pusher rod before compiling, then uses damped-least-squares IK per tick. `ScriptedLeaningController.step()` is designed as the base for a residual RL policy.
- **`perception.py`**: `ShelfEstimate` = only what a wrist RGB-D camera could see (spine faces, gap profile). `GroundTruthPerception` / `NoisyPerception`. Decision code should consume a `ShelfEstimate`, not `MjData`.
- `docs/code-translations/` holds pseudocode Markdown companions for each Python file; update the matching `.md` when changing a file's logic.
- `PBBM-robot-librarian/` is an external (ROS) reference repo for book assets, not part of this codebase. Its `librarian_common/librarian_resources/books.csv` (`depth` there = our `thickness`) and `spines/id_N.png` are used as spine labels in `notebooks/01_mujoco_librarian_pomdp.ipynb`, which also holds the prototype POMDP env (`LibrarianEnv`); `notebooks/02_how_pbbm_shelves_books.ipynb` explains PBBM's pipeline.

## Invariants worth knowing

- Shelf frame: x along the row, y out of the shelf (back panel at −y), z up, origin on the floor at the bookcase centre; board front edge at y = depth/2.
- `qpos` layout: 9 robot values (7 arm + 2 fingers), then 7 per target-row book; with `book_pages=N`, each book adds `N−1` hinge angles right after its 7. Filler-level books are static (no joint) unless `static_other_levels=False`.
- Controllers reconstruct books from `<book>_geom` box `geom_size`. Covers and page sections must not change that: covers are massless non-colliding visuals; with pages the box stays but becomes non-colliding and page sections carry collision/mass. Use `shelf_scene.book_children` when attributing contacts to a book.
- Score "toppled" as tilt *change* from the settled pose, not absolute tilt (some settled books lean >45°).
- Measure gaps from the books (as `perception.py` does); `row.gap_right - row.gap_left` can be up to 3 mm short.
- `scenes/` is git-ignored (XML contains absolute paths). `shelf_scene.py` writes `panda_shelf.xml` into `mujoco_menagerie/franka_emika_panda/` so mesh paths resolve; the submodule ignores untracked files.
- Robot base height tracks the target board (board top − 0.40 m); don't lower `target_band` below 0.40 m without raising the base.
