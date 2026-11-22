# The simulator

`bila/sim/shelf_scene.py` builds the scene: a randomized bookcase standing on the floor, one target row in one of
five scenarios, and a Franka Panda (from the `mujoco_menagerie` submodule) with thin fingers, a wrist camera and a
force/torque sensor at the flange. The shelf started as the one in ContactExplorer's singulation task
(`src/tasks/allegro_singulation.py`, Isaac Gym Preview 4).

## The bookcase

```
  top board
  ┌──────────────────┐
  │ level 4  (filler)│   ← books welded in place (static) by default
  ├──────────────────┤
  │ level 3  (TARGET)│   ← free books, scenario row, robot works here
  ├──────────────────┤
  │ level 2  (filler)│
  ├──────────────────┤
  │ ...              │
  ├──────────────────┤   ← board_0
  └──plinth──────────┘   floor
```

- **3 to 5 levels**, each 0.25 to 0.40 m tall, on a 5 to 10 cm plinth. Side panels and the back panel run from the
  floor to the top board.
- **One target level** per episode, from the levels whose board top is 0.40 to 1.10 m off the floor (within the
  Panda's reach). It holds the scenario row.
- **Books line up at the front**, as on a library shelf: each spine sits 10 to 30 mm behind the shelf's front
  edge, and every book is at least 30 mm shallower than the shelf, so no back goes past the back panel.
- **Every other level** holds an ordinary upright row, 50 to 100% full, packed against a random side panel. These
  books are static by default (no joint), so they cost almost nothing to simulate.
  `ShelfConfig(static_other_levels=False)` makes them free.
- **The robot base follows the target level**, like a lift: its height is the target board's top less 0.40 m,
  0.80 m in front of the shelf. The arm then always sees the same geometry relative to the row it works on. (The
  Panda can only hold its hand level about 0.5 m or more above its base, because joint 6 bends one way only.)
- Frame: x along the row, y out of the shelf (back panel at −y), z up, origin on the floor at the bookcase centre.

## The target row

| Scenario | The slot | What the controller does |
| --- | --- | --- |
| clear | 1.2 to 2 times the book's thickness | inserts the book |
| tight | 0.6 to 1.1 times the book's thickness | widens the gap, then inserts |
| leaning | a neighbour leans into it, 12° to 35° | stands the neighbour up, then inserts |
| packed | none: the slot's neighbours touch | shoves the row into its free space, then inserts |
| collapsed | a run of 3 to 6 books has toppled over | nothing (not handled) |

Books are placed upright, then the leaning and collapsed scenarios push some of them over during a short settling
phase (the pushes are stored in `row.nudges`). So every lean is a real resting pose found by physics, not a hand-set
angle that might overlap other books. A leaning row's gap is sized from the lean wanted, so the book comes to rest
against its neighbour. A packed row's only room is its spare space: 3 to 15 cm at one end, plus 0 to 3 mm between
books.

## Generating and viewing scenes

```bash
uv run python -m bila.sim.shelf_scene --seed 0 --panda mujoco_menagerie/franka_emika_panda/panda.xml --out scenes
uv run python -m bila.sim.shelf_scene --scenario leaning --seed 3 --out scenes        # one scenario only
uv run python -m bila.viz.view_scene --scenario leaning --seed 0                     # add --no-robot for the bookcase only
uv run mjpython -m bila.viz.view_scene --scenario leaning --seed 0 --show-estimate    # what the robot knows, over the scene
```

`shelf_scene` writes one `.xml` (the MuJoCo scene) and one `.json` (levels, book sizes, target slot, gap edges) per
scenario, settles each scene and prints whether it is stable. `scenes/` is git-ignored because the XML holds
absolute paths. `view_scene` settles the row first, so leaning books show slanted, and starts the camera on the gap.
In the viewer, Rendering > Camera > `scene_cam` shows the whole target row.

## Real books, covers and pages

The report's experiments use all three of these options: `--catalogue --book-pages 4 --book-meshes`.

- **`--catalogue`** fills the rows with the PBBM robot librarian's real books (`shelf_scene.CATALOGUE`). Each copy
  keeps the book's size, mass and own cover, and the Voynich and Lagemann books are hinged into their real page
  sections. Positions, order, friction and scenario stay random, and the same book can appear more than once.
- **`--book-meshes`** draws the real covers. They are converted once from the PBBM submodule:

  ```bash
  uv run python -m bila.sim.dae_to_obj --src PBBM-robot-librarian/librarian_common/tams_books_world/meshes
  ```

  This writes `generated_assets/books/` (git-ignored). MuJoCo cannot load Collada, so each cover cube becomes six
  one-face OBJs (one material per mesh). The covers are massless, non-colliding geoms on the book's box, so physics
  is unchanged, and the box moves to render group 3 so it doesn't show through. Missing assets quietly fall back to
  plain boxes.
- **`--book-pages N`** cuts every book into `N` page sections hinged on the spine, after the PBBM repo's articulated
  books. The back section is welded to the book and the `N - 1` in front of it each hang on a revolute joint about
  the spine's vertical line. Each section carries its share of the book's mass. The book's `<book>_geom` box stays,
  with the same size but non-colliding, so code that reconstructs a book from its box is unaffected. Count a page
  section as part of its book with `shelf_scene.book_children`. With `--catalogue`, each section is drawn as a slice
  of the book's own cover (written once to `generated_assets/books/slices/`).

## Using it from Python

```python
import numpy as np
from bila.sim.shelf_scene import ShelfConfig, make_episode

model, data, row, report = make_episode(ShelfConfig(), "leaning", np.random.default_rng(0),
                                        panda_xml="mujoco_menagerie/franka_emika_panda/panda.xml")
start_qpos = data.qpos.copy()      # the settled state is the episode's start
print(row.target_level, row.target_board_top)
```

- `make_episode` throws away the occasional row that keeps sliding and samples again. Settling takes a while, so
  for RL store settled states and reset from them.
- `qpos` layout: 9 robot values (7 arm, 2 fingers), then 7 per target-row book (position, quaternion), then 7 per
  filler book only if `static_other_levels=False`. With `book_pages=N`, each book adds `N - 1` hinge angles right
  after its own 7.
- To change the scene before compiling, build the XML with `sample_row` and `build_scene`, edit it (`xml.etree` or
  `mujoco.MjSpec`), compile it, then call `settle`. `bila/control/insert_book.py` does this in its own
  `make_episode` and `build_model`, which add the book to insert and gravity compensation.

## Settings (`ShelfConfig`)

| Setting | Default | Meaning |
| --- | --- | --- |
| `width`, `depth` | 0.60 to 1.00 m, 0.22 to 0.32 m | Inner size of the bookcase |
| `num_levels` | 3 to 5 | Shelf boards above the plinth |
| `level_height` | 0.25 to 0.40 m | Clear height of each level |
| `plinth_height` | 0.05 to 0.10 m | Bottom kick board |
| `target_band` | 0.40 to 1.10 m | Allowed height of the target board's top |
| `fill_other_levels` | 50 to 100% | How full the other rows are |
| `static_other_levels` | `True` | Weld the other rows' books in place, for speed |
| `thickness`, `book_depth`, `book_height` | 15 to 60 mm, 12 to 22 cm, 17 to 30 cm | Sampled book sizes (unused with `catalogue`) |
| `book_friction` | 0.25 to 0.7 | Books' friction |
| `spine_inset` | 10 to 30 mm | How far each spine sits behind the shelf's front edge |
| `depth_clearance` | 30 mm | Every book is at least this much shallower than the shelf |
| `end_slack` | 3 to 15 cm | Free space left at one end of the row |
| `clear_gap_factor`, `tight_gap_factor` | 1.2 to 2.0, 0.6 to 1.1 | Gap over the book's thickness |
| `lean_angle_deg` | 12° to 35° | Lean of leaning rows |
| `catalogue`, `book_meshes`, `book_pages` | off | The options above |
| `robot_xy`, `robot_z_offset` | (0, 0.80), −0.40 m | Robot base position (its height follows the target board) |

## Things to know

- **Robot start pose.** The Panda starts in a tucked "ready" pose, its hand about 25 cm in front of the shelf.
- **Robot height.** On a high target level the robot stands on a tall pedestal (up to about 0.7 m), modelling a lift
  or mobile base. Don't lower `target_band` below 0.40 m without raising the base, or it ends up below the floor.
- **Topple threshold.** Some settled books lean more than 45°, so score toppling as each book's tilt *change* from
  its settled pose, not its absolute tilt.
- **Gap width.** `row.gap_right - row.gap_left` can be up to 3 mm short of the real face-to-face gap. Measure gaps
  from the books, as `bila/vision/perception.py` does.
- **Generated robot file.** `shelf_scene` writes `panda_shelf.xml` (Menagerie's Panda with the thin fingers, wrist
  camera and F/T sensor) into `mujoco_menagerie/franka_emika_panda/` so the robot's mesh paths resolve. The
  submodule ignores untracked files, so it doesn't show as a change.
- **Two leaning neighbours** (one each side of the gap) are not generated: pushing both at once made one fall flat.
- **Shelf width.** `sample_row` raises an error when fewer than 4 books fit.
