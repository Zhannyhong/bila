# MuJoCo bookshelf, ported from ContactExplorer

`shelf_scene.py` builds the bookshelf scene for the gap-creation project in MuJoCo.
It starts from the shelf in ContactExplorer's singulation task
(`src/tasks/allegro_singulation.py`, Isaac Gym Preview 4) and turns it into a full,
randomized bookcase standing on the floor. It also adds a Franka Panda with a wrist
camera and force/torque sensor.

![scenarios](scenarios_preview.png)

## The bookcase

The bookcase is built from the floor up:

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

- **3 to 5 levels**, each 0.25 to 0.40 m tall, on a 5 to 10 cm plinth. Side panels and the back panel run from the floor to the top board.
- **One target level** is chosen per episode, from the levels whose board top is 0.40 to 1.10 m off the floor (within the Panda's reach).
  It holds the scenario row: clear, tight, leaning or collapsed.
- **Every other level** holds an ordinary upright row, 50 to 100% full, packed against a random side panel.
  These books are **static** by default (no joint), so they add almost no simulation cost. Set
  `ShelfConfig(static_other_levels=False)` to make them free, e.g. to check in evaluation that nothing gets knocked off.
- **The robot base follows the target level**, like a lift: base height = target board top − 0.21 m.
  The arm then always sees the same geometry relative to the row it works on.

## What was taken from the repo, and what changed

| Isaac Gym repo | MuJoCo version | Why it changed |
| --- | --- | --- |
| `_define_table`, `_define_upper_shelf`: one board 0.5 m up, one above | `board_0` … `board_N`, stacked from a plinth on the floor, plus `top_board` | A real bookcase, with books above and below the target row |
| `_define_back_wall`, `_define_side_wall`: walls for one cubby | `back_panel` and side panels from floor to top | Same parts, full height |
| Shelf 0.344 × 0.26 m, fixed | 0.6 to 1.0 m wide, 0.22 to 0.32 m deep, randomized levels and pose | Room for 12 to 20 books per row and space to push them |
| 5 boxes, 4 × 16 × 24 cm, fixed size | Every book a different, random size, mass and friction | Real shelves are mixed |
| Neighbour boxes have `fix_base_link=True` and friction `1e-5` | Target-row books are **free bodies** with normal friction (0.25 to 0.7) | Your task is to push neighbours, so they must be movable |
| PhysX `friction`, `restitution` | MuJoCo `friction`, `solref`, `solimp`, `condim=4` | Different contact model |
| Two fixed cameras, 640×480, 58° | `scene_cam` (fixed, off to the side, aimed at the target row) + `wrist_cam` on the Panda hand | Robot blocked the front view; wrist camera sees the gap up close |
| xArm6 + Allegro hand | Franka Panda from MuJoCo Menagerie + F/T sensor at the flange | Your chosen robot |

`python shelf_scene.py --repo-defaults` rebuilds the original single cubby (board top at 0.51 m) and box size, for comparison.

## Implementation steps

1. **Install**
   ```bash
   git clone --recurse-submodules <this repo>     # or, in an existing clone:
   git submodule update --init                     # fetches mujoco_menagerie
   uv sync                                         # installs mujoco and numpy from pyproject.toml
   ```

2. **Generate scenes**
   ```bash
   uv run python shelf_scene.py --seed 0 --panda mujoco_menagerie/franka_emika_panda/panda.xml --out scenes
   ```
   This writes one `.xml` (the MuJoCo scene) and one `.json` (levels, book sizes, target slot, gap edges) per scenario,
   then settles each scene and prints whether it is stable. `scenes/` is git-ignored: the XML holds absolute paths.

3. **Look at a scene**
   ```bash
   uv run python view_scene.py --scenario leaning --seed 0       # add --no-robot for the bookcase only
   ```
   This settles the row first, so leaning and collapsed books show slanted, then opens the viewer.
   (Opening an XML directly with `python -m mujoco.viewer --mjcf ...` shows the start state, with every book upright.)

4. **How the target row is made (the key idea)**
   Books are placed upright, then the leaning and collapsed scenarios push some of them over during a short
   settling phase. This way every lean is a real resting pose found by physics, not a hand-set angle that
   might overlap other books.
   - Leaning: the gap is sized from the lean angle you want (12° to 35°), so the book comes to rest against its neighbour.
   - Collapsed: a run of 3 to 6 books topples into a hole next to it. The hole is sized the same way, so the run
     leans far enough to stay down and lands on the book across the hole below its top.
   - In both, the spare row space goes to the end away from the toppling. The books that catch the lean are
     packed against a side panel and can't slide away.

5. **Use it in your training code**
   ```python
   import numpy as np
   from shelf_scene import ShelfConfig, make_episode
   cfg = ShelfConfig()
   rng = np.random.default_rng(0)
   model, data, row, report = make_episode(cfg, "collapsed", rng,
                                           panda_xml="mujoco_menagerie/franka_emika_panda/panda.xml")
   start_qpos = data.qpos.copy()      # settled state = the episode's start
   print(row.target_level, row.target_board_top)
   ```
   `make_episode` throws away the occasional row that keeps sliding and samples again.
   For RL, generate a few thousand settled `qpos` vectors once and reset from that list. Settling inside every
   reset is too slow.

   `qpos` layout: 9 robot values (7 arm + 2 fingers), then 7 per target-row book (position + quaternion),
   then 7 per filler book only if `static_other_levels=False`.

6. **Next pieces to build on top** (from the design doc)
   - Swap the Panda's position actuators for torque (`motor`) actuators and write the impedance controller.
   - Spawn the target book in the gripper (it is sampled in `row.target_book` but not placed yet).
   - Add the gap-creation moves and insertion on top of that controller.
   - Replace the black fiducial plate with an AprilTag texture, and add spine textures for perception.
   - For GPU training, load the same XML in MJX or MuJoCo Warp and check it still settles the same way.

## Settings (`ShelfConfig`)

| Setting | Default | Meaning |
| --- | --- | --- |
| `num_levels` | 3 to 5 | Shelf boards above the plinth |
| `level_height` | 0.25 to 0.40 m | Clear height of each level |
| `plinth_height` | 0.05 to 0.10 m | Bottom kick board |
| `target_band` | 0.40 to 1.10 m | Allowed height of the target board's top surface |
| `fill_other_levels` | 50 to 100% | How full the non-target rows are |
| `static_other_levels` | `True` | Weld non-target books in place, for speed |
| `robot_xy`, `robot_z_offset` | (0, 0.62), −0.21 m | Robot base position; its height follows the target board |
| `lean_angle_deg` | 12° to 35° | Lean for leaning and collapsed rows (raised when a book needs more to stay down) |

## Tested (MuJoCo 3.15.0)

| Scenario | Stable on first try | Notes |
| --- | --- | --- |
| clear | 40/40 | Nothing moves (< 0.2 mm) |
| tight | 40/40 | Nothing moves (< 0.2 mm) |
| leaning | 134/140 | Leans of 14° to 67°; failures are re-sampled by `make_episode` |
| collapsed | 136/140 | Runs lean 17° to 52°; failures re-sampled |
| all four, free filler books | 60/60 | Nothing on the other levels moves (< 0.5 mm) |

- Bookcases came out 0.96 to 1.99 m tall, with the target on levels 1 to 3 and its board top between 0.40 and 1.10 m.
- The robot touched nothing at the start in 160/160 scenes.

The physics tests ran against MuJoCo's C library with the same steps as `settle()`, because the Python package
could not be installed in the build environment. Run step 2 once on your machine to confirm the Python path.

## Things to know

- **Robot start pose.** Menagerie's own "home" pose reaches 0.55 m forward and hits the shelf, which knocks
  books over. The scene uses a tucked "ready" pose instead (hand about 10 cm in front of the shelf).
- **Robot height.** On a high target level the robot stands on a tall pedestal (up to about 0.9 m). That models a
  lift or mobile base. To use a fixed base instead, narrow `target_band` and set `robot_z_offset` accordingly.
- **Topple threshold.** Some settled books lean more than 45°. When scoring "no neighbour toppled", measure each
  book's tilt *change* from its settled pose, not its absolute tilt.
- **Two leaning neighbours** (one on each side of the gap) are not generated: pushing both at once made one fall
  flat. Use the collapsed scenario for several leaning books.
- **Generated robot file.** `shelf_scene.py` writes `panda_shelf.xml` into `mujoco_menagerie/franka_emika_panda/`
  so the robot's mesh paths still work. The submodule is set to ignore untracked files, so this doesn't show as a change.
- **Shelf width.** If you set the shelf very narrow, `sample_row` raises an error when fewer than 4 books fit.
