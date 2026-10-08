# MuJoCo bookshelf, ported from ContactExplorer

`shelf_scene.py` builds the bookshelf scene for the gap-creation project in MuJoCo.
It reuses the shelf from ContactExplorer's singulation task
(`src/tasks/allegro_singulation.py`, Isaac Gym Preview 4), makes it bigger and
randomized, and adds a Franka Panda with a wrist camera and force/torque sensor.

![scenarios](scenarios_preview.png)

## What was taken from the repo, and what changed

| Isaac Gym repo | MuJoCo version | Why it changed |
| --- | --- | --- |
| `_define_table`, `_define_upper_shelf`, `_define_back_wall`, `_define_side_wall`: separate box actors | One static `shelf` body with 5 box geoms, same layout and frame (x along row, y depth, back wall at −y) | Simpler; static geoms are cheap in MuJoCo |
| Shelf 0.344 × 0.26 m, fixed | 0.6 to 1.0 m wide, 0.22 to 0.32 m deep, randomized height and pose | Room for 12 to 20 books and space to push them |
| 5 boxes, 4 × 16 × 24 cm, fixed size | Every book a different, random size, mass and friction | Real shelves are mixed |
| Neighbour boxes have `fix_base_link=True` and friction `1e-5` | Neighbours are **free bodies** with normal friction (0.25 to 0.7) | Your task is to push neighbours, so they must be movable |
| PhysX `friction`, `restitution` | MuJoCo `friction`, `solref`, `solimp`, `condim=4` | Different contact model (see step 4) |
| Two fixed cameras, 640×480, 58° | `scene_cam` (fixed, off to the side) + `wrist_cam` on the Panda hand | Robot blocked the front view; wrist camera sees the gap up close |
| xArm6 + Allegro hand | Franka Panda from MuJoCo Menagerie + F/T sensor at the flange | Your chosen robot |

`python shelf_scene.py --repo-defaults` rebuilds the original cubby and box size, for comparison.

## Implementation steps

1. **Install**
   ```bash
   pip install mujoco numpy
   git clone https://github.com/google-deepmind/mujoco_menagerie
   ```

2. **Generate scenes**
   ```bash
   python shelf_scene.py --seed 0 --panda mujoco_menagerie/franka_emika_panda/panda.xml --out scenes
   ```
   This writes one `.xml` (the MuJoCo scene) and one `.json` (book sizes, target slot, gap edges) per scenario,
   then settles each scene and prints whether it is stable.

3. **Look at a scene**
   ```bash
   python -m mujoco.viewer --mjcf scenes/leaning_seed0.xml
   ```
   Note: the viewer shows the start state, before settling. Leaning and collapsed books only fall into place once `settle()` runs (step 5).

4. **How the row is made (the key idea)**
   Books are placed upright, then the leaning and collapsed scenarios push some of them over during a short
   settling phase. This way every lean is a real resting pose found by physics, not a hand-set angle that
   might overlap other books.
   - Leaning: the gap is sized from the lean angle you want (12° to 35°), so the book comes to rest against its neighbour.
   - Collapsed: a hole is left next to a run of 3 to 6 books, and the whole run topples into it.

5. **Use it in your training code**
   ```python
   import numpy as np
   from shelf_scene import ShelfConfig, make_episode
   cfg = ShelfConfig()
   rng = np.random.default_rng(0)
   model, data, row, report = make_episode(cfg, "collapsed", rng,
                                           panda_xml="mujoco_menagerie/franka_emika_panda/panda.xml")
   start_qpos = data.qpos.copy()      # settled state = the episode's start
   ```
   `make_episode` throws away the rare row that keeps sliding and samples again.
   For RL, generate a few thousand settled `qpos` vectors once and reset from that list. Settling inside every
   reset is too slow.

6. **Next pieces to build on top** (from the design doc)
   - Swap the Panda's position actuators for torque (`motor`) actuators and write the impedance controller.
   - Add the gap-creation moves and insertion on top of that controller.
   - Replace the black fiducial plate with an AprilTag texture, and add spine textures for perception.
   - For GPU training, load the same XML in MJX or MuJoCo Warp and check it still settles the same way.

## Tested (MuJoCo 3.15.0, 160 generated scenes)

| Scenario | Stable on first try | Notes |
| --- | --- | --- |
| clear | 40/40 | Nothing moves (< 0.3 mm) |
| tight | 40/40 | Nothing moves (< 0.2 mm) |
| leaning | 35/40 | Leans of 12° to 67°; failures are re-sampled by `make_episode` |
| collapsed | 36/40 | Runs lean 12° to 62°; failures re-sampled |

The robot touched nothing at the start in 160/160 scenes.

The physics tests ran against MuJoCo's C library with the same steps as `settle()`, because the Python package
could not be installed in the build environment. Run step 2 once on your machine to confirm the Python path.

## Things to know

- **Robot start pose.** Menagerie's own "home" pose reaches 0.55 m forward and hits the shelf's top board,
  which knocks books over. The scene uses a tucked "ready" pose instead (hand about 10 cm in front of the shelf).
- **Topple threshold.** Some settled books lean more than 45°. When scoring "no neighbour toppled", measure each
  book's tilt *change* from its settled pose, not its absolute tilt.
- **Two leaning neighbours** (one on each side of the gap) are not generated: pushing both at once made one fall
  flat. Use the collapsed scenario for several leaning books.
- **Shelf width.** If you set the shelf very narrow, `sample_row` raises an error when fewer than 4 books fit.
