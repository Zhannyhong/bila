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
- **Books line up at the front**, as on a library shelf: each spine sits 10 to 30 mm behind the shelf's front edge,
  and every book is at least 30 mm shallower than the shelf, so no back goes past the back panel (the deepest
  book, set furthest in, just touches it).
- **Every other level** holds an ordinary upright row, 50 to 100% full, packed against a random side panel.
  These books are **static** by default (no joint), so they add almost no simulation cost. Set
  `ShelfConfig(static_other_levels=False)` to make them free, e.g. to check in evaluation that nothing gets knocked off.
- **The robot base follows the target level**, like a lift: base height = target board top − 0.40 m,
  0.80 m in front of the shelf. The arm then always sees the same geometry relative to the row it works on,
  and can hold its hand level to slide a book in. (The Panda can only point its hand level with the hand
  about 0.5 m or more above its base, because joint 6 bends one way only.)

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
   Either way the camera starts on the gap, from the front and to one side, so the robot isn't in the way.
   The scene sets this as MuJoCo's default free camera (`<statistic>` and `<visual><global>`), so you can still
   orbit, and the viewer's Align button brings you back to it. MuJoCo's viewer only applies it by itself when it
   opens a file, so `view_scene.py` and `insert_book.py --view` apply it themselves.

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

6. **Scripted insertion (clear and tight scenarios)**
   ```bash
   uv run python insert_book.py --scenario clear --seed 0                # one episode: prints each step and the score
   uv run python insert_book.py --scenario tight --seed 0 --trials 50    # success rate over seeds 0 to 49
   uv run mjpython insert_book.py --scenario tight --seed 0 --view       # watch it (the macOS viewer needs mjpython)
   ```
   A hand-written baseline. The target book is held in the gripper at a pre-insertion pose in front of the gap.
   The robot rests the book's back on the board's edge and slides it in while holding it. It lets go before its
   fingers reach the neighbours' spines (each finger is wider than the clearance beside the book), turns its hand
   90°, and pushes the spine low down with the lower fingertip until the book is in place (no push if the slide
   already got it there). A book going in tightly drags its neighbours back, so it measures where the book has to
   go again after the slide, and pushes again if the push dragged them further. It scores the result as the proposal does: upright within 5°, in place, released, plus
   how far the other books moved. In place means:
   - spine within 10 mm of the spine of the neighbour that sticks out more, or
   - if the book is deeper than that neighbour, so it can't line up without its back passing the back panel:
     at least 10 mm behind the shelf's front edge. With the default rows that's rare (about 1 row in 400: a book
     near the maximum depth beside neighbours set more than 25 mm in, as the inserted book stops 5 mm short of
     the back panel), but it also covers other layouts such as `--repo-defaults`.
   - Control: Menagerie's position actuators, fed by differential IK along min-jerk paths, with gravity compensation.
     The arm configuration is chosen so the whole sequence stays clear of the joint limits and doesn't run into
     the shelf. The slide and the final push stop when the wrist force sensor feels the book stop giving way.
   - The script adds the target book to the scene, and gives the gripper the real Franka Hand's 70 N (Menagerie's
     squeezes a book with 1 to 2 N). It only changes its own copy of the model.
   - The fingers pinch the book 11.5 cm above its bottom. Higher up, board friction tips the book over the pads
     while it slides.
   - Clear, seeds 0 to 99: all 100 succeed, in place after 10 s on average. No episode moved another book more
     than 0.2 mm.

   **Tight**: the gap is first widened to the book's thickness plus 6 mm.
   - Fingertip: the empty hand puts its fingers (stacked vertically, 21 mm wide) into the gap and pushes a
     neighbour aside. By default (`--strategy one-sided`) it pushes the side with more free space, as far as the
     widening needs or that side's space allows. If the gap at the spines is then still short of the book plus
     6 mm, it pushes the other side too, by what is missing, provided the fingertips reach that neighbour and the
     arm can carry on from where it is. `split` pushes each neighbour half as far. Then the book is handed over
     (not modelled) and inserted as above. Used when the fingers fit the gap and reach at least
     1 cm past a neighbour's spine (the 63 mm palm stops 46 mm behind the fingertips, at the most protruding spine
     nearby). There is no force limit: the arm pushes as hard as it can (joint 6 reaches its 12 N m at 55 to
     100 N, depending on the pose).
   - Skipped otherwise, or when the push leaves the gap at the spines narrower than the book: no insertion,
     counted as a failure. Nothing else on the stock hand can widen a gap narrower than the fingers:
     - Pushed sideways at its back corner, the held book turns in the grasp: 2° at 9 to 15 N, 3 to 19° at 20 N,
       and at 40 N up to 31°, much of it for good. Sliding the books takes more than that. A 140 N grip barely helps.
     - A book wedge (held turned, back corner first, then straightened) was tried and dropped. Its slanted face
       pushed the neighbour back into the shelf instead of aside (31 mm back and 2.5 mm aside on seed 21). It
       placed none of its 18 rows, and dragged neighbours up to 12 cm into the shelf.
     - A thin fingertip (10 mm or less) would fit every such row.
   - Seeds 0 to 49, default `one-sided`: 37 succeed (one needed no widening). The fingertip handles 39 rows and
     places 36. Every push left the gap at least as wide as the book. The other 10 rows, all with gaps narrower
     than the fingers (12 to 23 mm), are skipped, and nothing in them moves.
   - In 7 fingertip rows the first push left the gap 1.4 to 2.3 mm short, so the other side was pushed too. That
     placed seed 38, which used to jam at the entrance. The 5 of them that already worked still do, 2 to 4 s slower.
   - Dropping the old 35 N limit on the books' estimated friction gave the fingertip 8 rows that used to go to the
     wedge (3 of them now placed), and let 6 rows that had to split push one side instead (4 of them failed before
     and now succeed). Seed 27 now
     pushes the side the estimate put at 93 N, and succeeds. Nothing that worked before fails.
   - Measuring where the book has to go again after the slide, and pushing again when the push drags the
     neighbours back, placed 2 more (seeds 0 and 46, which had stopped 12 to 15 mm proud of the neighbour) and 4
     more with `split`. Rows that already worked take 0.1 s longer on average.
   - One side versus splitting, on the same 39 fingertip rows: one side places more, 36 vs 24 (25/50 overall
     with `split`). Splitting swivels both neighbours, so the book's back corners catch on both sides and it
     jams at the entrance more often. In 6 rows it leaves the gap narrower than the book, so they are skipped.
     Where both worked, the most-moved book moved about the same (6.1 vs 6.6 mm, one side
     vs split) and one side was 1.5 s faster (17.9 vs 19.4 s). Splitting needs much less force, as fewer books
     must slide at once.
   - Books have room behind them, so a book sliding in tightly can also drag its neighbours back. On seed 0 it
     drags the pushed neighbour 82 mm back, and still counts as placed: success only checks the inserted book.
   - Why it's hard: the fingertips push a neighbour only near its spine, so it swivels instead of sliding. The gap
     opens at the spines but barely at the back, and the book has to widen the rest as it slides in. That is the
     contact-rich part the proposal's residual policy is meant to learn.

7. **Next pieces to build on top** (from the design doc)
   - Swap the Panda's position actuators for torque (`motor`) actuators and write the impedance controller.
   - Spawn the target book in the gripper for other code too (`insert_book.py` does it for its own runs;
     `build_scene` still leaves `row.target_book` out).
   - Add the gap-creation moves and insertion on top of that controller (`insert_book.py` scripts them with
     position control, as a baseline).
   - Give the hand a thin fingertip (10 mm or less), so it can widen tight gaps narrower than the stock fingers
     (skipped for now).
   - Replace the black fiducial plate with an AprilTag texture, and add spine textures for perception.
   - For GPU training, load the same XML in MJX or MuJoCo Warp and check it still settles the same way.

## Perception (`perception.py`)

What the robot knows about the target row, limited to what a wrist RGB-D camera could report. Code that decides
what to do (the RL policy, and ideally the scripted sequences) should read a `ShelfEstimate` rather than `data`,
so a camera-based source can replace the ground truth later without changing it.

```python
from perception import GroundTruthPerception, NoisyPerception
est = GroundTruthPerception(model, row).estimate(data)       # or NoisyPerception(GroundTruthPerception(...), rng)
est.book(-1).lean, est.book(+1).spine_y                      # the slot's left and right neighbours
est.min_gap_width(0.25)                                      # narrowest free width up to 25 cm above the board
```

- **Each book is its spine face**: four corners in the shelf frame, and the values derived from them (thickness,
  height, lean, swivel, how far out the spine sits). `slot_offset` names it relative to the slot: -1 and +1 are
  the neighbours.
- **The gap** is the free width at the spine plane at 8 heights above the board (1 to 29 cm), between everything
  left and right of the slot, side panels included. A leaning neighbour shows as a gap that narrows with height.
- **Not measured**: a book's depth is a prior (17 cm, capped to fit the shelf), since the camera only sees spines.
  Mass, friction, contact forces and velocities are left out; they are for the reward and critic only.
- `NoisyPerception` makes the ground truth look like the wrist camera's estimate, without rendering (use it to
  train): each book's position, size, lean and swivel get the camera's measured error for its distance from the
  slot (`CAMERA_ERRORS`, from `eval_perception.py`), and books beyond the camera's view along the row
  (`VIEW_REACH`, 53 cm from the slot) are left out. `scale=1.5` trains on worse estimates than the camera's.
- Note: `row.gap_right - row.gap_left` is up to 3 mm short of the actual face-to-face gap, because `gap_left`
  is set after the left neighbour's spacing. Measure the gap from the books, as `perception.py` does.

**Seeing what the robot knows** (`visualize.py`): both pictures read only a `ShelfEstimate`, so they work the
same for every perception source.

```bash
uv run python visualize.py --scenario leaning --seed 0 --noisy                  # front + top view -> scenes/*.png
uv run mjpython view_scene.py --scenario leaning --seed 0 --show-estimate       # the estimate over the 3D scene
```

- Front view: spines (the slot's neighbours orange, labelled by place relative to the slot and lean), and the
  gap's width at each measured height: green where the book to insert fits, red where it doesn't, grey above
  its top. With `--noisy`, the ground truth is drawn dashed.
- Top view: each book's box as the robot models it. Its depth is the prior, so the backs are a guess.
- In the viewer: see-through boxes, spine corners as dots, and the gap bars, over the real books.
  `draw_in_viewer` also works on `mujoco.Renderer`'s scene, for off-screen pictures.

**Where the robot looks from** (`inspection.py`): before touching the row, the wrist camera looks straight in
from 15 cm above the target board, as far back as the arm allows: about 50 cm from the spines, else 45 or 40 cm.

```python
from inspection import solve_inspection, look_from
q, standoff = solve_inspection(model, data, row)   # arm angles and distance used, or None if none is reachable
look_from(model, data, q)                          # puts the arm there (the move itself is not modelled)
solve_inspection(model, data, row, dx=0.2)         # a look 20 cm further along the row
```

- Tested on 48 scenes (12 seeds of each scenario, target boards 0.43 to 1.10 m up): the camera got to 50 cm in
  43 and to 45 cm in the other 5, and could also move 30 cm either way along the row. The arm showed as a thin
  strip at the image edge (at most 0.5% of the image), never over the neighbours.
- On 32 of those scenes, the books whose whole spine was in view:

  | Camera from the spines | Both neighbours | Share of the row | Whole row | Fewest on one side (median) |
  | --- | --- | --- | --- | --- |
  | about 50 cm (used) | 100% | 93% | 41% of scenes | 5 |
  | about 30 cm | 100% | 69% | 3% of scenes | 4 |

  Further back costs resolution: about 1.5 mm per pixel at 640 px wide, against 0.9 mm from 30 cm.
- Finding the pose takes up to about 20 s (many starting guesses, and closer distances if 50 cm fails), so
  solve it once per scene and store it, e.g. with the scene's settled start state.
- The camera is rolled 180°, so **its image comes out upside down**: rotate it before use. Held the right way
  up, the arm reached the centre view from 30 cm in only half the scenes (its wrist bends one way only).
- Without gravity compensation the held arm sags about 5 mm in 2 s. Read the camera's pose from `data` when
  the image is taken, not from the target.
- Rendering off-screen in WSL: `MUJOCO_GL=egl`.

**The wrist camera** (`camera.py`): RGB-D images in the conventions a real camera pipeline (OpenCV,
RealSense) uses, so code written against them carries over to real images.

```python
from camera import WristCamera, points, project, robot_mask
with WristCamera(model) as cam:           # 1280 x 720, ~89 deg wide like a D405 (87)
    frame = cam.capture(data)             # rgb, depth (m), seg (geom ids), K, pose, stamp
xyz = points(frame, mask)                 # world points for the pixels in mask that have depth
u, v, z = project(frame, xyz)             # world points back to pixels
```

- Camera frame x right, y down, z forward; pixel row 0 at the top. Depth is along the camera's z, and 0 means
  no reading (beyond 1.5 m, or dropped by the noise model).
- Images come out upright: at the inspection pose the camera is rolled 180°, so `capture` turns the image and
  the camera frame round. The pose is read from the simulation when the image is taken.
- Markers (sites, such as the green dot at the target slot) are hidden: no real camera would see them.
- `WristCamera(model, noise=DepthNoise())` adds D405-like depth errors: random error growing with distance
  squared (about 1.5 mm at 50 cm) and no reading at object edges. The numbers are a starting guess.
- Checked: every book pixel turned into a 3D point lies on that book's box (99% within 2 mm). A capture takes
  about 0.25 s; points for a whole image about 0.15 s.
- `uv run python visualize.py --scenario leaning --seed 0 --camera` saves the wrist camera's image from the
  inspection pose with the estimate's spines outlined over it, and the depth image.

**What the robot knows from the camera** (`rgbd_perception.py`): a `ShelfEstimate` built from one wrist-camera
image, the same kind of estimate as the ground truth's, so anything that reads estimates can use it.

```python
from rgbd_perception import RGBDPerception
look_from(model, data, solve_inspection(model, data, row)[0])    # the arm at the inspection pose first
with RGBDPerception(model, row) as perception:
    est = perception.estimate(data)
```

- Per book: its pixels (here from the simulation's object ids, `OracleSegmenter`; a learned segmenter would
  replace it), the 3D points on the target level that face out of the shelf (the spine), a plane facing their
  median normal, and the smallest rectangle around them in it. Its corners are the spine corners.
- Normals come from depth averaged over 5 x 5 pixels, and the plane keeps points within 3 mm or 3 times their
  spread: both matter with depth noise.
- Books are named by where they stand relative to the target slot: the boundary between adjacent books closest
  to the slot's place on the shelf map (books can shift a little while settling).
- A side panel out of the camera's view is `None`, and if a slot neighbour isn't seen the gap on that side is
  unknown (NaN), rather than measured past books the camera didn't see.
- About 1.5 to 2 s per estimate (capture plus fitting).
- The camera renders without anti-aliasing: blended pixels at object edges decoded to wrong object ids.

**How well it works** (`eval_perception.py`, 48 scenes, 12 of each scenario, 722 books matched to the truth):

```bash
MUJOCO_GL=egl uv run python eval_perception.py collect   # ~20 min: stores the estimates in scenes/
uv run python eval_perception.py report                  # error tables
uv run python eval_perception.py calibrate               # CAMERA_ERRORS and VIEW_REACH for NoisyPerception
uv run python eval_perception.py validate                # NoisyPerception's errors next to the camera's
```

- All 722 books named as in the truth. Each side of the slot: at least 3 books seen, median 7, the whole side in 93%.
- With D405-like depth noise, the 95% error for the slot's neighbours (books further out in brackets, up to 6 out):
  corners 2.6 mm (5.4), thickness 2.5 mm (7.3; further books come out too thick), lean 0.3° (0.4°), swivel 0.7°
  (1.4°). The narrowest gap up to the book's height: within 1.6 mm in 95% of scenes.
- A gap's width at one height can be far off just above a book's top: a 2 mm height error decides whether that
  book bounds the gap there. The narrowest gap doesn't suffer from this; prefer it.
- `NoisyPerception` matches the camera's error, per quantity and distance from the slot, within a few tenths of a
  mm or degree. Its narrowest gap is noisier than the camera's (95% within 3.3 mm): it errs each book's centre
  and thickness separately, where the camera's errors at a spine's two edges partly cancel.
- `uv run python visualize.py --scenario leaning --seed 0 --camera [--noisy]` saves the camera image with the
  estimate outlined, and the estimate's front and top view with the ground truth dashed.

Tests: `uv run pytest` (with `MUJOCO_GL=egl` in WSL for the rendering tests).

## Settings (`ShelfConfig`)

| Setting | Default | Meaning |
| --- | --- | --- |
| `num_levels` | 3 to 5 | Shelf boards above the plinth |
| `level_height` | 0.25 to 0.40 m | Clear height of each level |
| `plinth_height` | 0.05 to 0.10 m | Bottom kick board |
| `target_band` | 0.40 to 1.10 m | Allowed height of the target board's top surface |
| `fill_other_levels` | 50 to 100% | How full the non-target rows are |
| `static_other_levels` | `True` | Weld non-target books in place, for speed |
| `spine_inset` | 10 to 30 mm | How far each spine sits behind the shelf's front edge |
| `depth_clearance` | 30 mm | Every book is at least this much shallower than the shelf |
| `robot_xy`, `robot_z_offset` | (0, 0.80), −0.40 m | Robot base position; its height follows the target board |
| `lean_angle_deg` | 12° to 35° | Lean for leaning and collapsed rows (raised when a book needs more to stay down) |

## Tested (MuJoCo 3.15.0)

| Scenario | Stable on first try | Notes |
| --- | --- | --- |
| clear | 40/40 | Nothing moves (< 0.2 mm) |
| tight | 40/40 | Nothing moves (< 0.2 mm) |
| leaning | 132/140 | Leans of 14° to 56°; failures are re-sampled by `make_episode` |
| collapsed | 136/140 | Runs lean 15° to 49°; failures re-sampled |
| all four, free filler books | 60/60 | Stable with the other levels' books free too |

- Bookcases came out 0.96 to 1.99 m tall, with the target on levels 1 to 3 and its board top between 0.40 and 1.10 m.
- The robot touched nothing at the start in 360/360 scenes.

Run with the Python package, with the books front-aligned (seeds 0 to 39, or 0 to 139 for leaning and collapsed,
and 0 to 14 per scenario with free filler books).

## Things to know

- **Robot start pose.** The scene starts the Panda in a tucked "ready" pose (hand about 25 cm in front of the shelf).
  Menagerie's own "home" pose reaches 0.55 m forward. It knocked books over when the base stood 0.62 m out,
  and clears the shelf at 0.80 m.
- **Robot height.** On a high target level the robot stands on a tall pedestal (up to about 0.7 m), and on the
  lowest one (board top 0.40 m) it stands on the floor. That models a lift or mobile base. To use a fixed base
  instead, narrow `target_band` and set `robot_z_offset` accordingly. Don't lower `target_band` below 0.40 m
  without raising the base, or it ends up below the floor.
- **Topple threshold.** Some settled books lean more than 45°. When scoring "no neighbour toppled", measure each
  book's tilt *change* from its settled pose, not its absolute tilt.
- **Two leaning neighbours** (one on each side of the gap) are not generated: pushing both at once made one fall
  flat. Use the collapsed scenario for several leaning books.
- **Generated robot file.** `shelf_scene.py` writes `panda_shelf.xml` into `mujoco_menagerie/franka_emika_panda/`
  so the robot's mesh paths still work. The submodule is set to ignore untracked files, so this doesn't show as a change.
- **Shelf width.** If you set the shelf very narrow, `sample_row` raises an error when fewer than 4 books fit.
