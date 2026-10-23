# MuJoCo bookshelf, ported from ContactExplorer

`shelf_scene.py` builds the bookshelf scene for the gap-creation project in MuJoCo.
It starts from the shelf in ContactExplorer's singulation task
(`src/tasks/allegro_singulation.py`, Isaac Gym Preview 4) and turns it into a full,
randomized bookcase standing on the floor. It also adds a Franka Panda with thin
10 mm fingers, a wrist camera and a force/torque sensor.

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
  It holds the scenario row: clear, tight, leaning, collapsed or packed.
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
| xArm6 + Allegro hand | Franka Panda from MuJoCo Menagerie, with thin 10 mm fingers, + F/T sensor at the flange | Your chosen robot; Menagerie's 21 mm fingers don't fit the narrower tight gaps |

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
   - Packed: no gap at all. The slot's neighbours stand upright, face to face (no spacing between them either), so
     the only room is the row's spare space: 3 to 15 cm at one end, plus 0 to 3 mm between books. In 97% of rows
     that is enough on one side for the book plus 6 mm (all of them, using both sides).

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

6. **Scripted insertion (clear, tight and packed scenarios)**
   ```bash
   uv run python insert_book.py --scenario clear --seed 0                # one episode: prints each step and the score
   uv run python insert_book.py --scenario tight --seed 0 --trials 50    # success rate over seeds 0 to 49
   uv run mjpython insert_book.py --scenario tight --seed 0 --view       # watch it (the macOS viewer needs mjpython)
   uv run mjpython insert_book.py --scenario packed --seed 7 --view      # a packed row it manages
   uv run python insert_book.py --scenario packed --seed 0 --trials 50 --packed stagger   # the older packed baseline
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
   - The fingers are thin bars in place of Menagerie's (`shelf_scene.py`, `make_panda_with_sensors`): 10 mm wide
     across their travel, 12 mm thick, as long as Menagerie's, with a pad strip along all 46 mm of each finger
     that reaches past the palm. The real Franka Hand takes custom fingers, so these could be machined.
   - The book is held right up against the palm (its spine 2 mm off it), so the whole pad strip grips it. Held
     3 cm off the board for 3 s, books then turn 3.1° in the hand on average (5.1° for the 8 that weigh over
     1.2 kg), against 4.7° (7.8°) with the spine 8 mm out and a 35 mm pad. Insertion results are the same. A short
     pad is worse: with a 19 mm one, the board's drag turned a book 6° in the grasp on one clear row (seed 9) and
     it fell over.
   - Clear, seeds 0 to 99: all 100 succeed, in place after 10 s on average. No episode moved another book more
     than 0.25 mm.

   **Tight**: the gap is first widened to the book's thickness plus 6 mm.
   - Fingertip: the empty hand puts its fingers (stacked vertically, 10 mm wide) into the gap and pushes a
     neighbour aside. By default (`--strategy one-sided`) it pushes the side with more free space, as far as the
     widening needs or that side's space allows. If the gap at the spines is then still short of the book plus
     6 mm, it pushes the other side too, by what is missing, provided the fingertips reach that neighbour and the
     arm can carry on from where it is. `split` pushes each neighbour half as far. Then the book is handed over
     (not modelled) and inserted as above. Used when the fingers fit the gap and reach at least
     1 cm past a neighbour's spine (the 63 mm palm stops 46 mm behind the fingertips, at the most protruding spine
     nearby). There is no force limit: the arm pushes as hard as it can (joint 6 reaches its 12 N m at 55 to
     100 N, depending on the pose).
   - Skipped otherwise, or when the push leaves the gap at the spines narrower than the book: no insertion,
     counted as a failure. The fingers need 12 mm (1 mm clear each side) and the generator's tight gaps are at
     least 12.2 mm, so none are skipped for that. With Menagerie's 21 mm fingers, 10 of the 50 rows below were
     (gaps of 12 to 23 mm), and these other ways to widen them were tried and dropped:
     - Pushed sideways at its back corner, the held book turns in the grasp: 2° at 9 to 15 N, 3 to 19° at 20 N,
       and at 40 N up to 31°, much of it for good. Sliding the books takes more than that. A 140 N grip barely helps.
     - A book wedge (held turned, back corner first, then straightened) was tried and dropped. Its slanted face
       pushed the neighbour back into the shelf instead of aside (31 mm back and 2.5 mm aside on seed 21). It
       placed none of its 18 rows, and dragged neighbours up to 12 cm into the shelf.
   - Seeds 0 to 49, default `one-sided`: 48 succeed (one needed no widening), in place after 19 s on average. The
     fingertip handles the other 49 rows and places 47. Every push left the gap at least as wide as the book. The
     2 that fail (seeds 16 and 31) end up 4 to 5 cm proud of the neighbour. With Menagerie's 21 mm fingers, 37
     succeeded.
   - In 14 fingertip rows the first push left the gap short, so the other side was pushed too (13 of them placed).
   - Earlier changes, measured with Menagerie's 21 mm fingers:
     - Dropping the old 35 N limit on the books' estimated friction gave the fingertip 8 rows that used to go to
       the wedge (3 of them now placed), and let 6 rows that had to split push one side instead (4 of them failed
       before and now succeed). Seed 27 now pushes the side the estimate put at 93 N, and succeeds. Nothing that
       worked before fails.
     - Measuring where the book has to go again after the slide, and pushing again when the push drags the
       neighbours back, placed 2 more (seeds 0 and 46, which had stopped 12 to 15 mm proud of the neighbour) and
       4 more with `split`. Rows that already worked take 0.1 s longer on average.
   - One side versus splitting, on the same 49 fingertip rows: one side places more, 47 vs 34 (35/50 overall
     with `split`). Splitting swivels both neighbours, so the book's back corners catch on both sides and it
     jams at the entrance more often. In 9 rows it leaves the gap narrower than the book, so they are skipped.
     Where both worked, the most-moved book moved about the same (6.2 vs 5.6 mm, one side
     vs split) and one side was 1.6 s faster (18.3 vs 19.9 s). Splitting needs much less force, as fewer books
     must slide at once.
   - Books have room behind them, so a book sliding in tightly can also drag its neighbours back. On seed 0 it
     drags the pushed neighbour 82 mm back, and still counts as placed: success only checks the inserted book.
   - Why it's hard: the fingertips push a neighbour only near its spine, so it swivels instead of sliding. The gap
     opens at the spines but barely at the back, and the book has to widen the rest as it slides in. That is the
     contact-rich part the proposal's residual policy is meant to learn.

   **Packed**: the neighbours touch, so room has to be made first, and it can only come from the row's free
   space at one end: every book between the slot and that space has to slide along.
   - Shove (`--packed shove`, the default):
     1. The closed fingers, stacked, push the books in front of which the fingers will go (the other neighbour,
        and the book past the neighbour on the side with more free space) straight back, until their spines are
        35 mm behind that neighbour's. If they have less room behind them than 20 mm, the other side is tried.
     2. That neighbour now sticks out of the row. The fingers go either side of it, 4 mm off its faces, with the
        hand turned to its yaw and the fingertips 7 mm in front of the books pushed back (under the load the arm
        sags 6 to 10 mm deeper, and a fingertip on one of their spines pins it by friction), and close on it at
        5 cm/s. At the simulated gripper's own speed (over 2 m/s from a few mm away) the pads went 18 mm into the
        book before their contact stopped them, and squeezed it 3 cm back into the shelf.
     3. They shove it, and every book beyond it, away from the slot until the gap is the book plus 6 mm at its
        narrowest over the depth the book will take up, pushing on by what is missing up to 3 times. The
        neighbour still turns about 6° in the fingers (the run pushes back far behind where they hold it), so
        the gap at the spines is wider than that.
     4. The whole run slides, which takes 30 to 110 N (the books' weight times their friction, 52 N in the
        median row). Along the row the arm pushes with 55 N (joint 6 at its 12 N m) to 130 N depending on its
        configuration, and the strong configurations have joint 5 near its limit. So for this step the arm
        configuration is chosen for force, keeping 0.02 rad from the joint limits instead of 0.05. Without that,
        59 of the 100 rows below succeed instead of 69 (14 lost, 4 gained).
     5. The arm lags behind its target under the load. It holds where it got to, lets go, eases the fingers off
        the book's faces (else they drag it out with them), takes them out, and pushes the book back in line,
        square to its spine. Then the book is inserted as in a clear row, or, if the gap is still narrower than
        the book (the arm couldn't push the run far enough), after widening it as in a tight row.
   - Seeds 0 to 99: 69 succeed (39 of seeds 0 to 49), against 10 with the stagger. In the 62 rows where the shove
     opens the gap by itself, 59 are placed. In 25 the arm can't push the run far enough, and the tight-row
     widening places 10 of them. 13 rows aren't shoved: in 11 the books beside the neighbour have less than
     20 mm of room behind them, and 2 don't have room for the book on either side alone (they would need both).
     Of the 18 shoved rows that fail, 9 find no arm configuration (or joint move) for the widening and
     insertion, 4 jam at the entrance, 3 end up out of place, and 2 are left with a gap too narrow. The
     most-moved book moves 58 mm on average (the run moves by the widening, 38 mm on average, plus what the arm
     overshoots, and two books stay 35 mm back), against 69 mm with the stagger. It takes longer: in place after
     45 s on average, against 25 s, as the arm changes configuration 3 or 4 times.
   - Tried and dropped, on seeds 0 to 49: turning the hand against the book's turn while shoving, and pulling
     the book out and taking it again with the whole pad (43 mm instead of 28). It still turns, and no more rows
     are placed. A stronger grip (140 N, the Franka Hand's maximum) or grippier pads (friction 1.5) didn't help
     either, nor did moving the robot base (0.65 m from the shelf instead of 0.8, or beside the slot as in the
     gripper study, which helps the stagger but not this).
   - What would help, measured on the same 100 rows with hardware changes that aren't in this code (on a study
     version of the shove that places 67): a suction cup on a plunger between the fingers, to pull the
     neighbour out where the books beside it can't go back, 77; wrist joints as strong as a KUKA iiwa's (40 N m
     instead of 12), 73; both, 83.
   - Stagger (`--packed stagger`): the closed fingers, stacked, push the neighbour on the side with less room 2 cm
     straight back into the shelf. The side of the other neighbour then shows in front of it, and the fingers push
     that one aside from there, until the gap at the spines is 4 mm wider than the fingers. They back off its face
     before coming out, or friction drags it out of the shelf. Then the tight-row widening and insertion take over.
     Seeds 0 to 49: 6 succeed (in place after 24 s on average, but the most-moved book moves 9 cm on average).
     The neighbour is pushed at its front corner, so it swivels, comes partly out of the shelf, and then takes up
     more of the row. An earlier study found the same with other moves: dragging a book's top or spine by
     friction slides the books rather than tipping them, while a sideways push at the top tips the run 12° at 6
     to 58 N and it mostly stays tipped, leaving a V to wedge into.
   - `free_space` counts each side's room as its length less the books' thicknesses, not from the books'
     outlines: a pushed book that turned would otherwise seem to fill the row, and the widening would pick the
     wrong side. For upright books (clear and tight rows) both give the same.

7. **Next pieces to build on top** (from the design doc)
   - Swap the Panda's position actuators for torque (`motor`) actuators and write the impedance controller.
   - Spawn the target book in the gripper for other code too (`insert_book.py` does it for its own runs;
     `build_scene` still leaves `row.target_book` out).
   - Add the gap-creation moves and insertion on top of that controller (`insert_book.py` scripts them with
     position control, as a baseline).
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
- `NoisyPerception` adds 2 mm of noise to each corner and reads gaps under 4 mm as closed. These are starting
  guesses for a wrist camera.
- Note: `row.gap_right - row.gap_left` is up to 3 mm short of the actual face-to-face gap, because `gap_left`
  is set after the left neighbour's spacing. Measure the gap from the books, as `perception.py` does.

Tests: `uv run pytest`.

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
| packed | 40/40 | Nothing moves (≤ 0.2 mm); the slot's neighbours end up at most 1.1 mm apart |
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
- **Generated robot file.** `shelf_scene.py` writes `panda_shelf.xml` (Menagerie's Panda with the thin fingers,
  wrist camera and F/T sensor) into `mujoco_menagerie/franka_emika_panda/` so the robot's mesh paths still work.
  The submodule is set to ignore untracked files, so this doesn't show as a change.
- **Shelf width.** If you set the shelf very narrow, `sample_row` raises an error when fewer than 4 books fit.
