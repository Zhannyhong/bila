# Library robot: making room on the shelf

`shelf_scene.py` builds the bookshelf scene for the gap-creation project in MuJoCo.
It starts from the shelf in ContactExplorer's singulation task
(`src/tasks/allegro_singulation.py`, Isaac Gym Preview 4) and turns it into a full,
randomized bookcase standing on the floor. It also adds a Franka Panda with thin
10 mm-wide, 12 cm-long fingers, a wrist camera and a force/torque sensor. Optionally it draws the real book covers
ported over from the PBBM robot librarian repo (see [Book covers](#book-covers-optional)) and cuts every
book into page sections hinged on the spine (see [Book pages](#book-pages-optional)).

## What's in the repo

| File | What it does |
| --- | --- |
| `shelf_scene.py` | Builds a randomized bookcase and target row for one of five scenarios, adds the Panda, settles the row with physics; can split every book into page sections hinged on the spine |
| `dae_to_obj.py` | Converts the PBBM repo's Collada book covers (and, for the shaped books, their page sections) to the OBJ/PNG assets `shelf_scene.py` draws (run once, optional) |
| `view_scene.py` | Opens a settled scenario in the MuJoCo viewer |
| `cart_demo.py` | Book-return cart: the Panda on a lift, a tray of real books standing spine-up, a row camera on the lift carriage, call-number labels on every spine. Reads the tray, looks the book's row up in the shelf map, scans the row for its slot, makes the gap (`insert_book`'s skills), picks the book from the tray and inserts it (`uv run mjpython cart_demo.py --scenario tight --view`) |
| `insert_book.py` | Scripted baseline for the **clear**, **tight**, **leaning** and **packed** rows: makes room (widens a tight gap, stands a leaning neighbour up, shoves a packed row aside), then slides the book in |
| `perception.py` | What a wrist camera could report about the target row (`ShelfEstimate`), from ground truth or with noise |
| `make_vision_dataset.py` | Training images for the learned perception: RGB-D, a mask per book, and every sticker's corners and text |
| `learned_segmenter.py` | Book masks from the colour image (Mask R-CNN), in place of the simulation's object ids |
| `label_reader.py` | Finds and reads call-number stickers in an image and matches them to the catalogue (`cart_demo.py --labels learned`) |
| `notebooks/` | Beginner's guide to the simulator and the RL/POMDP setup with PBBM's book labels (`01_...`), and a breakdown of how the PBBM robot shelved books (`02_...`) |
| `mujoco_menagerie/` | Git submodule with the Panda model |
| `scenes/` | Generated scenes (git-ignored) |
| `generated_assets/books/` | Converted book covers and textures (git-ignored, written by `dae_to_obj.py`) |

| Scenario | Target row | Scripted baseline |
| --- | --- | --- |
| clear | Slot already wide enough | `insert_book.py --scenario clear` |
| tight | Slot narrower than the book | `insert_book.py --scenario tight` |
| leaning | A neighbour leans into the slot | `insert_book.py --scenario leaning` |
| packed | No gap: the slot's neighbours touch | `insert_book.py --scenario packed` |
| collapsed | A run of 3 to 6 books has toppled over | none yet |

## Quick start

```bash
git clone --recurse-submodules <this repo>     # or, in an existing clone:
git submodule update --init                     # fetches mujoco_menagerie
uv sync                                         # installs mujoco and numpy from pyproject.toml

# Generate and look at scenes
uv run python shelf_scene.py --seed 0 --panda mujoco_menagerie/franka_emika_panda/panda.xml --out scenes
uv run python view_scene.py --scenario leaning --seed 0          # add --no-robot for the bookcase only

# Run the scripted baselines
uv run python insert_book.py --scenario clear --seed 0
uv run python insert_book.py --scenario tight --seed 0 --trials 50
uv run python insert_book.py --scenario leaning --seed 0
```

Optional, once, to draw the real book covers and their page sections (see
[Book covers](#book-covers-optional) and [Book pages](#book-pages-optional)):

```bash
git clone --depth 1 --filter=blob:none --sparse \
    https://github.com/TAMS-Group/PBBM-robot-librarian /tmp/pbbm
git -C /tmp/pbbm sparse-checkout set librarian_common/tams_books_world
uv run python dae_to_obj.py --src /tmp/pbbm/librarian_common/tams_books_world/meshes
```

On macOS, anything with `--view` must run under `mjpython` instead of `python`
(for example `uv run mjpython insert_book.py --scenario leaning --seed 0 --view`), because the passive viewer needs it.
With a uv-installed Python, `mjpython` can fail with `Library not loaded: @rpath/libpython3.12.dylib`. Link the
library into the venv once (again after recreating `.venv`):

```bash
ln -sf "$(uv run python -c 'import sysconfig; print(sysconfig.get_config_var("LIBDIR"))')/libpython3.12.dylib" .venv/
```

## The simulator

### The bookcase

`shelf_scene.py` starts from the shelf in ContactExplorer's singulation task
(`src/tasks/allegro_singulation.py`, Isaac Gym Preview 4) and turns it into a full, randomized bookcase
standing on the floor, with a Franka Panda that has a wrist camera and a force/torque sensor.

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
- Frame convention (same as the Isaac Gym task): x along the row, y out of the shelf (back panel at −y), z up.

### How the target row is made

Books are placed upright, then the leaning and collapsed scenarios push some of them over during a short
settling phase (the pushes are stored in `row.nudges`). This way every lean is a real resting pose found by
physics, not a hand-set angle that might overlap other books.

- **Leaning**: the gap is sized from the lean angle you want (12° to 35°), so the book comes to rest against its neighbour.
- **Collapsed**: a run of 3 to 6 books topples into a hole next to it. The hole is sized the same way, so the run
  leans far enough to stay down and lands on the book across the hole below its top.
- In both, the spare row space goes to the end away from the toppling. The books that catch the lean are
  packed against a side panel and can't slide away.

### Generating and viewing scenes

```bash
uv run python shelf_scene.py --seed 0 --panda mujoco_menagerie/franka_emika_panda/panda.xml --out scenes
uv run python shelf_scene.py --scenario leaning --seed 3 --out scenes      # one scenario only
```

This writes one `.xml` (the MuJoCo scene) and one `.json` (levels, book sizes, target slot, gap edges) per
scenario, then settles each scene and prints whether it is stable. `scenes/` is git-ignored: the XML holds absolute paths.

```bash
uv run python view_scene.py --scenario leaning --seed 0       # add --no-robot for the bookcase only
```

`view_scene.py` settles the row first, so leaning and collapsed books show slanted, then opens the viewer.
(Opening an XML directly with `python -m mujoco.viewer --mjcf ...` shows the start state, with every book upright.)
Either way the camera starts on the gap, from the front and to one side, so the robot isn't in the way.
The scene sets this as MuJoCo's default free camera (`<statistic>` and `<visual><global>`), so you can still
orbit, and the viewer's Align button brings you back to it. MuJoCo's viewer only applies it by itself when it
opens a file, so `view_scene.py` and `insert_book.py --view` apply it themselves. In the viewer,
Rendering > Camera > `scene_cam` shows the whole target row.

### Book covers (optional)

By default every book is a plain box. `dae_to_obj.py` brings over the real covers from the
[PBBM robot librarian](https://github.com/TAMS-Group/PBBM-robot-librarian) repo
(`librarian_common/tams_books_world`), which models each book as a unit cube whose six faces carry
a texture: front and back cover, top and bottom edges, spine and fore edge.

```bash
uv run python dae_to_obj.py --src /tmp/pbbm/librarian_common/tams_books_world/meshes
uv run python shelf_scene.py --scenario leaning --seed 3 --book-meshes --out scenes
uv run python insert_book.py --scenario clear --seed 0 --book-meshes
uv run mjpython insert_book.py --scenario tight --seed 0 --view --book-meshes   # watch it
```

It writes `generated_assets/books/` (git-ignored: 512 px PNGs plus six one-face OBJs per book, and the
same again for each page section of the shaped books; see [Book pages](#book-pages-optional)).
MuJoCo cannot load Collada, so each cube is converted to OBJ; the six faces are split because MuJoCo
allows one material per mesh. Turn the covers on with `ShelfConfig(book_meshes=True)`, or `--book-meshes`
on `shelf_scene.py` and `insert_book.py` (the other entry points build with covers off).

- **Physics is untouched.** The covers are extra `mass="0"`, non-colliding geoms (`contype=0`,
  `conaffinity=0`) sitting on the book's box, which keeps all of the controller's logic
  (`insert_book.py` reconstructs the book's corners from `geom_size`, the box half-extents). Models
  built with and without covers have identical masses, inertias, `nq` and keyframe `qpos`.
- **No z-fighting.** When a book has covers its box moves to render group 3, which the default
  `geomgroup` does not draw; the covers are drawn instead, in group 2.
- **Missing assets are not an error.** If `generated_assets/books/` is absent, `book_meshes=True`
  quietly falls back to the plain boxes.
- The covers are decoration only: perception still reads the box geometry, so a camera-based
  `ShelfEstimate` is unaffected.

### Book pages (optional)

The repo also models some books (the Voynich facsimile, the Caterpillar, the Lagemann book) as a stack of
page sections hinged on the spine, so their fore edges can fan open (`urdf/book_part.xacro`, loaded by
`urdf/articulated_*.xacro`). `ShelfConfig(book_pages=N)` ports that design: every book's pages are cut into
`N` sections that hinge on the spine, and `dae_to_obj.py` converts each section's own mesh alongside the
book's covers.

```bash
uv run python shelf_scene.py --scenario leaning --seed 3 --book-pages 4 --out scenes
uv run python insert_book.py --scenario tight --seed 0 --book-pages 4
uv run mjpython insert_book.py --scenario tight --seed 0 --view --book-pages 4   # watch them fan
```

- **How it is built.** The back section is welded to the book; the `N - 1` in front of it each hang on a
  revolute joint about the spine's vertical line (`range="-π 0"`), chaining from the back cover forward, as
  in the repo. Each section carries its share of the book's mass, so the row still weighs what it did.
- **The box stays.** As with the covers, `<book>_geom` remains the flat box with the same name and
  `geom_size`, but non-colliding: the page sections carry all of the collision and the mass. So
  `insert_book.py` and `perception.py`, which reconstruct a book from its box, are
  unaffected — the code that reads contacts only has to count a page section as part of its book
  (`shelf_scene.book_children` does that).
- **Closed to start.** The keyframe starts every hinge at 0, so an articulated book stands as the box it
  replaced and the pages fan only when something pushes them.
- `book_pages=0` or `1` is normalised to the one rigid box, so the default scene is unchanged.

### Using it from Python

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
then 7 per filler book only if `static_other_levels=False`. With `book_pages=N`, each articulated book
adds `N - 1` hinge angles right after its own 7 values (`insert_book.py` keys the parked target book the
same way).

To change the scene before compiling (as `insert_book.build_model` does to add the held book and gravity
compensation), build the XML with `sample_row` + `build_scene`, edit it (`xml.etree` or
`mujoco.MjSpec.from_string(...)`), compile, then call `settle`.

### What was taken from ContactExplorer, and what changed

| Isaac Gym repo | MuJoCo version | Why it changed |
| --- | --- | --- |
| `_define_table`, `_define_upper_shelf`: one board 0.5 m up, one above | `board_0` … `board_N`, stacked from a plinth on the floor, plus `top_board` | A real bookcase, with books above and below the target row |
| `_define_back_wall`, `_define_side_wall`: walls for one cubby | `back_panel` and side panels from floor to top | Same parts, full height |
| Shelf 0.344 × 0.26 m, fixed | 0.6 to 1.0 m wide, 0.22 to 0.32 m deep, randomized levels and pose | Room for 12 to 20 books per row and space to push them |
| 5 boxes, 4 × 16 × 24 cm, fixed size | Every book a different, random size, mass and friction | Real shelves are mixed |
| Neighbour boxes have `fix_base_link=True` and friction `1e-5` | Target-row books are **free bodies** with normal friction (0.25 to 0.7) | The task is to push neighbours, so they must be movable |
| PhysX `friction`, `restitution` | MuJoCo `friction`, `solref`, `solimp`, `condim=4` | Different contact model |
| Two fixed cameras, 640×480, 58° | `scene_cam` (fixed, off to the side, aimed at the target row) + `wrist_cam` on the Panda hand | Robot blocked the front view; wrist camera sees the gap up close |
| xArm6 + Allegro hand | Franka Panda from MuJoCo Menagerie, with thin 10 mm fingers, + F/T sensor at the flange | Your chosen robot; Menagerie's 21 mm fingers don't fit the narrower tight gaps |

`python shelf_scene.py --repo-defaults` rebuilds the original single cubby (board top at 0.51 m) and box size, for comparison.

## Scripted baselines

### Clear and tight rows: `insert_book.py`

```bash
uv run python insert_book.py --scenario clear --seed 0                # one episode: prints each step and the score
uv run python insert_book.py --scenario tight --seed 0 --trials 50    # success rate over seeds 0 to 49
uv run mjpython insert_book.py --scenario tight --seed 0 --view       # watch it (macOS)
```

Options: `--strategy one-sided|split` (how a tight gap is widened, default `one-sided`), `--margin` (room left
around the book in a tight gap, default 6 mm), `--panda` (path to Menagerie's `panda.xml`), `--book-meshes`
(draw the converted book covers; see [Book covers](#book-covers-optional)), `--book-pages N` (split every
book into `N` page sections hinged on the spine; see [Book pages](#book-pages-optional)).

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

**Scoring** follows the proposal: upright within 5°, in place, released, plus how far the other books moved.
In place means:
- spine within 10 mm of the spine of the neighbour that sticks out more, or
- if the book is deeper than that neighbour, so it can't line up without its back passing the back panel:
  at least 10 mm behind the shelf's front edge. With the default rows that's rare (about 1 row in 400: a book
  near the maximum depth beside neighbours set more than 25 mm in, as the inserted book stops 5 mm short of
  the back panel), but it also covers other layouts such as `--repo-defaults`.

**Control and setup**
- Menagerie's position actuators, fed by differential IK along min-jerk paths, with gravity compensation.
  The arm configuration is chosen so the whole sequence stays clear of the joint limits and doesn't run into
  the shelf. The slide and the final push stop when the wrist force sensor feels the book stop giving way.
- The script adds the target book to the scene, and gives the gripper the real Franka Hand's 70 N (Menagerie's
  squeezes a book with 1 to 2 N). It only changes its own copy of the model.
- The fingers pinch the book 11.5 cm above its bottom. Higher up, board friction tips the book over the pads
  while it slides.

6. **Scripted insertion (clear, tight, leaning and packed scenarios)**
   ```bash
   uv run python insert_book.py --scenario clear --seed 0                # one episode: prints each step and the score
   uv run python insert_book.py --scenario tight --seed 0 --trials 50    # success rate over seeds 0 to 49
   uv run mjpython insert_book.py --scenario tight --seed 0 --view       # watch it (the macOS viewer needs mjpython)
   uv run mjpython insert_book.py --scenario leaning --seed 12 --view    # a leaner it has to clamp
   uv run mjpython insert_book.py --scenario packed --seed 7 --view      # a packed row it manages
   uv run python insert_book.py --scenario tight --seed 0 --perception camera   # decide from the wrist camera
   uv run python insert_book.py --scenario tight --seed 0 --perception camera --camera-noise basic   # cleaner camera
   uv run mjpython insert_book.py --scenario tight --seed 1 --perception camera --view   # watch it, and what it saw
   ```
   One controller for every scenario (`Episode.reshelve`): it looks at the slot, takes the high-level step the
   slot needs next, and looks again. The scenario only picks the row.

   | The slot | Step |
   | --- | --- |
   | a neighbour leans into it (more than 5°) | `stand_up`: lift it upright, or clamp it if the fingers can't get under it |
   | narrower than the book plus 6 mm (at the spines; at its narrowest after a shove or a stand-up, which can leave a neighbour turned), and the fingers (stacked, 10 mm) fit in it where they go | `widen`: push a neighbour aside with them |
   | narrower, and they don't (the neighbours touch) | `shove`: pinch a neighbour, shove it and the books beyond into the row's free space |
   | wide enough, or nothing more would widen it and the book fits | `insert` |

   Each step starts from wherever the arm is and leaves the hand out of the shelf, so the steps follow one another
   in any order, and one that falls short is simply taken again: a second stand-up; a second widening, on the
   other side if it has the room; a third on the first side with the arm configured for force, while the book
   still doesn't fit. A clear row comes out as `insert`, a tight one `widen, insert`, a leaning one
   `stand_up, insert`, a packed one `shove, insert` or `shove, widen, insert`. These are the high-level steps a
   policy could choose between; `push_back`, `walk_out`, a single pinch (`shove_stage`), `lift_up`, `clamp_up` and
   `fingertip_push` are the smaller moves they are built from.

   Seeds 0 to 199 of each scenario, ground truth:

   | | clear | tight | leaning | packed | all four |
   | --- | --- | --- | --- | --- | --- |
   | per-scenario scripts (`0ff9d92`) | 199 | 185 | 195 | 183 | 762 |
   | one controller | 200 | 197 | 199 | 184 | 780 |
   | in place after, s, on rows both place | 10.5 → 11.0 | 18.9 → 21.6 | 22.0 → 22.9 | 90.1 → 87.9 | |
   | actions (moves and waits), same rows | 10 → 10 | 17.6 → 19.8 | 17.0 → 17.5 | 69.1 → 68.3 | |

   Clear's time now includes the 0.5 s the gripper takes to close at the start. Tight rows take 2.7 s longer: a
   second widening is a step of its own (the fingers come out and go in again), and wedged books slow down near
   the neighbour's face. The most-moved other book moves as much as before (tight 9.1 mm on average, leaning 3.3,
   packed 64).

   What changed, besides the shared controller:
   - Retries come free: a leaner that fell back gets a second stand-up (leaning 142: 5.3° after the clamp); a
     widening that fell short gets the other side, then the first side again with the arm configured for force
     (`choose_strong`), which pushes heavy runs the arm gave way to (tight 92, 119, 125: the first push moved
     the neighbour only a few mm, and the old top-up could only go on from the arm configuration it was in).
   - Every row is inserted along the middle of the straight way in, or, if that is narrower than the book plus
     3 mm, wedged past the swivelled or turned neighbour (`wedge_slot`), as packed rows already were. Tight rows
     that jammed at the entrance now mostly go in. The wedge slows down 15 mm before the face it will meet:
     sliding on at full speed and wedging only once the book stopped lost 6 tight rows (it hits the face and
     jams). Its slow part runs at 4 cm/s (was 2): the same tight rows go in, 4 s sooner. Aiming the tight
     widening at the narrowest gap instead of the spines placed 199 but took 47 s a book: the fingertips swivel
     the neighbour, so the back never opens and nearly every row was widened three times.
   - Rows the old scripts didn't expect: a clear row (167) and a tight one (also 167) where a thin neighbour
     tipped into the gap as the row settled get stood up; a tight row with a gap narrower than the fingers (160)
     gets shoved; a leaning row whose other neighbour leans away (86), so the gap is 10 mm at the board but 45 mm
     where the fingers go, gets widened (the widening now measures the gap at the fingers' height).
   - Fixes in shared code: carrying on into the next step planned and ran with the arm's posture from before the
     move to its start (after a lift, 1.6 rad off on the wrist), and the IK's pull drove the arm toward a
     singularity mid-slide, which the wrist felt as a jam (leaning 93). It now plans the move to the start with
     the old posture and the step with the one it starts in, and runs them so. The clamp closed at the gripper's
     own speed and squeezed the 56 mm book out of the grasp (leaning 117); it now closes at 5 cm/s like the
     shove's pinch. If neither hand turn for the final push stays clear of the joint limits, the arm changes
     configuration for it instead of giving up (leaning 136). Choosing a configuration for force counts the
     moment of the pushed books' resistance 6 cm behind the fingers, so the arm yields less and turns the book
     less (on the 11 packed rows whose neighbour turned, 5 placed against 3 without it, none lost).
   - Heavy runs are split when they turn: the first shove of a run of 5 books or more stops as soon as the
     neighbour has turned 15° (the camera sees it), and the run is shoved in two, its far half first. Before, the
     split came only after the whole first shove, by when the neighbour had turned 20 to 40° and the gap was shut
     at the back. The stop fires in 11 of the 200 packed rows: 38 and 176 now go in, 96 doesn't (184 against
     183), at the same time per book. How much a run turns does grow with its load (the books' weight times
     their friction): 5° on average under 30 N, 25° at 75 to 90 N. But splitting by load up front, into
     segments of at most 35 or 50 N shoved farthest first (with the true masses), cut the turning in the worst
     rows (38: 40° to 9°) and still placed only 2 or 3 of the 11 turned-run failures, lost a row that worked,
     and took heavy rows that one or two pinches did before 5 or 6 (up to 3 times as long): each segment ends
     a little short, and the next can only go as far as the gap ahead. Stopping at 4° fired in 160 rows (most
     runs turn that much early and go in anyway) and placed 181. 4 of the 11 failures weren't heavy at all
     (33 to 44 N).
   - Removed: the packed stagger (10 and 5 of 100 placed) and `--strategy split` (34 of 49 against one side's 47).
   - Tried and dropped: the widening aimed at the spines after a shove too (packed 175: the book then goes in
     without the widening the turned neighbour needs); levelling only the books on the side pushed (packed 180:
     pushing the shoved neighbour back is what lets the fingers reach the other one in some rows, though in
     packed 128 it shuts the gap); taking the force configuration nearest the arm's (0.7 s faster per packed
     book); the slide's reach from the fingers' real opening instead of fully open (5 of 40 clear rows then had
     no arm configuration); shoving the side with too little room for the book just far enough for the fingers
     (no row gained).

   Left, of seeds 0 to 199:
   - Packed, 16. Five neighbours can't be pinched even to walk them out (3, 79, 152, 156, 166): the books beside
     them stand at the back panel, or a light neighbour goes back with them when they are pushed, and the other
     side has less room than the fingers need; in 111, pushing its 16 mm flanks back drags the neighbour 30 mm
     back with them. These need a second contact (a second arm or a suction cup, see below). The other 10 are
     runs whose neighbour turns in the shove (96, 119, 122, 123, 125, 128, 132, 165, 173, 195): the arm yields,
     with joint 6 at its 12 N m and joint 1 at its 87 N m, and the hand turns with the book; afterwards the
     wedge jams or the gap stays short. Which of these rows go in flips from run to run: the per-scenario script
     placed 4 of them (128, 165, 173, 195), and lost 5 others that now go in (47, 92, 94, 176, 178).
   - Tight, 3 (51, 132, 181), and leaning, 1 (139, where the lift swivelled the leaner 42° about the vertical, out
     of the shelf in front of the slot): books that jam at the entrance or stop short. The per-scenario script
     placed 132 and 139 and lost 51 and 181. Correcting the insertion as it goes is what the residual policy is for.

   **Insert**: a hand-written baseline. The target book is held in the gripper at a pre-insertion pose in front of the gap.
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
     across their travel, 12 mm thick and 12 cm long (Menagerie's are 5.4 cm), with a pad strip along all of
     each finger from the palm to the tip. The long fingers let the hand grasp a book from above past taller
     neighbours (see `cart_demo.py`). The real Franka Hand takes custom fingers, so these could be machined.
   - The book is held right up against the palm (its spine 2 mm off it), so the whole pad strip grips it. Held
     3 cm off the board for 3 s, books then turn 3.1° in the hand on average (5.1° for the 8 that weigh over
     1.2 kg), against 4.7° (7.8°) with the spine 8 mm out and a 35 mm pad. Insertion results are the same. A short
     pad is worse: with a 19 mm one, the board's drag turned a book 6° in the grasp on one clear row (seed 9) and
     it fell over.
   - The fingers start 0.2 mm off the book when they close on it, inside its 0.5 mm contact margin. Right on
     the margin, one pad's contact could start a step late, and that finger ended up 9 mm inside the book (on
     leaning seed 5, which then flew off when let go).
   - Clear, seeds 0 to 99: all 100 succeed, in place after 10 s on average. No episode moved another book more
     than 0.25 mm.

   **Widen** (tight rows): the gap is first widened to the book's thickness plus 6 mm at the spines. Measured
   with the per-scenario script unless it says otherwise.
   - Fingertip: the empty hand puts its fingers (stacked vertically, 10 mm wide) into the gap and pushes a
     neighbour aside. It pushes the side with room for all of the widening (the one with more free space if
     both have), as far as the widening needs or that side's space allows. If the gap at the spines is then
     still short of the book plus 6 mm, the next widening pushes the other side, by what is missing. Then the
     book is handed over (not modelled) and inserted as above. Used when the fingers fit the gap and reach at least
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
     with the split, which has been removed). Splitting swivels both neighbours, so the book's back corners
     catch on both sides and it jams at the entrance more often. In 9 rows it leaves the gap narrower than the book, so they are skipped.
     Where both worked, the most-moved book moved about the same (6.2 vs 5.6 mm, one side
     vs split) and one side was 1.6 s faster (18.3 vs 19.9 s). Splitting needs much less force, as fewer books
     must slide at once.
   - Books have room behind them, so a book sliding in tightly can also drag its neighbours back. On seed 0 it
     drags the pushed neighbour 82 mm back, and still counts as placed: success only checks the inserted book.
   - Why it's hard: the fingertips push a neighbour only near its spine, so it swivels instead of sliding. The gap
     opens at the spines but barely at the back, and the book has to widen the rest as it slides in. That is the
     contact-rich part the proposal's residual policy is meant to learn.

   **Stand up** (leaning rows): a neighbour leans into the slot (more than 5°), so it is stood back up first.
   Then the row is handled as a tight one: widened if the gap is short, then the insertion.
   - Lift: the closed fingers, stacked so they take 10 mm along the row, go under the leaning book as far up its
     face as they fit, and lift it along an arc about the bottom corner it pivots on. They press square on its
     face, so it turns rather than slides. They stop just short of upright, within its tipping angle
     (atan(thickness / height)) and still leaning slightly into the slot, so it stands up the rest of the way by
     itself instead of going over onto its other side. Up to 3 lifts, each closer to upright.
   - Clamp, when the fingers can't get under it at any height: they pinch it across its thickness near the spine,
     with the hand rolled to its lean, and swing it about its bottom edge to just short of upright. Only one
     finger has to fit between it and the far book. The other goes between it and the book on its own side,
     which gets pushed along a little.
   - A thin book let go in a gap more than twice its thickness can lean over, so it goes 3 mm off the taller
     upright neighbour instead of in the middle.
   - Seeds 0 to 49: every leaner is stood up, 47 by the lift and 3 by the clamp (seeds 12, 26 and 37), and all
     50 books are placed, in place after 21.9 s on average. 45 rows need no widening after the stand-up and 5 are
     widened with the fingertip. Leaving out the book that was stood up, the most-moved book moves 3.1 mm on
     average (median 0.3 mm, worst 41 mm on seed 26, where the clamp pushes the leaner's own side along).
   - The lift's fingers come in up to 4 mm off the leaning book's face, and as much lower, as far as the room
     beside them allows. When the room was only checked where they press on the book, they passed within
     0.1 mm of the far neighbour on seed 5 and dragged it 4 cm into the shelf. In 6 rows the most-moved book
     then moved 27 to 44 mm, against 0 to 10 mm now.
   - The rod on `feat/leaning-controller` (8 cm past the fingertips, pushing the leaner sideways at the top of
     the wedge under it) stands up 46 of these 50 and doesn't insert. By its test right after the stand-up
     (leaner under 8°, no other book tilted 10° or more, gap at least the book plus 4 mm without widening), 45
     pass here. The other 5 just need the widening that follows. The same fingers stand the book up,
     widen the gap and insert it, with no tool change.

   **Shove** (packed rows): the neighbours touch, so room has to be made first, and it can only come from the
   row's free space at one end: every book between the slot and that space has to slide along.
   - Shove:
     1. The fingers, stacked and 1 mm apart, push the books in front of which the fingers will go (the other
        neighbour, and the book past the neighbour on the side with more free space) straight back, until their
        spines are 35 mm behind that neighbour's. If they can't go back 12 mm, the other side is tried. (Fully
        closed, the fingers press on each other with all the gripper's force, and a knock pushed one pad into the
        other, where they stuck.) If neither can, or what is left to pinch is too little, the neighbour is
        walked out of the row (`walk_out`): the books beside it go back as far as they can, the fingers pinch
        what little of it sticks out, their tips 1.5 mm in front of those books (it only pulls, so the arm
        doesn't sag in), pull it 10 mm out, and pinch again deeper, until it is 35 mm out. Pinches of 2 mm
        held it. It goes back to where it stood once shoved.
     2. That neighbour now sticks out of the row. The fingers go either side of it, 4 mm off its faces, with the
        hand turned to its yaw, the fingertips 7 mm in front of the books pushed back (down to 4 mm if they could go
        back less than 19 mm) and the palm 12 mm in front of the other books: under the load the arm sags 6 to 10 mm
        deeper, and a fingertip on a spine pins that book by friction, a palm drags it along. They close on it
        at 5 cm/s. At the simulated gripper's own speed (over 2 m/s from a few mm away) the pads went 18 mm into
        the book before their contact stopped them, and squeezed it 3 cm back into the shelf.
     3. They shove it, and every book beyond it, away from the slot until the gap is the book plus 6 mm at its
        narrowest over the depth the book will take up. The whole run slides, which takes 30 to 110 N (the
        books' weight times their friction, 52 N in the median row). Along the row the arm pushes with 55 N
        (joint 6 at its 12 N m) to 130 N depending on its configuration, so the configuration is chosen for
        force (`choose_strong`). The strong ones have joint 5 near its limit, so they keep 0.02 rad from the
        joint limits instead of 0.05, and they reach it 1.5 to 2 cm into the push.
     4. If the gap is still short, the arm lets go and pinches again, up to 5 times, each with the
        configuration chosen afresh: the hand level or pointing 15° or 30° down into the shelf (which moves the
        load off joint 5), and pushing only 15 mm if that is much stronger. Before each, the books the fingers
        will be in front of go back, as the run has turned and fanned out.
     5. The neighbour still turns, 6° in the median row: the run pushes back far behind where the fingers hold
        it, and a 70 N pinch resists about 5 N m of turning (with 140 N it turned nearly as much). Where the run
        takes more than about 60 N, the neighbour pivots on its back corner rather than sliding the run, and
        pushing harder only turns it more.
        So if the gap is still short and the run has 5 books or more, the far half of the run is shoved on its
        own (pinching the book halfway along), then the near half into the gap that leaves. The first shove of
        such a run stops as soon as the neighbour has turned 15°, so the split comes before the gap shuts at the
        back.
     6. The arm holds where it got to (it lags behind its target under the load), lets go, eases the fingers off
        the book's faces (else they drag it out with them), takes them out, and pushes the book back in line,
        square to its spine. Then the book goes in as in a clear row, along the middle of the straight way in
        (a turned neighbour narrows it and moves its middle off the gap's at the spines) if that clears the
        book by 3 mm.
     7. Otherwise the neighbour has turned: its front went with the fingers and its back stayed, so the gap is
        wide at the spines and shut at the back. The fingers can't open it: they reach only about 4 cm behind
        the spines, and pushing a spine turns the book further. The tight-row widening first pushes the other
        neighbour into the room on its side, aiming at the narrowest gap. That neighbour went back for the
        shove, and the palm has to stay in front of the books it sweeps past, so the books there that stand
        proud of it (the shoved neighbour among them) first go back level with it: else the fingertips reach
        only its front. In rows where the side shoved has barely room for the book, the run's turned books
        take up more of it than when upright, and the room on the other side is needed too. Then the book
        itself opens the rest (`wedge_slot`): held, it slides in where its back corners meet a neighbour's
        face deepest (along the straighter neighbour's face if only one turned, between the two if both did),
        as usual until 15 mm before that, then at 2 cm/s, stopping only above 60 N (at 90 N it slipped back
        in the fingers). Its back corner meets the turned face deep in the shelf, near the middle of the
        books' friction, and pushes it aside, so the neighbour and the run beyond straighten. Let go, the book
        is pushed home until the wrist feels 100 N (at 40 N, wedged books stayed 3 to 9 cm out). If the
        fingertips can't reach a neighbour, or no arm configuration does the widening, the book is wedged in
        straight away.
     8. If neither side alone has room for the book, one side is shoved as far as its room allows, and the
        widening takes the rest from the other.
   - Results, seeds 0 to 199 (ground truth, the Panda with the thin fingers):

     | | seeds 0 to 99 | seeds 100 to 199 |
     | --- | --- | --- |
     | stagger | 10 | 5 |
     | first shove (one pinch, level; commit `eb33682`) | 69 | 60 |
     | shove, finished with the tight-row widening | 82 | 72 |
     | shove, finished with the widening and the wedge | 90 | 78 |
     | ... the books beside the other neighbour levelled first | 92 | 80 |
     | ... the widening planned with the books it pushes moving | 94 | 81 |
     | ... and the neighbour walked out where it can't be pinched | 96 | 89 |

     The staged pinches and the split were made looking at seeds 0 to 99; on seeds 100 to 199 they place 15
     rows the first shove didn't and lose 3 (paired exact test, p = 0.008). The wedge was made looking at the
     turned-neighbour failures of both halves, so seeds 100 to 199 are no longer a fresh test for it: it
     places 14 rows more and loses none; levelling the books beside the other neighbour, 5 more, losing 1;
     planning the widening with the books it pushes moving, 3 more; walking the neighbour out, 10 more,
     losing none. The most-moved book moves 65 mm on
     average, and a book is in place after about 88 s (the first shove: 45 s, the stagger: 25 s): the arm
     changes configuration for each pinch and push.
   - Of the 15 rows lost, 6 have a neighbour that can't be pinched even to walk it out: in 5 it stands level
     with or behind the books beside it once they are back, and in 1 a book further along stands so far out
     that the palm can't come close enough. The other 9 follow a turned neighbour: 4 jam at the entrance, 3
     are left narrower than the book even at the spines, and 2 wedged books stay 3 to 9 cm out.
   - Fixes made on the way, outside the packed code:
     - `reposition` keeps the arm's posture while it takes the hand out of the shelf. Choosing the next
       configuration had already set the IK's rest posture to it, and its pull swung the arm round mid-move
       (2.9 rad off in 30 steps), into the shelf. Clear, tight and stagger trials are unchanged.
     - A joint move that can't go direct or by way of the tucked home pose tries random waypoints, and
       `choose_arm` tries 3 more rounds of starts if the first 3 found nothing.
     - The tight-row widening plans the fingers free to touch every book on a side they push, not only the
     neighbour: those go ahead of the fingers, but not in the planner's static scene, where a push longer
     than the neighbour is thick ran the fingers into the next book (the same as `push_back` ignoring the
     book it pushes). Tight rows are unchanged (48 of seeds 0 to 49).
   - `face_x` takes a book's face from its own axes. It took the two corners furthest along the row, which
       are the back face's once the book has turned more than atan(thickness / depth), about 7°.
   - Tried and dropped: shoving long runs in two from the start (they lose rows a single shove manages);
     turning the hand against the book's turn; pulling the book out and taking it again with the whole pad
     (43 mm instead of 28); a 140 N grip; grippier pads; moving the robot base; turning the neighbour further
     before the wedge, so the book meets it deeper (the run's fronts have used up the row's free space by
     then: 0 to 13 mm were left); planning the hand around turned books' outlines rather than their bounding
     boxes (no change); shoving heavy runs in segments of 3 books, the farthest first (they pivot as well:
     one row more, two fewer); squeezing the wedged book at 140 N to slide it harder, or pushing it home
     with up to 150 N (no change); widening again, both sides, when the first widening falls short (no
     change); dragging a tall neighbour out by its top under a low board (it tips 6 to 7° until its top
     meets the board, then the fingers slip over it).
   - Tilting the neighbour out by its top (fingers pressing just behind its spine and pulling) works: its top
     comes out with the hand, 5 cm at 16°, its bottom staying put. But let go, it falls back upright, and a
     parallel gripper can't hold it tilted and pinch it at once. Rocking it (pushing its spine high up, then
     letting it fall) only walks it back into the shelf. So rows where the books beside the neighbour have no
     room behind them need another way to pull it out, such as a suction cup or a second arm.
   - A second Panda (a study, not in this code), on a base it places beside or above the first, as a mobile
     arm would, and out of reach until it is needed (parked beside the first, its tucked arm blocked the
     first's moves). Where the neighbour can't be walked out (it stands level with or behind the books
     beside it), the first arm puts its open fingers where the neighbour's top will come out, the second
     presses its closed fingers on the top just behind the spine and tips the neighbour forward about its
     front bottom edge (up to 20°, less under a low board above: its top back corner rises), the first
     closes on it, the second leaves, and the first pulls it 35 mm out while standing it up. And where the
     wedged book stalls with its middle still out of the shelf, the second arm pinches its top from above,
     outside the shelf, and both push it in. Seeds 0 to 199: 190 (one arm: 185): it places the 5 rows
     that can't be walked out and one stalled wedge, and 3 rows flip either way between the runs. Before the walk-out, the second arm placed 11 of the 20 rows
     one arm couldn't pinch; on the 5 with a tall neighbour under a low board, a search over 8 pinches of the
     first and 24 bases of the second (192 pairs per row) found none that both arms can do, but walking
     the neighbour out places 3 of them with one arm. Pinching the neighbour with both arms, one above the
     other, to shove harder gained nothing (-6 to +0.1 mm): a turned run's back doesn't follow pushes at
     its front however hard.
   - What would help, measured with hardware changes that aren't in this code (on an earlier version of the
     shove that placed 67 of seeds 0 to 99): a suction cup on a plunger between the fingers, to pull the
     neighbour out where the books beside it can't go back, 77; wrist joints as strong as a KUKA iiwa's (40 N m
     instead of 12), 73; both, 83. On the current sequence (168 of seeds 0 to 199): the suction cup (60 kPa,
     4 cm past the fingertips, pulling the neighbour 35 mm out where one arm can't pinch it), 182: 14 of the
     20 rows one arm can't pinch, 3 more than a second arm, losing none, but it loses its hold on two thin
     books (on a 16 mm spine it holds only 8 to 9 N); the gripper at the Franka Hand's 140 N peak throughout,
     169. Of the 18 rows the cup leaves, 16 are runs whose neighbour turns and 2 are those thin books.
   - A stagger was tried first and removed (10 and 5 of seeds 0 to 99 and 100 to 199): the closed fingers pushed
     one neighbour 2 cm back so the other's side showed in front of it, and pushed that one aside from there.
     Pushed at its front corner, it swivelled, came partly out of the shelf and took up more of the row. An
     earlier study found the same with other moves: dragging a book's top or spine by friction slides the books
     rather than tipping them, while a sideways push at the top tips the run 12° at 6 to 58 N and it mostly
     stays tipped, leaving a V to wedge into.
   - `free_space` counts each side's room as its length less the books' thicknesses, not from the books'
     outlines: a pushed book that turned would otherwise seem to fill the row, and the widening would pick the
     wrong side. For upright books (clear and tight rows) both give the same.

   **Perception** (`--perception truth` or `camera`): where the script's knowledge of the row comes from.
   - `truth` (default): the simulation, read whenever it is needed. All results above are with it.
   - `camera`: the wrist camera's RGB-D images (`rgbd_perception.py`). `--camera-noise realistic` (the default)
     gives the camera a D405's stereo errors and a calibration error drawn per episode (see the camera section
     below); `--camera-noise basic` only independent depth noise per pixel. Before
     touching the row the arm sweeps along it, looking from 30 cm left of the slot, in front of it and 30 cm right
     (`SWEEP`), and fuses the three images. It looks again from the middle view each time the hand has left the
     shelf after changing the row: after each widening, lift and clamp, each book pushed back, each pinch of the
     shove and each pull of the walk-out, after letting go of the book and after each push of it. The arm really
     moves there and back, in joint space, and comes in from just in front of the next step in a straight line.
   - On a look again, a book past the slot's neighbours that hasn't moved (within 3 mm and 1.5°) keeps what the
     sweep saw of it, and one that has keeps the size it saw: from the middle view, books further out come out
     too thick, and summed along the row that took 20 to 30 mm off the free space.
   - While the fingers push a neighbour aside, how far it went comes from where the fingertips are (joint angles;
     within 0.7 mm of the truth on the pushes checked): the camera can't see a face the hand is on. A book the
     shove (or the walk-out) pinches moves and turns with the hand until it lets go, if the fingers stopped on it
     (more than half its seen thickness apart; closed further, they missed it and it stays where it was seen).
     So the shove's stop at `TURN_STOP_DEG` comes from how far the hand has turned (joint angles).
   - The arm's paths are planned around the books where the last look saw them (books not seen are left out), and
     once let go of, the inserted book is in the way of its joint moves too. Known without looking: the shelf map
     (bookcase, side panels, where the slot is and how many books stand left of it), and the book being inserted
     (its size, and how it is held). The other books' depth is a prior.
   - With `--view`, after each look the viewer draws what the camera saw over the real books, as `view_scene.py
     --show-estimate` does: see-through boxes (the slot's neighbours orange), spine corners as dots, and the gap's
     width at each height (green where the book fits). It stays until the next look, so while the arm works you
     see what it is acting on. Books the camera didn't see aren't drawn.
   - The score is always from the simulation. (On the per-scenario scripts below, "flush" was judged against
     where the last look saw the neighbours' spines.)
   - Results with the controller (`Episode.reshelve`), seeds 0 to 199 of each scenario, realistic camera. With
     `--perception truth` the controller's episodes are the same as before the camera was added (18 episodes
     compared line by line).

     | | clear | tight | leaning | packed | all four |
     | --- | --- | --- | --- | --- | --- |
     | truth | 200 | 197 | 199 | 184 | 780 |
     | camera, realistic | 189 | 139 | 188 | 94 | 610 |
     | looks per row, on rows both place | 3.2 | 5.3 | 4.5 | 10.6 | |
     | in place after, s, same rows (truth, on its own: 11, 22, 23, 88) | 48 | 87 | 73 | 182 | |

     The camera places 5 rows the truth doesn't (tight 132 and 181, leaning 139, packed 96 and 173) and loses
     175 that it does. Why, on those 175:
     - The row's free space seen too small (packed 53). The camera sees each book 3 to 4 mm too thick (the
       D405's edge spread and `fit_spine`'s fixed 2-pixel grow, see below), and the free space is the side's
       length less the books' thicknesses, so on a side of 10 to 15 books the shove sees 30 to 60 mm too
       little room (packed 13: 105 mm on the right, seen as 42) and gives up.
     - The gap still narrower than the book at the end (tight 21, packed 15): the pushes go to faces the
       camera placed a few mm off where the hand is (the calibration error, see below).
     - The book left out of place (51): stuck partway, more than 20 mm out (31), or 10 to 20 mm proud (20). Seen
       a few mm to one side, the book catches a neighbour's face and drags it in (clear 13: the slot seen 6 mm
       to the left, a 15 mm book for a 23 mm gap, the neighbour dragged 13 cm back); seen spines a few mm too
       far out, it is pushed flush with where the camera saw them (tight 0: 9 mm, left 10.1 mm proud).
     - Jammed at the entrance (20), too little of the neighbour to pinch (11: the fingers stop short of a spine
       seen too far out, or close on nothing), other (4).
     Seeds 0 to 49: 154 (46, 36, 46 and 26), against the per-scenario scripts' 128 below.
   - Results with the per-scenario scripts (`b7675f1`), seeds 0 to 49 of each scenario. "Truth at the looks" is
     a control: the same sweep, looks and arm moves as with the camera, but each look reports the exact ground
     truth.

     | Scenario | truth | truth at the looks | camera, basic noise | camera, realistic |
     | --- | --- | --- | --- | --- |
     | clear | 50 | 50 | 50 | 49 |
     | tight | 48 | 47 | 44 | 33 |
     | leaning | 50 | 49 | 49 | 42 |
     | packed | 6 | 7 | 6 | 4 |
     | all | 154 | 153 | 149 | 128 |

     The packed row was the stagger then (since removed): the camera didn't drive the shove yet.

     Looking (going to look, and acting on what a look saw rather than on the row as it is now) costs about one
     book; leaning seed 37 and the packed rows lost (4, 7, 19, 37) fail the same with exact looks. The basic
     camera's errors cost about four more, all in tight rows (not significant at 50 rows: paired exact test,
     p = 0.13 for tight). The realistic camera costs 21 more than the basic one.
   - Which of the realistic errors costs what, on the 50 tight rows (each seed has the same errors in every
     column: the camera draws each kind from its own random stream): basic camera 44; with the stereo errors
     instead 41 without the depth offset, 43 with it; with the calibration error added to the basic camera 35;
     both 33. Each row's result depends a lot on the error it draws: another set of draws gave 28 for both, so
     take these to about ±5 rows. The calibration error is what matters. It puts what the camera sees a few mm
     to one side of where the hand is, and the script mixes the two: it plans the widening to a face position
     the camera measured and pushes there with the hand, and while pushing it measures the gap from the face it
     feels to the other face as the camera saw it. Tight seed 12, with a smaller than average draw (0.3° and
     1.8 mm, against 0.5° and 2.4 mm on average): the
     camera saw the slot 4 to 5 mm to the left; once the push had stopped, the robot believed the gap 25.4 mm
     wide when it was 19.8 mm, and the look after it rightly found the gap narrower than the 20.2 mm book. 8 of
     the 17 tight rows lost end that way. The stereo errors make the gap look 2 to 3 mm narrower (mostly the
     edge spread), and cost 1 to 3 rows.
   - What a look gets wrong (basic camera, 95% of errors): the gap at the spines 2.0 mm (median 0.7 mm too wide), the
     neighbours' faces 1.3 mm, their spines' depth 0.5 mm. Free space on each side 22 mm, and 9 mm too little in
     the median. Both point at how the spine fit (`fit_spine`) places a spine's edges: it grows every spine by 2
     pixels on each side for the pixels the normals lose at its edges, the same wherever it is. Measured at the
     books' bottom corners from the middle view (18 scenes), an edge beside another book comes out 2 mm too far
     out (0.9 mm if that book is set more than 1 cm further back), and one beside the open slot 0.5 mm too far
     in. So the gap comes out a little wide, and the other books 1 to 2 mm too thick, which adds up along the row.
   - The tight rows only the basic camera loses are at the margin, and about 1 mm decides them. Seeds 32 and 38: the
     gap was seen 0.8 to 1.2 mm too wide, so the neighbour was pushed that much less, and the book then dragged a
     neighbour 10 cm into the shelf (32) or jammed at the entrance (38). Seed 0: seen 1.4 mm too narrow, so the
     push is planned that much longer, and it then pushes the fingertips into the book behind the neighbour in
     the planner's scene, where nothing moves, which counts as running into it (with the truth the plan is only
     let through because the planner checks every third pose). In an earlier run seed 6 was lost too: free space
     seen too small stopped the other neighbour from being pushed as well.
   - Time, per-scenario scripts: looking takes 28 s (clear) to 48 s (packed) per row, 3 to 6 looks, so a book is
     in place after 43 to 96 s instead of 10 to 24 s. About 30 to 50 s of computer time per row instead of 10 to
     15. With the controller: 30 s (clear) to 84 s (packed) of looking, in place after 48 to 182 s.

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
  the camera frame round. The pose is read from the simulation when the image is taken (with a calibration
  error, it is where the robot believes the camera is).
- Markers (sites, such as the green dot at the target slot) are hidden: no real camera would see them.
- `WristCamera(model, noise=DepthNoise())` adds basic depth errors: random error growing with distance squared
  (Intel's formula for D400 depth with 0.08 px matching error: about 1.7 mm at 50 cm), independent per pixel, and
  no reading at object edges.
- **A more realistic camera**: `WristCamera(model, noise=DepthNoise.d405(), calibration=CalibrationError(), rng=...)`.
  - Stereo errors (`DepthNoise.d405()`): matching error 0.1 px (Intel: below 0.1 on well-textured targets),
    smooth over 2 px rather than independent per pixel; a disparity offset fixed per camera, 0.7% of the depth
    at 50 cm (std), kept within the D405's ±2% spec; at depth edges the nearer surface's depth spreads 2 px over
    the farther one; and no reading where the second imager, 18 mm to the right, can't see the surface (the
    stereo shadow left of nearer things, and the image's left edge). Intel publishes neither the smoothing nor
    the edge spread: those two are assumptions.
  - Calibration error (`CalibrationError()`), drawn once per camera: where the robot thinks the camera sits on
    the hand is off by 1.5 mm and 0.3° per axis (about 2.6 mm and 0.5° in all; published eye-in-hand RGB-D
    calibrations are a few mm and 0.5 to 1° off), and the focal length by 0.3% and the image centre by 1 px. Images
    are taken from where the camera really is; `frame.pose` and `frame.K` are what the robot believes.
  - Each of a camera's errors comes from its own random stream, split off `rng` without drawing from it: a seed
    gives the same calibration error with or without the stereo errors, the same depth offset with or without
    the calibration error, and the same per-image noise as the basic camera. Comparisons of one seed across
    error models are then paired.
  - The calibration error is fixed for a camera's life (every image of an episode shares it) and drawn anew per
    episode (per `WristCamera`): results average over calibrations, as training should. To see how success falls
    with the error's size, scale `CalibrationError`'s standard deviations; to model one robot's calibration, set
    the camera's `mount_error` and `K_believed` after making it.
  - What it does to an estimate (sweep, 48 scenes, against the basic camera): corners 14 mm off at 95% (2.5 to
    3.7), spines' depth 9 mm (0.5), books 2.4 to 3.8 mm too thick (−0.6 to +1.9), the gap 3 mm too narrow in the
    median (0.7 mm too wide). One error at a time, from the middle view: the stereo shadow changes nothing (it
    hides background, not spines); the edge spread makes books 2.4 mm thicker (the book's own side face, next to
    its spine, takes the spine's depth); the depth offset moves spines in or out by 7 mm (95%); the calibration
    error moves everything sideways, 11 mm at 95%.
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
- Books are named by where they stand relative to the target slot. If the camera sees the whole row (both side
  panels, and as many books as the shelf map lists), by counting: the map knows how many books stand left of the
  slot. Otherwise the slot is the boundary between adjacent books closest to the slot's place on the map. Books
  can shift while settling: in 2 of 50 leaning rows they settled 3 to 7 cm from the map's place, and the
  boundary rule named every book one place off.
- A side panel out of the camera's view is `None`, and if a slot neighbour isn't seen the gap on that side is
  unknown (NaN), rather than measured past books the camera didn't see.
- About 1.5 to 2 s per estimate (capture plus fitting).
- The camera renders without anti-aliasing: blended pixels at object edges decoded to wrong object ids.
- **Several images** (`estimate_from_frames`, e.g. from views 30 cm either side of the slot and in front of it):
  each book is taken from the view whose camera is nearest it along the row, and from the others only if those
  views missed it. Seen from far to the side, books come out too thick, and a book that sticks out past its
  neighbour shows its side, which the fit can take for a thin spine 15 mm further back (keeping the best view of
  every book instead counted 2 books twice in 32 scenes). On 48 scenes (12 of each of clear, tight, leaning and
  packed, with depth noise), against one image from in front of the slot: books named right in 48 (48), the whole
  row and both side panels in view in 48 (19); for books 4 or more from the slot, 95% of corner errors under
  3.7 mm (8.6) and thicknesses 1.2 mm too thick in the median (4.6). The neighbours and the books next to them
  come from the middle view either way.
- After the robot has put a book into the slot, `placed_x=` (where it let go) names the book seen there
  slot offset 0, and the others around it; offset 0 counts on neither side of the gap.

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

**Learned book masks** (`learned_segmenter.py`): the camera estimate's one cheat is step 1, reading book masks
off the simulation's object ids (`OracleSegmenter`, still the default). `LearnedSegmenter` finds them in the
colour image instead, with Mask R-CNN (torchvision, fine-tuned from COCO, one class "book"):

```bash
uv sync --group rl                                                      # torch, torchvision
# training data (WSL; ~1.8 h on 5 workers, 3.3 GB): shelf and cart scenes, seeds from 2000
MUJOCO_GL=egl uv run python make_vision_dataset.py --out ~/bila_data/train --first-seed 2000 --scenes 670 --workers 5
MUJOCO_GL=egl uv run python make_vision_dataset.py --out ~/bila_data/val --first-seed 9000 --scenes 70 --workers 5
uv run python learned_segmenter.py train --data ~/bila_data/train --val ~/bila_data/val --out ~/bila_data/segmenter
uv run python learned_segmenter.py eval --data ~/bila_data/val --weights ~/bila_data/segmenter/segmenter.pt
MUJOCO_GL=egl uv run python eval_perception.py collect --workers 3 --segmenter ~/bila_data/segmenter/segmenter.pt \
    --path ~/bila_data/perception_eval_learned.pkl                     # adds "robust" and "learned" kinds
```

```python
from learned_segmenter import LearnedSegmenter
with RGBDPerception(model, row, segmenter=LearnedSegmenter("segmenter.pt"), robust_fit=True) as perception: ...
```

- Data (`make_vision_dataset.py`): 2938 training and 309 validation images, 1280 x 720, from real arm poses (the
  sweep views, random views, row and tray cameras, sticker close-ups) in all five scenarios. Half the shelves are
  plain boxes, half have covers (sampled, the PBBM books, or page sections); lighting and colours vary; D405 depth
  noise. Each image has an instance mask (one id per book, sticker included) and the stickers' corners and text.
- Training: 8 epochs, 2.1 h on an RTX 3050 Ti (4 GB; mixed precision, 2 images a batch). Inference 0.24 s an image.
- Against the oracle's masks on the validation images (a book is found if a mask overlaps it with IoU >= 0.5):
  93% of target-row books showing 2000+ pixels, mean IoU 0.93; in the sweep views 93% found and 97% of masks are a
  book. Small far books do worse (tray camera 53%).
- `robust_fit=True` is needed with learned masks. The spine fit takes the smallest rectangle around a mask's spine
  pixels; a learned mask bleeds a few pixels onto the next spine, in about the same plane, and on a tilted book
  that stretched the rectangle by 10 to 50 mm. The robust fit drops strips across the spine that hold few points
  (a real spine fills each from bottom to top) and fits again. It is off by default, so the oracle pipeline (and
  everything calibrated on it) is unchanged; it would help the oracle's masks a little too (narrowest gap in
  leaning rows, 95%: 4.1 mm against 6.3).
- Whole estimate, `eval_perception.py` (80 scenes; the realistic camera's own images, with the robust fit for
  both, so only the masks differ): 1144 of 1151 books found, 2 estimates that are no book, 1125 of 1144 named
  right (oracle masks: 1150 of 1152). Narrowest gap up to the book's height, 95% within 4.4 mm (oracle masks 3.8):
  per scenario 3.8 (3.8) clear, 2.7 (2.7) tight, 1.5 (1.5) packed, and 8.8 (4.1) leaning, where masks of books
  leaning on each other are still the weak point. Neighbours' thickness, 95% within 4.3 mm (3.6).
- Learned masks carry no label; `OracleLabelled` gives each the id of the oracle book it overlaps most, only so
  the evaluations can match books.
- To see what the models see in one scene (~1.5 min each; weights in `~/bila_data`, `--weights` otherwise):

  ```bash
  MUJOCO_GL=egl uv run python visualize.py --scenario leaning --seed 3 --learned          # scenes/learned_*.png
  MUJOCO_GL=egl uv run python visualize.py --scenario tight --seed 1 --learned --cart     # scenes/stickers_*.png
  ```

  `--learned`: the first look's three views with the learned masks in colour, books missed outlined red, masks
  that are no book magenta, each target-row book's mask IoU, and the fitted spines; plus
  `estimate_learned_*.png`, its estimate with the one from the simulation's masks dotted and the truth dashed.
  `--learned --cart`: row_cam, tray_cam, a wrist close-up at the slot and a wrist view of the tray, every sticker
  found labelled with what was read and how sure (green right, red wrong, grey unreadable), missed ones dashed.
- `NoisyPerception.learned(source, rng)` imitates this pipeline instead of the camera with the simulation's masks
  (`perception.LEARNED_CALIBRATION`, from `eval_perception.py calibrate --kind learned` on those 80 scenes;
  `validate --kind learned` checks it). The default `NoisyPerception`, which `gap_env.py` trains on, is unchanged.
  Against it: each book's own error spreads more (position x, z about 1.3x; lean about 2x, from masks of books
  leaning on each other), the whole-row shift a little more along the row and up (std 3.2 and 3.4 mm, against 2.6
  and 2.5), and thickness is less biased (+2.1 to +2.5 mm, against +2.5 to +4.0: the robust fit). Not imitated:
  the learned pipeline misses about 0.6% of the books within the camera's reach and names 2.7% of those four or
  more from the slot a place off (the camera with the simulation's masks: none, 0.3%).

**Reading call-number stickers** (`label_reader.py`): `cart_demo.py` reads the stickers' numbers off the
simulation by default; `--labels learned` reads them from the camera images.

```bash
uv run python label_reader.py train-finder --data ~/bila_data/train --out ~/bila_data/labels   # ~17 min
uv run python label_reader.py train-reader --data ~/bila_data/train --val ~/bila_data/val --out ~/bila_data/labels --epochs 10
uv run python label_reader.py eval --data ~/bila_data/val --weights ~/bila_data/labels
MUJOCO_GL=egl uv run python cart_demo.py --scenario tight --labels learned    # weights from ~/bila_data/labels
```

- Find: a small segmentation network (LR-ASPP MobileNetV3, from COCO) marks the middle 60% of each sticker, so
  that stickers on neighbouring books come out apart; each piece's outline, grown back, gives its corners. A colour
  rule doesn't do: white covers and printed labels merge with the stickers.
- Read: the sticker is straightened from its corners and its two lines (class, author mark) cut out where the
  label prints them; a small CRNN reads each, in all four orientations (tilted books, the tray from above). It is
  trained on crops at the true corners, jittered, and at the finder's own corners.
- Match: every catalogue entry (each sticker's text, as the library knows it) is scored by the network's
  probability of reading it; the confidence is the best entry's share. Under 0.99 it says "unreadable": a misread
  puts a book in the wrong place, an unreadable sticker only costs another look.
- On the validation images (146 with stickers, 0.23 s an image), of the stickers in view:

  | camera | distance | found | read right | read wrong | unreadable |
  | --- | --- | --- | --- | --- | --- |
  | wrist close-up of the row | < 25 cm | 99% | 79% | 0% | 20% |
  | wrist close-up of the tray | < 25 cm | 100% | 85% | 0% | 15% |
  | wrist close-up of the tray | 25-50 cm | 100% | 76% | 0% | 24% |
  | wrist, inspection sweep | 25-50 cm | 94% | 63% | 0.7% | 30% |
  | row_cam | 50-100 cm | 55% | 12% | 0.1% | 43% |
  | tray_cam | 50-100 cm | 46% | 19% | 0% | 28% |

  0.15% of the stickers found are read wrong (0.8% at confidence 0.9). Most close-ups left unreadable are thin
  books: the label shrinks its text to the sticker's width. tray_cam gets no depth on half the stickers, and finds
  few of those.
- So the cart's cameras mostly find stickers and the wrist reads them: in the tray, a view of the whole tray from
  ~30 cm, then a close-up of any sticker still unread; in the row, close-ups (~18 cm) near the slot, at the unread
  sticker nearest the map's place for it, until the two books whose numbers bracket the book's are read side by
  side. A sticker read is tied to its book by the simulation's object ids under it (bookkeeping: the demo names
  books by body).
- In 12 cart scenes (clear, tight, leaning, packed; seeds 0-2): all 5 tray books read in every one, no number
  misread, and the right slot found in 10. In the other two the one book between the bracketing pair stayed
  unreadable (a thin book), and the demo stops rather than guess.

**What the policy sees** (`observation.py`): fixed-size vectors from a `ShelfEstimate` and the robot's readings.

```python
from observation import ObservationBuilder, robot_state, book_names_by_offset
obs = ObservationBuilder(est, target=(thickness, height))       # once per episode, after the robot looks
x = obs.actor(est, robot_state(model, data), now=data.time)     # every control step: 117 numbers
v = obs.critic(model, data, truth, book_names_by_offset(row))   # privileged extras for training: 126 numbers
x[obs.actor_layout["gap_narrowest"]]                            # any part by name
```

- Everything is in the slot frame, fixed when the robot looks: origin at the slot, the board's front edge and
  its top, with the shelf's axes. The same row gives the same numbers wherever the shelf, level or robot is.
- Actor (what a real robot has): books -3..-1 and +1..+3 (seen flag; spine centre, thickness, height, lean,
  swivel, visible), the gap (widths at 8 heights capped at 10 cm, which heights the book to insert spans, the
  narrowest gap, the widening still needed for book + 6 mm), free space behind each neighbour, the book to
  insert, the estimate's age, and the robot (joint angles and speeds, gripper, tool tip pose and velocity in
  the slot frame, wrist force/torque). The environment appends its own: base controller phase and action,
  time left, the last action.
- Free space (`perception.free_space`): walking out from the slot, the books before the first space of 10 mm
  or more and its width, capped at 6 books, and whether the camera saw far enough to tell. It is the cascade
  risk: those books move together when the neighbour is pushed, and the space is where they can fall.
- Critic: the same book and gap features from the ground truth now (not from the look), and each nearby book's
  mass, friction and velocity.
- SI units; normalise in the RL wrapper. The tool tip defaults to the fingertips (`tool_offset=` for a rod).

Tests: `uv run pytest` (with `MUJOCO_GL=egl` in WSL for the rendering tests).

## Settings (`ShelfConfig`)

| Setting | Default | Meaning |
| --- | --- | --- |
| `width`, `depth` | 0.60 to 1.00 m, 0.22 to 0.32 m | Inner size of the bookcase |
| `num_levels` | 3 to 5 | Shelf boards above the plinth |
| `level_height` | 0.25 to 0.40 m | Clear height of each level |
| `plinth_height` | 0.05 to 0.10 m | Bottom kick board |
| `target_band` | 0.40 to 1.10 m | Allowed height of the target board's top surface |
| `fill_other_levels` | 50 to 100% | How full the non-target rows are |
| `static_other_levels` | `True` | Weld non-target books in place, for speed |
| `spine_inset` | 10 to 30 mm | How far each spine sits behind the shelf's front edge |
| `depth_clearance` | 30 mm | Every book is at least this much shallower than the shelf |
| `book_meshes` | `False` | Draw the converted book covers (from `dae_to_obj.py`) on the boxes as non-colliding visuals |
| `book_pages` | `0` | Split every book's pages into this many sections hinged on the spine (0 or 1 = one rigid box) |
| `catalogue` | `False` | Fill the rows with PBBM's real books (`shelf_scene.CATALOGUE`, `--catalogue`): each copy keeps the book's size, mass and own cover, the Voynich and Lagemann books are hinged into their real page sections, and the same book can appear more than once. Positions, order, friction and scenario stay random. The sampled size, density and `book_pages` settings are then unused |
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
  book's tilt *change* from its settled pose, not its absolute tilt (`insert_book.py` does this).
- **Two leaning neighbours** (one on each side of the gap) are not generated: pushing both at once made one fall
  flat. Use the collapsed scenario for several leaning books.
- **Generated robot file.** `shelf_scene.py` writes `panda_shelf.xml` (Menagerie's Panda with the thin fingers,
  wrist camera and F/T sensor) into `mujoco_menagerie/franka_emika_panda/` so the robot's mesh paths still work.
  The submodule is set to ignore untracked files, so this doesn't show as a change.
- **Shelf width.** If you set the shelf very narrow, `sample_row` raises an error when fewer than 4 books fit.
