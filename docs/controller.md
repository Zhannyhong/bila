# The controller

`bila/control/insert_book.py` holds one closed-loop controller for every scenario, `Episode.reshelve`. It looks at
the slot, takes the high-level step the slot needs next, and looks again. The scenario only picks the row. The module
docstring describes every step in detail.

```bash
uv run python -m bila.control.insert_book --scenario clear --seed 0                    # one row: each step and the score
uv run python -m bila.control.insert_book --scenario tight --seed 0 --trials 50        # success over seeds 0 to 49
uv run mjpython -m bila.control.insert_book --scenario packed --seed 7 --view          # watch it (macOS needs mjpython)
uv run python -m bila.control.insert_book --scenario tight --seed 0 --perception camera   # decide from the wrist camera
```

Options: `--perception truth|camera|noisy|noisy-learned`, `--camera-noise realistic|basic`, `--segmenter WEIGHTS`
(learned book masks, see [perception](perception.md)), `--catalogue`, `--book-meshes`, `--book-pages N` (see
[the simulator](simulator.md)), `--margin` (room left beside the book, default 6 mm), `--panda` (Menagerie's
`panda.xml`).

## The steps

| The slot | Step |
| --- | --- |
| a neighbour leans into it (more than 5°) | `stand_up`: lift it upright with the closed fingers, or clamp it if they can't get under it |
| narrower than the book plus 6 mm, and the stacked fingers (10 mm) fit in it | `widen`: push a neighbour aside with them |
| narrower, and they don't fit (the neighbours touch) | `shove`: pinch a neighbour and shove it, and the books beyond it, into the row's free space |
| wide enough, or nothing more would widen it and the book fits | `insert` |

Each step starts from wherever the arm is and leaves the hand out of the shelf, so the steps follow one another in
any order, and a step that falls short is simply taken again. A clear row comes out as `insert`, a tight one as
`widen, insert`, a leaning one as `stand_up, insert` and a packed one as `shove, insert` or `shove, widen, insert`.

- **Insert.** The book is held at a pre-insertion pose in front of the gap. The robot rests its back on the board's
  edge and slides it in, stopping when the wrist feels a force rise of 35 N. It lets go before the fingers reach the
  neighbours' spines, turns the hand 90° and pushes the spine home with the lower fingertip, stopping at a 40 N
  change. A tight book drags its neighbours back as it goes, so the controller looks again and pushes again if needed.
  If the straight way in is narrower than the book plus 3 mm, the book is wedged past the turned neighbour instead.
- **Widen.** The stacked fingers go into the gap and push one neighbour aside, as hard as the arm can, until the gap
  at the spines is the book plus 6 mm. The neighbour is pushed only near its spine, so it swivels as well as slides,
  and the book widens the rest as it slides in.
- **Stand up.** The fingers go under the leaning book as far up its face as they fit and lift it along an arc about
  the corner it pivots on, to just short of upright, so it stands up the rest of the way by itself. If they can't get
  under it, they pinch it across its thickness and swing it up about its bottom edge.
- **Shove.** The books beside the neighbour go straight back so it sticks out of the row. The fingers pinch it and
  shove it, and every book beyond, into the free space, with the arm configured for force, pinching again up to 5
  times while the row gives. A run of 5 or more books that stalls, or whose neighbour turns 15°, is shoved again in
  two halves, the far half first.
  A neighbour that can't be pinched is first walked out of the row in small pulls.

## Scoring

A row counts as placed when the book is upright within 5°, released, and in place: its spine within 10 mm of the
spine of the neighbour that sticks out more (or, for a book too deep to line up, at least 10 mm behind the shelf's
front edge). The score also reports how far the other books moved and how much they tilted (measured as a change from
their settled pose). `Runner` records any contact of the arm with the shelf, cart or floor over 1 N, and
`safe_success` is a success without one.

## Control and the hand

- Menagerie's position actuators, fed by differential IK along min-jerk paths, with gravity compensation. The arm
  configuration is chosen so each step stays clear of the joint limits and the shelf. Joint-space moves are checked
  for collisions every 1 cm of arm motion and keep 1 cm from the fixed structure (`Episode.joint_route`).
- The controller adds the book to insert to its own copy of the model and gives the gripper the real Franka Hand's
  70 N (Menagerie's squeezes a book with 1 to 2 N).
- The fingers are thin bars in place of Menagerie's (`shelf_scene.make_panda_with_sensors`): 10 mm wide, 12 mm thick
  and 12 cm long, with a pad strip along each. The long fingers let the hand reach a book from above past taller
  neighbours. The pads hold the book's first 44 mm behind its spine, so the spine sits 68 mm off the palm
  (`PALM_GAP`), and they pinch it 11.5 cm above its bottom.

## Perception

`--perception` sets where the controller's knowledge of the row comes from (the score always comes from the
simulation):

- `truth` (default): the simulation, read whenever it is needed.
- `camera`: the wrist camera's RGB-D images. Before touching the row the arm sweeps along it, looking from 30 cm left
  of the slot, in front of it and 30 cm right, and fuses the three images. It looks again from the middle view each
  time the hand has left the shelf after changing the row. While the fingers push or pinch a book, where it went
  comes from the joint angles, since the camera can't see a face the hand is on. See [perception](perception.md).
- `noisy` and `noisy-learned`: no rendering. Each look gives the ground truth with the camera's measured errors
  (`perception.NoisyEyes`), as the residual's training environment does.

Known without looking: the shelf map (bookcase, side panels, where the slot is and how many books stand left of it)
and the book being inserted. Other books' depth is a prior. With `--view`, the viewer draws what each look saw over
the real books.

## The book-return cart

`bila/control/cart_demo.py` puts the Panda on a lift on a book-return cart, with a tray of returned books and a call
number on every spine. With its wrist camera only, it reads the tray, looks each book's row up in the shelf map,
raises the lift to it, reads the row's stickers to find the slot, makes room with the controller's steps, then picks
the book out of the tray and inserts it. `CartEpisode` is `Episode` with `take_book` replaced, so `reshelve` runs
unchanged.

```bash
uv run mjpython -m bila.control.cart_demo --scenario tight --view                  # watch it (macOS needs mjpython)
uv run python -m bila.viz.live_view --scenario tight                               # on a web page: cameras and what it recognised
uv run python -m bila.viz.record_cart cart_tight.mp4 --scenario tight              # as a video
uv run python -m bila.control.cart_demo --scenario tight --labels learned          # read the stickers from the images
uv run python -m bila.control.cart_demo --scenario tight --policy runs/hl/q.npz    # the learned skill selector
```

Books over 1.5 kg (`MAX_KG`) go to the librarian rather than the tray.

## Many rows

`bila/control/eval_reshelve.py` runs the controller alone on many rows, in parallel, with any perception. It skips
rows already in `--out`, so an interrupted run continues, and `--shard i/n` splits the rows for a cluster.
`compare` puts result files side by side: placed per scenario, how the rest ended, and row by row.

```bash
uv run python -m bila.control.eval_reshelve run --perception truth --catalogue --book-pages 4 --book-meshes \
    --scenarios clear tight leaning packed --seeds 0:50 --workers 4 --out results/slide_check/truth.jsonl
uv run python -m bila.control.eval_reshelve compare results/paper/truth.jsonl results/slide_check/truth.jsonl
```

The design studies behind these steps (what was tried and dropped, and why) are in the git history of `README.md`.
