# Why the catalogue run did worse, and why calibration error breaks insertion

## Summary

- Looking each book up in the catalogue by its call number makes the books'
  sizes **exact**. The cart demo still succeeded *less* often with it (5 of 12
  runs, against 6 of 12 without it).
- The cause is the wrist camera's **calibration error**. It moves everything
  the camera sees sideways by **4–9 mm**, and the shift is different in each
  view. The book needs about **5–7 mm** of room on each side, so an error that
  size is enough to make it hit a neighbour.
- Without the catalogue, the camera saw every book a few millimetres too thick.
  The gap looked narrower than it was, so the robot widened it before
  inserting. That extra room hid the calibration error. The catalogue removed
  this accidental safety margin.
- **Recommended fix: a fiducial marker on the shelf as a reference.** The
  robot already knows exactly where the marker is, so it can measure how far
  off its view is and correct for it. What makes this work, according to the
  measurements below:
  1. Correct only the **shift along the shelf's front** (sideways and up–down),
     measured from the marker's corners in the colour image. Don't solve for
     the camera's full position and rotation.
  2. The marker has to be **close to the slot in the image**. Put markers
     along the board's front edge, so that one is always within about 10 cm of
     the slot.

## Status: implemented

The markers and the correction are now in the code. This is the recommendation
in section 3.

- **The markers.** `shelf_scene.fiducials()` is the shelf map's record of them.
  There is one ArUco marker every 15 cm along every board's front edge. Each is
  20 mm square and stays below the board's top, so it never hides a spine or a
  sticker.
- **The correction.** `vision/fiducials.py` finds the markers in each image
  with `cv2.aruco`. It takes the whole marker nearest the slot and removes that
  marker's in-plane offset from the camera's position. `make_eyes` turns this
  on by default. The live view logs every correction, and has
  `--no-fiducials` to switch it off.

Results:

| | sideways error per view | cart demo, 12 runs |
|---|---|---|
| no correction | −9 to +3 mm | 5 succeed |
| **in-plane shift from the nearest marker (used)** | **−2.7 to +2.2 mm** in the left and middle views; +6 mm in the right view | **7 succeed**, which is every scene the arm can reach |
| full pose from every marker in view (PnP) | −2.3 to +3.8 mm, but 5–7 mm in/out | 6 succeed |

In the right-hand view, the marker nearest the slot sits at the edge of the
image. Books on the far right are then still about 6 mm off. That doesn't
affect the slot itself, which the middle view sees.

## 1. The two runs

Both runs are cart scene `clear`, seed 0. The held book is 55 mm thick. Its true
gap is 68.4 mm, which leaves 6.7 mm on each side.

All diagrams in this document are top-down views of the target row: x runs
along the row, and the robot is at the bottom.

```
 truth                 |<---------------- 68.4 mm gap --------------->|
               ████████|                                              |████████████
           left book   -2.4                                        66.0   right book
           (14 mm)                                                         (34 mm)
```

**Without the catalogue (succeeded).** The camera fits each spine from the
image. It saw both neighbours about 2.5 mm too thick, so the gap looked
narrower than it is, and the robot decided to widen it before inserting.
Widening pushed the right neighbour further out than needed, so the true gap
ended up with room to spare. The book went in despite the calibration error.

```
 seen by the camera     ██████████|<------ gap looks narrower ------>|██████████████
                                    → "too narrow for 55 mm + margin: widen first"
 after widening         █████████|<----------- wider than it needs ---------->|█████
                                    → inserted with room to spare
```

**With the catalogue (jammed).** Each neighbour's sticker was read, so its
width is now exact. Its position, though, still comes from the camera, which
saw everything 5–7 mm to the left:

```
                        sticker read     true     error
 left neighbour (#190)     -14.3 mm     -9.4 mm   -4.9 mm
 right neighbour (#180)     76.1 mm     83.0 mm   -6.9 mm

 believed               |<------------- 66.4 mm --------------->|
               ████████|                                        |████████████
                     -7.3                                     59.1
 truth                     |<---------------- 68.4 mm ---------------->|
                   ████████|                                           |████████████
                         -2.4                                        66.0
```

The believed gap is the right width but sits about 6 mm too far left. The robot
judged it wide enough, skipped widening, and aimed the book at the believed
centre, 25.9 mm. The true centre is 31.8 mm.

```
 book as aimed            |<----------- 55 mm ----------->|
                        -1.6                             53.4
 left neighbour's face   -2.4    → 0.8 mm overlap: the book hits it
                                   ("the book jammed at the slot's entrance")
```

The catalogue did what it should. The positions were wrong because of the
calibration error, and the catalogue removed the margin that had been
covering for it.

**All 12 runs** (cart demo, scenarios × seeds 0–2):

| | with catalogue | without |
|---|---|---|
| succeeded | 5 | 6 |
| better with the catalogue | tight 1 | |
| worse with the catalogue | clear 0 (jammed), leaning 1 (ended 16 mm short) | |
| failed either way, "no collision-free joint move" | seeds 2 of every scenario, packed 0, packed 1 | same |

The last row is an arm-reach problem in the cart scene. It is separate from
perception.

## 2. Why a few millimetres of calibration error matter

### What the error is

The camera is bolted to the robot's hand. To turn a pixel into a point on the
shelf, the robot has to know exactly where the camera is relative to the hand.
That is called its **hand-eye calibration**. Real calibrations are off by a
couple of millimetres and a fraction of a degree. The simulation draws one such
error per episode (`camera.CalibrationError`): about 1.5 mm and 0.3° per axis,
plus a small lens error.

The two parts of the error behave differently:

- **A position error** moves everything in the image by the same amount.
- **A rotation error** grows with distance, like a lever.

```
      camera, believed  ●
                         ╲  0.3° off
      camera, truly     ● ╲
                         ╲ ╲
                          ╲ ╲                at 0.3 m:   ~1.6 mm
                           ╲ ╲               at 0.5 m:   ~2.6 mm
     shelf ─────────────────●─●──────────    (per axis; three axes add up)
                       true   believed
```

### What it does here

On 12 first looks (3 views each) in cart scenes, the stickers' measured
positions compared with the truth:

| | sideways error per view |
|---|---|
| perfect calibration (depth noise only) | −2 to +3.5 mm |
| with the calibration error | **−9 to +3 mm** |

- **The error differs from view to view.** In one scene it was −8.6, −6.4 and
  −3.5 mm across the three views. The calibration error is fixed relative to
  the hand, but the arm is in a different pose for each view, so what it does
  at the shelf changes.
- **It's not just the stickers.** The camera's own spine fits are shifted by
  the same amount, since both come from the same image and the same believed
  camera pose.

### Why the insertion has no room for it

The insertion is open-loop: the camera can't see while the hand holds the
book. The robot aims for where it *believes* the slot is, and has nothing to
correct with until the force limit stops it. The room it has is half the
difference between the gap and the book:

```
  |<-c->|<----------- book ----------->|<-c->|        c = (gap − book) / 2
  █████ |                              | █████
                                                      clear 0:  c = 6.7 mm
  the calibration error shifts the aim by 4–9 mm  →   the book's edge lands on the neighbour
```

In the clear scenes, `c` was 5.6–6.9 mm. A shift of 4–9 mm uses up most or all
of it, and the rest of the camera noise is added on top.

## 3. Using a shelf fiducial as reference

The scene already has a fiducial: a 50 mm black square on the target board's
front edge, 4 cm from the left side panel (`shelf_scene.py`, geom
`fiducial`). Its position is on the shelf map, so the robot knows exactly
where it is. Today nothing looks for it. The robot simply assumes the shelf's
pose.

The idea: on every look, find the marker in the image, compare where the
camera *says* it is with where the map *says* it is, and correct everything
else in that image by the difference.

### What was tried, and what happened

All three variants were tested on the same first looks, against the sideways
error with no correction (−9 to +3 mm):

| correction from the marker | sideways error left (marker near the slot in the image) | (marker at the image's edge) | in/out error |
|---|---|---|---|
| shift by the offset of the marker's depth points | −3 to +1 mm | −10 to −28 mm when partly in view | worse |
| full camera pose from the marker's 4 corners (PnP) | −2 to +1 mm | +2 to +6 mm | **worse: +3 to +19 mm** |
| **in-plane shift from the marker's corners** | **−1.6 to +1.4 mm** | +4 to +6 mm | unchanged |

- **Depth-point offset fails** because the marker is often only partly in view
  (at the image's edge, or only a few hundred pixels of it), and then the
  average of its depth points isn't its centre.
- **The full pose fails** because a 5 cm square seen from 20–40 cm pins down
  the camera's rotation poorly. The slot is 30–50 cm from the marker, so that
  small rotation error becomes a large position error at the slot (the lever
  in section 2).
- **The in-plane shift** projects the marker's centre, found from its corners
  in the colour image, onto the shelf's front plane. That plane is known, so
  no depth is needed. Sideways and up–down are corrected; in-and-out is left
  alone. It works whenever the marker is near what's being measured.

```
   marker near the slot in the image           marker far from the slot
   ┌────────────────────────────┐              ┌────────────────────────────┐
   │      ▣  ▯▯▯▯ ▯ ▯▯▯▯        │              │▣             ▯▯▯▯ ▯ ▯▯▯▯   │
   │      ↑  slot here          │              │↑             slot here     │
   │ error at ▣ ≈ error at slot │              │ error differs across image │
   └────────────────────────────┘              └────────────────────────────┘
        corrected to ±1.6 mm                       still 4–6 mm off
```

### Recommendation

1. **Put markers along the board's front edge**, roughly every 15–20 cm, so
   that one is close to the slot in every view. A single marker at the shelf's
   end is too far away: it was 28–50 cm from the slot in these scenes, and the
   right-hand view never saw it at all. Give each marker an ID, e.g. small
   AprilTags. The map then holds each marker's ID and position, and the robot
   uses the one nearest the slot in each image.
2. **In each image, correct only the in-plane shift**, measured from that
   nearest marker's corners. Apply it to every point taken from that image:
   spine fits, stickers and the gap profile. In this code that means
   `RGBDPerception.estimate_from_frames` and `ReadingEyes.read`, before
   anything is fused across views.
3. **Use the corners, not the depth.** Corners are sharp in the colour image.
   Depth on a small black square is unreliable.
4. **Skip the correction when no whole marker is in view.** Fall back to the
   uncorrected estimate, and keep a margin for the calibration error, i.e. widen
   the gap first.

With the marker near the slot, the sideways error came down to under about
1.6 mm. That is well inside the 5–7 mm of room a book has on each side.

### What stays

- **About ±2–3 mm comes from the depth camera itself.** It appears even with
  perfect calibration and differs between views. The fiducial doesn't remove
  it.
- **The in-and-out error (+4 to +9 mm) isn't corrected.** It matters less for
  the slide, which stops on contact, but it does affect how far the robot
  pushes the book home.
- **The arm-reach failures** in section 1 are unrelated.

## How these numbers were measured

The measurement scripts were throwaway and aren't in the repo. To reproduce:

- **Success counts:** run `cart_demo.run` on each scene twice, once with
  `ReadingEyes(..., database=...)` and once with the database empty.
- **Error per view:** on each episode's first look, compare every sticker's
  measured centre (`ReadingEyes.seen`, shelf frame) with its `<book>_label`
  geom's true position.
- **The marker corrections:** use the views where the `fiducial` geom shows
  whole, with its corners found from the segmentation image's outline. A real
  robot would find them with a tag detector.
- **"Perfect calibration":** the same, with the camera made without a
  `CalibrationError`.
