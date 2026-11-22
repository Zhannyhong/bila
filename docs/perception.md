# Perception

What the robot knows about the target row is a `ShelfEstimate` (`bila/vision/perception.py`): only what a wrist
RGB-D camera could report. Decision code reads an estimate rather than `MjData`, so the ground truth, a noise model
or the rendered camera can stand behind it without changing that code.

```python
from bila.vision.perception import GroundTruthPerception
est = GroundTruthPerception(model, row).estimate(data)
est.book(-1).lean, est.book(+1).spine_y        # the slot's left and right neighbours
est.min_gap_width(0.25)                        # narrowest free width up to 25 cm above the board
```

- **Each book is its spine face**: four corners in the shelf frame and what follows from them (thickness, height,
  lean, swivel, how far out the spine sits). `slot_offset` names it relative to the slot, -1 and +1 being the
  neighbours.
- **The gap** is the free width at the spine plane at 8 heights above the board, between everything left and right
  of the slot, side panels included. A leaning neighbour shows as a gap that narrows with height.
- **Not measured**: a book's depth is a prior (17 cm, capped to fit the shelf), since the camera only sees spines.
  Mass, friction, contact forces and velocities are left out (they are for the critic only).

## The wrist camera

`bila/vision/camera.py` renders RGB-D images in the conventions of a real pipeline (OpenCV, RealSense), so code
written against them carries over to real images.

```python
from bila.vision.camera import WristCamera, DepthNoise, CalibrationError, points, project
with WristCamera(model, noise=DepthNoise.d405(), calibration=CalibrationError(), rng=rng) as cam:
    frame = cam.capture(data)       # rgb, depth (m), seg (geom ids), K, pose, stamp; 1280 x 720, ~89° wide like a D405
xyz = points(frame, mask)           # world points for the pixels in mask that have depth
```

- Camera frame x right, y down, z forward. Depth is along z, and 0 means no reading. Images come out upright (the
  camera is rolled 180° at the inspection pose, and `capture` turns the image round).
- **Basic noise** (`DepthNoise()`): depth error growing with distance squared (Intel's D400 formula), independent per
  pixel, and no reading at object edges.
- **Realistic noise** (`--camera-noise realistic`, the default for `--perception camera`):
  - Stereo errors (`DepthNoise.d405()`): a 0.1 px matching error smoothed over 2 px, a disparity offset fixed per
    camera, the nearer surface's depth spreading 2 px over the farther one at depth edges, and no reading where the
    second imager can't see the surface.
  - A calibration error (`CalibrationError()`) drawn once per camera: where the robot thinks the camera sits on the
    hand is off by 1.5 mm and 0.3° per axis, the focal length by 0.3% and the image centre by 1 px. Images are taken
    from where the camera really is, while `frame.pose` and `frame.K` are what the robot believes.
- Each kind of error has its own random stream split off `rng`, so one seed gives the same calibration error with or
  without the stereo errors, and comparisons across noise models are paired.

## From images to an estimate

`bila/vision/rgbd_perception.py` builds a `ShelfEstimate` from wrist-camera images, the same kind of estimate as the
ground truth's.

- Per book: its pixels (from the simulation's object ids, or from the learned segmenter below), the 3D points on the
  target level that face out of the shelf (the spine), a plane through them and the smallest rectangle around them in
  it. Its corners are the spine corners.
- Books are named by where they stand relative to the slot: by counting from a side panel when the whole row is in
  view (the shelf map knows how many books stand left of the slot), else from the slot's place on the map.
- Several images (the sweep 30 cm either side of the slot and in front of it) are fused: each book comes from the
  view nearest it along the row.
- Every image is corrected from the ArUco markers on the boards' front edges (`vision/fiducials.py`): the marker
  nearest the slot shows how far the camera's view is shifted along the shelf's front, and that shift is removed
  from the estimate. `eval_reshelve run --no-fiducials` turns it off. See
  [calibration and insertion](calibration-and-insertion.md).
- The inspection pose (`bila/control/inspection.py`) looks straight in from 15 cm above the target board, as far back
  as the arm allows (about 50 cm from the spines).

## Learned book masks

`bila/vision/learned_segmenter.py` finds the book masks in the colour image instead of reading the simulation's
object ids: Mask R-CNN (torchvision, fine-tuned from COCO, one class "book"). Learned masks need the robust spine fit
(`robust_fit=True`), which drops strips of a mask that bled onto the next spine. Why masks rather than boxes:
[explain-segmenter-choice.md](explain-segmenter-choice.md). The trained weights are in the weights-v1 release (see the
README). To train them again (`uv sync --group rl` first):

```bash
# training data: shelf and cart scenes, about 2900 and 300 images (on Linux, set MUJOCO_GL=egl)
uv run python -m bila.vision.make_vision_dataset --out ~/bila_data/train --first-seed 2000 --scenes 670 --workers 5
uv run python -m bila.vision.make_vision_dataset --out ~/bila_data/val --first-seed 9000 --scenes 70 --workers 5
uv run python -m bila.vision.learned_segmenter train --data ~/bila_data/train --val ~/bila_data/val --out bila_weights/segmenter
uv run python -m bila.vision.learned_segmenter eval --data ~/bila_data/val --weights bila_weights/segmenter/segmenter.pt
```

The full recipe trains 8 epochs (about 2 hours on a 4 GB laptop GPU). `notebooks/04_train_segmenter.ipynb` walks
through a shorter run. The segmenter picks CUDA, then Apple's Metal, then the CPU (`--segmenter-device` overrides).

## Reading call-number stickers

`bila/vision/label_reader.py` finds and reads the call-number stickers for the cart (`cart_demo --labels learned`). A
small segmentation network (LR-ASPP MobileNetV3) marks each sticker, the sticker is straightened from its corners, a
small CRNN reads its two lines in all four orientations, and the reading is matched to the catalogue. Under 0.99
confidence it reports "unreadable" rather than guess, since a misread puts a book in the wrong place. To train it again
(on the same data as the segmenter):

```bash
uv run python -m bila.vision.label_reader train-finder --data ~/bila_data/train --out bila_weights/labels
uv run python -m bila.vision.label_reader train-reader --data ~/bila_data/train --val ~/bila_data/val --out bila_weights/labels --epochs 10
uv run python -m bila.vision.label_reader eval --data ~/bila_data/val --weights bila_weights/labels
```

## Noise models

`NoisyPerception` makes the ground truth look like the camera's estimate without rendering: each book's position,
size, lean and swivel get the camera's measured error for its distance from the slot (`CAMERA_ERRORS`), and books
beyond the camera's reach along the row are left out. `NoisyPerception.learned` imitates the learned-mask pipeline
instead. `bila/vision/eval_perception.py` measures the camera's errors and fits these models:

```bash
uv run python -m bila.vision.eval_perception collect      # stores the camera's estimates of 48 scenes in scenes/
uv run python -m bila.vision.eval_perception report       # error tables
uv run python -m bila.vision.eval_perception calibrate    # CAMERA_ERRORS and the camera's reach, for NoisyPerception
uv run python -m bila.vision.eval_perception validate     # NoisyPerception's errors next to the camera's
```

## Seeing what the robot knows

`bila/viz/visualize.py` draws an estimate: a front view (spines, and the gap's width at each height, green where the
book fits) and a top view, with the ground truth dashed when the estimate is noisy.

```bash
uv run python -m bila.viz.visualize --scenario leaning --seed 0 --noisy          # front and top view -> scenes/*.png
uv run python -m bila.viz.visualize --scenario leaning --seed 0 --camera         # the camera image with the estimate outlined
uv run python -m bila.viz.visualize --scenario leaning --seed 3 --learned        # the learned masks of the first look
```
