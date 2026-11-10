# Why masks and not boxes: how the camera finds a book's edges

## The question

The robot's camera has to work out, for each book, **where the spine is,
how far the book leans, how it is turned, and how thick it is**. From
that it works out the gap the held book has to fit into.

The first step is finding *which pixels belong to which book*. There are
two common ways to do that:

| | **Box** (YOLOv5, as PBBM used) | **Mask** (Mask R-CNN, what we use) |
|---|---|---|
| What the model outputs | A rectangle around each book, always level with the image edges | Exactly which pixels are that book |
| Cost of labelling | Cheap: drag a rectangle | Expensive by hand: trace the outline |
| Fits a leaning book | Badly | Exactly |

## 1. Upright books: boxes are fine

```
   image, seen from the front

   ┌───┐┌───┐┌───┐┌───┐
   │ A ││ B ││ C ││ D │      every spine is straight up,
   │   ││   ││   ││   │      so a rectangle around it
   │   ││   ││   ││   │      holds only that book
   └───┘└───┘└───┘└───┘
```

When spines stand straight, a rectangle around a spine is almost the same
as its outline. **This is the case PBBM was built for.**

## 2. A leaning book: the box takes in its neighbours

```
   B leans right

   ┌───┐   ╱‾‾╱┌───┐
   │ A │  ╱  ╱ │ C │
   │   │ ╱  ╱  │   │
   └───┘╱__╱   └───┘

   the box around B:                  B's mask:
   ┌─────────┐
   │ A ╱‾‾╱ C│  ← includes slices         ╱‾‾╱
   │  ╱  ╱   │    of A, C and the        ╱  ╱
   │ ╱  ╱    │    empty gap             ╱  ╱
   │╱__╱     │                         ╱__╱
   └─────────┘                     only B's pixels
```

A rectangle that has to stay level with the image edges *must* grow to
cover the whole slanted spine. It then also covers parts of the books
beside it and the gap. That's a wedge of pixels that aren't B.

## 3. Why the extra pixels matter here: we measure from depth

Every pixel of the wrist camera also has a **depth**: how far away it
is. Each pixel the model gives for a book becomes a 3D point. The
pipeline then fits:

1. a **flat surface** through those points (the spine's face), and
2. the **smallest rectangle** that holds them (the spine's corners).

Lean, turn, thickness, and the size of the gap all come from that
rectangle.

```
   points from a MASK               points from a BOX

        ● ● ●                          ○ ● ● ●     ○
        ● ● ●                        ○ ○ ● ● ●   ○ ○
        ● ● ●                        ○ ○ ● ● ● ○ ○ ○
        ● ● ●                          ○ ● ● ●   ○
     ┌───────┐                    ┌─────────────────┐
     │ fits B│                    │ stretched by A's│
     └───────┘                    │ and C's points  │
                                  └─────────────────┘
   ● = B   ○ = neighbours / back of the shelf
```

The fitted rectangle **stretches toward the stray points**. The book then
looks thicker, more tilted, or in the wrong place, and the gap looks
smaller or larger than it really is.

We need **millimetre** accuracy: a book has to slide into a gap only a
few mm wider than itself. Even a mask that spills a few pixels onto a
neighbour stretched the fit by 10–50 mm on tilted books. That's why
`robust_fit` exists. A box spills much more than a mask does.

## 4. So how did PBBM manage with boxes?

```
   PBBM's pipeline

   camera image ──► warp the image ──► YOLOv5 boxes ──► average the
                    so the shelf                         depth points
                    looks square-on                      in each box
                                                              │
                                                              ▼
                                                   smooth over many frames
                                                   (cluster book positions)
```

- **They warp each image first.** The shelf is made to look square-on, so
  upright spines line up with the image edges and fit a box well.
- **They only average the points.** A book's centre is roughly right even
  when a few stray points get in. They don't need its exact edges.
- **They smooth over time.** Book positions are clustered over many
  frames, which averages out the errors.
- **Their books were mostly upright.** Warping fixes the *camera's*
  angle, not a book's own lean.
- **They labelled real photos by hand,** so cheap box labels mattered.

## 5. Why masks cost us nothing extra

```
   our pipeline

   simulator ──► renders an image ──► gives every pixel's book id for free
                                               │
                                               ▼
                                     perfect masks to train on
                                               │
                                               ▼
   wrist image ──► Mask R-CNN ──► per-book masks ──► depth points
                                                         │
                                                         ▼
                                         plane + rectangle fit (robust_fit)
                                                         │
                                                         ▼
                                       spine corners, lean, turn, gap
```

The simulator already knows which pixel belongs to which book. So the
one real advantage of boxes, cheap labelling, doesn't apply to us. And
our scenarios (`leaning`, `collapsed`, swivelled books) are exactly where
boxes go wrong.

## 6. What about YOLO-seg (newer YOLO that outputs masks)?

It produces masks too, so it would work. It isn't worth switching yet:

| | Mask R-CNN (now) | YOLO-seg |
|---|---|---|
| Speed | ~0.4–0.5 s per image on the Mac's GPU | Faster |
| Already built, trained, evaluated | Yes (93% of target-row books found) | No |
| Licence | BSD (torchvision) | AGPL-3.0 (Ultralytics) |
| Work to switch | none | convert labels, retrain, redo evaluation |

The camera looks only a few times per episode, so speed isn't the
bottleneck. **Switch if perception ever has to run on every frame,**
for example if a policy reads the camera continuously while it moves.

## Summary

- **Boxes** are level rectangles. They're fine for upright books. On a
  leaning book they take in the neighbours.
- We measure books **from the depth pixels inside the shape**, so extra
  pixels mean wrong measurements.
- PBBM straightened the images, only averaged positions, and had mostly
  upright books, so boxes were good enough for them.
- The simulator gives us masks for free, so we use them.
- **YOLO-seg** is a faster drop-in for later, not a fix for anything now.
