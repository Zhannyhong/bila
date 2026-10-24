# [OPEN] perception-camera

## Goal

Understand and debug how perception currently observes the bookshelf scene, especially from the robot-mounted camera, and determine whether the current camera placement should be changed.

## User Questions

- Can we test and debug perception first?
- Can we observe what the robot camera sees in RGB-D terms?
- Is there a better camera placement for observing the bookshelf?

## Hypotheses

1. The current codebase has enough scene camera and wrist camera definitions to render useful RGB views, but there is no dedicated tool yet to inspect them side by side.
2. The current perception pipeline is mostly geometric and simulator-backed, so the user cannot yet “see” the same representation the controller uses without an added visualization/debug layer.
3. The wrist camera placement is optimized for insertion-time local view, not for broad row understanding, so it may be suboptimal for initial shelf perception.
4. A fixed scene camera may currently provide better whole-row observability than the wrist camera for debugging slot state and leaning books.
5. Small camera placement changes, especially moving the wrist camera less off-axis or farther back, may improve shelf visibility but could trade off occlusion from the hand during contact.

## Current Status

- Session opened.
- Existing camera definitions inspected:
  - `scene_cam` is a fixed external camera aimed at the target row.
  - `wrist_cam` is attached to the Panda hand as `panda/wrist_cam`.
- Current perception implementation confirmed:
  - `perception.py` is geometric and simulator-backed, not image-driven yet.
- Added a debug-only script `debug_perception.py` to render:
  - RGB views,
  - depth previews and raw depth arrays,
  - camera metadata,
  - current `ShelfEstimate` JSON.
- Evidence from first render:
  - `scene_cam` gives a useful whole-row view.
  - `wrist_cam` from the current home pose is mostly self-occluded by the robot and is poor for initial shelf perception.
- Evidence from a simple wrist-camera mount override:
  - changing the camera mount alone slightly improves the top-of-shelf view,
  - but the arm still dominates the image,
  - suggesting that arm observation pose is a bigger factor than camera mount alone.
