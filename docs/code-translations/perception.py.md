# `perception.py` Pseudocode

Small context: this file defines what the robot is allowed to know about the shelf from a front-facing perception view.

## Main idea

Represent the row using only things a camera could plausibly estimate.

Do not expose hidden simulator state such as:

- true book depth,
- mass,
- friction,
- contact forces,
- hidden geometry behind the spines.

## Data structures

Define `BookEstimate` to store:

- where the book is relative to the slot,
- the four spine corners,
- estimated thickness,
- estimated height,
- estimated lean,
- estimated yaw,
- average spine depth position,
- assumed depth prior,
- visibility score.

Define `ShelfEstimate` to store:

- shelf pose,
- target board top height,
- front edge position,
- clear height,
- shelf depth,
- side panel positions,
- visible books,
- the heights where the gap is measured,
- the gap widths at those heights,
- the time stamp,
- the source name.

Inside `ShelfEstimate.book(slot_offset)`:

- Search the stored books.
- Return the one whose slot offset matches.
- If none matches, return nothing.

Inside `ShelfEstimate.min_gap_width(up_to)`:

- Keep only gap measurements up to the requested height.
- If there are no such measurements, return not-a-number.
- Otherwise return the smallest gap width in that range.

## Geometry helpers

Inside `book_from_spine(corners, slot_offset, depth, visible)`:

- Convert the corners into an array.
- Estimate the spine direction across the book by averaging the bottom edge and top edge.
- Estimate the up direction by averaging the left edge and right edge.
- Compute thickness from the length of the across direction.
- Compute height from the length of the up direction.
- Compute lean from how much the up direction shifts along shelf x.
- Compute yaw from how much the across direction rotates in the shelf x-y plane.
- Compute average spine y position from the corners.
- Return a `BookEstimate`.

Inside `slice_x(corners, z)`:

- Look at each edge of the spine face.
- For each edge:
  - If the edge crosses the requested height:
    - Compute the x position where it crosses.
  - If the edge lies exactly at that height:
    - Use both x endpoints.
- If no edge reaches that height:
  - Return nothing.
- Otherwise:
  - Return the minimum and maximum x values.

Inside `gap_profile(books, panel_x, board_top, heights)`:

- For each requested height above the board:
  - Convert that height into absolute shelf z.
  - Find every visible left-side book slice at that z.
  - Find every visible right-side book slice at that z.
  - Include left or right side panel positions if available.
  - If both sides exist:
    - Compute free width as the nearest right boundary minus the nearest left boundary.
    - Clamp the width to zero if needed.
  - If one side is missing:
    - Record the width as not-a-number.
- Return all widths as an array.

## Perception source interface

Define a `Perception` protocol:

- Any perception source must provide `estimate(data)`.

## Ground-truth perception

Inside `GroundTruthPerception.__init__`:

- Store the model and row.
- Build a shelf-frame helper.
- Find the MuJoCo geom for every row book.
- Set the assumed book depth using the depth prior, capped so it still fits the shelf.
- Store the gap heights.
- Compute each book’s slot offset relative to the target slot.

Inside `GroundTruthPerception.spine_face(data, geom)`:

- Build the four local corners of the book’s spine face.
- Transform those corners into world coordinates using the geom pose.
- Transform them again into the shelf frame.
- Check whether the spine normal faces enough toward the camera side.
- Return the spine corners and a visibility value.

Inside `GroundTruthPerception.estimate(data)`:

- For each row book:
  - Read its spine corners from the simulator.
  - Convert those corners into a `BookEstimate`.
- Build a shelf pose matrix.
- Set panel positions from the row width.
- Compute the gap profile from the perceived books.
- Return a `ShelfEstimate` with source set to `ground_truth`.

## Noisy perception

Inside `NoisyPerception.__init__`:

- Store the wrapped perception source.
- Store the random number generator.
- Store the corner noise standard deviation.
- Store the minimum resolvable gap size.

Inside `NoisyPerception.estimate(data)`:

- Get the base estimate from the wrapped source.
- For each perceived book:
  - Add Gaussian noise to the four spine corners.
  - Recompute the book estimate from those noisy corners.
- Recompute the gap profile using the noisy books.
- For any gap narrower than the resolution threshold:
  - Set that gap width to zero.
- Return an updated shelf estimate with source set to `noisy`.
