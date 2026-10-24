# `test_perception.py` Pseudocode

Small context: this file checks that the perception layer reports shelf geometry correctly.

## Helper for handmade spine geometry

Inside `spine(t, h, lean, yaw, base)`:

- Build a simple rectangle representing a book spine.
- Rotate it by the requested lean and yaw.
- Shift it by the requested base position.
- Return the four spine corners.

## Test: book estimate recovers size and angles

Inside `test_book_from_spine_recovers_size_and_angles(lean, yaw)`:

- Build a synthetic spine with known thickness, height, lean, and yaw.
- Convert it into a `BookEstimate`.
- Check that the recovered thickness matches.
- Check that the recovered height matches.
- Check that the recovered lean matches.
- Check that the recovered yaw matches.

## Test: slice x of a leaning spine

Inside `test_slice_x_of_a_leaning_spine()`:

- Build a spine leaning by 30 degrees.
- Slice it halfway up the book.
- Compute the expected sideways shift.
- Check that the left edge moved by the expected amount.
- Check that the width at that slice is correct.
- Check that slicing above the book top returns nothing.

## Test: gap profile of a V-shaped gap

Inside `test_gap_profile_of_a_v_shaped_gap()`:

- Build a left book that leans into the slot.
- Build a right upright neighbor.
- Measure the gap profile at several heights.
- Check that:
  - the gap is full width at the board,
  - the gap narrows higher up,
  - above both books the width becomes the panel-to-panel distance.

## Helper for comparing with sampled rows

Inside `placed_gap(row)`:

- Read the left and right neighbors of the target slot.
- Compute the true face-to-face gap from their placed positions.
- Return that gap.

## Fixtures

Inside `clear_scene()`:

- Build one settled clear scenario.
- Return model, data, and row.

Inside `leaning_scene()`:

- Build one settled leaning scenario.
- Return model, data, and row.

## Test: slot offsets name the neighbours

Inside `test_slot_offsets_name_the_neighbours(clear_scene)`:

- Run ground-truth perception on the clear scene.
- Check that the slot offsets run from the left side up to the right side with zero omitted.
- Check that slot offset `-1` matches the true left neighbor thickness.
- Check that slot offset `+1` matches the true right neighbor thickness.
- Check that no book uses slot offset `0`.

## Test: upright books match the generator

Inside `test_upright_books_match_the_generator(clear_scene)`:

- Run ground-truth perception on the clear scene.
- Check that the source is `ground_truth`.
- Check that the time stamp equals the simulation time.
- For each perceived book and true sampled book:
  - check thickness,
  - check height,
  - check that lean and yaw are near zero,
  - check that spine depth matches the generator,
  - check that depth equals the depth prior rather than the true simulator depth,
  - check that visibility equals one.

## Test: clear gap matches the generator

Inside `test_clear_gap_matches_the_generator(clear_scene)`:

- Run ground-truth perception on the clear scene.
- Keep only gap heights below both neighboring book tops.
- Check that those measured gaps match the placed gap.
- Check that the minimum gap width also matches the placed gap.

## Test: leaning neighbour closes the gap toward its top

Inside `test_leaning_neighbour_closes_the_gap_toward_its_top(leaning_scene)`:

- Run ground-truth perception on the leaning scene.
- Determine which neighbor was pushed to lean.
- Read that leaning neighbor from the estimate.
- Check that it leans into the slot by more than a small threshold.
- Check that the gap near the board is close to the originally placed gap.
- Find the top height up to which the leaning book intrudes.
- Check that the gap widths up to that height are non-increasing.
- Check that the top of that range is meaningfully narrower than the bottom.

## Test: noisy perception stays close and is repeatable

Inside `test_noisy_perception_stays_close_and_is_repeatable(leaning_scene)`:

- Build a ground-truth perception source.
- Wrap it twice with noisy perception using the same random seed.
- Estimate the scene with both wrappers.
- Also estimate the exact scene.

- Check that the noisy source label is `noisy`.
- Check that the two noisy estimates are identical to each other.

- For each noisy book and exact book:
  - check that the slot offset stayed the same,
  - check that the noisy corners stayed within a reasonable bound,
  - check that the noisy lean stayed within a reasonable bound.

- Check that every noisy gap width is either:
  - zero because the gap was too small to resolve, or
  - at least the minimum resolvable gap threshold.
