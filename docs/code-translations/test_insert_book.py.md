# `test_insert_book.py` Pseudocode

Small context: this file checks the handoff from a leaning-row recovery stage to the ordinary insertion stage.

## Helper

Inside `_geom_names(model)`:

- Return the set of all geom names in the MuJoCo model.

## Test: rebuilding the insertion scene removes the pusher and preserves state

Inside `test_rebuild_insertion_scene_removes_pusher_and_preserves_state()`:

- Build one leaning insertion episode.
- Check that the original model contains a geom named `pusher`.

- Rebuild the scene into a normal insertion scene while keeping the current simulation state.

- Check that the rebuilt model no longer contains `pusher`.
- Check that rebuilt simulation time matches the old simulation time.
- Check that rebuilt joint positions match the old joint positions.
- Check that rebuilt joint velocities match the old joint velocities.

Meaning:

- temporary recovery hardware should disappear,
- but the physical world state should stay unchanged.

## Test: leaning prep hands off to an insertable route

Inside `test_leaning_prep_hands_off_to_insertable_route()`:

- Build one leaning insertion episode.
- Run the leaning preparation stage.
- Check that the preparation reports it finished.

- Rebuild the scene into a normal insertion scene.
- Run the observation-and-routing logic again.
- Check that the new route is either:
  - `clear`, or
  - `tight`.

Meaning:

- after recovery, the row should no longer be treated as a leaning special case,
- it should become a normal insertable case.
