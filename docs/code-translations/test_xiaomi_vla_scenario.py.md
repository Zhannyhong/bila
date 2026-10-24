# `test_xiaomi_vla_scenario.py` Pseudocode

Small context: this file checks that a leaning bookshelf episode can be packaged into a valid VLA task description.

## Test: build Xiaomi VLA scenario for a leaning row

Inside `test_build_xiaomi_vla_scenario_for_leaning_row()`:

- Build one leaning episode.
- Build the Xiaomi VLA scenario package from that episode.

- Check that the initial route is `leaning`.
- Check that the chosen model id is preserved.
- Check that the state summary says the leaning slot must be recovered first.
- Check that the observed minimum gap is positive.
- Check that the task instruction contains the word `between`.
- Check that the left-neighbor label appears in the instruction.
- Check that the first two allowed skills are:
  - `recover_leaning_slot`,
  - `reobserve_gap`.

Meaning:

- the package should describe the leaning-row problem correctly,
- and it should present recovery before insertion.

## Test: Xiaomi VLA scenario is JSON serializable

Inside `test_xiaomi_vla_scenario_is_json_serializable()`:

- Build one demo scenario.
- Convert it to a dictionary.

- Check that the initial route is `leaning`.
- Check that the controller bindings include `insert_if_tight`.
- Check that the observation sources include `scene_cam`.
- Check that the payload can be turned into JSON text.

Meaning:

- the scenario package should be exportable and usable by downstream systems.
