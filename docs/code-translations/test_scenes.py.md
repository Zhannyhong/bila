# `test_scenes.py` Pseudocode

Small context: this file checks that scene generation really produces the packed-slot and thin-finger conditions the rest of the project assumes.

## Helper

Inside `_geom_names(model)`:

- Return the set of every geom name in the MuJoCo model.

## Test: packed slot is closed

Inside `test_packed_slot_is_closed(seed)`:

- Build one `packed` episode for the given seed.
- Check that the settle report says the scene is stable.
- Run ground-truth perception on the settled row.
- Find the shorter of the two books beside the slot.
- Measure the narrowest gap width up to just below that shorter height.
- Check that this minimum gap is almost zero.

Meaning:

- the packed scenario should not leave a usable slot.

## Test: Panda has thin fingers

Inside `test_panda_has_thin_fingers()`:

- Sample one clear row.
- Build the scene with the Panda attached.

- For each finger body:
  - collect all collision geoms on that finger body.
  - check that there are exactly two collision geoms.
  - check that both collision geoms are boxes.
  - check that the finger width matches the custom thin-finger constant.
  - check that the total finger reach matches the custom finger length.

Meaning:

- the generated Panda model should actually use the custom thin-finger geometry.
