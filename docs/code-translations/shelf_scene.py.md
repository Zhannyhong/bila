# `shelf_scene.py` Pseudocode

Small context: this file generates bookshelf scenes and turns them into MuJoCo episodes.

## Main constants and configuration

Define the scenario names:

- clear,
- tight,
- leaning,
- collapsed,
- packed.

Define `ShelfConfig` to hold all adjustable ranges for:

- shelf size,
- number of shelf levels,
- target level height,
- robot base placement,
- book dimensions,
- book density and friction,
- spacing and end slack,
- contact parameters,
- scenario-specific gap and leaning settings.

Define `REPO_DEFAULTS` as one fixed configuration that reproduces the original cubby-style setup.

## Core data structures

Define `Book` to store:

- name,
- thickness,
- depth,
- height,
- mass,
- friction,
- x position along the row,
- color,
- spine inset.

Define `FillerLevel` to store:

- which level it belongs to,
- which books are on that level.

Define `Row` to store one full episode description:

- scenario type,
- shelf dimensions,
- target level geometry,
- shelf pose,
- target row books,
- target slot location,
- target book to insert,
- pushes used during settling,
- filler rows on the other levels.

Inside `Row.to_json()`:

- Convert the row data into a JSON string.

## Sampling helpers

Inside `_u(rng, lohi)`:

- If the low and high values are the same:
  - return that fixed value.
- Otherwise:
  - sample uniformly from the range.

Inside `_make_book(cfg, rng, name)`:

- Sample thickness, depth, height, density, friction, and color.
- Compute mass from density times volume.
- Return a `Book`.

Inside `board_tops(cfg, plinth, level_heights)`:

- Start from the plinth height.
- For each level:
  - add board thickness,
  - record the board top height,
  - add the clear level height.
- Return all board top heights.

Inside `sample_levels(cfg, rng)`:

- Repeatedly sample:
  - number of levels,
  - level heights,
  - plinth height.
- Compute all board top heights.
- Keep only target levels whose board top lies inside the target reach band.
- If at least one valid target level exists:
  - choose one and return the sampled shelf layout.
- If repeated attempts fail:
  - raise an error.

Inside `sample_filler(cfg, rng, level, H, W, D)`:

- Choose how full the filler row should be.
- Keep making upright books until the row budget is full enough.
- Clamp each filler book so it fits the level height and shelf depth.
- Randomly choose whether to pack books from the left side or the right side.
- Place the books one after another along the row.
- Return the filler-level description.

## Main row sampler

Inside `sample_row(cfg, scenario, rng)`:

- Check that the scenario name is valid.

- Sample shelf width and depth.
- Sample shelf levels and choose the target level.
- Compute the target level height.

- Make the target book.
- Clamp the target book so it fits the level height and shelf depth.

- Choose the target gap width:
  - if scenario is clear:
    - set the gap to target thickness times a clear-gap factor.
  - if scenario is tight:
    - set the gap to target thickness times a tight-gap factor.
  - if scenario is leaning:
    - reserve space for a leaning obstruction.
  - otherwise:
    - start with zero gap.

- If the scenario is collapsed:
  - reserve a hole for the collapsed run.

- Compute how much row width remains for ordinary books.

- Keep sampling books until the target row is nearly full.
- If too few books fit:
  - raise an error.

- Choose a target slot index roughly in the middle third of the row.

- If the scenario is collapsed:
  - choose how many books will fall into the hole.

- If the scenario is leaning:
  - choose which neighbor will lean.
  - choose a lean angle.
  - compute how much gap is needed so that the leaning book can physically rest against the far side.
  - cap that gap by the leaning reserve.

- If the scenario is collapsed:
  - choose the lean angle for the falling run.
  - compute how much hole space is needed for a stable collapsed run.
  - cap that hole by the collapsed reserve.

- Decide which row end gets the spare slack.

- Place the books from left to right:
  - if the current position is the target slot:
    - insert the chosen gap,
    - except in the packed scenario where the slot closes.
  - if the current position is after a collapsed run:
    - insert the reserved hole.
  - place each book center.

- Sample a small random shelf position jitter and yaw jitter.

- Define a helper called `nudge`:
  - compute a push strong enough to tip a chosen book,
  - store that push as force plus torque over a duration.

- If the scenario is leaning:
  - add a nudge for the chosen leaning neighbor.

- If the scenario is collapsed:
  - add domino-style nudges for the run of falling books.

- Sample filler rows for the non-target levels.

- For every book:
  - sample how far its spine sits behind the shelf front edge.

- Return a `Row` containing the full episode description.

## Panda robot modification

Define custom finger dimensions:

- finger half-width,
- finger thickness,
- finger length,
- pad length.

Inside `make_panda_with_sensors(panda_xml)`:

- Load the Panda XML.
- Remove any existing keyframe.
- Find the hand body.
- Add a force/torque sensor site at the flange.
- Add a wrist camera.
- Replace the original finger geoms with thin custom bar-and-pad geoms.
- Ensure force and torque sensors exist in the sensor section.
- Write the modified Panda XML next to the original Panda file.
- If the file already exists with identical content:
  - reuse it.
- Return the path to the modified Panda XML.

Define `PANDA_HOME` as the tucked starting arm pose.

## Scene builder

Inside `_f(...)`:

- Convert numbers into a compact MuJoCo-friendly string.

Inside `build_scene(cfg, row, panda_xml)`:

- Compute shelf and target-level geometry.
- Compute where the robot base should be mounted.

- Create the root MuJoCo XML.
- Add compiler, option, and memory settings.

- Set camera statistics so the default viewer starts pointed at the target gap.
- Add visual settings, shadows, and lighting defaults.

- Define MuJoCo defaults for:
  - books,
  - shelf parts.

- Add assets:
  - floor texture,
  - floor material,
  - Panda model asset if requested.

- Build the worldbody.
- Add a light.
- Add the floor plane.

- If a Panda is requested:
  - create a robot mount in front of the shelf,
  - add a cylinder stand,
  - attach the Panda model.

- Build the shelf body:
  - add every shelf board,
  - add the top board,
  - add both side panels,
  - add the back panel,
  - add the plinth,
  - add a fiducial marker,
  - add a site for the shelf frame,
  - add a site for the target slot,
  - add a fixed scene camera aimed at the target row.

- Define a helper called `add_books`:
  - place books on a chosen board top,
  - if the books should be free-moving:
    - give them free joints,
    - record their initial pose for the keyframe.

- Add the target-row books first.
- Add the filler-row books next.

- Build the `init` keyframe:
  - include the Panda home pose if the Panda exists,
  - include the pose of every free-moving book.

- Indent the XML and return it as a string.

## Settling and stability checking

Inside `book_tilt_deg(quat)`:

- Convert the book orientation into an angle from upright.
- Return that angle in degrees.

Inside `settle(model, data, row, settle_time, check_time)`:

- Reset the simulation to the `init` keyframe.
- Find the body ids of the target-row books.
- Find the body ids of any movable filler books.
- For the settle phase:
  - apply any scripted nudges at their scheduled times,
  - step the simulation.
- Clear any remaining applied forces.
- Zero the velocities.

- Record the positions of every movable book.

- For the stability-check phase:
  - step the simulation,
  - track the largest drift from the recorded positions.

- Measure the tilt of every movable book.

- Return:
  - whether the scene is stable,
  - maximum drift,
  - all tilt values,
  - maximum tilt.

Inside `make_episode(cfg, scenario, rng, panda_xml, max_tries)`:

- Repeatedly:
  - sample a row,
  - build the MuJoCo scene XML,
  - compile it,
  - create MuJoCo data,
  - settle the scene.
- If the scene is stable:
  - return model, data, row, and the settle report.
- If repeated attempts fail:
  - raise an error.

## Command-line entry point

Define `main`.

- Create a command-line parser.
- Add arguments for:
  - scenario,
  - seed,
  - Panda XML path,
  - whether to use repository defaults,
  - output folder.
- Parse the arguments.

- Create the output folder.
- Choose either the repo defaults or the normal config.
- Decide which scenarios to generate.

- For each chosen scenario:
  - create a random number generator from the seed,
  - sample a row,
  - build the XML,
  - write the XML file,
  - write the JSON metadata file,
  - print a summary line.

- If MuJoCo is installed:
  - compile the XML,
  - settle the scene,
  - print whether it was stable.

If this file is being run directly:

- Call `main`
