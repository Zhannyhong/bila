# `xiaomi_vla_scenario.py` Pseudocode

Small context: this file packages one bookshelf episode into a VLA-style task description for high-level skill selection.

## Data structures

Define `MacroSkill` with:

- skill name,
- short description,
- backend function name.

Define `XiaomiVLAScenario` with:

- scenario id,
- model id,
- task instruction,
- initial route,
- observation sources,
- state summary,
- allowed skills,
- success criteria,
- controller bindings,
- notes.

Inside `XiaomiVLAScenario.to_dict()`:

- Convert the dataclass to a dictionary.
- Convert each allowed skill to a dictionary too.
- Return the finished dictionary.

## Main scenario builder

Inside `build_xiaomi_vla_leaning_insert_scenario(...)`:

- Ask the runtime to observe the current scene and choose a route.
- If the chosen route is not `leaning`:
  - raise an error.

- Convert the observed leaning angle into degrees.
- Read the target book.

- Build a `state_summary` dictionary containing:
  - seed,
  - target level,
  - left neighbor label,
  - right neighbor label,
  - target book label,
  - target book dimensions,
  - observed minimum gap,
  - seat rule,
  - observed lean into the slot,
  - whether the leaning book must be fixed first,
  - the recommended next step.

- Build a list of allowed macro-skills:
  - recover the leaning slot,
  - re-observe the gap,
  - insert if clear,
  - insert if tight,
  - abort and ask for help.

- Build and return a `XiaomiVLAScenario` containing:
  - a scenario id based on the seed,
  - the requested model id,
  - either the provided instruction or a generated instruction,
  - the initial route,
  - the available observation sources,
  - the state summary,
  - the allowed skills,
  - success criteria,
  - controller bindings,
  - notes explaining that the VLA is a high-level skill selector.

## Demo scenario helper

Inside `build_demo_scenario(...)`:

- Create the default shelf configuration.
- Build one leaning episode.
- Pass that episode into `build_xiaomi_vla_leaning_insert_scenario`.
- Return the finished scenario package.

## Command-line entry point

Define `main`.

- Create a command-line parser.
- Add arguments for:
  - seed,
  - Panda XML path,
  - perception mode,
  - model id,
  - optional instruction override,
  - optional output path.
- Parse the arguments.

- Build the demo scenario using the parsed arguments.
- Convert the scenario to JSON text.

- If an output path was provided:
  - write the JSON to that file.
- Otherwise:
  - print the JSON.

If this file is being run directly:

- Call `main`
