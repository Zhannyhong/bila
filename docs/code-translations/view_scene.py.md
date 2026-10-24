# `view_scene.py` Pseudocode

Small context: this file opens a settled shelf scene in the MuJoCo viewer so you can inspect it visually.

## Pseudocode

Define `main`.

- Create a command-line parser.
- Add arguments for:
  - scenario,
  - random seed,
  - Panda XML path,
  - whether to hide the robot.
- Parse the arguments.

- Create a random number generator from the seed.

- Build one settled episode by calling `make_episode`.
  - If `--no-robot` was given:
    - do not attach the Panda.
  - Otherwise:
    - attach the Panda XML.

- Print a short summary:
  - which scenario was built,
  - how many shelf levels there are,
  - which level is the target,
  - how many books are in the target row.

- For each book in the row:
  - Measure its tilt angle.
  - If the tilt is larger than a small threshold:
    - print that the book is leaning and by how much.

- Print a note telling the user that the camera starts on the gap.

- Build a scene name string from the scenario and seed.

- Launch the MuJoCo viewer.
  - Pass a loader function that returns:
    - the model,
    - the settled data,
    - the scene name.

If this file is being run directly:

- Call `main`
