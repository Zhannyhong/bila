# Learned components

Two learned pieces sit on top of the controller: a residual PPO policy that corrects the insertion, and a skill
selector that picks the controller's next step. Both need `uv sync --group rl`. The report compares each with the
controller alone.

## Residual PPO for the insertion

The controller prepares the row and holds the book. During the slide and the final push, PPO adds five bounded
corrections to the script's pose: x, y, z, yaw and pitch. Stopping stays scripted. An action is the correction
itself, not a change of it: 0 is the script's pose and ±1 the bounds on that side, approached at 3 mm and 0.4° per
decision at most. The bounds (`insert_book.SLIDE_LO`, `SLIDE_HI`, `PUSH_LO`, `PUSH_HI`) are ±4 mm along the row,
−8 to +2 mm in depth and ±1.5° of yaw and pitch, and the book is never lifted during the slide nor the hand held
back from the script's push.

- **Environment**: `bila/rl/gap_env.py` (`GapEnv`) runs the script in a thread and hands control to the policy in
  the phases chosen with `--control-phases`. A rendered camera's OpenGL context on macOS is tied to the thread that
  made it, so that thread makes the episode and its eyes.
- **Observation** (`bila/rl/observation.py`): fixed-size vectors in the slot's frame. The actor gets what a real robot
  has: the slot's three nearest books on each side, the gap's widths at 8 heights, the free space behind each
  neighbour, the book to insert, the estimate's age, and the robot's joints, gripper, tool pose and wrist
  force/torque. The critic also gets the ground truth and each nearby book's mass, friction and velocity.
- **Reward**: the book's progress toward the accepted placement, less penalties for tilting or lifting it, wrist force,
  the action's size, time and the other books moving, plus a terminal reward. `log.jsonl` and TensorBoard record
  each term.
- **Training** (`bila/rl/train_residual.py`, `bila/rl/residual_ppo.py`): PPO on pools of stored, settled rows,
  validated against the script on identical cases.

### The report's pipeline

The policy trains on rows the camera actually sees, half of them rows the controller fails:

```bash
bash results/find_hard.sh     # candidate rows (layouts from seed 10000), labelled by the controller alone, mixed 50/50
                              # into scenes/gap_pool_<scenario>_mix.pkl and _mix_val.pkl
bash results/train_mix.sh     # train_residual on those pools -> runs/insertion_mix
bash results/paper/run.sh     # among Table 1: test pools from seed 30000, labelled and mixed the same way,
                              # then hard_cases test -> results/slide_check/residual_test.json
```

`bila/rl/hard_cases.py` does the labelling (`label`: which rows the controller alone fails), the mixing (`mix`) and
the held-out test (`test`). Pool creation refuses to overwrite a pool, and `mix` refuses to overwrite its output.

### Training options and outputs

| Option | Meaning |
| --- | --- |
| `--control-phases slide push` | Let PPO correct only the slide and the final push (the report's setting) |
| `--no-learn-stop` | Five continuous actions: the controller decides when each contact stops |
| `--phases widen` | Sample only rows whose first contact is a widening |
| `--perception camera --segmenter W` | Train on the rendered camera with the learned masks |
| `--eval-action-scale` | The action scale used for validation, fixed across checkpoints |

A run writes `runs/<name>/`: `baseline.json` and `val_*.json` (per-case outcomes, with `val_recovered`, `val_lost`
and `val_success_delta` against the script on the same cases), `episodes.jsonl`, `log.jsonl` and TensorBoard logs
(`uv run --group rl tensorboard --logdir runs`). A pool records how its rows were seen, and `GapEnv` refuses a pool
seen with other eyes. An exact case replays with
`GapEnv.reset(options={"scenario": "tight", "geometry_seed": 10000, "variant": 0})`.

## The learned skill selector

`bila/control/high_level.py` lets a policy choose the controller's next step instead of its hand-written rules, from
a few numbers in the camera's estimate. The skills are the controller's. The choices are `insert`, `widen`,
`widen_more` (a margin for what the camera got wrong), `shove`, `stand_up_left`, `stand_up_right`, `set_aside` (after
a jam: back the book out, put it back and look again) and `give_up`. The policy is a Q-function over (features,
action) fitted offline by fitted Q iteration on episodes of the rules with random deviations. Its reward is 1 if the
book ends up placed, less a cost per step.

```bash
bash results/hl.sh            # episodes of the rules with deviations (seeds 5000-5059), then the fit -> runs/hl/q.npz
bash results/hl_eval.sh       # the rules and the selector on the same 100 rows
bash results/hl_cart.sh       # the same on the cart, 12 scenes
uv run python -m bila.control.cart_demo --scenario tight --policy runs/hl/q.npz
```
