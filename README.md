# BILA: Book Insertion Library Assistant

A simulated library robot that puts returned books back on the shelf. A Franka Panda in MuJoCo makes room in the
book's row (it stands a leaning neighbour up, widens a tight gap or shoves a packed row aside), then slides the book
in, deciding from a wrist RGB-D camera. On a book-return cart it also reads the call numbers to find the slot.
CS4278 project, Team 44. The report is [report/main.pdf](report/main.pdf).

## Setup

```bash
git clone --recurse-submodules https://github.com/Zhannyhong/bila
cd bila
uv sync          # add --group rl for the learned perception and PPO (torch)
# trained perception weights (spine segmenter, call-number reader) -> bila_weights/{segmenter,labels}/
curl -L -O https://github.com/Zhannyhong/bila/releases/download/weights-v1/bila_weights.zip
unzip bila_weights.zip
# real book covers from the PBBM submodule (for --book-meshes) -> generated_assets/books/
uv run python -m bila.sim.dae_to_obj --src PBBM-robot-librarian/librarian_common/tams_books_world/meshes
```

On macOS, anything with a viewer (`--view`) runs under `mjpython` instead of `python`. If `mjpython` fails with
`Library not loaded: @rpath/libpython3.12.dylib`, link the library into the venv once:

```bash
ln -sf "$(uv run python -c 'import sysconfig; print(sysconfig.get_config_var("LIBDIR"))')/libpython3.12.dylib" .venv/
```

## Run it

```bash
uv run mjpython -m bila.control.cart_demo --scenario tight --view        # the cart: read the labels, make room, insert
uv run python -m bila.viz.live_view --scenario tight                     # the same on a web page, with the cameras
uv run python -m bila.viz.record_cart cart_tight.mp4 --scenario tight    # the same as a video
uv run python -m bila.control.insert_book --scenario tight --seed 0      # one row, the controller alone
```

The scenarios are `clear`, `tight`, `leaning` and `packed`.

## Reproducing the report

Table 1, with exact perception and then with the camera (each skips rows already in its `--out`, so an interrupted
run continues):

```bash
uv run python -m bila.control.eval_reshelve run --perception truth \
    --catalogue --book-pages 4 --book-meshes \
    --scenarios clear tight leaning packed --seeds 0:50 \
    --out results/slide_check/truth.jsonl
uv run --group rl python -m bila.control.eval_reshelve run \
    --perception camera \
    --segmenter bila_weights/segmenter/segmenter.pt \
    --catalogue --book-pages 4 --book-meshes \
    --scenarios clear tight leaning packed --seeds 0:50 \
    --out results/slide_check/camera.jsonl
```

| Script | Runs |
| --- | --- |
| `results/paper/run.sh` | Table 1, the fiducial comparison and the residual's held-out test |
| `results/find_hard.sh`, `results/train_mix.sh` | The residual's training pools, then its training (`runs/insertion_mix`) |
| `results/hl.sh`, `results/hl_eval.sh` | The skill selector's fit (`runs/hl/q.npz`), then its comparison with the rules |
| `results/hl_cart.sh` | The rules and the selector on the cart |

- Reruns write to `results/slide_check/` (git-ignored). The rows the report was computed from are kept in
  `results/paper/` and `results/`, and `notebooks/08_paper_draft.ipynb` recomputes the report's numbers from them.
- `runs/` is not in the repo: train the residual and fit the selector before the steps that use them.
- Table 1 alone takes about 140 CPU-hours, so `--seeds 0:1` is a quick check.
- On one machine, reruns repeat bit for bit (checked on 20 exact-perception rows and a camera row). On another
  machine, a row's measurements can differ by micrometres, and a row that shoves books by millimetres or in its
  outcome: sliding a run of touching books for a minute or more amplifies the smallest numerical difference. (Of
  seeds 0 to 4, only packed seed 0 changed outcome in a rerun on another Mac.)

## Layout

| Path | What it holds |
| --- | --- |
| `bila/sim/` | The scene: bookcase, target row, Panda, real book covers ([docs/simulator.md](docs/simulator.md)) |
| `bila/control/` | The controller, the cart and evaluation over many rows ([docs/controller.md](docs/controller.md)), and the skill selector ([docs/learning.md](docs/learning.md)) |
| `bila/vision/` | The wrist camera and everything that turns its images into an estimate of the row ([docs/perception.md](docs/perception.md)) |
| `bila/rl/` | The residual PPO: environment, observation, training, hard cases ([docs/learning.md](docs/learning.md)) |
| `bila/viz/` | Viewers, the live web view, the video recorder, figures |
| `tests/` | `uv run pytest` |
| `notebooks/` | Walkthroughs and analyses (`08_paper_draft.ipynb` recomputes the report's results) |
| `results/` | The recorded result rows and the scripts that make them |
| `report/` | The report's source and PDF |
| `docs/` | How each part works, plus [calibration and insertion](docs/calibration-and-insertion.md) and [why masks](docs/explain-segmenter-choice.md) |
| `mujoco_menagerie/`, `PBBM-robot-librarian/` | Submodules: the Panda model, and the real books' models and covers |

The design studies behind the controller and the earlier results are in the git history of this README.
