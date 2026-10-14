"""
Open a bookshelf scenario in the MuJoCo viewer, with the books already
settled (so leaning and collapsed rows show their slanted books).

    uv run python view_scene.py --scenario leaning --seed 0
    uv run python view_scene.py --scenario collapsed --seed 3 --no-robot
    uv run mjpython view_scene.py --scenario leaning --seed 0 --show-estimate [--noisy]

Why this is needed: the scene XML starts with every book upright. The lean
only appears after `settle()` pushes the chosen books over and lets physics
run, so this script settles first and then opens the viewer on that state.

--show-estimate draws what the robot knows (perception.py's ShelfEstimate)
over the scene: each book as a see-through box (orange: the slot's
neighbours), its spine corners, and the gap at each measured height (green
where the target book fits, red where it doesn't). It also saves the front and
top view to scenes/. It uses the passive viewer, so on macOS run it with
mjpython.
"""

import argparse
import os
import time

import numpy as np
import mujoco
import mujoco.viewer

from shelf_scene import SCENARIOS, ShelfConfig, book_tilt_deg, make_episode
from visualize import draw_in_viewer, estimate_row, plot_estimate

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_PANDA = os.path.join(HERE, "mujoco_menagerie", "franka_emika_panda", "panda.xml")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenario", choices=SCENARIOS, default="leaning")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--panda", default=DEFAULT_PANDA, help="Menagerie panda.xml")
    ap.add_argument("--no-robot", action="store_true", help="show the bookcase only")
    ap.add_argument("--show-estimate", action="store_true", help="draw what the robot knows over the scene")
    ap.add_argument("--noisy", action="store_true", help="with --show-estimate: the noisy estimate")
    a = ap.parse_args()

    rng = np.random.default_rng(a.seed)
    model, data, row, report = make_episode(ShelfConfig(), a.scenario, rng,
                                            panda_xml=None if a.no_robot else a.panda)

    print(f"{a.scenario}: {len(row.level_heights)} levels, target level {row.target_level} "
          f"(board top {row.target_board_top:.2f} m), {len(row.books)} books in the row")
    for b in row.books:
        tilt = book_tilt_deg(data.xquat[model.body(b.name).id])
        if tilt > 3:
            print(f"  {b.name} leans {tilt:.0f} deg")
    print("The camera starts on the gap. Rendering > Camera > scene_cam in the left panel shows the whole target row.")

    if a.show_estimate:
        show_estimate(model, data, row, a)
        return

    # Opens on the settled state and keeps simulating. Handed over through a
    # loader with a name, the model counts as a newly opened file, which is
    # when the viewer aims its camera with the scene's defaults (at the gap).
    # Passed directly, it would keep its own default camera instead.
    name = f"{a.scenario} seed {a.seed}"
    mujoco.viewer.launch(loader=lambda: (model, data, name))


def show_estimate(model, data, row, a):
    """Save the front and top view of the estimate, then open the passive
    viewer with the estimate drawn over the scene."""
    est, truth = estimate_row(model, data, row, a.noisy, a.seed)
    target = (row.target_book.thickness, row.target_book.height)
    path = os.path.join(HERE, "scenes", f"estimate_{a.scenario}_seed{a.seed}{'_noisy' if a.noisy else ''}.png")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plot_estimate(est, truth, target, path)
    print(f"{est.source} estimate: narrowest gap up to the target book's height is "
          f"{1000 * est.min_gap_width(target[1]):.0f} mm, for a {1000 * target[0]:.0f} mm book. Saved {path}")

    # The passive viewer only aims its camera with the scene's defaults (at
    # the gap) when told to, so do it here.
    with mujoco.viewer.launch_passive(model, data) as viewer:
        with viewer.lock():
            mujoco.mjv_defaultFreeCamera(model, viewer.cam)
            draw_in_viewer(viewer.user_scn, est, target)
        while viewer.is_running():
            start = time.perf_counter()
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(max(0.0, model.opt.timestep - (time.perf_counter() - start)))


if __name__ == "__main__":
    main()
