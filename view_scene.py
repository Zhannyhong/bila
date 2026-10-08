"""
Open a bookshelf scenario in the MuJoCo viewer, with the books already
settled (so leaning and collapsed rows show their slanted books).

    uv run python view_scene.py --scenario leaning --seed 0
    uv run python view_scene.py --scenario collapsed --seed 3 --no-robot

Why this is needed: the scene XML starts with every book upright. The lean
only appears after `settle()` pushes the chosen books over and lets physics
run, so this script settles first and then opens the viewer on that state.
"""

import argparse
import os

import numpy as np
import mujoco
import mujoco.viewer

from shelf_scene import SCENARIOS, ShelfConfig, book_tilt_deg, make_episode

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_PANDA = os.path.join(HERE, "mujoco_menagerie", "franka_emika_panda", "panda.xml")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenario", choices=SCENARIOS, default="leaning")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--panda", default=DEFAULT_PANDA, help="Menagerie panda.xml")
    ap.add_argument("--no-robot", action="store_true", help="show the bookcase only")
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
    print("Tip: in the viewer's left panel, Rendering > Camera > scene_cam shows the target row.")

    mujoco.viewer.launch(model, data)   # opens on the settled state and keeps simulating


if __name__ == "__main__":
    main()
