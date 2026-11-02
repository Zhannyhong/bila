"""Record cart_demo from a scene camera, row_cam and the hand's wrist_cam, side by side, to an mp4 (mediapy, needs ffmpeg).

    uv run python notebooks/record_cart.py notebooks/cart_scan_demo.mp4 --scenario tight --seed 0

or from Python: record(out, scenario, seed) returns out.
"""
import argparse, builtins, contextlib, pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))   # repo root: cart_demo
import mediapy, mujoco, mujoco.viewer, numpy as np
from PIL import Image, ImageDraw
import cart_demo

W, H, FPS = 1280, 960, 30     # the scene's offscreen buffer is 1280x960 (shelf_scene)
T_MAX = 300.0               # sim seconds: stop here even if the demo hasn't finished


def record(out, scenario="tight", seed=0):
    step, done = [""], [False]

    def show(*a, **k):                               # keep the latest step name; spot the demo's last line
        line = " ".join(map(str, a))
        if " s  " in line:
            step[0] = line.split(" s  ", 1)[1].strip()
        done[0] |= line.startswith("Close the viewer")
        print_(*a, **k)

    class Recorder:                                  # stands in for mujoco.viewer's passive viewer
        def __init__(self, model, data):
            self.d, self.n = data, 0
            self.cam = mujoco.MjvCamera()            # main() aims this one; the recording uses its own
            self.r = mujoco.Renderer(model, H, W)
            self.scene = mujoco.MjvCamera()
            self.scene.lookat[:] = data.xpos[model.body("lift").id] + [0.2, 0, 0.3]
            self.scene.azimuth, self.scene.elevation, self.scene.distance = -35, -18, 2.6

        def lock(self):
            return contextlib.nullcontext()

        def is_running(self):
            return not done[0] and self.d.time < T_MAX

        def sync(self):
            self.n += 1
            if self.n % 2:                           # Runner draws at 60 fps; keep 30
                return
            self.r.update_scene(self.d, self.scene); a = self.r.render()
            self.r.update_scene(self.d, "row_cam"); b = self.r.render()
            self.r.update_scene(self.d, "panda/wrist_cam"); c = self.r.render()
            img = Image.fromarray(np.concatenate([a, b, c], 1)); dr = ImageDraw.Draw(img)
            dr.text((10, 10), f"scene camera   t = {self.d.time:5.2f} s   {step[0]}", fill=(255, 255, 255))
            dr.text((W + 10, 10), "row_cam (on the lift carriage)", fill=(255, 255, 255))
            dr.text((2 * W + 10, 10), "wrist_cam (on the hand)", fill=(255, 255, 255))
            video.add_image(np.asarray(img))

    print_, launch, argv = builtins.print, mujoco.viewer.launch_passive, sys.argv
    builtins.print, mujoco.viewer.launch_passive = show, Recorder
    sys.argv = ["cart_demo.py", "--view", "--scenario", scenario, "--seed", str(seed)]
    try:
        with mediapy.VideoWriter(out, (H, 3 * W), fps=FPS) as video, contextlib.suppress(SystemExit):
            cart_demo.main()
    finally:
        builtins.print, mujoco.viewer.launch_passive, sys.argv = print_, launch, argv
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out")
    ap.add_argument("--scenario", choices=("clear", "tight", "leaning", "packed"), default="tight")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    record(a.out, a.scenario, a.seed)
