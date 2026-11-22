"""
A web page that shows the book-return cart at work as it runs: what the wrist
camera sees, with the books the robot recognised drawn over it, next to a
scene camera.

    uv run python -m bila.viz.live_view --scenario leaning --seed 0          # opens http://localhost:8000
    uv run python -m bila.viz.live_view --scenario tight --segmenter bila_weights/segmenter/segmenter.pt
    uv run python -m bila.viz.live_view --scenario tight --label-weights bila_weights/labels   # read stickers with label_reader
    uv run python -m bila.viz.live_view --scenario clear --seed 0 --no-fiducials   # the calibration error uncorrected: it jams

The episode is cart_demo's, started when the page is first opened: the Panda
on the cart's lift reads the return tray's stickers with its wrist camera,
lifts to the held book's row, reads the row's stickers on its first look to
find the slot, makes room, picks the book out of the tray and inserts it. Its
only camera is the wrist camera (RGBDPerception, the realistic D405). The page has:
  wrist camera  live, with the last look's estimate projected onto it: each
                spine as the robot believes it (the slot's neighbours orange,
                others blue, labelled by place from the slot). It is drawn
                with the camera's true pose, so where an outline is off its
                spine, that is the estimate's error.
  scene camera  live, from a free camera on the cart and the bookcase, with the
                estimate's boxes drawn into the scene (as --view does).
  last look     the images the estimate came from, every book mask the
                segmenter found tinted, the fitted spines, and the call-number
                stickers read (call_numbers_seen.py): the two whose numbers
                bracket the held book's, which put the slot between them, green.
  books, log    the estimate as numbers (a book whose sticker was read: its call
                number, and its size and mass from the catalogue), and the
                controller's steps.
No rendering happens unless the episode's Runner asks for a frame (15 fps of
sim time); the sim runs at most real time, slower when rendering can't keep up.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import mujoco
import numpy as np

from bila.vision.camera import CAMERA, TO_OPENCV, TURN_180, CameraFrame, intrinsics, project

LIVE_SIZE = (640, 360)      # wrist camera, live (the looks use the camera's own 1280 x 720)
SCENE_SIZE = (960, 540)
FRAME_EVERY = 4             # render one of every 4 viewer syncs (the Runner syncs at 60 Hz of sim time)
MASK_COLOURS = np.array([(230, 25, 75), (60, 180, 75), (255, 225, 25), (0, 130, 200), (245, 130, 48),
                         (145, 30, 180), (70, 240, 240), (240, 50, 230), (210, 245, 60), (250, 190, 212)], np.uint8)


class Live:
    """Latest JPEG per stream and the page's state, shared with the server threads."""

    def __init__(self):
        self.cond = threading.Condition()
        self.jpeg = {"wrist": b"", "scene": b"", "look": b""}
        self.version = 0
        self.state = {"status": "waiting for the page", "log": [], "books": [], "looks": 0}
        self.opened = threading.Event()

    def publish(self, **images):
        with self.cond:
            for k, rgb in images.items():
                self.jpeg[k] = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                                            [cv2.IMWRITE_JPEG_QUALITY, 80])[1].tobytes()
            self.version += 1
            self.cond.notify_all()

    def update(self, **state):
        with self.cond:
            self.state.update(state)

    def log(self, *parts):
        print(*parts)
        with self.cond:
            self.state["log"] = (self.state["log"] + [" ".join(map(str, parts))])[-500:]


def draw_spines(img, frame: CameraFrame, est, fill=0.25):
    """The estimate's spines over an image taken from `frame`'s camera (in place)."""
    if est is None:
        return img
    scale = img.shape[1] / frame.size[0]
    R, t = est.shelf_pose[:3, :3], est.shelf_pose[:3, 3]
    over, outlines = img.copy(), []
    for b in est.books:
        u, v, z = project(frame, b.spine_corners @ R.T + t)
        if not np.all(z > 0):
            continue
        pts = np.column_stack([u, v]) * scale
        if np.abs(pts).max() > 1e4:
            continue
        pts = np.round(pts).astype(np.int32)
        colour = (255, 130, 60) if abs(b.slot_offset) == 1 else (75, 120, 170)
        cv2.fillPoly(over, [pts], colour)
        outlines.append((pts, colour, f"{b.slot_offset:+d}"))
    img[:] = cv2.addWeighted(over, fill, img, 1 - fill, 0)
    for pts, colour, label in outlines:
        cv2.polylines(img, [pts], True, colour, 2, cv2.LINE_AA)
        x, y = int(pts[:, 0].mean()), int(pts[:, 1].min()) - 6
        cv2.putText(img, label, (x - 10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return img


class Screen:
    """Stands in for mujoco.viewer's passive viewer, which Episode and Runner
    drive (is_running, sync, lock, user_scn): every sync renders the wrist and
    scene cameras and publishes them."""

    def __init__(self, model, data, live: Live):
        self.m, self.data, self.live, self.syncs = model, data, live, 0
        self.est = None                                   # the last look's estimate, set by WatchedEyes
        self.user_scn = mujoco.MjvScene(model, maxgeom=2000)
        self.wrist = mujoco.Renderer(model, LIVE_SIZE[1], LIVE_SIZE[0])
        self.scene = mujoco.Renderer(model, SCENE_SIZE[1], SCENE_SIZE[0])
        self.cam_id = model.camera(CAMERA).id
        self.K = intrinsics(model.cam_fovy[self.cam_id], *LIVE_SIZE)
        self.hide_sites = mujoco.MjvOption()
        self.hide_sites.sitegroup[:] = 0
        self.free = mujoco.MjvCamera()                    # the cart and the bookcase both in view (as record_cart)
        self.free.lookat[:] = data.xpos[model.body("lift").id] + [0.2, 0, 0.3]
        self.free.azimuth, self.free.elevation, self.free.distance = -35, -18, 2.6

    def is_running(self):
        return True

    def lock(self):
        return contextlib.nullcontext()

    def sync(self):
        self.syncs += 1
        if self.syncs % FRAME_EVERY == 0:
            self.render(self.data)

    def render(self, data):
        self.wrist.update_scene(data, CAMERA, self.hide_sites)
        rgb = self.wrist.render()
        pose = np.eye(4)
        pose[:3, :3] = data.cam_xmat[self.cam_id].reshape(3, 3) @ TO_OPENCV
        pose[:3, 3] = data.cam_xpos[self.cam_id]
        if pose[2, 1] > 0:                                # upside down: turn it, as WristCamera does
            rgb, pose[:3, :3] = rgb[::-1, ::-1], pose[:3, :3] @ TURN_180
        rgb = np.ascontiguousarray(rgb)
        frame = CameraFrame(rgb, np.zeros(rgb.shape[:2], np.float32), None, self.K, pose, float(data.time))
        draw_spines(rgb, frame, self.est)

        self.scene.update_scene(data, self.free)
        scn = self.scene.scene
        for i in range(min(self.user_scn.ngeom, scn.maxgeom - scn.ngeom)):   # the estimate's boxes
            g = self.user_scn.geoms[i]
            mujoco.mjv_initGeom(scn.geoms[scn.ngeom], g.type, g.size, g.pos, g.mat.ravel(), g.rgba)
            scn.ngeom += 1
        self.live.publish(wrist=rgb, scene=self.scene.render().copy())
        self.live.update(sim_time=round(float(data.time), 2))

    def close(self):
        self.wrist.close()
        self.scene.close()


class WatchedEyes:
    """Episode's eyes (an RGBDPerception), reporting every look to the page."""

    def __init__(self, eyes, screen: Screen, live: Live, target):
        self.eyes, self.screen, self.live, self.target = eyes, screen, live, target   # eyes: a ReadingEyes
        self.looks = 0

    def capture(self, data):
        return self.eyes.capture(data)

    def estimate_from_frames(self, frames, x_slot=None, placed_x=None):
        est = self.eyes.estimate_from_frames(frames, x_slot=x_slot, placed_x=placed_x)
        self.looks += 1
        self.screen.est = est
        from bila.control.cart_demo import dewey
        slot = {s.number for s in self.eyes.slot or ()}
        views = []
        for f, stickers in zip(frames, self.eyes.seen):
            img = f.rgb.copy()
            for i, seg in enumerate(self.eyes.segment(f)):
                img[seg.mask] = (0.55 * img[seg.mask] + 0.45 * MASK_COLOURS[i % len(MASK_COLOURS)]).astype(np.uint8)
            draw_spines(img, f, est, fill=0.0)
            for s in stickers:
                colour = (40, 220, 60) if s.number in slot else (255, 255, 255)
                pts = np.round(s.corners).astype(np.int32)
                cv2.polylines(img, [pts], True, colour, 3, cv2.LINE_AA)
                if s.number in slot:                         # a crowded row's numbers overlap: label only these
                    x, y = int(pts[:, 0].min()), int(pts[:, 1].max()) + 30
                    cv2.putText(img, dewey(s.number), (x - 20, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, colour, 3, cv2.LINE_AA)
            views.append(cv2.resize(img, LIVE_SIZE))
        self.live.publish(look=np.hstack(views))
        if self.eyes.slot is not None:
            lo, hi = self.eyes.slot
            self.live.update(slot=f"{dewey(self.eyes.number)} goes between {dewey(lo.number)} and {dewey(hi.number)}")
        gap = est.min_gap_width(self.target[1])
        self.live.update(looks=self.looks, narrowest_gap_mm=None if math.isnan(gap) else round(1000 * gap, 1),
                         books=[{"offset": b.slot_offset, "x_mm": round(1000 * b.spine_corners[:2, 0].mean()),
                        "call": None if b.call_number is None else dewey(b.call_number),
                        "depth_mm": round(1000 * b.depth), "kg": None if b.mass is None else round(b.mass, 2),
                                 "thickness_mm": round(1000 * b.thickness, 1), "height_mm": round(1000 * b.height),
                                 "lean_deg": round(math.degrees(b.lean), 1), "yaw_deg": round(math.degrees(b.yaw), 1),
                                 "visible": round(b.visible, 2)} for b in est.books])
        return est

    def close(self):
        self.eyes.close()


PAGE = """<!DOCTYPE html>
<html lang="en" data-bs-theme="dark"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Shelf robot: live view</title>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css">
<style>
  .cam { object-fit: contain; background: #000; }
  .card-header { font-size: .85rem; font-weight: 600; color: var(--bs-secondary-color); }
  #books td, #books th { text-align: right; font-variant-numeric: tabular-nums; }
  #log { font: 12px ui-monospace, monospace; white-space: pre-wrap; overflow-wrap: anywhere; height: 360px; overflow-y: auto; }
</style></head><body class="bg-body-tertiary">
<nav class="navbar bg-body border-bottom sticky-top"><div class="container-fluid gap-2">
  <span class="navbar-brand mb-0 h1">Shelf robot <small class="text-body-secondary fw-normal">live view</small></span>
  <div class="d-flex flex-wrap gap-2 align-items-center small">
    <span class="badge text-bg-secondary" id="b-scn">–</span>
    <span class="badge text-bg-secondary" id="b-seg">–</span>
    <span class="badge text-bg-dark border" id="b-t">t = 0 s</span>
    <span class="badge text-bg-dark border" id="b-looks">0 looks</span>
    <span class="badge text-bg-info" id="status">connecting…</span>
  </div></div></nav>
<main class="container-fluid py-3">
  <div class="alert py-2 mb-3" id="slot" role="status">–</div>
  <div class="row g-3 mb-3">
    <div class="col-lg-6"><div class="card h-100"><div class="card-header">Wrist camera · live, last estimate projected</div>
      <div class="card-body p-2"><div class="ratio ratio-16x9"><img class="cam rounded" src="/stream/wrist" alt="wrist camera"></div></div></div></div>
    <div class="col-lg-6"><div class="card h-100"><div class="card-header">Scene camera</div>
      <div class="card-body p-2"><div class="ratio ratio-16x9"><img class="cam rounded" src="/stream/scene" alt="scene camera"></div></div></div></div>
  </div>
  <div class="card mb-3"><div class="card-header">Last look · segmenter masks, fitted spines, call numbers read</div>
    <div class="card-body p-2"><img class="cam rounded w-100" src="/stream/look" alt="no look yet"></div></div>
  <div class="row g-3">
    <div class="col-xl-7"><div class="card h-100"><div class="card-header">Books recognised <span id="gap" class="fw-normal"></span></div>
      <div class="table-responsive"><table class="table table-sm table-hover mb-0" id="books">
        <thead><tr><th>place</th><th>call no.</th><th>x mm</th><th>thick mm</th><th>height mm</th><th>depth mm</th><th>kg</th><th>lean °</th><th>yaw °</th><th>visible</th></tr></thead>
        <tbody id="books-body"></tbody></table></div></div></div>
    <div class="col-xl-5"><div class="card h-100"><div class="card-header">Controller log</div>
      <div class="card-body p-2"><div id="log" class="bg-body-secondary rounded p-2 text-body-secondary"></div></div></div></div>
  </div>
</main>
<script>
const $ = id => document.getElementById(id);
async function poll() {
  try {
    const s = await (await fetch('/state')).json();
    $('b-scn').textContent = `${s.scenario} · seed ${s.seed}`;
    $('b-seg').textContent = `${s.segmenter} masks`;
    $('b-t').textContent = `t = ${s.sim_time ?? 0} s`;
    $('b-looks').textContent = `${s.looks} looks`;
    $('status').textContent = s.status;
    $('slot').className = `alert py-2 mb-3 ${s.slot ? 'alert-success' : 'alert-secondary'}`;
    $('slot').textContent = s.slot ? `Slot from call numbers: ${s.slot}` :
      `Returning ${s.book_number}: the row's call numbers not read yet`;
    $('gap').textContent = s.narrowest_gap_mm == null ? '' :
      `· narrowest gap ${s.narrowest_gap_mm} mm for a ${s.book_mm} mm book`;
    $('books-body').innerHTML = s.books.map(b =>
      `<tr class="${Math.abs(b.offset) == 1 ? 'table-warning' : ''}"><td>${b.offset > 0 ? '+' : ''}${b.offset}</td><td>${b.call ?? '–'}</td><td>${b.x_mm}</td>
       <td>${b.thickness_mm}</td><td>${b.height_mm}</td><td>${b.depth_mm}</td><td>${b.kg ?? '–'}</td><td>${b.lean_deg}</td><td>${b.yaw_deg}</td><td>${b.visible}</td></tr>`).join('');
    const log = $('log'), atEnd = log.scrollTop + log.clientHeight >= log.scrollHeight - 4;
    log.textContent = s.log.join('\\n');
    if (atEnd) log.scrollTop = log.scrollHeight;
  } catch (e) { $('status').textContent = 'server stopped'; $('status').className = 'badge text-bg-danger'; }
  setTimeout(poll, 500);
}
poll();
</script></body></html>"""


def serve(live: Live, port: int) -> ThreadingHTTPServer:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def send(self, body: bytes, kind: str):
            self.send_response(200)
            self.send_header("Content-Type", kind)
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/":
                live.opened.set()
                self.send(PAGE.encode(), "text/html; charset=utf-8")
            elif self.path == "/state":
                with live.cond:
                    self.send(json.dumps(live.state).encode(), "application/json")
            elif self.path.startswith("/stream/") and self.path[8:] in live.jpeg:
                self.stream(self.path[8:])
            else:
                self.send_error(404)

        def stream(self, name):
            """MJPEG: a new JPEG part every time the stream's image changes."""
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            seen, sent = -1, None
            try:
                while True:
                    with live.cond:
                        live.cond.wait_for(lambda: live.version != seen, timeout=5)
                        seen, jpeg = live.version, live.jpeg[name]
                    if jpeg and jpeg is not sent:
                        self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: %d\r\n\r\n" % len(jpeg))
                        self.wfile.write(jpeg + b"\r\n")
                        sent = jpeg
            except (BrokenPipeError, ConnectionResetError):
                pass

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenario", choices=("clear", "tight", "leaning", "packed"), default="leaning")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--segmenter", help="learned segmenter weights (segmenter.pt); default: the simulation's masks")
    ap.add_argument("--label-weights", help="label_reader's folder (finder.pt, reader.pt) to read the stickers "
                                            "from the images; default: from the simulation")
    ap.add_argument("--camera-noise", choices=("realistic", "basic"), default="realistic")
    ap.add_argument("--no-fiducials", action="store_true",
                    help="don't correct the camera pose from the shelf's markers (see the calibration error's effect)")
    ap.add_argument("--policy", help="a fitted high-level policy (high_level.py fit's q.npz) instead of the rules")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--no-browser", action="store_true", help="don't open the page; wait for it to be opened")
    a = ap.parse_args()

    segmenter = None
    if a.segmenter:
        from bila.vision.learned_segmenter import LearnedSegmenter    # before any rendering (see learned_segmenter.py)
        segmenter = LearnedSegmenter(a.segmenter)
    if a.label_weights:
        from bila.vision import label_reader  # noqa: F401  (imports torch._dynamo, before any render)
    from bila.control import cart_demo
    from bila.control.insert_book import report

    live = Live()
    model, data, row, tray, numbers, shelf_map = cart_demo.setup(a.scenario, a.seed, live.log)
    book = row.target_book
    labels, reading_eyes = cart_demo.readers(model, row, tray, numbers, a.seed,
                                             a.label_weights and os.path.expanduser(a.label_weights),
                                             segmenter, a.camera_noise, not a.no_fiducials)
    live.update(scenario=a.scenario, seed=a.seed, segmenter="learned" if segmenter else "simulation",
                book_mm=round(1000 * book.thickness, 1), book_number=cart_demo.dewey(numbers[book.name]))
    screen = Screen(model, data, live)
    def eyes_for(n):
        reading = reading_eyes(n)
        if reading.eyes.reference is not None:
            reading.eyes.reference.log = live.log           # each image's correction, in the log
        return WatchedEyes(reading, screen, live, (book.thickness, book.height))
    screen.render(data)

    url = f"http://localhost:{a.port}"
    server = serve(live, a.port)
    print(f"live view at {url}: the episode starts when the page opens (Ctrl-C to stop)")
    if not a.no_browser:
        webbrowser.open(url)
    live.opened.wait()

    live.update(status="running")
    try:
        res = cart_demo.run(model, data, row, tray, shelf_map, labels, eyes_for, screen, live.log, a.seed, numbers,
                            a.policy)
        live.log(report(res))
        live.update(status="SUCCESS" if res["success"] else "FAIL")
    except RuntimeError as e:                     # no safe way to do it in this scene, or it jammed
        live.log(f"FAIL: {e}")
        live.update(status=f"FAIL: {e}")
    screen.render(data)
    try:
        threading.Event().wait()                  # keep serving the last frames
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()
        screen.close()


if __name__ == "__main__":
    main()
