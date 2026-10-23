"""
The wrist camera as an RGB-D sensor: images, the camera model, and depth turned
into 3D points.

    from camera import WristCamera, points, project
    cam = WristCamera(model)                  # 1280 x 720, like a D405 (~87 deg wide)
    frame = cam.capture(data)                 # rgb, depth, seg, K, pose, stamp
    xyz = points(frame)                       # (N, 3) world points, one per pixel with a reading
    u, v, z = project(frame, xyz)             # back into the image

Conventions, as a real camera pipeline (OpenCV, RealSense) uses them, so code
written against a CameraFrame works on real images too:
  - Camera frame: x right, y down, z forward (MuJoCo's own camera looks along
    its -z with y up; capture converts).
  - Pixels: (u, v) = (column, row), row 0 at the top, pixel centres at whole
    numbers. K = [[f, 0, cx], [0, f, cy], [0, 0, 1]].
  - Depth: metres along the camera's z (not along each pixel's ray). 0 means no
    reading: nothing within range, or (with noise) a pixel the sensor drops.

Only physical things are drawn: sites (markers such as the scene's green dot at
the target slot) are hidden, as no real camera would see them.

Images come out upright: if the camera is rolled so that its image would show
the world upside down (as at the inspection pose), the image and the camera
frame are both turned 180 deg about the viewing axis. The pose in the frame is
the camera's pose when the image was taken, read from the simulation; with a
CalibrationError, it and K are what the robot believes instead (the image is
still taken from where the camera really is).

Errors: WristCamera(model, noise=DepthNoise(), ...) is the basic depth model
(random error per pixel, no reading at edges). DepthNoise.d405() adds a stereo
camera's errors: smooth (not per-pixel) random error, a depth offset per
camera, edge fattening and the stereo shadow. calibration=CalibrationError()
adds a hand-eye and intrinsics error per camera.

Off-screen rendering needs an OpenGL context: in WSL, set MUJOCO_GL=egl.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import mujoco
import numpy as np

CAMERA = "panda/wrist_cam"
WIDTH, HEIGHT = 1280, 720     # 16:9 with the camera's 58 deg vertical view: ~89 deg wide, like a D405 (87)
MAX_RANGE = 1.5               # m: further than this reads as no depth (the scene's background is far away)
TO_OPENCV = np.diag([1.0, -1.0, -1.0])   # MuJoCo camera axes (x right, y up, z back) -> OpenCV (x right, y down, z forward)
TURN_180 = np.diag([-1.0, -1.0, 1.0])    # turn the camera frame 180 deg about its viewing axis


@dataclass
class CameraFrame:
    """One RGB-D image and the camera it came from."""
    rgb: np.ndarray             # (H, W, 3) uint8
    depth: np.ndarray           # (H, W) float32, metres along the camera's z; 0 = no reading
    seg: Optional[np.ndarray]   # (H, W) int32 MuJoCo geom id per pixel, -1 for none (simulation only)
    K: np.ndarray               # (3, 3) intrinsics
    pose: np.ndarray            # (4, 4) world <- camera (OpenCV axes)
    stamp: float                # sim time when it was taken

    @property
    def size(self) -> Tuple[int, int]:
        return self.depth.shape[1], self.depth.shape[0]          # (W, H)


def intrinsics(fovy_deg: float, width: int, height: int) -> np.ndarray:
    """Pinhole intrinsics of a MuJoCo camera: square pixels, centred, with the
    vertical field of view fovy."""
    f = (height / 2) / math.tan(math.radians(fovy_deg) / 2)
    return np.array([[f, 0.0, (width - 1) / 2], [0.0, f, (height - 1) / 2], [0.0, 0.0, 1.0]])


@dataclass
class DepthNoise:
    """Errors of a stereo depth camera like the D405, applied to a perfect depth
    image. The defaults are the basic model; DepthNoise.d405() adds the stereo
    errors after it:
      - random error growing with distance squared: std = z^2 * subpixel / (f * baseline),
        Intel's formula for D400 depth (its tuning guide uses 0.08 px; below 0.1 on
        well-textured targets), with the D405's 18 mm baseline and its focal length at
        1280 px wide (~670 px): about 1.7 mm at 50 cm;
      - no reading where depth jumps by more than `edge_jump` to a neighbouring pixel
        (stereo matching fails at object edges), and outside [min_range, max_range].
    Stereo errors (off by default):
      - `correlation_px`: the random error is smooth over about this many pixels
        (Gaussian), as matching over a window of pixels makes it, instead of
        independent per pixel, so fitting a plane to many pixels averages less of it away;
      - `offset_px`: std of a disparity offset fixed per camera (drawn by WristCamera):
        depth comes out too near or too far everywhere, by more further away. D405
        spec: depth within 2% at 50 cm;
      - `fatten_px`: at a depth edge, the nearer surface's depth spreads this far
        over the farther one (a matching window straddling the edge takes the nearer
        depth);
      - `shadow`: no reading where the second imager, `baseline` to the right of the
        first, can't see the surface: beside the left of anything nearer than what is
        behind it, and along the image's left edge.
    These act on the image as the camera takes it, before WristCamera turns it upright."""
    subpixel: float = 0.08
    baseline: float = 0.018
    focal_px: float = 670.0
    edge_jump: float = 0.01
    min_range: float = 0.07
    max_range: float = 1.0
    correlation_px: float = 0.0
    offset_px: float = 0.0
    fatten_px: int = 0
    shadow: bool = False

    @classmethod
    def d405(cls) -> "DepthNoise":
        """The stereo errors of a D405 on book spines (matte, partly printed): a
        matching error of 0.1 px, smooth over 2 px; a disparity offset of 0.17 px
        std (0.7% at 50 cm, so 95% of cameras within 1.4%, inside the 2% spec);
        edges fattened by 2 px; and the stereo shadow. The correlation and the
        fattening are assumptions: Intel publishes neither."""
        return cls(subpixel=0.10, correlation_px=2.0, offset_px=0.17, fatten_px=2, shadow=True)

    def draw_offset(self, rng: np.random.Generator) -> float:
        """A camera's disparity offset (px), kept within the 2% spec at 50 cm."""
        limit = 0.02 * self.focal_px * self.baseline / 0.5
        return float(np.clip(rng.normal(0.0, self.offset_px), -limit, limit)) if self.offset_px else 0.0

    def apply(self, depth: np.ndarray, rng: np.random.Generator, offset: float = 0.0) -> np.ndarray:
        """`offset`: this camera's disparity offset, px (see draw_offset)."""
        z = true = depth.astype(np.float64)
        valid = (z >= self.min_range) & (z <= self.max_range)
        fb = self.focal_px * self.baseline
        if self.fatten_px:
            far = np.where(z > 0, z, np.inf)
            k = 2 * self.fatten_px + 1
            nearest = cv2.erode(far, np.ones((k, k), np.uint8), borderType=cv2.BORDER_REPLICATE)
            spread = valid & (z - nearest > self.edge_jump)
            z = np.where(spread, nearest, z)
        visible = np.ones_like(valid)
        if self.shadow:
            # A pixel at column u shows up at u - disparity in the second image. It is
            # hidden there if something to its right lands at or left of it, or it lands
            # outside the image.
            u = np.arange(z.shape[1], dtype=np.float64)
            at = u - np.where(true > 0, fb / np.maximum(true, 1e-9), 0.0)
            ahead = np.minimum.accumulate(at[:, ::-1], axis=1)[:, ::-1]          # min over u' >= u
            ahead = np.concatenate([ahead[:, 1:], np.full((z.shape[0], 1), np.inf)], axis=1)   # u' > u
            visible = (ahead >= at - 0.5) & (at >= 0)
        std = z ** 2 * self.subpixel / fb
        err = rng.normal(0.0, 1.0, z.shape)
        if self.correlation_px:
            ks = int(8 * self.correlation_px + 1) | 1
            k1 = cv2.getGaussianKernel(ks, self.correlation_px)
            err = cv2.GaussianBlur(err, (ks, ks), self.correlation_px) / float(np.sum(k1 ** 2))   # back to unit std
        noisy = z + err * std
        if offset:
            noisy = fb / (fb / np.maximum(noisy, 1e-9) + offset)
        jump = np.zeros_like(valid)
        dz_x, dz_y = np.abs(np.diff(z, axis=1)) > self.edge_jump, np.abs(np.diff(z, axis=0)) > self.edge_jump
        jump[:, 1:] |= dz_x
        jump[:, :-1] |= dz_x
        jump[1:, :] |= dz_y
        jump[:-1, :] |= dz_y
        return np.where(valid & visible & ~jump, noisy, 0.0).astype(np.float32)


@dataclass
class CalibrationError:
    """How far off a camera's calibration is, as standard deviations; WristCamera
    draws one error per camera, and images then report where the robot thinks the
    camera is, and its believed intrinsics, while being taken from where it really
    is with its real lens:
      - hand-eye: where the camera sits on the hand. Published eye-in-hand RGB-D
        calibrations are off by a few mm and about 0.5 to 1 deg;
      - intrinsics: focal length and image centre (assumed: factory calibration)."""
    translation: float = 0.0015                  # m, per axis
    rotation: float = math.radians(0.3)          # rad, per axis (about 0.5 deg in all)
    focal: float = 0.003                         # fraction of the focal length
    centre_px: float = 1.0                       # px, per axis

    def draw(self, rng: np.random.Generator, K: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """(camera <- believed camera transform (4, 4), believed intrinsics)."""
        w = rng.normal(0.0, self.rotation, 3)
        R = np.zeros(9)
        q = np.zeros(4)
        mujoco.mju_axisAngle2Quat(q, w / max(np.linalg.norm(w), 1e-12), np.linalg.norm(w))
        mujoco.mju_quat2Mat(R, q)
        E = np.eye(4)
        E[:3, :3], E[:3, 3] = R.reshape(3, 3), rng.normal(0.0, self.translation, 3)
        Kb = K.copy()
        Kb[0, 0] = Kb[1, 1] = K[0, 0] * (1 + rng.normal(0.0, self.focal))
        Kb[:2, 2] += rng.normal(0.0, self.centre_px, 2)
        return E, Kb


class WristCamera:
    """Takes RGB-D images from a camera in the model (the wrist camera by
    default). One off-screen renderer, reused for every capture; close() it
    when done (or use `with WristCamera(model) as cam:`)."""

    def __init__(self, model: mujoco.MjModel, width: int = WIDTH, height: int = HEIGHT, name: str = CAMERA,
                 noise: Optional[DepthNoise] = None, rng: Optional[np.random.Generator] = None,
                 calibration: Optional[CalibrationError] = None):
        self.m, self.name = model, name
        self.cam = model.camera(name).id
        self.K = intrinsics(model.cam_fovy[self.cam], width, height)
        # No anti-aliasing: it blends neighbouring pixels at object edges, and a
        # blend of two objects' id colours decodes to some third object's id.
        samples = model.vis.quality.offsamples
        model.vis.quality.offsamples = 0
        try:
            self.renderer = mujoco.Renderer(model, height, width)
        finally:
            model.vis.quality.offsamples = samples
        self.options = mujoco.MjvOption()
        self.options.sitegroup[:] = 0           # hide markers (e.g. the target slot's dot)
        self.noise, self.rng = noise, rng if rng is not None else np.random.default_rng()
        # This camera's own errors, fixed for its life: its depth offset, and how far
        # off its calibration is (where the robot thinks it sits, and its intrinsics).
        # Each comes from its own stream, split off rng without drawing from it, so
        # a seed gives the same of each whichever others are on, and the same
        # per-image noise as without them.
        offset_rng, calibration_rng = self.rng.spawn(2)
        self.offset = noise.draw_offset(offset_rng) if noise is not None else 0.0
        self.mount_error, self.K_believed = (calibration.draw(calibration_rng, self.K) if calibration is not None
                                             else (np.eye(4), self.K))

    def capture(self, data: mujoco.MjData) -> CameraFrame:
        r = self.renderer
        r.update_scene(data, self.name, self.options)
        rgb = r.render().copy()
        r.enable_depth_rendering()
        r.update_scene(data, self.name, self.options)
        depth = r.render().copy()
        r.disable_depth_rendering()
        r.enable_segmentation_rendering()
        r.update_scene(data, self.name, self.options)
        seg = r.render()
        r.disable_segmentation_rendering()
        seg = np.where(seg[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM), seg[..., 0], -1).astype(np.int32)

        depth[depth > MAX_RANGE] = 0.0
        if self.noise is not None:
            depth = self.noise.apply(depth, self.rng, self.offset)

        pose = np.eye(4)
        pose[:3, :3] = data.cam_xmat[self.cam].reshape(3, 3) @ TO_OPENCV
        pose[:3, 3] = data.cam_xpos[self.cam]
        pose = pose @ self.mount_error      # where the robot thinks the camera is
        if pose[2, 1] > 0:                  # the image's "down" points up in the world: the camera is upside down
            rgb, depth, seg = rgb[::-1, ::-1], depth[::-1, ::-1], seg[::-1, ::-1]
            pose[:3, :3] = pose[:3, :3] @ TURN_180
        return CameraFrame(np.ascontiguousarray(rgb), np.ascontiguousarray(depth), np.ascontiguousarray(seg),
                           self.K_believed.copy(), pose, float(data.time))

    def close(self):
        self.renderer.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


# ---------------------------------------------------------------------------
# Geometry on a frame
# ---------------------------------------------------------------------------
def points(frame: CameraFrame, mask: Optional[np.ndarray] = None, world: bool = True) -> np.ndarray:
    """3D points (N, 3) for the pixels with a depth reading (and in `mask`, if
    given), in the world frame, or in the camera frame with world=False. Row-major
    pixel order; pixels() gives the matching (u, v)."""
    v, u = np.nonzero(_selected(frame, mask))
    z = frame.depth[v, u].astype(np.float64)
    K = frame.K
    cam = np.column_stack([(u - K[0, 2]) * z / K[0, 0], (v - K[1, 2]) * z / K[1, 1], z])
    return cam @ frame.pose[:3, :3].T + frame.pose[:3, 3] if world else cam


def pixels(frame: CameraFrame, mask: Optional[np.ndarray] = None) -> np.ndarray:
    """(N, 2) pixel coordinates (u, v) of the points that points() returns."""
    v, u = np.nonzero(_selected(frame, mask))
    return np.column_stack([u, v])


def project(frame: CameraFrame, xyz_world) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pixel coordinates (u, v) and depth z of world points (N, 3). Points
    behind the camera get z <= 0."""
    p = (np.atleast_2d(xyz_world) - frame.pose[:3, 3]) @ frame.pose[:3, :3]
    z = p[:, 2]
    K = frame.K
    with np.errstate(divide="ignore", invalid="ignore"):
        return K[0, 0] * p[:, 0] / z + K[0, 2], K[1, 1] * p[:, 1] / z + K[1, 2], z


def robot_mask(model: mujoco.MjModel, frame: CameraFrame, prefix: str = "panda/") -> np.ndarray:
    """Pixels showing the robot itself (any geom of a body named prefix...).
    In the simulation this comes from the geom ids; a real robot would project
    its own model into the image instead."""
    robot = [g for g in range(model.ngeom) if model.body(model.geom_bodyid[g]).name.startswith(prefix)]
    return np.isin(frame.seg, robot)


def _selected(frame: CameraFrame, mask: Optional[np.ndarray]) -> np.ndarray:
    has_depth = frame.depth > 0
    return has_depth if mask is None else has_depth & mask
