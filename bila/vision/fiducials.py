"""
The shelf's fiducial markers as the camera's reference: on every look, find
them in the colour image and correct where the robot believes its camera is.

    ref = FiducialReference(fiducials(cfg, row), ShelfFrame(row), slot_x(row))
    frame = ref(camera.capture(data))           # the same image, its pose corrected

The wrist camera's hand-eye calibration is off by a few mm and a fraction of
a degree, which shifts everything it sees at the shelf by 4-9 mm, differently
in every view (docs/calibration-and-insertion.md). Each marker is on the
shelf map (shelf_scene.fiducials): its ArUco id and where it is. In each
image, the whole marker nearest the slot is found (cv2.aruco); its centre,
seen through the believed camera pose, lands on the marker's known front
plane some way from where the map has it; that offset, along the row and up,
is taken out of the camera's position. Only the shift in the shelf's plane is
corrected: a 15 mm marker pins down the camera's rotation poorly, and the
in-plane error near the marker is what the slot needs. (Solving the full pose
from every marker in view, PnP over all their corners, fixed the far side of
the image better but added 5-7 mm in and out, and did worse on the cart demo:
6 of 12 against 7.) Images without a whole marker keep their pose.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from bila.sim.shelf_scene import FIDUCIAL_DICT, Fiducial
from bila.vision.camera import CameraFrame


class FiducialReference:
    def __init__(self, markers: List[Fiducial], shelf, near_x: float, log=None):
        """markers: the shelf map's; shelf: the ShelfFrame; near_x: where
        along the row it matters most (the slot, on the map)."""
        self.markers = {m.id: m for m in markers}
        self.shelf, self.near_x, self.log = shelf, near_x, log
        self.detector = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, FIDUCIAL_DICT)))
        self.last: Optional[Tuple[int, np.ndarray]] = None   # (marker id, shift taken out, shelf frame m), or None

    def detect(self, rgb: np.ndarray) -> Dict[int, np.ndarray]:
        """{marker id: its corners (4, 2) px} for the map's markers found whole."""
        corners, ids, _ = self.detector.detectMarkers(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY))
        if ids is None:
            return {}
        return {int(i): c.reshape(4, 2) for c, i in zip(corners, ids.ravel()) if int(i) in self.markers}

    def shift(self, frame: CameraFrame, marker: Fiducial, corners: np.ndarray) -> np.ndarray:
        """How far (shelf frame, y = 0) the believed pose puts the marker from
        where the map has it: its corners' centre in the image, along the
        believed camera's ray onto the marker's front plane."""
        ray = frame.pose[:3, :3] @ np.linalg.solve(frame.K, [*corners.mean(axis=0), 1.0])
        o, r = self.shelf.to_shelf(frame.pose[:3, 3]), ray @ self.shelf.R
        y = marker.centre[1]
        hit = o + r * (y - o[1]) / r[1]
        d = hit - np.asarray(marker.centre)
        d[1] = 0.0
        return d

    def __call__(self, frame: CameraFrame) -> CameraFrame:
        found = self.detect(frame.rgb)
        if not found:
            self.last = None
            if self.log:
                self.log("         no whole fiducial in view: the camera pose is not corrected")
            return frame
        i = min(found, key=lambda i: abs(self.markers[i].centre[0] - self.near_x))
        d = self.shift(frame, self.markers[i], found[i])
        self.last = (i, d)
        if self.log:
            self.log(f"         fiducial {i}: the camera saw it {1000 * d[0]:+.1f} mm along the row, "
                     f"{1000 * d[2]:+.1f} mm up; corrected")
        pose = frame.pose.copy()
        pose[:3, 3] -= self.shelf.R @ d
        return replace(frame, pose=pose)
