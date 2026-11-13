"""Per-objective calibration state: lateral scale, axial scale, dark and flat frames.

Statuses:
  NOMINAL     - derived from nameplate optics only; lengths are indicative
  CALIBRATED  - measured by the user with a reference (stage micrometer, knob, step standard)
  SIMULATED   - ground truth from the built-in simulator; never written to disk
"""

from __future__ import annotations

import datetime
import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .optics import OBJECTIVES, SENSOR_FULL_WIDTH_PX, Objective

logger = logging.getLogger("microtopos.calibration")


class CalStatus(str, Enum):
    NOMINAL = "NOMINAL"
    CALIBRATED = "CALIBRATED"
    SIMULATED = "SIMULATED"


@dataclass
class ObjectiveCalibration:
    objective: str
    lateral_um_per_px_full: float | None = None
    lateral_source: str = ""
    lateral_date: str = ""
    z_um_per_frame: float | None = None
    z_source: str = ""
    z_date: str = ""
    z_uncertainty_pct: float | None = None
    # dark/flat keyed by "WxH@exposure"
    darks: dict[str, np.ndarray] = field(default_factory=dict)
    flats: dict[str, np.ndarray] = field(default_factory=dict)
    lens_model: LensDistortionModel | None = None

    @property
    def lateral_status(self) -> CalStatus:
        return CalStatus.CALIBRATED if self.lateral_um_per_px_full else CalStatus.NOMINAL

    @property
    def z_status(self) -> CalStatus:
        return CalStatus.CALIBRATED if self.z_um_per_frame else CalStatus.NOMINAL

    def um_per_px(self, frame_width: int) -> float:
        obj = OBJECTIVES[self.objective]
        full = self.lateral_um_per_px_full or obj.nominal_um_per_px(SENSOR_FULL_WIDTH_PX)
        return full * SENSOR_FULL_WIDTH_PX / float(frame_width)


def radiometric_key(shape: tuple[int, int], exposure: float | None) -> str:
    h, w = shape[:2]
    return f"{w}x{h}@{'auto' if exposure is None else round(exposure)}"


def today() -> str:
    return datetime.datetime.now(datetime.timezone.utc).astimezone().date().isoformat()


class CalibrationStore:
    """JSON metadata plus one NPZ of dark/flat frames, stored in a user data directory."""

    def __init__(self, directory: Path | None):
        self.directory = directory
        self.cals: dict[str, ObjectiveCalibration] = {n: ObjectiveCalibration(n) for n in OBJECTIVES}
        if directory is not None:
            directory.mkdir(parents=True, exist_ok=True)
            self.load()

    def get(self, objective: Objective | str) -> ObjectiveCalibration:
        return self.cals[objective if isinstance(objective, str) else objective.name]

    @property
    def _json(self) -> Path:
        assert self.directory is not None
        return self.directory / "calibration.json"

    @property
    def _npz(self) -> Path:
        assert self.directory is not None
        return self.directory / "radiometric.npz"

    def save(self) -> None:
        if self.directory is None:
            return
        meta: dict[str, Any] = {}
        arrays: dict[str, Any] = {}
        for name, c in self.cals.items():
            meta[name] = {
                k: getattr(c, k)
                for k in ("lateral_um_per_px_full", "lateral_source", "lateral_date", "z_um_per_frame",
                          "z_source", "z_date", "z_uncertainty_pct")
            }
            if c.lens_model is not None:
                meta[name]["lens_model"] = c.lens_model.to_dict()
            for kind, d in (("dark", c.darks), ("flat", c.flats)):
                for key, arr in d.items():
                    arrays[f"{kind}|{name}|{key}"] = arr.astype(np.float32)
        tmp = self._json.with_suffix(".tmp")
        tmp.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        tmp.replace(self._json)
        np.savez_compressed(self._npz, **arrays)

    def load(self) -> None:
        if self._json.exists():
            try:
                meta = json.loads(self._json.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                logger.warning("Unable to load objective calibration metadata from %s: %s", self._json, exc)
                meta = {}
            for name, d in meta.items():
                if name in self.cals:
                    for k, v in d.items():
                        if k == "lens_model" and isinstance(v, dict):
                            self.cals[name].lens_model = LensDistortionModel.from_dict(v)
                        elif hasattr(self.cals[name], k):
                            setattr(self.cals[name], k, v)
        if self._npz.exists():
            with np.load(self._npz) as z:
                for key in z.files:
                    kind, name, rkey = key.split("|", 2)
                    if name in self.cals:
                        (self.cals[name].darks if kind == "dark" else self.cals[name].flats)[rkey] = z[key]


class RadiometricCorrector:
    """(raw - dark) / (flat - dark) * mean(flat - dark), applied to float luma."""

    def __init__(self, dark: np.ndarray | None = None, flat: np.ndarray | None = None):
        self.dark = dark
        self.flat = flat
        self._cache: dict[tuple[int, int], tuple[np.ndarray | None, np.ndarray | None]] = {}

    def _for_shape(self, shape):
        if shape not in self._cache:
            def fit(a):
                if a is None:
                    return None
                return a if a.shape == shape else cv2.resize(a, (shape[1], shape[0]), interpolation=cv2.INTER_AREA)

            dark = fit(self.dark)
            gain = None
            if self.flat is not None:
                flat = fit(self.flat)
                assert flat is not None
                signal = flat - (dark if dark is not None else 0.0)
                signal = np.maximum(signal, 1.0)
                gain = (float(np.mean(signal)) / signal).astype(np.float32)
            self._cache[shape] = (dark, gain)
        return self._cache[shape]

    def __call__(self, luma: np.ndarray) -> np.ndarray:
        dark, gain = self._for_shape(luma.shape)
        out = luma if dark is None else luma - dark
        if gain is not None:
            out = out * gain
        return out.astype(np.float32, copy=False)


def bgr_to_luma(bgr: np.ndarray, linearize: bool = False) -> np.ndarray:
    """Rec.601 luma Y' of gamma-encoded camera data, or linear-light Rec.709 luminance when linearize=True."""
    if bgr.ndim == 2:
        y = bgr.astype(np.float32)
        return 255.0 * (y / 255.0) ** 2.2 if linearize else y
    if linearize:
        lin = 255.0 * (bgr.astype(np.float32) / 255.0) ** 2.2
        return np.asarray(0.0722 * lin[..., 0] + 0.7152 * lin[..., 1] + 0.2126 * lin[..., 2], dtype=np.float32)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)


@dataclass
class LensDistortionModel:
    """Lens geometric distortion parameters (Brown-Conrady model) to correct optical aberrations."""

    camera_matrix: np.ndarray  # 3x3 float64
    dist_coeffs: np.ndarray  # 1x5 or 1x8 float64 [k1, k2, p1, p2, k3, ...]
    image_size: tuple[int, int] = (0, 0)  # (w, h)
    _map1: np.ndarray | None = field(default=None, repr=False)
    _map2: np.ndarray | None = field(default=None, repr=False)

    def undistort(self, img: np.ndarray) -> np.ndarray:
        """Undistort an image, using cached rectification maps for fast streaming execution."""
        h, w = img.shape[:2]
        if self._map1 is None or self._map2 is None or (w, h) != self.image_size:
            self.image_size = (w, h)
            new_cam_mat, _ = cv2.getOptimalNewCameraMatrix(self.camera_matrix, self.dist_coeffs, (w, h), 0)
            self._map1, self._map2 = cv2.initUndistortRectifyMap(
                self.camera_matrix, self.dist_coeffs, None, new_cam_mat, (w, h), cv2.CV_16SC2
            )
        return np.asarray(cv2.remap(img, self._map1, self._map2, cv2.INTER_LINEAR))

    def to_dict(self) -> dict:
        return {
            "camera_matrix": self.camera_matrix.tolist(),
            "dist_coeffs": self.dist_coeffs.tolist(),
            "image_size": list(self.image_size),
        }

    @classmethod
    def from_dict(cls, data: dict) -> LensDistortionModel:
        return cls(
            camera_matrix=np.array(data["camera_matrix"], dtype=np.float64),
            dist_coeffs=np.array(data["dist_coeffs"], dtype=np.float64),
            image_size=tuple(data.get("image_size", (0, 0))),
        )


def calibrate_lens_from_checkerboard(
    images: list[np.ndarray],
    board_size: tuple[int, int] = (9, 6),
    square_size_mm: float = 1.0,
) -> tuple[LensDistortionModel, float]:
    """Calibrate camera intrinsics and lens distortion from a set of checkerboard calibration images.

    Returns:
        (model, rms_reprojection_error)
    """
    objp = np.zeros((board_size[0] * board_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0 : board_size[0], 0 : board_size[1]].T.reshape(-1, 2) * square_size_mm

    objpoints = []
    imgpoints = []
    img_size: tuple[int, int] | None = None

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    for img in images:
        gray = bgr_to_luma(img) if img.ndim == 3 else img.astype(np.uint8)
        if img_size is None:
            img_size = (gray.shape[1], gray.shape[0])
        ret, corners = cv2.findChessboardCorners(gray.astype(np.uint8), board_size, None)
        if ret:
            objpoints.append(objp)
            corners_sub = cv2.cornerSubPix(gray.astype(np.uint8), corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners_sub)

    if not objpoints or img_size is None:
        raise ValueError(f"No checkerboard patterns found in the provided {len(images)} images.")

    rms, mtx, dist, _rvecs, _tvecs = cv2.calibrateCamera(
        objpoints,
        imgpoints,
        img_size,
        np.eye(3, dtype=np.float64),
        np.zeros((1, 5), dtype=np.float64),
    )
    model = LensDistortionModel(camera_matrix=mtx, dist_coeffs=dist, image_size=img_size)
    return model, float(rms)

