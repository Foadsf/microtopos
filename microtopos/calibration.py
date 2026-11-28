"""Per-objective calibration state: lateral scale, axial scale, dark and flat frames.

Statuses:
  NOMINAL     - derived from nameplate optics only; lengths are indicative
  ESTIMATED   - a scale carried over from another measurement or a model (a stored µm-per-frame value assumes
                the same knob speed as the sweep it came from; the DOF ruler is a model estimate)
  CALIBRATED  - measured with a reference for THIS data (stage micrometer; knob travel stated for this sweep)
  SIMULATED   - ground truth from the built-in simulator; never written to disk
"""

from __future__ import annotations

import datetime
import json
import os
import threading
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import cv2
import numpy as np

from .objectives import resolve
from .optics import SENSOR_FULL_WIDTH_PX, Objective


class CalStatus(str, Enum):
    NOMINAL = "NOMINAL"
    ESTIMATED = "ESTIMATED"
    CALIBRATED = "CALIBRATED"
    SIMULATED = "SIMULATED"


@dataclass
class ObjectiveCalibration:
    objective: str
    lateral_um_per_px_full: float | None = None
    lateral_source: str = ""
    lateral_date: str = ""
    lateral_magnification: float | None = None
    # (width, height) of the frame the lateral scale was measured on. A mode with the same aspect ratio is covered
    # (scaled by width); one with another aspect ratio is not: a camera cannot say whether a 4:3 mode is a crop or
    # a rescale of its 16:9 sensor, so no scale is guessed for it. None (files written before this was recorded):
    # width scaling for every mode, as before.
    lateral_frame: tuple[int, int] | list[int] | None = None
    z_um_per_frame: float | None = None
    z_source: str = ""
    z_date: str = ""
    z_uncertainty_pct: float | None = None
    revision: int = 0  # incremented on every change; results record the revision they used
    # dark/flat keyed by "WxH@exposure"
    darks: dict[str, np.ndarray] = field(default_factory=dict)
    flats: dict[str, np.ndarray] = field(default_factory=dict)
    # how each flat was obtained, by the same key: FLAT_RECORDED or a description starting with FLAT_ESTIMATED
    flat_sources: dict[str, str] = field(default_factory=dict)
    # measured lateral chromatic aberration (chromatic.LateralCA JSON) keyed by capture size "WxH"
    lateral_ca: dict[str, str] = field(default_factory=dict)
    # PSF measured on a straight edge (edgemtf.measure_edge) keyed by capture size "WxH": psf_sigma_px, mtf50, date
    measured_psf: dict[str, dict] = field(default_factory=dict)
    # estimated radial distortion (distortion.RadialModel JSON + corner_u_px, date) keyed by capture size "WxH"
    distortion: dict[str, dict] = field(default_factory=dict)

    @property
    def lateral_status(self) -> CalStatus:
        return CalStatus.CALIBRATED if self.lateral_um_per_px_full else CalStatus.NOMINAL

    @property
    def z_status(self) -> CalStatus:
        """A stored µm-per-frame value is only an estimate for a new sweep (knob speed varies between sweeps)."""
        return CalStatus.ESTIMATED if self.z_um_per_frame else CalStatus.NOMINAL

    @property
    def has_data(self) -> bool:
        """Anything measured or recorded (a revision alone, e.g. after clearing, does not count)."""
        return bool(self.lateral_um_per_px_full or self.z_um_per_frame or self.darks or self.flats)

    ASPECT_TOL = 0.01  # relative aspect-ratio difference below which two frame sizes are the same camera mode shape

    def covers(self, frame_width: int, frame_height: int | None = None) -> bool:
        """Whether the stored lateral calibration applies to a frame of this size: a lateral scale exists and the
        frame has the aspect ratio it was measured at (or no frame size was recorded, or no height is given)."""
        if not self.lateral_um_per_px_full:
            return False
        if not self.lateral_frame or not frame_height or len(self.lateral_frame) != 2:
            return True
        cw, ch = self.lateral_frame
        return abs(frame_width * ch - frame_height * cw) <= self.ASPECT_TOL * frame_height * cw

    def lateral_for(self, frame_width: int, frame_height: int | None = None,
                    objective: Objective | None = None) -> tuple[float, CalStatus]:
        """THE decision of micrometres per pixel for a frame, shared by every call site (live, GUI, batch).

        CALIBRATED: the stored scale, converted by width. NOMINAL: the objective's nameplate scale by width,
        when nothing is calibrated or the calibration does not cover this camera mode (different aspect ratio)."""
        obj = objective or resolve(self.objective)
        if self.covers(frame_width, frame_height):
            return (self.lateral_um_per_px_full * SENSOR_FULL_WIDTH_PX / float(frame_width), CalStatus.CALIBRATED)
        return obj.nominal_um_per_px(frame_width), CalStatus.NOMINAL

    def um_per_px(self, frame_width: int, objective: Objective | None = None,
                  frame_height: int | None = None) -> float:
        """Calibrated scale, else the nominal one of `objective` (default: this calibration's name, resolved)."""
        return self.lateral_for(frame_width, frame_height, objective)[0]

    def lateral_status_for(self, frame_width: int, frame_height: int | None = None) -> CalStatus:
        """CALIBRATED only where the calibration covers this camera mode, else NOMINAL."""
        return CalStatus.CALIBRATED if self.covers(frame_width, frame_height) else CalStatus.NOMINAL


def radiometric_key(shape: tuple[int, int], exposure: float | None) -> str:
    h, w = shape[:2]
    return f"{w}x{h}@{'auto' if exposure is None else round(exposure)}"


def today() -> str:
    return datetime.date.today().isoformat()


def lateral_ca_for(cal: ObjectiveCalibration, shape: tuple[int, int]):
    """This objective's measured lateral chromatic aberration (chromatic.LateralCA) for frames of `shape`, or None.
    Only the capture size it was measured at: another mode may be a crop, which shows a different field."""
    text = cal.lateral_ca.get(f"{shape[1]}x{shape[0]}")
    if not text:
        return None
    from .chromatic import LateralCA

    try:
        return LateralCA.from_json(text)
    except (KeyError, TypeError, ValueError):
        return None


def psf_for(cal: ObjectiveCalibration, objective: Objective, shape: tuple[int, int], um_per_px: float) -> tuple[float, str]:
    """The PSF sigma (px) to deconvolve frames of `shape` with, and where it comes from: the one measured on an
    edge for this objective and capture size, else the objective's theoretical one (optics.psf_sigma_px)."""
    m = cal.measured_psf.get(f"{shape[1]}x{shape[0]}")
    if m and m.get("psf_sigma_px", 0) > 0:
        return float(m["psf_sigma_px"]), (f"measured on an edge ({m.get('date', '?')}, MTF50 "
                                          f"{m.get('mtf50_cyc_per_px', float('nan')):.3f} cyc/px)")
    return objective.psf_sigma_px(um_per_px), "nominal"


def distortion_for(cal: ObjectiveCalibration, shape: tuple[int, int]):
    """This objective's estimated radial distortion (distortion.RadialModel) for frames of `shape`, or None."""
    d = cal.distortion.get(f"{shape[1]}x{shape[0]}")
    if not d:
        return None
    from .distortion import RadialModel

    try:
        return RadialModel.from_json(d)
    except (KeyError, TypeError, ValueError):
        return None


FLAT_RECORDED = "recorded on a blank, defocused view"
FLAT_ESTIMATED = "estimated from panning"


def flat_source(cal: ObjectiveCalibration, key: str) -> str:
    """How the flat stored under `key` was obtained (files written before this was recorded: recorded)."""
    return cal.flat_sources.get(key, FLAT_RECORDED) if key in cal.flats else ""


class CalibrationStore:
    """JSON metadata plus one NPZ of dark/flat frames, stored in a user data directory.

    Keyed by objective name. An entry is created on first use; entries of objectives that are no longer in the
    objective set are loaded and saved unchanged, so removing an objective never deletes its calibration."""

    def __init__(self, directory: Path | None):
        self.directory = directory
        self._lock = threading.RLock()
        self.cals: dict[str, ObjectiveCalibration] = {}
        if directory is not None:
            directory.mkdir(parents=True, exist_ok=True)
            self.load()

    def get(self, objective: Objective | str) -> ObjectiveCalibration:
        name = objective if isinstance(objective, str) else objective.name
        with self._lock:
            c = self.cals.get(name)
            if c is None:
                c = self.cals[name] = ObjectiveCalibration(name)
            return c

    def has_data(self, name: str) -> bool:
        c = self.cals.get(name)
        return c is not None and c.has_data

    def rename_many(self, renames: dict[str, str]) -> None:
        """Move calibrations to new names (a swap is fine: all are taken out before any is put back)."""
        with self._lock:
            moved = {b: self.cals.pop(a) for a, b in renames.items() if a in self.cals}
            for b, c in moved.items():
                c.objective = b
                c.revision += 1
                self.cals[b] = c

    @property
    def _json(self) -> Path:
        return self.directory / "calibration.json"

    @property
    def _npz(self) -> Path:
        return self.directory / "radiometric.npz"

    @property
    def _responses_json(self) -> Path:
        return self.directory / "camera_response.json"

    # ---------------------------------------------------------------- measured camera responses
    # A camera's response belongs to the camera and its capture mode (the sensor mode changes with the size on the
    # MikrOkular), not to an objective: stored per (camera name, "WxH") in camera_response.json.
    @staticmethod
    def _response_key(camera: str, shape: tuple[int, int]) -> str:
        return f"{camera}|{shape[1]}x{shape[0]}"

    def get_response(self, camera: str, shape: tuple[int, int]):
        """The measured response (response.ResponseCurve) of this camera in this capture size, or None."""
        if not camera:
            return None
        with self._lock:
            d = self._responses().get(self._response_key(camera, shape))
        if d is None:
            return None
        from .response import ResponseCurve

        try:
            return ResponseCurve.from_json(d)
        except (KeyError, TypeError, ValueError):
            return None  # a damaged entry is not a response

    def get_camera_value(self, kind: str, camera: str, shape: tuple[int, int]) -> dict | None:
        """A per-camera, per-capture-size measurement other than the response (e.g. kind "readout": the rolling
        shutter's readout fraction), stored next to the responses under its own kind."""
        if not camera:
            return None
        with self._lock:
            return self._responses().get(f"{kind}:{self._response_key(camera, shape)}")

    def set_camera_value(self, kind: str, camera: str, shape: tuple[int, int], value: dict) -> None:
        with self._lock:
            self._responses()[f"{kind}:{self._response_key(camera, shape)}"] = {**value, "date": today()}
            self._save_responses()

    def _save_responses(self) -> None:
        if self.directory is not None:
            tmp = self._responses_json.with_name(f".camera_response-{os.getpid()}.tmp")
            tmp.write_text(json.dumps(self._response_cache, indent=1), encoding="utf-8")
            tmp.replace(self._responses_json)

    def set_response(self, camera: str, shape: tuple[int, int], curve) -> None:
        with self._lock:
            self._responses()[self._response_key(camera, shape)] = {**curve.to_json(), "date": today()}
            self._save_responses()

    def _responses(self) -> dict:
        if getattr(self, "_response_cache", None) is None:
            self._response_cache = {}
            if self.directory is not None and self._responses_json.exists():
                try:
                    d = json.loads(self._responses_json.read_text(encoding="utf-8"))
                    self._response_cache = d if isinstance(d, dict) else {}
                except (OSError, json.JSONDecodeError):
                    pass
        return self._response_cache

    def touch(self, objective: Objective | str) -> int:
        """Mark a calibration as changed; returns its new revision."""
        c = self.get(objective)
        c.revision += 1
        return c.revision

    def save(self) -> None:
        """Atomic, serialised write: each file goes to a temporary name and replaces the old one."""
        if self.directory is None:
            return
        with self._lock:
            meta, arrays = {}, {}
            for name, c in self.cals.items():
                if not c.has_data and not c.revision:
                    continue  # never used: nothing to keep
                meta[name] = {
                    k: getattr(c, k)
                    for k in ("lateral_um_per_px_full", "lateral_source", "lateral_date",
                              "lateral_magnification", "lateral_frame",
                              "z_um_per_frame", "z_source", "z_date", "z_uncertainty_pct", "revision",
                              "flat_sources", "lateral_ca", "measured_psf", "distortion")
                }
                for kind, d in (("dark", c.darks), ("flat", c.flats)):
                    for key, arr in list(d.items()):
                        arrays[f"{kind}|{name}|{key}"] = arr.astype(np.float32)
            tmp_npz = self._npz.with_name(f".radiometric-{os.getpid()}.tmp.npz")
            with open(tmp_npz, "wb") as f:
                np.savez_compressed(f, **arrays)
                f.flush()
                os.fsync(f.fileno())
            tmp_npz.replace(self._npz)
            tmp = self._json.with_name(f".calibration-{os.getpid()}.tmp")
            tmp.write_text(json.dumps(meta, indent=2), encoding="utf-8")
            tmp.replace(self._json)

    def load(self) -> None:
        if self._json.exists():
            try:
                meta = json.loads(self._json.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                meta = {}
            for name, d in meta.items():
                if isinstance(d, dict):
                    c = self.get(name)
                    for k, v in d.items():
                        if hasattr(c, k) and k not in ("objective", "darks", "flats"):
                            setattr(c, k, v)
        if self._npz.exists():
            try:
                z = np.load(self._npz)
            except (OSError, ValueError):  # a damaged file must not stop the program; calibrations reset
                return
            with z:
                for key in z.files:
                    parts = key.split("|", 2)
                    if len(parts) == 3:
                        kind, name, rkey = parts
                        c = self.get(name)
                        (c.darks if kind == "dark" else c.flats)[rkey] = z[key]


class RadiometricCorrector:
    """(raw - dark) / (flat - dark) * mean(flat - dark), applied to float luma or 3-channel BGR."""

    def __init__(self, dark: np.ndarray | None = None, flat: np.ndarray | None = None):
        self.dark = dark
        self.flat = flat
        self._cache: dict[tuple[int, int], tuple[np.ndarray | None, np.ndarray | None]] = {}

    def _for_shape(self, shape: tuple[int, int]):
        h, w = shape[:2]
        if (h, w) not in self._cache:
            def fit(a):
                if a is None:
                    return None
                return a if a.shape[:2] == (h, w) else cv2.resize(a, (w, h), interpolation=cv2.INTER_AREA)

            dark = fit(self.dark)
            gain = None
            if self.flat is not None:
                flat = fit(self.flat)
                signal = flat - (dark if dark is not None else 0.0)
                signal = np.maximum(signal, 1.0)
                gain = (float(np.mean(signal)) / signal).astype(np.float32)
            self._cache[(h, w)] = (dark, gain)
        return self._cache[(h, w)]

    def __call__(self, img: np.ndarray) -> np.ndarray:
        dark, gain = self._for_shape(img.shape[:2])
        if img.ndim == 3:
            out = img.astype(np.float32, copy=False)
            if dark is not None:
                out = out - (dark[..., None] if dark.ndim == 2 else dark)
            if gain is not None:
                out = out * (gain[..., None] if gain.ndim == 2 else gain)
            return out.astype(np.float32, copy=False)
        out = img if dark is None else img - dark
        if gain is not None:
            out = out * gain
        return out.astype(np.float32, copy=False)


def bgr_to_luma(bgr: np.ndarray, linearize: bool = False, response=None) -> np.ndarray:
    """Rec.601 luma Y' of gamma-encoded camera data (8-bit or 16-bit), or linear-light Rec.709 luminance.

    16-bit (`uint16`) inputs are normalised to the 0..255 float32 scale while preserving full sub-DN precision.
    linearize: with `response` (a measured response.ResponseCurve of this camera and mode) the camera's own curve
    is inverted, scaled so that DN 255 in every channel gives 255; without one, gamma 2.2 is assumed.
    """
    if linearize and response is not None:
        return linear_luma(bgr, response)
    scale = (1.0 / 257.0) if bgr.dtype == np.uint16 else 1.0
    if bgr.ndim == 2:
        y = bgr.astype(np.float32) * scale
        return 255.0 * np.power(np.clip(y / 255.0, 0.0, None), 2.2) if linearize else y
    if linearize:
        lin = 255.0 * np.power(np.clip(bgr.astype(np.float32) * scale / 255.0, 0.0, None), 2.2)
        return (0.0722 * lin[..., 0] + 0.7152 * lin[..., 1] + 0.2126 * lin[..., 2]).astype(np.float32)
    if bgr.dtype == np.uint16:
        f = bgr.astype(np.float32) * scale
        return (0.114 * f[..., 0] + 0.587 * f[..., 1] + 0.299 * f[..., 2]).astype(np.float32)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)


def linear_luma(bgr: np.ndarray, response) -> np.ndarray:
    """Linear-light Rec.709 luminance through a measured camera response, on the 0..255 scale (DN 255 in every
    channel -> 255). A luma curve (measured on Y') is applied to the Rec.601 luma Y'."""
    img = bgr
    if img.dtype == np.uint16:
        img = img.astype(np.float32) / 257.0
    elif img.dtype != np.uint8:
        img = img.astype(np.float32)
    lut = response.lut
    if response.channels == "luma":
        y = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return (response.linearize(y) * (255.0 / float(lut[0, 255]))).astype(np.float32)
    if img.ndim == 2:  # a grey frame: every channel holds that value
        img = np.repeat(img[..., None], 3, axis=2)
    w = np.array([0.0722, 0.7152, 0.2126], np.float32)
    return (response.linearize(img) @ w * (255.0 / float(np.dot(w, lut[:, 255])))).astype(np.float32)


WB_LOW_DN, WB_HIGH_DN = 10.0, 245.0  # pixels darker or brighter than this in any channel do not set white balance


def white_balance_gains(bgr: np.ndarray) -> tuple[float, float, float]:
    """Per-channel gains (B, G, R; G = 1) that make the mean of the usable pixels neutral (grey world).

    Pixels near black or clipped in any channel are left out: a clipped channel no longer carries its colour. The
    gains act on the camera's encoded values; for a power-law tone curve a neutral grey stays neutral in linear
    light too. Raises ValueError when fewer than 1 % of the pixels are usable (a dark or saturated view)."""
    f = bgr.astype(np.float32) * ((1.0 / 257.0) if bgr.dtype == np.uint16 else 1.0)
    if f.ndim != 3 or f.shape[2] != 3:
        raise ValueError("white balance needs a colour (3-channel) frame")
    ok = np.all((f > WB_LOW_DN) & (f < WB_HIGH_DN), axis=2)
    if ok.mean() < 0.01:
        raise ValueError(f"only {ok.mean():.1%} of the pixels are between {WB_LOW_DN:.0f} and {WB_HIGH_DN:.0f} DN in "
                         "every channel: adjust the exposure first")
    m = f[ok].mean(axis=0)
    return float(m[1] / m[0]), 1.0, float(m[1] / m[2])


def apply_white_balance(bgr: np.ndarray, gains: tuple[float, float, float] | None) -> np.ndarray:
    """Multiply each channel by its gain (B, G, R). uint8 stays uint8 (rounded, clipped); other types become float32.
    None returns the input unchanged."""
    if gains is None or bgr.ndim != 3:
        return bgr
    g = np.asarray(gains, np.float32)
    if bgr.dtype == np.uint8:
        return np.clip(bgr.astype(np.float32) * g + 0.5, 0, 255).astype(np.uint8)
    return (bgr.astype(np.float32) * g).astype(np.float32)


def bgr_to_rgb_u8(bgr: np.ndarray) -> np.ndarray:
    """Convert BGR or grayscale array (uint8, uint16, float32) to uint8 RGB."""
    scale = (1.0 / 257.0) if bgr.dtype == np.uint16 else 1.0
    u8 = np.clip(bgr.astype(np.float32) * scale + 0.5, 0, 255).astype(np.uint8)
    if u8.ndim == 2:
        return np.dstack([u8, u8, u8])
    return cv2.cvtColor(u8, cv2.COLOR_BGR2RGB)
