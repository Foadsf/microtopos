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
    z_um_per_frame: float | None = None
    z_source: str = ""
    z_date: str = ""
    z_uncertainty_pct: float | None = None
    revision: int = 0  # incremented on every change; results record the revision they used
    # dark/flat keyed by "WxH@exposure"
    darks: dict[str, np.ndarray] = field(default_factory=dict)
    flats: dict[str, np.ndarray] = field(default_factory=dict)

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

    def um_per_px(self, frame_width: int, objective: Objective | None = None) -> float:
        """Calibrated scale, else the nominal one of `objective` (default: this calibration's name, resolved)."""
        obj = objective or resolve(self.objective)
        full = self.lateral_um_per_px_full or obj.nominal_um_per_px(SENSOR_FULL_WIDTH_PX)
        return full * SENSOR_FULL_WIDTH_PX / float(frame_width)


def radiometric_key(shape: tuple[int, int], exposure: float | None) -> str:
    h, w = shape[:2]
    return f"{w}x{h}@{'auto' if exposure is None else int(round(exposure))}"


def today() -> str:
    return datetime.date.today().isoformat()


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
                    for k in ("lateral_um_per_px_full", "lateral_source", "lateral_date", "z_um_per_frame",
                              "z_source", "z_date", "z_uncertainty_pct", "revision")
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
        return (0.0722 * lin[..., 0] + 0.7152 * lin[..., 1] + 0.2126 * lin[..., 2]).astype(np.float32)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
