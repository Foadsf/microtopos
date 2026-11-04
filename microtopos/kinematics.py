"""Live kinematic state detection from video alone: DWELL, PAN, Z_SWEEP, OBJECTIVE_CHANGE.

Every decision uses features normalised by their own recent noise (median / MAD over quiet frames),
with entry and exit counts (hysteresis), so thresholds do not depend on scene, objective or exposure.
"""

from __future__ import annotations

import collections
import math
from dataclasses import dataclass
from enum import Enum

import cv2
import numpy as np

from .focus import brenner
from .registration import phase_shift


class State(str, Enum):
    DWELL = "DWELL"
    PAN = "PAN"
    Z_SWEEP = "Z_SWEEP"
    OBJECTIVE_CHANGE = "OBJECTIVE_CHANGE"


class Override(str, Enum):
    AUTO = "Auto"
    PAN = "Pan"
    Z = "Z capture"
    PAUSE = "Pause"


@dataclass
class Features:
    shift_px: float  # in the analysis image's pixels, relative to the previous frame
    response: float  # phase-correlation peak
    dlog_brenner: float  # change of log sharpness vs previous frame
    luma_change: float  # relative change of mean brightness
    brenner: float
    z_score: float


@dataclass
class Event:
    kind: str  # "sweep_started", "sweep_ended", "objective_change"
    detail: dict


def spectral_centroid(img: np.ndarray) -> float:
    """Radial centroid of the magnitude spectrum: a texture-scale signature used to guess magnification."""
    f = np.abs(np.fft.fftshift(np.fft.fft2(img * cv2.createHanningWindow(img.shape[::-1], cv2.CV_32F))))
    h, w = f.shape
    yy, xx = np.mgrid[0:h, 0:w]
    r = np.hypot((xx - w / 2) / w, (yy - h / 2) / h)
    f[r < 0.02] = 0
    return float((f * r).sum() / (f.sum() + 1e-9))


class StateMachine:
    def __init__(self, analysis_width: int = 320):
        self.analysis_width = analysis_width
        self.state = State.DWELL
        self.override = Override.AUTO
        self.prev = None
        self.prev_b = None
        self.prev_luma = None
        self.quiet_shift = collections.deque(maxlen=120)
        self.quiet_db = collections.deque(maxlen=120)
        self.quiet_resp = collections.deque(maxlen=120)
        self.cut_before = None
        self.sweep_complete = True  # set by the caller each frame during a sweep (live completeness >= 0.5)
        self.trend = collections.deque(maxlen=6)
        self.enter_count = 0
        self.exit_count = 0
        self.lost_count = 0
        self.centroid_before = None
        self.centroid_hist = collections.deque(maxlen=10)

    def reset(self) -> None:
        self.__init__(self.analysis_width)

    def _small(self, luma: np.ndarray) -> np.ndarray:
        h, w = luma.shape
        sw = self.analysis_width
        return cv2.resize(luma, (sw, max(8, int(round(h * sw / w)))), interpolation=cv2.INTER_AREA)

    @staticmethod
    def _robust(values, floor):
        if len(values) < 10:
            return None, None
        a = np.asarray(values)
        med = float(np.median(a))
        return med, max(floor, 1.4826 * float(np.median(np.abs(a - med))))

    def update(self, luma: np.ndarray) -> tuple[State, Features, list[Event]]:
        s = self._small(luma)
        b = brenner(s)
        luma_mean = float(s.mean())
        events: list[Event] = []
        if self.prev is None:
            self.prev, self.prev_b, self.prev_luma = s, b, luma_mean
            self.centroid_hist.append(spectral_centroid(s))
            return self.state, Features(0, 1, 0, 0, b, 0), events
        dx, dy, resp = _phase(self.prev, s)
        shift = math.hypot(dx, dy)
        dlb = math.log((b + 1e-6) / (self.prev_b + 1e-6))
        dl = abs(luma_mean - self.prev_luma) / max(self.prev_luma, 1.0)
        before = self.prev
        self.prev, self.prev_b, self.prev_luma = s, b, luma_mean

        base_resp, _ = self._robust(self.quiet_resp, 0.01)
        cut = resp < max(0.08, 0.25 * (base_resp or 0.5))  # the view decorrelated from one frame to the next
        if cut:
            # a fast, motion-blurred pan also lowers the phase peak: if the band-passed frames still match well
            # at some shift, the stage moved; only a view that matches nowhere is a cut
            mdx, mdy, q = motion_match(before, s)
            if q > 0.2:  # unrelated views score |q| < 0.1
                cut, dx, dy, resp = False, mdx, mdy, max(resp, 0.081)
                shift = math.hypot(dx, dy)
        _, sd_db = self._robust(self.quiet_db, 0.004)
        sd_db = sd_db or 0.02
        if cut or dl > 0.05:
            self.trend.clear()  # a discontinuity or an illumination change is not a focus trend
        else:
            self.trend.append(dlb)
        # sustained same-sign sharpness change over the last frames, in units of the quiet noise
        trend = float(np.sum(self.trend)) / (sd_db * math.sqrt(len(self.trend))) if self.trend else 0.0
        _, sd_sh = self._robust(self.quiet_shift, 0.05)
        sd_sh = sd_sh or 0.3
        moving = shift > max(1.0, 6 * sd_sh) and resp > 0.08
        z_like = abs(trend) > 4.0 and not moving and dl < 0.05 and not cut
        feats = Features(shift, resp, dlb, dl, b, trend)

        if self.override != Override.AUTO:
            forced = {Override.PAN: State.PAN, Override.Z: State.Z_SWEEP, Override.PAUSE: State.DWELL}[self.override]
            if forced != self.state:
                if self.state == State.Z_SWEEP:
                    events.append(Event("sweep_ended", {}))
                if forced == State.Z_SWEEP:
                    events.append(Event("sweep_started", {}))
                self.state = forced
            return self.state, feats, events

        if self.state == State.DWELL and not moving and not cut and abs(dlb) < 3 * sd_db + 0.02:
            self.quiet_shift.append(shift)
            self.quiet_db.append(dlb)
            self.quiet_resp.append(resp)
            self.centroid_hist.append(spectral_centroid(s))

        # scene cut (objective turned, light blocked, stage jumped): wait for the view to settle, then compare
        # it with the last view before the cut
        if cut and self.state != State.OBJECTIVE_CHANGE:
            self.cut_before = before
            self.centroid_before = float(np.median(self.centroid_hist)) if self.centroid_hist else None
            if self.state == State.Z_SWEEP:
                events.append(Event("sweep_ended", {"aborted": True}))
            self.state, self.exit_count = State.OBJECTIVE_CHANGE, 0
            return self.state, feats, events
        if self.state == State.OBJECTIVE_CHANGE:
            self.exit_count = 0 if cut else self.exit_count + 1
            if self.exit_count >= 8:
                _, _, r_back = _phase(self.cut_before, s)
                if r_back < max(0.08, 0.25 * (base_resp or 0.5)):  # a different view, not a brief occlusion
                    after = spectral_centroid(s)
                    ratio = (self.centroid_before / after) if self.centroid_before and after > 0 else None
                    events.append(Event("objective_change", {"magnification_ratio": ratio}))
                self.state = State.DWELL
                self.trend.clear()
                self.quiet_shift.clear()
                self.quiet_db.clear()
                self.quiet_resp.clear()
                self.centroid_hist.clear()
            return self.state, feats, events

        if self.state == State.DWELL:
            if moving:
                self.enter_count += 1
                if self.enter_count >= 2:
                    self.state, self.enter_count = State.PAN, 0
            elif z_like:
                self.enter_count += 1
                if self.enter_count >= 2:
                    self.state, self.enter_count, self.exit_count = State.Z_SWEEP, 0, 0
                    events.append(Event("sweep_started", {}))
            else:
                self.enter_count = 0
        elif self.state == State.PAN:
            self.exit_count = 0 if moving else self.exit_count + 1
            if self.exit_count >= 5:
                self.state, self.exit_count = State.DWELL, 0
        elif self.state == State.Z_SWEEP:
            self.enter_count = self.enter_count + 1 if moving else 0
            if self.enter_count >= 3:  # the stage really moved during the sweep: abort it
                events.append(Event("sweep_ended", {"aborted": True}))
                self.state, self.exit_count, self.enter_count = State.PAN, 0, 0
            elif not moving:
                self.exit_count = self.exit_count + 1 if abs(dlb) < 3 * sd_db + 0.01 else 0
                # an incomplete sweep (field not yet sharp-then-blurry) stays open longer: in the blank,
                # fully defocused zone the image barely changes even while the knob is still turning
                if self.exit_count >= (16 if self.sweep_complete else 40):
                    events.append(Event("sweep_ended", {}))
                    self.state, self.exit_count = State.DWELL, 0
                    self.trend.clear()
        return self.state, feats, events


def _phase(a: np.ndarray, b: np.ndarray):
    return phase_shift(a, b)


def _band(img: np.ndarray) -> np.ndarray:
    f = img.astype(np.float32)
    return cv2.GaussianBlur(f, (0, 0), 2.5) - cv2.GaussianBlur(f, (0, 0), 10.0)


def motion_match(a: np.ndarray, b: np.ndarray) -> tuple[float, float, float]:
    """Shift of b relative to a from band-passed phase correlation, and the NCC of the overlap there.
    Returns q = 0 when the overlap is under 40 % of the frame (too fast to tell motion from a cut)."""
    h, w = a.shape
    ba, bb = _band(a), _band(b)
    win = cv2.createHanningWindow((w, h), cv2.CV_32F)
    dx, dy, _ = phase_shift(ba, bb, win)
    ix, iy = int(round(dx)), int(round(dy))
    ow, oh = w - abs(ix), h - abs(iy)
    if ow * oh < 0.4 * w * h:
        return dx, dy, 0.0
    # phase_shift convention: b(x) = a(x - dx)
    pa = ba[max(0, -iy):max(0, -iy) + oh, max(0, -ix):max(0, -ix) + ow]
    pb = bb[max(0, iy):max(0, iy) + oh, max(0, ix):max(0, ix) + ow]
    pa, pb = pa - pa.mean(), pb - pb.mean()
    q = float((pa * pb).sum() / (math.sqrt(float((pa * pa).sum()) * float((pb * pb).sum())) + 1e-9))
    return dx, dy, q


def guess_objective(current_mag: float, ratio: float | None, magnifications: list[float]) -> int | None:
    """Index of the objective whose magnification ratio to the current one best matches the measured ratio."""
    if not ratio or ratio <= 0:
        return None
    errs = [abs(math.log(m / current_mag) - math.log(ratio)) for m in magnifications]
    return int(np.argmin(errs))
