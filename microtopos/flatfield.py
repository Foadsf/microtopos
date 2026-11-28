"""Estimated flat field from a panned specimen, without a blank reference.

When the stage moves by a known displacement d between two frames, the same specimen point is seen at field
position x - d in the earlier frame and at x in the later one. The specimen's reflectance cancels in the ratio:

    log(raw_k(x) - dark) - log(raw_j(x - d) - dark) = log V(x) - log V(x - d) + c_jk

where V is the fixed shading of the optical path (vignetting, uneven Köhler illumination) and c_jk absorbs a
global brightness change between the two frames (lamp flicker, auto exposure). With log V modelled as a smooth
polynomial, every overlapping pair adds linear equations in the polynomial's coefficients; the estimator keeps only
the normal equations (a 14 x 14 matrix and a vector) and the previous accepted frame at reduced resolution, so the
live engine can feed it frame by frame. The result is an ESTIMATED flat field: it is derived from the specimen
and from the frame-to-frame motion the caller reports, not recorded on a uniform reference, and it is only as
good as that motion and its registration.

Choices (each stated so a reviewer can disagree with a reason):

* Pairwise log ratios rather than a per-pixel statistic (mean or median of log intensity over frames). A
  per-pixel statistic relies on every pixel seeing statistically identical specimen: its residual is the
  specimen's own large-scale contrast divided by the square root of the number of independent specimen regions
  a pixel has seen. In a development trial on the package's simulated specimen (grains about a fifth of the
  field across), the mean of log intensity still had 2-5 % RMS gain error after 4.5 field widths of travel; the
  ratio of registered frames cancels the specimen and, on the same frames, was limited to 0.1-0.4 % by noise and
  registration error. A median has the same dependence on specimen statistics and needs every frame in memory.
* Brightness: a global brightness change between two frames (lamp flicker) is indistinguishable from a linear
  tilt of log V along a chain of frames, so the exposure and lamp must be steady; flicker is treated as noise and
  its size is reported (`brightness_rms`). Automatic exposure violates this assumption.
* Per-pixel exclusion: pixels at or above `SATURATED_LUMA` and pixels within `BLACK_FLOOR` of the dark level are
  excluded; a reduced-resolution cell enters a pair only when every pixel under it is valid in both frames. (A
  safety margin around saturated pixels made no measurable difference on simulated glints, so there is none;
  blooming on a real sensor has not been examined.)
* Motion gate: a frame is accepted only after the view has moved at least `min_step_fraction` of the shorter field
  side since the last accepted frame. Least squares already gives a short-step pair little weight, so the gate
  hardly changes the estimate of a real pan (a pan that stopped for half its frames gave the same error either
  way on the simulator). What it prevents is a stopped view counting as motion: without it, hand tremor makes
  every frame a pair with a sub-pixel step in a random direction, and after a few hundred frames a view that never
  moved passes the travel and two-direction tests (tests/test_flatfield.py shows this). It also skips the work
  for frames that add nothing.
* Coverage: a pair only constrains V along its displacement. Steps all along one line leave the profile across
  that line undetermined, so readiness requires the steps to point in at least two directions (`isotropy`).
* Smooth model: a 2-D polynomial of total degree 4 in coordinates normalised to [-1, 1]. Vignetting (cos^4-like
  falloff) and an illumination tilt are smooth and, in the log domain, closely represented by such a polynomial;
  15 terms cannot follow specimen texture or a local defect. A heavy Gaussian blur of a per-pixel statistic was
  rejected: it is biased at the field border, where vignetting is strongest.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

# Luma at or above this is treated as saturated (0..255 luma scale, as calibration.bgr_to_luma returns).
# live.py computes its saturation fraction with the same literal (raw_luma >= 250); the two must stay equal.
SATURATED_LUMA = 250.0
# Pixels with raw - dark below this many luma DN carry no usable shading information (the log is mostly noise).
BLACK_FLOOR = 4.0
# Total degree of the polynomial model of log V (15 terms; the constant term is not observable from ratios).
POLY_DEGREE = 4
# A pair enters the normal equations only when at least this fraction of the cells overlap validly.
MIN_PAIR_OVERLAP = 0.2

# Readiness thresholds (reasoning in FlatEstimator.progress).
MIN_PAIRS = 20
MIN_TRAVEL_FIELDS = 2.0
MIN_ISOTROPY = 0.3


def _exponents(degree: int) -> list[tuple[int, int]]:
    """(i, j) of the terms x^i y^j with 1 <= i + j <= degree (no constant term)."""
    return [(i, d - i) for d in range(1, degree + 1) for i in range(d, -1, -1)]


def _terms(x: np.ndarray, y: np.ndarray, exps: list[tuple[int, int]]) -> np.ndarray:
    return np.stack([x**i * y**j for i, j in exps], axis=-1)


class FlatEstimator:
    """Accumulate an estimated flat field from frames of a panned specimen and their measured motion.

    Feed it frame by frame with `add(luma, shift_px)`, ask `progress()` whether the motion so far suffices, and
    call `result()` for a flat field in the form `calibration.RadiometricCorrector` expects.

    shape: (h, w) of the luma frames (the same luma the corrector is applied to).
    downsample: integer reduction factor of the analysis grid.
    dark: dark frame (2-D, resized to `shape` if needed) or None for a zero dark level. It is subtracted before
        the logarithm, so it has to be known when accumulation starts; `result()` builds the flat on it.
    min_step_fraction: motion needed since the last accepted frame, as a fraction of the shorter field side.
        0 accepts every frame (no motion gate).
    """

    def __init__(self, shape: tuple[int, int], downsample: int = 4, dark: np.ndarray | None = None,
                 min_step_fraction: float = 0.1, saturation: float = SATURATED_LUMA):
        h, w = int(shape[0]), int(shape[1])
        if downsample < 1 or h // downsample < 16 or w // downsample < 16:
            raise ValueError(f"FlatEstimator needs a frame of at least 16 cells per side after downsampling, "
                             f"got shape {shape} with downsample {downsample}")
        self.shape = (h, w)
        self.downsample = int(downsample)
        gh, gw = h // self.downsample, w // self.downsample
        self.grid = (gh, gw)
        if dark is None:
            self.dark = None
        else:
            d = np.asarray(dark, np.float32)
            self.dark = d if d.shape == (h, w) else cv2.resize(d, (w, h), interpolation=cv2.INTER_AREA)
        self.min_step_px = float(min_step_fraction) * min(h, w)
        self.saturation = float(saturation)
        self._exps = _exponents(POLY_DEGREE)
        # analysis-cell centres in normalised full-frame coordinates (INTER_AREA cell i covers full-res pixels
        # i*w/gw .. (i+1)*w/gw, centre (i + 0.5) * w / gw - 0.5; pixel 0 maps to -1 and pixel w-1 to +1)
        xc = 2.0 * ((np.arange(gw) + 0.5) * w / gw - 0.5) / (w - 1) - 1.0
        yc = 2.0 * ((np.arange(gh) + 0.5) * h / gh - 0.5) / (h - 1) - 1.0
        self._xg, self._yg = np.meshgrid(xc, yc)
        self._tg = _terms(self._xg, self._yg, self._exps)
        n = len(self._exps)
        self._N = np.zeros((n, n))
        self._b = np.zeros(n)
        self._rr = 0.0       # sum of squared log ratios
        self._Smm, self._Smb, self._Sbb = np.zeros((n, n)), np.zeros(n), 0.0  # per-pair means (brightness check)
        self._nobs = 0       # number of cell equations
        self._pairs = 0
        self._frames = 0
        self._travel = 0.0   # sum of |step| over the pairs used, field units (x in widths, y in heights)
        self._steps = np.zeros(3)  # sums of sx^2, sy^2, sx*sy over the pairs used (field units)
        self._level = 0.0    # sum over accepted frames of the mean valid raw - dark
        self._pending = np.zeros(2)
        self._prev: tuple[np.ndarray, np.ndarray] | None = None

    # ------------------------------------------------------------------ accumulation
    def _reduce(self, img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(mean log(raw - dark) over the valid pixels of each cell, fraction of valid pixels per cell)."""
        sig = img - self.dark if self.dark is not None else img
        valid = (img < self.saturation) & (sig >= BLACK_FLOOR)
        if valid.any():
            self._level += float(np.mean(sig[valid]))
        logv = np.where(valid, np.log(np.maximum(sig, BLACK_FLOOR)), 0.0).astype(np.float32)
        gh, gw = self.grid
        frac = cv2.resize(valid.astype(np.float32), (gw, gh), interpolation=cv2.INTER_AREA)
        mean_log = cv2.resize(logv, (gw, gh), interpolation=cv2.INTER_AREA) / np.maximum(frac, 1e-6)
        return mean_log.astype(np.float32), frac

    def add(self, luma: np.ndarray, shift_px: tuple[float, float] | None) -> bool:
        """Offer one frame. shift_px: content shift (dx, dy) in full-resolution px since the previously OFFERED
        frame, in the convention of registration.register_pair (template = previous, moving = this frame):
        this(x) = previous(x - d). None for the first frame. Returns True when the frame was accepted."""
        img = np.asarray(luma, np.float32)
        if img.shape != self.shape:
            raise ValueError(f"frame shape {img.shape} does not match the estimator's {self.shape}")
        if shift_px is not None:
            self._pending += np.asarray(shift_px, np.float64)
        if self._prev is not None and math.hypot(*self._pending) < self.min_step_px:
            return False
        cur = self._reduce(img)
        if self._prev is not None:
            self._add_pair(self._prev, cur, self._pending.copy())
        self._pending[:] = 0.0
        self._prev = cur
        self._frames += 1
        return True

    def break_chain(self) -> None:
        """Forget the previous frame (the caller lost track of the motion): the next offered frame starts a new
        chain with shift None. Pairs already accumulated are kept."""
        self._prev = None
        self._pending[:] = 0.0

    def _add_pair(self, prev: tuple[np.ndarray, np.ndarray], cur: tuple[np.ndarray, np.ndarray],
                  d: np.ndarray) -> None:
        h, w = self.shape
        gh, gw = self.grid
        dxg, dyg = d[0] * gw / w, d[1] * gh / h
        # dst(x) = src(x - d): the previous frame resampled onto the current frame's specimen positions
        M = np.array([[1.0, 0.0, dxg], [0.0, 1.0, dyg]], np.float32)
        p_log = cv2.warpAffine(prev[0], M, (gw, gh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        p_ok = cv2.warpAffine(prev[1], M, (gw, gh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                              borderValue=0.0)
        ok = (cur[1] >= 0.999) & (p_ok >= 0.999)
        n = int(ok.sum())
        if n < MIN_PAIR_OVERLAP * gh * gw:
            return
        dxn, dyn = 2.0 * d[0] / (w - 1), 2.0 * d[1] / (h - 1)
        rows = self._tg[ok] - _terms(self._xg[ok] - dxn, self._yg[ok] - dyn, self._exps)
        rhs = (cur[0][ok] - p_log[ok]).astype(np.float64)
        # A global brightness change between the two frames (lamp flicker) adds a constant to rhs. It is NOT
        # projected out: a linear term of log V adds a . d, also a constant per pair, and along a chain of frames
        # it equals a per-frame brightness a . position, so a free brightness per frame would make the
        # illumination tilt unobservable. Brightness changes are treated as zero-mean noise instead, and their
        # size is reported (brightness_rms) from the per-pair mean residuals.
        mr, mb = rows.mean(axis=0), float(rhs.mean())
        self._Smm += np.outer(mr, mr)
        self._Smb += mr * mb
        self._Sbb += mb * mb
        self._N += rows.T @ rows
        self._b += rows.T @ rhs
        self._rr += float(rhs @ rhs)
        self._nobs += n
        self._pairs += 1
        sx, sy = d[0] / w, d[1] / h
        self._travel += math.hypot(sx, sy)
        self._steps += (sx * sx, sy * sy, sx * sy)

    # ------------------------------------------------------------------ readiness
    def _isotropy(self) -> float:
        """sqrt(minor / major eigenvalue) of the step second-moment matrix: 0 for steps along one line, 1 for
        steps spread evenly over all directions."""
        sxx, syy, sxy = self._steps
        tr, det = sxx + syy, sxx * syy - sxy * sxy
        if tr <= 0:
            return 0.0
        disc = math.sqrt(max(tr * tr / 4 - det, 0.0))
        lo, hi = tr / 2 - disc, tr / 2 + disc
        return math.sqrt(max(lo, 0.0) / hi)

    def progress(self) -> dict:
        """frames (accepted), pairs (overlapping accepted pairs used), travel (summed step length of those pairs in
        field units: x in widths, y in heights), isotropy (0 = all steps along one line, 1 = all directions
        alike), ready, and missing (what readiness still lacks, empty when ready).

        Thresholds: >= 20 pairs and >= 2 fields of travel, so that registration errors and noise, which differ
        from pair to pair, average down (on the simulated specimen with register_pair shifts the gain error
        is then below 1 %); isotropy >= 0.3, because the profile across a line of motion is unobservable and
        the gain along the weak direction is determined only in proportion to the motion along it."""
        iso = self._isotropy()
        missing = []
        if self._pairs < MIN_PAIRS:
            missing.append(f"pairs {self._pairs} < {MIN_PAIRS}")
        if self._travel < MIN_TRAVEL_FIELDS:
            missing.append(f"travel {self._travel:.2f} < {MIN_TRAVEL_FIELDS} fields")
        if iso < MIN_ISOTROPY:
            missing.append(f"isotropy {iso:.2f} < {MIN_ISOTROPY} (move in two directions, not along one line)")
        return {"frames": self._frames, "pairs": self._pairs, "travel": self._travel, "isotropy": iso,
                "ready": not missing, "missing": missing}

    # ------------------------------------------------------------------ result
    def result(self, level: float | None = None) -> dict:
        """Smooth gain field and a flat for RadiometricCorrector. Raises ValueError unless progress() is ready.

        level: flat-minus-dark level written into `flat` (default: mean raw - dark over the accepted frames).
        RadiometricCorrector divides by flat - dark normalised to its mean, so the level matters only for the
        corrector's clamp of flat - dark at 1 DN.

        Returns flat (float32, full resolution: dark + gain * level), gain (float32, mean 1), residual_rms (RMS
        of the pairwise log ratios about the fitted model, a relative quantity per analysis cell: noise,
        registration error and model misfit together), brightness_rms (RMS over pairs of the mean log ratio left
        after the fit: the frame-to-frame brightness change the tilt estimate had to absorb as noise),
        frames_used, pairs, travel, isotropy."""
        p = self.progress()
        if not p["ready"]:
            raise ValueError("estimated flat field not ready: " + "; ".join(p["missing"]))
        coef = np.linalg.solve(self._N, self._b)
        ssr = self._rr - 2.0 * float(coef @ self._b) + float(coef @ self._N @ coef)
        residual_rms = math.sqrt(max(ssr, 0.0) / self._nobs)
        bsr = self._Sbb - 2.0 * float(coef @ self._Smb) + float(coef @ self._Smm @ coef)
        brightness_rms = math.sqrt(max(bsr, 0.0) / self._pairs)
        h, w = self.shape
        x = np.linspace(-1.0, 1.0, w)
        y = np.linspace(-1.0, 1.0, h)
        log_gain = np.zeros((h, w))
        for c, (i, j) in zip(coef, self._exps):
            log_gain += c * np.outer(y**j, x**i)  # separable evaluation: memory h*w, not h*w*terms
        gain = np.exp(log_gain - log_gain.max())
        gain = (gain / gain.mean()).astype(np.float32)
        lvl = float(level) if level is not None else self._level / self._frames
        dark = self.dark if self.dark is not None else np.zeros(self.shape, np.float32)
        return {"flat": (dark + gain * lvl).astype(np.float32), "gain": gain, "residual_rms": residual_rms,
                "brightness_rms": brightness_rms, "frames_used": self._frames, "pairs": self._pairs, "travel": self._travel, "isotropy": p["isotropy"]}
