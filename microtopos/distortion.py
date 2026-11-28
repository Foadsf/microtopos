"""Estimated radial lens distortion from a panned specimen, without a calibration target.

When the stage translates by t between two frames, a specimen point at undistorted field position u in the earlier
frame is at u + t in the later one. The camera observes D(u) and D(u + t), where D is the lens distortion. Tracking
features between the two frames gives pairs of observed positions (p_A, p_B), and for the true D

    U(p_B) - U(p_A) = t        (U = D^-1, the undistortion)

for every match of the pair. A translation is the same at every field position, so any field dependence of
p_B - p_A that U does not remove is distortion. Many pairs constrain D; no target is needed. The result is an
ESTIMATED radial distortion: it is derived from the specimen and the frame-to-frame motion and rests on the
assumptions below; it is not a calibration against a reference artefact.

Model and conventions (each stated so a reviewer can disagree with a reason):

* Brown's radial polynomial in the forward direction, undistorted -> observed:

      p = c + R * x * (1 + k1 r^2 + k2 r^4),   x = (u - c) / R,   r = |x|

  c = ((w - 1) / 2, (h - 1) / 2) is the image centre (pixel-centre coordinates) and R = |c| the half-diagonal, so
  r = 1 exactly at the centres of the four corner pixels. This is OpenCV's model with camera matrix
  [[R, 0, c_x], [0, R, c_y]] and distortion coefficients (k1, k2, 0, 0, 0), so the coefficients can be handed to
  cv2.undistort / cv2.initUndistortRectifyMap unchanged (tests/test_distortion.py checks the maps against
  OpenCV's). The forward direction is what an undistortion map needs: for every pixel of the undistorted output
  it gives the observed position to sample, in closed form, so `undistort_maps` involves no approximation beyond
  the interpolation cv2.remap does. The division model was not chosen: with one coefficient both directions are
  closed form, but it is not OpenCV's convention, and with two coefficients its forward direction is not closed
  form either.
* The estimation needs U (observed -> undistorted). It is computed per point by Newton's method on the radius,
  r_u (1 + k1 r_u^2 + k2 r_u^4) = r_d, iterated until the radius changes by less than 1e-12 (normalised units,
  about 1e-9 px); this converges because the model is required to be monotonic over the field
  (|k1|, |k2| <= K_BOUND, whose derivative 1 + 3 k1 r^2 + 5 k2 r^4 stays >= 0.2 out to r = 1).
* Distortion centre fixed at the image centre: a decentred lens (or a sensor offset) is not modelled; its
  effect shows as a larger residual, not as a separate parameter.
* Per-pair translation only: each pair has its own free translation t (the reported stage shift only seeds the
  feature tracker), so neither hand tremor nor registration error of the reported shift biases k. A rotation of
  the specimen or a magnification change between the two frames (focus breathing) is NOT modelled; the focus
  must be held during the pan.
* Readiness counts pairs, matches and where in the field the matches lie (`field_coverage` over an 8 x 6 grid,
  and each of the four corner cells, where radial distortion is largest). A view that has not moved gives no
  pairs at all (motion gate, as in flatfield.FlatEstimator), because U(p_B) - U(p_A) = 0 holds for every D.
* Tracking image: features are tracked (Shi-Tomasi corners, pyramidal Lucas-Kanade, forward-backward checked) on
  the high-passed logarithm of the frame, because field-fixed shading (vignetting) otherwise biases the tracker
  in a pattern that mimics barrel distortion (numbers in _tracking_image).
* Uncertainty and significance: the statistical standard uncertainty of the corner displacement is a
  cluster-robust (sandwich) estimate with the matches of one pair as a cluster, because matches from one frame
  pair share that pair's noise, JPEG blocks and tracking errors. A Type B relative term (TRACKER_REL_U) for the
  tracker's systematic error is added in quadrature. The result is refused as "no measurable distortion" when the
  corner displacement does not exceed its expanded uncertainty, COVERAGE_FACTOR = 3 standard uncertainties.
* Verified only on the package's simulated specimen with a distortion applied by resampling; nothing here has run
  on a real camera.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np
import scipy.sparse as sp
from scipy.optimize import least_squares

# Bound on |k1| and |k2| in the fit: keeps the model monotonic out to r = 1 (see module docstring) and the
# Newton inversion convergent. 0.1 is far beyond microscope distortion (typically below 1-2 % at the corner).
K_BOUND = 0.1
# Coverage factor of the expanded uncertainty used for the significance decision.
COVERAGE_FACTOR = 3.0
# Type B relative standard uncertainty of the corner displacement for the tracker's systematic error, which the
# statistical (cluster-robust) part does not see. Measured on simulated pans with a known distortion of about 3 px
# at the corner (320 x 240, 34 pairs each), the magnitude error was +0.6 to +2.6 % for distortion applied by
# remapping at render resolution (seeds 1-7), -1.6 to +3.0 % for distortion applied at 3x resolution and
# area-reduced (seeds 1-3), +0.6 to +3.6 % for noise-free frames (seeds 1-3), while the statistical part was
# 0.6-1.3 %. The error depends on how the frames were formed (resampling blur and aliasing differ between the two
# ends of a match), so it is not corrected, only accounted for.
TRACKER_REL_U = 0.02
# Robust loss scale (px, undistorted coordinates): matches off by more than this are progressively down-weighted.
ROBUST_SCALE_PX = 0.3
# A match whose final residual exceeds this many robust scales is an outlier for the before/after statistics.
OUTLIER_SCALES = 5.0

# Feature tracking (per accepted pair).
MAX_FEATURES = 300          # corners sought on the earlier frame of a pair
FB_MAX_PX = 0.2             # forward-backward tracking disagreement above this rejects a match
LK_WIN = 21
LK_LEVELS = 3
LK_ITERS = 40
LK_EPS = 0.003
MIN_PAIR_MATCHES = 40       # a pair with fewer valid matches is discarded
TRACK_GAIN = 100.0          # 8-bit DN per unit of log intensity in the tracking image (+/-1.28 before clipping)
TRACK_SIGMA = LK_WIN / 3.0  # high-pass scale of the tracking image (px): the tracking window's own scale
EDGE_MARGIN = LK_WIN / 2.0  # px from the frame border inside which no match endpoint is kept

# Readiness thresholds (reasoning in DistortionEstimator.progress).
MIN_PAIRS = 12
MIN_MATCHES = 2000
MIN_COVERAGE = 0.75
GRID = (8, 6)               # field-coverage grid, columns x rows
MIN_CELL_MATCHES = 10


@dataclass(frozen=True)
class RadialModel:
    """Brown radial distortion k1, k2 of a frame of `shape` (h, w), normalised to the half-diagonal (r = 1 at the
    corner pixel centres) about the image centre. Forward direction: undistorted -> observed."""

    shape: tuple[int, int]
    k1: float
    k2: float = 0.0

    @property
    def centre(self) -> tuple[float, float]:
        h, w = self.shape
        return (w - 1) / 2.0, (h - 1) / 2.0

    @property
    def radius(self) -> float:
        cx, cy = self.centre
        return math.hypot(cx, cy)

    @property
    def corner_displacement_px(self) -> float:
        """Signed radial displacement of a corner, D(u) - u at r = 1, in px: negative = barrel (towards the
        centre), positive = pincushion."""
        return self.radius * (self.k1 + self.k2)

    def distort_points(self, u_px: np.ndarray) -> np.ndarray:
        """Observed positions of undistorted points (N x 2 px). Closed form."""
        return _distort(np.asarray(u_px, np.float64), self.centre, self.radius, self.k1, self.k2)

    def undistort_points(self, p_px: np.ndarray) -> np.ndarray:
        """Undistorted positions of observed points (N x 2 px), by Newton's method on the radius (converged to
        1e-12 in normalised radius)."""
        return _undistort(np.asarray(p_px, np.float64), self.centre, self.radius, self.k1, self.k2)[0]

    def undistort_maps(self, shape: tuple[int, int] | None = None) -> tuple[np.ndarray, np.ndarray]:
        """(map_x, map_y) float32 for cv2.remap(observed, map_x, map_y, ...) -> undistorted image.

        Each output pixel u samples the observed frame at D(u): the forward model, exact (no inversion). The
        output has the model's own geometry (same centre and scale); for pincushion distortion the output corners
        sample outside the observed frame and take the remap border value. shape must be the model's shape."""
        if shape is not None and tuple(shape[:2]) != tuple(self.shape):
            raise ValueError(f"maps requested for shape {tuple(shape[:2])}, model is for {self.shape}")
        h, w = self.shape
        cx, cy = self.centre
        R = self.radius
        x = (np.arange(w, dtype=np.float64) - cx) / R
        y = (np.arange(h, dtype=np.float64) - cy) / R
        xx, yy = np.meshgrid(x, y)
        r2 = xx * xx + yy * yy
        f = 1.0 + self.k1 * r2 + self.k2 * r2 * r2
        return (cx + R * xx * f).astype(np.float32), (cy + R * yy * f).astype(np.float32)

    def undistort(self, img: np.ndarray, border: int = cv2.BORDER_REPLICATE) -> np.ndarray:
        """The undistorted image (bilinear resampling of the observed frame at D(u))."""
        mx, my = self.undistort_maps(img.shape[:2])
        return cv2.remap(img, mx, my, cv2.INTER_LINEAR, borderMode=border)

    def to_json(self) -> dict:
        h, w = self.shape
        return {"model": "brown-radial", "normalisation": "half-diagonal", "width": int(w), "height": int(h),
                "k1": float(self.k1), "k2": float(self.k2)}

    @classmethod
    def from_json(cls, d: dict) -> RadialModel:
        if d.get("model") != "brown-radial" or d.get("normalisation") != "half-diagonal":
            raise ValueError(f"not a half-diagonal Brown radial model: {d.get('model')!r}, {d.get('normalisation')!r}")
        return cls((int(d["height"]), int(d["width"])), float(d["k1"]), float(d["k2"]))


def _distort(u: np.ndarray, c, R: float, k1: float, k2: float) -> np.ndarray:
    x = (u - np.asarray(c)) / R
    r2 = np.sum(x * x, axis=-1, keepdims=True)
    return np.asarray(c) + R * x * (1.0 + k1 * r2 + k2 * r2 * r2)


def _undistort(p: np.ndarray, c, R: float, k1: float, k2: float):
    """(undistorted px, dU/dk1 px, dU/dk2 px) for observed points p (N x 2)."""
    xd = (p - np.asarray(c)) / R
    rd = np.sqrt(np.sum(xd * xd, axis=-1))
    ru = rd.copy()
    for _ in range(30):
        r2 = ru * ru
        f = ru * (1.0 + k1 * r2 + k2 * r2 * r2) - rd
        fp = 1.0 + 3.0 * k1 * r2 + 5.0 * k2 * r2 * r2
        step = f / fp
        ru -= step
        if np.max(np.abs(step), initial=0.0) < 1e-12:
            break
    r2 = ru * ru
    fp = 1.0 + 3.0 * k1 * r2 + 5.0 * k2 * r2 * r2
    safe = np.where(rd > 0, rd, 1.0)
    s = np.where(rd > 0, ru / safe, 1.0)  # U = c + R * xd * (ru / rd); ru / rd -> 1 at r = 0
    # implicit differentiation of ru (1 + k1 ru^2 + k2 ru^4) = rd at fixed rd
    dru1, dru2 = -ru**3 / fp, -ru**5 / fp
    u = np.asarray(c) + R * xd * s[:, None]
    du1 = R * xd * (dru1 / safe)[:, None]
    du2 = R * xd * (dru2 / safe)[:, None]
    return u, du1, du2


@dataclass
class DistortionResult:
    """An estimated radial distortion and the statistics it rests on (see DistortionEstimator.result)."""

    model: RadialModel
    k_cov: tuple[tuple[float, float], tuple[float, float]]   # statistical covariance of (k1, k2); k2 zero if not fitted
    corner_u_px: float            # combined standard uncertainty of the corner displacement (statistical + TRACKER_REL_U)
    residual_before_px: float     # RMS match residual under pure per-pair translation (no distortion model)
    residual_after_px: float      # RMS match residual with the estimated distortion
    pairs: int
    matches: int
    inliers: int
    field_coverage: float

    @property
    def k1(self) -> float:
        return self.model.k1

    @property
    def k2(self) -> float:
        return self.model.k2

    @property
    def k1_u(self) -> float:
        return math.sqrt(self.k_cov[0][0])

    @property
    def k2_u(self) -> float:
        return math.sqrt(self.k_cov[1][1])

    @property
    def corner_displacement_px(self) -> float:
        return self.model.corner_displacement_px

    @property
    def corner_expanded_px(self) -> float:
        return COVERAGE_FACTOR * self.corner_u_px

    def undistort_maps(self, shape: tuple[int, int] | None = None):
        return self.model.undistort_maps(shape)

    def undistort(self, img: np.ndarray, border: int = cv2.BORDER_REPLICATE) -> np.ndarray:
        return self.model.undistort(img, border)

    def to_json(self) -> dict:
        return {"kind": "estimated radial distortion (from panning)", "distortion": self.model.to_json(),
                "k_cov": [[float(v) for v in row] for row in self.k_cov], "corner_u_px": float(self.corner_u_px),
                "coverage_factor": COVERAGE_FACTOR,
                "residual_before_px": float(self.residual_before_px),
                "residual_after_px": float(self.residual_after_px), "pairs": int(self.pairs),
                "matches": int(self.matches), "inliers": int(self.inliers),
                "field_coverage": float(self.field_coverage)}

    @classmethod
    def from_json(cls, d: dict) -> DistortionResult:
        cov = d["k_cov"]
        return cls(RadialModel.from_json(d["distortion"]),
                   ((float(cov[0][0]), float(cov[0][1])), (float(cov[1][0]), float(cov[1][1]))),
                   float(d["corner_u_px"]), float(d["residual_before_px"]), float(d["residual_after_px"]),
                   int(d["pairs"]), int(d["matches"]), int(d["inliers"]), float(d["field_coverage"]))


class NoMeasurableDistortion(ValueError):
    """The data sufficed, but the estimated corner displacement does not exceed its expanded uncertainty.

    Not a failure: the measurement says the distortion is below `bound_px` (|corner displacement| + expanded
    uncertainty) at the corner. Carries the rejected estimate for display, never for correction."""

    def __init__(self, corner_displacement_px: float, expanded_px: float, pairs: int):
        self.corner_displacement_px = corner_displacement_px
        self.expanded_px = expanded_px
        self.bound_px = abs(corner_displacement_px) + expanded_px
        self.pairs = pairs
        super().__init__(f"no measurable distortion: corner displacement {corner_displacement_px:+.3f} px is within "
                         f"its expanded uncertainty {expanded_px:.3f} px (coverage factor {COVERAGE_FACTOR:g}, "
                         f"{pairs} pairs); |distortion| at the corner is below {self.bound_px:.3f} px")


def _tracking_image(img: np.ndarray, dark: float) -> np.ndarray:
    """8-bit image for the tracker: log(frame - dark) minus its Gaussian blur at TRACK_SIGMA, scaled by TRACK_GAIN
    and offset to mid-grey.

    Shading fixed to the field (vignetting) is a factor g(x) on the specimen's reflectance. As a feature moves
    across the field the shading ramp inside its tracking window changes, and Lucas-Kanade turns that change into
    a displacement error that grows towards the field edge, i.e. it mimics barrel distortion. In the logarithm the
    factor becomes an additive smooth term, and a high-pass at the scale of the tracking window removes what is
    left of it inside the window. Measured on the simulator's vignette (distortion-free pans, three seeds, corner
    displacement): -0.035 to -0.078 px tracking the raw frames; -0.025 to -0.040 px with the log high-passed at
    registration.remove_shading's frame scale (1/8 of the short side, too coarse to remove the shading's curvature
    within a window); -0.004 to -0.019 px with this step, against -0.003 to -0.012 px for the same pans rendered
    without vignetting."""
    sig = np.maximum(np.asarray(img, np.float32) - np.float32(dark), 1.0)
    log = np.log(sig)
    return np.clip(TRACK_GAIN * (log - cv2.GaussianBlur(log, (0, 0), TRACK_SIGMA)) + 128.5, 0, 255).astype(np.uint8)


class DistortionEstimator:
    """Accumulate feature matches between frames of a panned specimen and estimate radial distortion from them.

    Feed it frame by frame with `add(luma, shift_px)` (as flatfield.FlatEstimator), ask `progress()` whether the
    data so far suffices, and call `result()`.

    shape: (h, w) of the luma frames.
    terms: 1 fits k1 only, 2 fits k1 and k2.
    min_step_fraction: motion needed since the last accepted frame, as a fraction of the shorter field side.
    max_pairs: accepted pairs kept; once reached, further frames are ignored (bounded memory and solve time).
    dark: dark level (luma DN) subtracted before the logarithm of the tracking image; 0 if unknown (an offset
        left in makes the vignetting removal approximate, see _tracking_image).
    """

    def __init__(self, shape: tuple[int, int], terms: int = 2, min_step_fraction: float = 0.1,
                 max_pairs: int = 200, dark: float = 0.0):
        h, w = int(shape[0]), int(shape[1])
        if min(h, w) < 4 * LK_WIN:
            raise ValueError(f"DistortionEstimator needs frames of at least {4 * LK_WIN} px per side, got {shape}")
        if terms not in (1, 2):
            raise ValueError("terms must be 1 (k1) or 2 (k1, k2)")
        self.shape = (h, w)
        self.terms = int(terms)
        self.min_step_px = float(min_step_fraction) * min(h, w)
        self.max_pairs = int(max_pairs)
        self.dark = float(dark)
        self._pairs: list[tuple[np.ndarray, np.ndarray]] = []   # (p_A, p_B) observed px, float64 N x 2
        self._cells = np.zeros((GRID[1], GRID[0]), np.int64)
        self._frames = 0
        self._pending = np.zeros(2)
        self._prev: tuple[np.ndarray, np.ndarray] | None = None  # (u8 frame, its corners N x 1 x 2 float32)
        self._min_dist = max(4.0, min(h, w) / 40.0)
        # match endpoints keep this far from the frame border, in both frames (see EDGE_MARGIN)
        self._margin = float(EDGE_MARGIN)
        m = math.ceil(self._margin)
        self._mask = np.zeros((h, w), np.uint8)
        self._mask[m:h - m, m:w - m] = 255

    # ------------------------------------------------------------------ accumulation
    def add(self, luma: np.ndarray, shift_px: tuple[float, float] | None) -> bool:
        """Offer one frame. shift_px: content shift (dx, dy) in px since the previously OFFERED frame, in the
        convention of registration.register_pair (template = previous, moving = this frame): a specimen point at
        p in the previous frame is near p + d in this one. None for the first frame. It seeds the tracker and
        drives the motion gate; the fit has its own translation per pair. Returns True when the frame was
        accepted."""
        img = np.asarray(luma)
        if img.shape[:2] != self.shape or img.ndim != 2:
            raise ValueError(f"frame shape {img.shape} does not match the estimator's {self.shape} (2-D luma)")
        if len(self._pairs) >= self.max_pairs:
            return False
        if shift_px is not None:
            self._pending += np.asarray(shift_px, np.float64)
        if self._prev is not None and math.hypot(*self._pending) < self.min_step_px:
            return False
        cur = _tracking_image(img, self.dark)
        if self._prev is not None:
            self._add_pair(self._prev, cur, self._pending.copy())
        self._pending[:] = 0.0
        corners = cv2.goodFeaturesToTrack(cur, MAX_FEATURES, 0.01, self._min_dist, mask=self._mask, blockSize=7)
        self._prev = (cur, corners if corners is not None else np.zeros((0, 1, 2), np.float32))
        self._frames += 1
        return True

    def _add_pair(self, prev: tuple[np.ndarray, np.ndarray], cur: np.ndarray, d: np.ndarray) -> None:
        a_img, pa = prev
        if len(pa) == 0:
            return
        h, w = self.shape
        guess = (pa + d.astype(np.float32)[None, None, :]).astype(np.float32)
        margin = self._margin
        inside = ((guess[:, 0, 0] >= margin) & (guess[:, 0, 0] <= w - 1 - margin)
                  & (guess[:, 0, 1] >= margin) & (guess[:, 0, 1] <= h - 1 - margin))
        pa, guess = pa[inside], guess[inside]
        if len(pa) < MIN_PAIR_MATCHES:
            return
        crit = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, LK_ITERS, LK_EPS)
        lk = {"winSize": (LK_WIN, LK_WIN), "maxLevel": LK_LEVELS, "criteria": crit,
              "flags": cv2.OPTFLOW_USE_INITIAL_FLOW}
        pb, st, _ = cv2.calcOpticalFlowPyrLK(a_img, cur, pa, guess.copy(), **lk)
        back, st2, _ = cv2.calcOpticalFlowPyrLK(cur, a_img, pb, pa.copy(), **lk)
        fb = np.linalg.norm((back - pa)[:, 0, :], axis=1)
        q = pb[:, 0, :]
        ok = ((st[:, 0] == 1) & (st2[:, 0] == 1) & (fb < FB_MAX_PX)
              & (q[:, 0] >= margin) & (q[:, 0] <= w - 1 - margin) & (q[:, 1] >= margin) & (q[:, 1] <= h - 1 - margin))
        if int(ok.sum()) < MIN_PAIR_MATCHES:
            return
        A = pa[ok, 0, :].astype(np.float64)
        B = q[ok].astype(np.float64)
        self._pairs.append((A, B))
        for P in (A, B):
            gx = np.clip((P[:, 0] * GRID[0] / w).astype(int), 0, GRID[0] - 1)
            gy = np.clip((P[:, 1] * GRID[1] / h).astype(int), 0, GRID[1] - 1)
            np.add.at(self._cells, (gy, gx), 1)

    # ------------------------------------------------------------------ readiness
    def _coverage(self) -> float:
        return float(np.mean(self._cells >= MIN_CELL_MATCHES))

    def progress(self) -> dict:
        """frames (accepted), pairs (accepted pairs with enough matches), matches (over those pairs),
        field_coverage (fraction of the 8 x 6 field cells holding >= 10 match endpoints), corners (how many of the
        four corner cells do), ready, and missing (what readiness still lacks, empty when ready).

        Thresholds: >= 12 pairs, because the significance test treats a pair as one cluster of correlated errors
        and a cluster-robust variance from fewer clusters is itself too uncertain to decide on; >= 2000 matches;
        coverage >= 0.75 and all four corner cells, because radial distortion grows with r^3 and is determined
        mainly by matches far from the centre."""
        c = self._cells
        corners = int(sum(c[i, j] >= MIN_CELL_MATCHES for i in (0, -1) for j in (0, -1)))
        matches = sum(len(a) for a, _ in self._pairs)
        cov = self._coverage()
        missing = []
        if len(self._pairs) < MIN_PAIRS:
            missing.append(f"pairs {len(self._pairs)} < {MIN_PAIRS} (pan the specimen; a still view gives none)")
        if matches < MIN_MATCHES:
            missing.append(f"matches {matches} < {MIN_MATCHES}")
        if cov < MIN_COVERAGE:
            missing.append(f"field coverage {cov:.2f} < {MIN_COVERAGE}")
        if corners < 4:
            missing.append(f"matches in {corners} of 4 field corners")
        return {"frames": self._frames, "pairs": len(self._pairs), "matches": matches, "field_coverage": cov,
                "corners": corners, "ready": not missing, "missing": missing}

    # ------------------------------------------------------------------ solve
    def _residuals(self, k: np.ndarray, t: np.ndarray, want_jac: bool):
        """Residuals U(p_B) - U(p_A) - t_pair (px), stacked as [x0, y0, x1, y1, ...] per match, and their Jacobian
        with respect to (k[:terms], t)."""
        geom = RadialModel(self.shape, 0.0)  # the one definition of centre and normalisation radius
        c, R = geom.centre, geom.radius
        k1 = float(k[0])
        k2 = float(k[1]) if self.terms == 2 else 0.0
        A = self._A_all
        B = self._B_all
        uA, a1, a2 = _undistort(A, c, R, k1, k2)
        uB, b1, b2 = _undistort(B, c, R, k1, k2)
        e = uB - uA - t[self._pid]
        if not want_jac:
            return e.ravel(), None
        cols = [(b1 - a1).ravel()]
        if self.terms == 2:
            cols.append((b2 - a2).ravel())
        Jk = np.stack(cols, 1)
        return e.ravel(), Jk

    def _jac_full(self, Jk: np.ndarray) -> sp.csr_matrix:
        n = len(self._pid)
        m, K = 2 * n, Jk.shape[1]
        rows = np.arange(m)
        tcols = K + 2 * np.repeat(self._pid, 2) + np.tile([0, 1], n)
        Jt = sp.csr_matrix((-np.ones(m), (rows, tcols)), shape=(m, K + 2 * len(self._pairs)))
        Jkk = sp.csr_matrix(np.hstack([Jk, np.zeros((m, 2 * len(self._pairs)))]))
        return (Jt + Jkk).tocsr()

    def result(self) -> DistortionResult:
        """Estimate k (and a translation per pair) by robust nonlinear least squares.

        Raises ValueError when progress() is not ready or the fit fails, and NoMeasurableDistortion (a ValueError
        subclass) when the corner displacement does not exceed COVERAGE_FACTOR standard uncertainties.

        Returns a DistortionResult: the model (k1, k2), the covariance of k, the standard uncertainty of the
        corner displacement, the RMS match residual (px, inliers of the final fit) before (pure per-pair
        translation) and after (with the distortion model), pairs, matches, inliers and field coverage."""
        p = self.progress()
        if not p["ready"]:
            raise ValueError("estimated distortion not ready: " + "; ".join(p["missing"]))
        self._A_all = np.concatenate([a for a, _ in self._pairs])
        self._B_all = np.concatenate([b for _, b in self._pairs])
        self._pid = np.concatenate([np.full(len(a), i) for i, (a, _) in enumerate(self._pairs)])
        P, K = len(self._pairs), self.terms
        t0 = np.array([np.median(b - a, axis=0) for a, b in self._pairs])

        def split(x):
            k = np.zeros(2)
            k[:K] = x[:K]
            return k, x[K:].reshape(P, 2)

        def fun(x):
            return self._residuals(*split(x), want_jac=False)[0]

        def jac(x):
            return self._jac_full(self._residuals(*split(x), want_jac=True)[1])

        x0 = np.concatenate([np.zeros(K), t0.ravel()])
        lb = np.concatenate([np.full(K, -K_BOUND), np.full(2 * P, -np.inf)])
        ub = -lb
        sol = least_squares(fun, x0, jac=jac, bounds=(lb, ub), method="trf", loss="soft_l1",
                            f_scale=ROBUST_SCALE_PX, x_scale="jac", tr_solver="lsmr", max_nfev=200)
        if not sol.success or np.any(np.abs(sol.x[:K]) >= 0.999 * K_BOUND):
            raise ValueError(f"estimated distortion: fit did not converge to an admissible model ({sol.message})")
        k, t = split(sol.x)
        e, Jk = self._residuals(k, t, want_jac=True)
        cov_k = self._cluster_cov(e, Jk)
        cov = np.zeros((2, 2))
        cov[:K, :K] = cov_k
        model = RadialModel(self.shape, float(k[0]), float(k[1]))
        g = np.ones(K) * model.radius
        corner_u_stat = float(math.sqrt(max(g @ cov_k @ g, 0.0)))
        corner_u = math.hypot(corner_u_stat, TRACKER_REL_U * model.corner_displacement_px)
        # before / after statistics over the final fit's inliers
        mag = np.hypot(e[0::2], e[1::2])
        inl = mag < OUTLIER_SCALES * ROBUST_SCALE_PX
        after = float(np.sqrt(np.mean(mag[inl] ** 2)))
        e0, _ = self._residuals(np.zeros(2), np.zeros((P, 2)), want_jac=False)
        d0 = np.stack([e0[0::2], e0[1::2]], 1)
        before_sq = []
        for i in range(P):
            m = (self._pid == i) & inl
            if m.any():
                before_sq.append(np.sum((d0[m] - d0[m].mean(axis=0)) ** 2, axis=1))
        before = float(np.sqrt(np.mean(np.concatenate(before_sq))))
        res = DistortionResult(model, ((float(cov[0, 0]), float(cov[0, 1])), (float(cov[1, 0]), float(cov[1, 1]))),
                               corner_u, before, after, P, len(self._pid), int(inl.sum()), p["field_coverage"])
        if abs(model.corner_displacement_px) <= COVERAGE_FACTOR * corner_u:
            raise NoMeasurableDistortion(model.corner_displacement_px, COVERAGE_FACTOR * corner_u, P)
        return res

    def _cluster_cov(self, e: np.ndarray, Jk: np.ndarray) -> np.ndarray:
        """Cluster-robust (sandwich) covariance of k with each pair's translation profiled out.

        Linearised at the solution with least_squares' soft-l1 weights, w = (1 + (e / s)^2)^-1/2 per residual
        component. Within a pair the translation absorbs the weighted mean of each Jacobian column (separately for
        the x and y components), so the k columns are centred per pair and component; A = sum Jc^T W Jc,
        B = sum over pairs of g g^T with g = Jc^T W e, cov = A^-1 B A^-1 with the small-sample factor P / (P - 1)."""
        P, K = len(self._pairs), Jk.shape[1]
        n = len(self._pid)
        E = e.reshape(n, 2)
        J = Jk.reshape(n, 2, K)
        wgt = 1.0 / np.sqrt(1.0 + (E / ROBUST_SCALE_PX) ** 2)   # n x 2
        Amat = np.zeros((K, K))
        G = np.zeros((P, K))
        for i in range(P):
            m = self._pid == i
            wi, Ji, Ei = wgt[m], J[m], E[m]
            sw = np.sum(wi, axis=0)                               # per component
            Jc = Ji - np.einsum("nc,ncj->cj", wi, Ji)[None] / sw[None, :, None]
            Ec = Ei - np.sum(wi * Ei, axis=0) / sw
            Amat += np.einsum("ncj,nc,nck->jk", Jc, wi, Jc)
            G[i] = np.einsum("ncj,nc,nc->j", Jc, wi, Ec)
        Ainv = np.linalg.inv(Amat)
        Bmat = G.T @ G * P / max(P - 1, 1)
        return Ainv @ Bmat @ Ainv
