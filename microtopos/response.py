"""Measured camera response from an exposure series, and exposure-bracket merging with it.

The camera delivers 8-bit values after an in-camera tone curve that nobody has published. This module measures
the inverse of that curve, g: DN -> relative linear exposure, from frames of a still scene taken at known
exposure times, instead of assuming a power law (`calibration.bgr_to_luma(..., linearize=True)` assumes 2.2).

Method: Debevec & Malik (1997), solved by OpenCV's `cv2.createCalibrateDebevec` (log-domain least squares with
a second-difference smoothness term and OpenCV's triangle weights). OpenCV places its sample pixels on a fixed
grid regardless of content, so their levels follow the image histogram (mostly dark for this camera). This module
chooses the sample pixels itself, spread evenly over the brightness range of the scene, and hands them to the
solver as a 1 x P image per exposure; each channel is solved separately as a 1-channel problem.
`cv2.createCalibrateRobertson` was tried and is not used: on a synthetic sRGB camera (11 exposures, 320 x 240) its
default 30 iterations left 4.3 % maximum error between 10 and 245 DN and 300 iterations 2.2 % (11 s), against
about 1 % for Debevec in under a second; and on OpenCV 5.0.0 it aborts the process (double free) when given
1-channel images.

What the result is and is not:
* A measured camera response for the exposure settings, sensor mode and in-camera processing in force when the
  series was taken. Auto exposure, auto gain or a different resolution (which changes the sensor mode on this
  camera) may change it.
* Relative: each channel is scaled so that g(reference_dn) = 1. Ratios between channels are therefore not a
  colour calibration, and the absolute radiance unit is arbitrary.
* Supported by data only over `observed_range` (per channel). Outside it the LUT is extrapolated by an explicit
  rule (proportional to DN below, linear continuation above) and `measured` is False there; `merge_hdr` gives such
  levels zero weight. The levels 0 and 255 are censored (the true value may lie beyond them) and are never counted
  as measured.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np

LEVELS = 256
MIN_DISTINCT_EXPOSURES = 3
MIN_EXPOSURE_SPAN = 4.0  # longest / shortest exposure
MIN_MEASURED_LEVELS = 16  # a curve fitted to fewer distinct DN levels is not a curve
DEFAULT_SAMPLES = 400  # sample pixels per channel handed to the solver
DEFAULT_SMOOTHNESS = 30.0  # Debevec lambda; OpenCV's default 10 left up to 3.4 % error at 10-15 DN on the synthetic cameras of tests/test_response.py, 30 about 2 %
GAMMA_FIT_MIN_DN = 10  # gamma_equivalent ignores the noise-dominated levels below this
MIN_LEVEL_COUNT = 2  # a level is measured when at least this many solver observations have it
REFERENCE_DN = 128  # g(REFERENCE_DN) = 1 when that level is measured in every channel
MONOTONIC_TOLERANCE = 0.02  # a raw solution that falls by more than this fraction is refused, not repaired
BLACK_DN = 4  # merge weight is zero at and below this level ...
CLIP_DN = 250  # ... and at and above this one (specular highlights clip at 255)
STEP_CLIP_DN = 240  # pixels at or above this in either frame do not judge whether an exposure step brightened
MIN_STEP_RISE_DN = 1.0  # a longer exposure that raises the median unclipped pixel by less than this did not apply
CONSISTENCY_DN = (16, 240)  # levels used for the still-scene check
MAX_INCONSISTENCY_DN = 4.0  # worst per-exposure median |DN - predicted DN| accepted as a still scene
CONSISTENCY_PIXELS = 250_000  # pixels used for that check
MIN_CONSISTENCY_OBSERVATIONS = 100  # an exposure with fewer checkable pixels is not judged


@dataclass
class ResponseCurve:
    """A measured inverse camera response g: DN -> relative linear exposure, one 256-entry LUT per channel.

    `lut` has shape (channels, 256) and is normalised so that lut[c, reference_dn] == 1. `observed_range[c]` is the
    inclusive DN range the estimate is supported by; `measured[c, dn]` is False for extrapolated levels.
    `inconsistency_dn` is, for the worst exposure of the series, the median absolute difference in DN between that
    exposure and the value the merge of the whole series predicts for it through this curve (a quality figure in
    DN: noise, compression, and any residual scene change).
    """

    lut: np.ndarray
    channels: str  # "luma" (one curve) or "bgr" (three)
    exposures_s: tuple[float, ...]
    observed_range: tuple[tuple[int, int], ...]
    reference_dn: int
    method: str
    inconsistency_dn: float
    measured: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.lut = np.asarray(self.lut, np.float32).reshape(-1, LEVELS)
        if self.channels not in ("luma", "bgr") or self.lut.shape[0] != (1 if self.channels == "luma" else 3):
            raise ValueError(f"channels={self.channels!r} does not match a LUT of shape {self.lut.shape}")
        if len(self.observed_range) != self.lut.shape[0]:
            raise ValueError("one observed range per channel is required")
        self.observed_range = tuple((int(a), int(b)) for a, b in self.observed_range)
        self.exposures_s = tuple(float(t) for t in self.exposures_s)
        dn = np.arange(LEVELS)
        self.measured = np.stack([(dn >= a) & (dn <= b) for a, b in self.observed_range])
        if np.any(np.diff(self.lut, axis=1) < 0):
            raise ValueError("a camera response LUT must be non-decreasing")

    def to_json(self) -> dict:
        """Plain lists and numbers, for the calibration JSON. float32 values survive the round trip exactly."""
        return {
            "kind": "measured camera response",
            "version": 1,
            "channels": self.channels,
            "lut": [[float(v) for v in row] for row in self.lut],
            "exposures_s": list(self.exposures_s),
            "observed_range": [list(r) for r in self.observed_range],
            "reference_dn": self.reference_dn,
            "method": self.method,
            "inconsistency_dn": self.inconsistency_dn,
        }

    @classmethod
    def from_json(cls, d: dict) -> ResponseCurve:
        if d.get("kind") != "measured camera response" or d.get("version") != 1:
            raise ValueError(f"not a version-1 measured camera response: kind={d.get('kind')!r}")
        return cls(
            lut=np.asarray(d["lut"], np.float32),
            channels=d["channels"],
            exposures_s=tuple(d["exposures_s"]),
            observed_range=tuple(tuple(r) for r in d["observed_range"]),
            reference_dn=int(d["reference_dn"]),
            method=str(d["method"]),
            inconsistency_dn=float(d["inconsistency_dn"]),
        )

    def linearize(self, img: np.ndarray) -> np.ndarray:
        """Relative linear exposure (float32) of an 8-bit image, or of a float image in DN units (interpolated).

        A luma curve is applied to every value of an image of any shape; a BGR curve needs a last axis of 3.
        Levels outside `observed_range` get the extrapolated LUT values (see the module docstring).
        """
        img = np.asarray(img)
        if img.dtype != np.uint8 and not np.issubdtype(img.dtype, np.floating):
            raise ValueError(f"linearize takes uint8 or float DN values, not {img.dtype}")
        if self.channels == "bgr" and (img.ndim < 1 or img.shape[-1] != 3):
            raise ValueError(f"a BGR response needs an image with a last axis of 3, got shape {img.shape}")
        out = np.empty(img.shape, np.float32)
        grid = np.arange(LEVELS, dtype=np.float64)
        for c in range(self.lut.shape[0]):
            src = img if self.channels == "luma" else img[..., c]
            if img.dtype == np.uint8:
                v = self.lut[c][src]
            else:
                v = np.interp(np.clip(src.astype(np.float64), 0.0, LEVELS - 1.0), grid, self.lut[c].astype(np.float64))
            if self.channels == "luma":
                out[...] = v
            else:
                out[..., c] = v
        return out

    def gamma_equivalent(self) -> float:
        """Exponent of the best-fit power law g ~ DN**gamma over the measured levels >= GAMMA_FIT_MIN_DN (all
        channels), for display.

        Least squares of log g against log DN with a free offset. The levels below GAMMA_FIT_MIN_DN are left out
        because there one DN of read noise is a large fraction of the signal and log g is poorly determined. A
        summary number only: a measured response with a linear toe or a shoulder is not a power law, which is why
        the LUT, not this number, is used for linearising.
        """
        xs, ys = [], []
        for c in range(self.lut.shape[0]):
            dn = np.flatnonzero(self.measured[c] & (np.arange(LEVELS) >= GAMMA_FIT_MIN_DN))
            if dn.size < 2:
                raise ValueError(f"fewer than 2 measured levels at or above {GAMMA_FIT_MIN_DN} DN; no power law fit")
            xs.append(np.log(dn.astype(np.float64)))
            ys.append(np.log(self.lut[c, dn].astype(np.float64)))
        x, y = np.concatenate(xs), np.concatenate(ys)
        return float(np.polyfit(x, y, 1)[0])


def _check_series(frames, exposures_s) -> tuple[np.ndarray, np.ndarray]:
    """Validated (n, H, W, C) uint8 stack and exposure array, sorted by exposure (shortest first)."""
    frames = [np.asarray(f) for f in frames]
    t = np.asarray(exposures_s, np.float64).ravel()
    if len(frames) != t.size:
        raise ValueError(f"{len(frames)} frames but {t.size} exposure times")
    if not np.all(np.isfinite(t)) or np.any(t <= 0):
        raise ValueError("exposure times must be finite and positive")
    if any(f.dtype != np.uint8 for f in frames):
        raise ValueError("the response is measured from 8-bit (uint8) frames as the camera delivers them")
    if len({f.shape for f in frames}) != 1:
        raise ValueError("all frames must have the same shape")
    if np.unique(t).size < MIN_DISTINCT_EXPOSURES:
        raise ValueError(f"at least {MIN_DISTINCT_EXPOSURES} distinct exposure times are needed, got {np.unique(t).size}")
    if t.max() / t.min() < MIN_EXPOSURE_SPAN:
        raise ValueError(f"exposures span a factor {t.max() / t.min():.2f}; at least {MIN_EXPOSURE_SPAN:g} is needed")
    stack = np.stack(frames)
    if stack.ndim == 3:
        stack = stack[..., None]
    if stack.ndim != 4 or stack.shape[-1] not in (1, 3):
        raise ValueError(f"frames must be 2-D (luma) or H x W x 3 (BGR), got shape {frames[0].shape}")
    order = np.argsort(t, kind="stable")
    return stack[order], t[order]


def _hat(black_dn: int, clip_dn: int) -> np.ndarray:
    """Merge weight per DN level: a triangle that is zero at and below black_dn and at and above clip_dn."""
    z = np.arange(LEVELS, dtype=np.float64)
    return np.clip(np.minimum(z - black_dn, clip_dn - z), 0.0, None)


def _choose_samples(z: np.ndarray, n: int) -> np.ndarray:
    """Indices (into the flattened pixels) of n pixels whose summed DN over the series is spread evenly.

    The sum over a still series grows with scene radiance whatever the response is, so even spacing in it covers
    the dark, middle and bright parts of the scene even when most pixels are dark.
    """
    s = z.sum(axis=0, dtype=np.int64)  # z: (n_exposures, pixels)
    order = np.argsort(s, kind="stable")
    targets = np.linspace(s[order[0]], s[order[-1]], n)
    pos = np.clip(np.searchsorted(s[order], targets), 0, order.size - 1)
    return np.unique(order[pos])


def _solve_channel(z: np.ndarray, t: np.ndarray, samples: int, smoothness: float):
    """Raw Debevec log-response for one channel; returns (g, observed (lo, hi))."""
    idx = _choose_samples(z, samples)
    pts = z[:, idx]  # (n_exposures, P)
    imgs = [np.ascontiguousarray(p.reshape(1, -1)) for p in pts]
    cal = cv2.createCalibrateDebevec(samples=idx.size, lambda_=float(smoothness), random=False)
    g = cal.process(imgs, t.astype(np.float32)).reshape(-1).astype(np.float64)
    if g.size != LEVELS or not np.all(np.isfinite(g)) or np.any(g <= 0):
        raise ValueError("the response solver returned a non-finite or non-positive curve")
    counts = np.bincount(pts.ravel(), minlength=LEVELS)
    counts[0] = counts[LEVELS - 1] = 0  # censored levels
    lv = np.flatnonzero(counts >= MIN_LEVEL_COUNT)
    if lv.size < MIN_MEASURED_LEVELS:
        raise ValueError(f"only {lv.size} distinct DN levels were observed; at least {MIN_MEASURED_LEVELS} are needed")
    return g, (int(lv[0]), int(lv[-1]))


def _finish_channel(g: np.ndarray, lo: int, hi: int) -> np.ndarray:
    """Validate monotonicity on the measured range, then replace unmeasured levels by the explicit extrapolation."""
    seg = g[lo:hi + 1]
    run_max = np.maximum.accumulate(seg)
    if np.any(seg < run_max * (1.0 - MONOTONIC_TOLERANCE)):
        k = int(np.argmax(seg < run_max * (1.0 - MONOTONIC_TOLERANCE))) + lo
        raise ValueError(f"the estimated response falls by more than {MONOTONIC_TOLERANCE:.0%} at DN {k}; "
                         "the series is inconsistent (moving scene, changing light or auto exposure?)")
    out = g.copy()
    out[lo:hi + 1] = run_max
    dn = np.arange(LEVELS, dtype=np.float64)
    out[:lo] = out[lo] * dn[:lo] / lo if lo > 0 else out[:lo]
    m = min(8, hi - lo)
    slope = (out[hi] - out[hi - m]) / m
    out[hi + 1:] = out[hi] + slope * (dn[hi + 1:] - hi)
    return out


def estimate_response(frames, exposures_s, *, samples: int = DEFAULT_SAMPLES,
                      smoothness: float = DEFAULT_SMOOTHNESS) -> ResponseCurve:
    """Measure the inverse camera response from uint8 frames of a STILL scene at known exposure times (seconds).

    frames: 2-D frames give one luma curve, H x W x 3 frames give B, G and R curves. The UVC control
    exposure_time_absolute is in units of 100 us, so seconds = units * 1e-4. Several frames per exposure are allowed.

    Raises ValueError for fewer than 3 distinct exposures, a span of exposures below a factor 4, frames of different
    shapes, too few observed DN levels, a non-monotonic solution, or a series whose exposures disagree after
    linearisation by more than MAX_INCONSISTENCY_DN (worst per-exposure median DN difference; a moving scene,
    flicker or auto exposure), so that such a series never silently yields a curve. The check is in DN rather than
    in relative linear units because sensor and compression noise is roughly constant in DN, while its relative
    size in linear units explodes in the dark frames this camera delivers.
    """
    stack, t = _check_series(frames, exposures_s)
    _check_exposures_take_effect(stack, t)
    n, h, w, nc = stack.shape
    flat = stack.reshape(n, h * w, nc)
    raw = []
    for c in range(nc):
        raw.append(_solve_channel(flat[:, :, c], t, samples, smoothness))
    ranges = tuple(r for _, r in raw)
    lo_all, hi_all = max(r[0] for r in ranges), min(r[1] for r in ranges)
    if lo_all <= REFERENCE_DN <= hi_all:
        ref = REFERENCE_DN
    elif lo_all <= hi_all:
        ref = round((lo_all + hi_all) / 2)
    else:
        raise ValueError(f"the channels' measured ranges do not overlap: {ranges}")
    luts = []
    for g, (lo, hi) in raw:
        lut = _finish_channel(g, lo, hi)
        luts.append(lut / lut[ref])
    curve = ResponseCurve(
        lut=np.stack(luts).astype(np.float32),
        channels="luma" if nc == 1 else "bgr",
        exposures_s=tuple(t),
        observed_range=ranges,
        reference_dn=ref,
        method=(f"Debevec-Malik via cv2.createCalibrateDebevec (lambda={smoothness:g}, up to {samples} sample pixels "
                "per channel spread over the scene brightness, 1-channel solves)"),
        inconsistency_dn=float("nan"),
    )
    curve.inconsistency_dn = _inconsistency_dn(stack, t, curve)
    if not curve.inconsistency_dn <= MAX_INCONSISTENCY_DN:
        raise ValueError(f"an exposure differs from what the rest of the series predicts by {curve.inconsistency_dn:.1f}"
                         f" DN (median); at most {MAX_INCONSISTENCY_DN:g} DN is accepted for a still scene. Was the scene, the light or "
                         "the camera's automatic exposure changing during the series?")
    return curve


def _check_exposures_take_effect(stack: np.ndarray, t: np.ndarray) -> None:
    """Every longer exposure must make the frames brighter. Wrong exposure ratios cannot be caught after the fact:
    a different smooth curve absorbs them and still looks self-consistent (the response and the exposure ratios are
    determined only together). But two distinct exposures that give the same brightness prove that the camera did
    not apply the exposure it reports; the MikrOkular, for one, gives identical frames for every setting from 7.8 to
    20.4 ms (measured 2026-11-28). Compared on pixels below STEP_CLIP_DN in both, so saturation does not hide a step.
    """
    times = np.unique(t)
    means = []
    for tj in times:
        means.append(stack[t == tj].astype(np.float32).mean(axis=0).mean(axis=-1))  # per pixel, over frames/channels
    for a, b, ta, tb in zip(means, means[1:], times, times[1:]):
        ok = (a < STEP_CLIP_DN) & (b < STEP_CLIP_DN) & (b > BLACK_DN)
        if ok.mean() < 0.01:
            continue  # nothing comparable: everything clipped (judged by the other checks)
        rise = float(np.median(b[ok] - a[ok]))
        if rise < MIN_STEP_RISE_DN:
            raise ValueError(f"the exposures {ta * 1e3:.1f} ms and {tb * 1e3:.1f} ms gave the same brightness (median "
                             f"rise {rise:.2f} DN): the camera does not apply the exposure time it reports, so its "
                             "response cannot be measured from its exposure setting")


def _inconsistency_dn(stack: np.ndarray, t: np.ndarray, curve: ResponseCurve) -> float:
    """Worst over exposures j of median |Z_j - g^-1(E t_j)| in DN, E the merged radiance.

    Pixels are judged in exposure j where the PREDICTED level g^-1(E t_j) is well exposed (CONSISTENCY_DN) and is so
    in at least one other exposure. Selecting by the observed Z_j instead would keep only the upward noise just above
    the lower bound (and the downward noise below the upper one) and report that selection bias as disagreement.
    Per exposure, so that one moved frame among many is not outvoted by the others."""
    n = stack.shape[0]
    step = max(1, math.ceil(math.sqrt(stack.shape[1] * stack.shape[2] / CONSISTENCY_PIXELS)))
    sub = stack[:, ::step, ::step]
    merged = merge_hdr(list(sub[..., 0] if curve.channels == "luma" else sub), t, curve)
    e = merged["radiance"]
    if curve.channels == "luma":
        e = e[..., None]
    grid = np.arange(LEVELS, dtype=np.float64)
    pred = np.empty(sub.shape, np.float64)
    for j in range(n):
        for c in range(e.shape[-1]):
            pred[j, ..., c] = np.interp(e[..., c] * t[j], curve.lut[c].astype(np.float64), grid)
    lo, hi = CONSISTENCY_DN
    good = (pred >= lo) & (pred <= hi) & np.isfinite(e)[None]
    good &= good.sum(axis=0, keepdims=True) >= 2
    if not np.any(good):
        raise ValueError("no pixel is well exposed in two exposures; the series cannot be checked for a still scene")
    worst = 0.0
    for j in range(n):
        if np.count_nonzero(good[j]) < MIN_CONSISTENCY_OBSERVATIONS:
            continue  # this exposure has (almost) no well-exposed pixel seen in another exposure too
        worst = max(worst, float(np.median(np.abs(sub[j].astype(np.float64) - pred[j])[good[j]])))
    return worst


def merge_hdr(frames, exposures_s, curve: ResponseCurve, *, black_dn: int = BLACK_DN,
              clip_dn: int = CLIP_DN) -> dict:
    """Merge an exposure bracket into a relative radiance map with a measured camera response.

    A weighted mean of g(Z_j) / t_j over the exposures j, implemented here rather than with cv2.MergeDebevec /
    MergeRobertson: those weight the clipping levels 0 and 255 above zero, take no per-level mask for unmeasured
    levels and return no weight sum. The weight is a triangle (hat) in DN that is zero at and below `black_dn` and at
    and above `clip_dn`, and every level outside the curve's observed range (0 and 255 included) gets zero weight, so
    clipped highlights of the long exposures and unmeasured levels contribute nothing.

    Two passes. The first weights each observation by the hat of its observed level. Weighting by the observed level
    is biased near the thresholds (of a pixel whose true level is 5 DN, only the noisy observations above `black_dn`
    are kept), so the second pass weights each observation by the hat of the level the first-pass radiance predicts
    for that exposure, and drops only observations at unmeasured levels. Pixels the second pass cannot weight keep
    the first-pass value.

    Returns {"radiance": float32 (same shape as a frame; NaN where unusable), "weight": float32 weight sum of the
    pass that gave the value, "unusable": bool, True where no exposure gave a usable value}. Radiance is relative:
    curve units per second.
    """
    frames = [np.asarray(f) for f in frames]
    t = np.asarray(exposures_s, np.float64).ravel()
    if len(frames) != t.size or t.size == 0:
        raise ValueError(f"{len(frames)} frames but {t.size} exposure times")
    if not np.all(np.isfinite(t)) or np.any(t <= 0):
        raise ValueError("exposure times must be finite and positive")
    if any(f.dtype != np.uint8 for f in frames) or len({f.shape for f in frames}) != 1:
        raise ValueError("frames must be uint8 and all of the same shape")
    if curve.channels == "bgr" and frames[0].shape[-1:] != (3,):
        raise ValueError("a BGR response needs H x W x 3 frames")
    nc = curve.lut.shape[0]
    hat = np.asarray(_hat(black_dn, clip_dn), np.float64)
    grid = np.arange(LEVELS, dtype=np.float64)
    luts = curve.lut.astype(np.float64)

    def per_channel(fn, img):
        """fn(channel, data) for each channel of img (the whole image for a luma curve)."""
        if nc == 1:
            return fn(0, img)
        return np.stack([fn(c, img[..., c]) for c in range(nc)], axis=-1)

    def observed_weight(c, z):
        return hat[z] * curve.measured[c][z]

    def measured_level(c, z):
        return curve.measured[c][z]

    num = np.zeros(frames[0].shape, np.float64)
    den = np.zeros(frames[0].shape, np.float64)
    for f, tj in zip(frames, t, strict=True):
        w = per_channel(observed_weight, f)
        num += w * (curve.linearize(f) / tj)
        den += w
    unusable = den <= 0
    with np.errstate(invalid="ignore", divide="ignore"):
        first = np.where(unusable, np.nan, num / den)

    e0 = np.where(unusable, 0.0, first)
    num2 = np.zeros(frames[0].shape, np.float64)
    den2 = np.zeros(frames[0].shape, np.float64)
    for f, tj in zip(frames, t, strict=True):
        pred = per_channel(lambda c, e, tj=tj: np.interp(e * tj, luts[c], grid), e0)
        w = np.interp(pred, grid, hat) * per_channel(measured_level, f)
        num2 += w * (curve.linearize(f) / tj)
        den2 += w
    second = (den2 > 0) & ~unusable
    with np.errstate(invalid="ignore", divide="ignore"):
        radiance = np.where(second, num2 / den2, first)
    weight = np.where(second, den2, den)
    return {"radiance": radiance.astype(np.float32), "weight": weight.astype(np.float32), "unusable": unusable}


def tonemap_for_display(radiance: np.ndarray) -> np.ndarray:
    """DISPLAY ONLY: map a relative radiance map to uint8 by log scaling between its 0.5 and 99.5 percentiles.

    The output is for looking at a high-dynamic-range merge on screen. It is not a measurement and must not be fed
    back into any analysis. Non-finite (unusable) pixels are shown black.
    """
    r = np.asarray(radiance, np.float64)
    ok = np.isfinite(r) & (r > 0)
    out = np.zeros(r.shape, np.uint8)
    if not np.any(ok):
        return out
    lg = np.log(r[ok])
    a, b = np.percentile(lg, [0.5, 99.5])
    span = b - a if b > a else 1.0
    out[ok] = np.clip(np.round((lg - a) / span * 255.0), 0, 255).astype(np.uint8)
    return out
