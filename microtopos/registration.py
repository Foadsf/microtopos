"""Frame registration: phase correlation + ECC on a reduced pyramid, chained over a focus sweep.

Conventions (pinned by tests):
  * phase_shift(a, b) (cv2.phaseCorrelate) returns (dx, dy) such that b(x) = a(x - d): content moved by +d.
  * A warp W (2x3) maps reference-frame coordinates to moving-frame coordinates, and
    aligned = cv2.warpAffine(moving, W, size, flags=WARP_INVERSE_MAP) satisfies aligned(x) = moving(W x).
"""

from __future__ import annotations

import math
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import cv2
import numpy as np


PHASE_MIN_RESPONSE = 0.05  # below this a phase-correlation peak is indistinguishable from noise
PHASE_MAX_FRACTION = 0.25  # a fallback shift beyond this fraction of the frame is not trusted
# Illumination shading (vignette, lamp fall-off) varies on the scale of the whole frame and is the same in both
# frames of a pair, so it correlates at zero shift whatever the content does: it pulled ECC toward (0, 0) on
# low-contrast pairs and gave unrelated frames a spurious correlation. A Gaussian high-pass removes any quadratic
# shading exactly (up to a constant) whatever its sigma, so sigma is chosen large, 1/8 of the frame's short side,
# to keep all texture up to that scale (heavily defocused sweep frames have little else).
SHADING_SIGMA_FRACTION = 1 / 8
# Ambiguity: a correlation peak is accepted only if it beats every competing peak of the same correlation surface
# by this many robust standard deviations of that surface ("3 sigma"). The surface values at wrong shifts are
# samples of what the frames' own texture, noise and JPEG produce without a true match, so this asks whether the
# chosen shift is distinguishable from the alternatives the data offers. Measured (seeds 1-12, 10x and 40x, 4 draws,
# 320 x 240): periodic gratings separate by <= 0.02 (every period is an equally good peak), unrelated frames by
# <= 3.1, every valid condition of the accuracy matrix by >= 5.4 (+3 DOF defocus; 5 % contrast >= 5.9), and 2 %
# contrast pairs, all correct after shading removal, by >= 3.5.
PEAK_SEPARATION_MIN = 3.0
# Signal-to-noise: ECC's correlation coefficient on the shading-free frames is the fraction of their energy that
# the two frames share, rho = S / (S + N) for equal noise, so rho >= 0.5 means the shared texture is at least as
# strong as everything that differs (noise, defocus difference, blur): SNR >= 0 dB at the analysis scale. Below it
# the correlation maximum is displaced by noise (1 % contrast: rho 0.10-0.21, 3 of 16 significant pairs off by
# 0.54-0.74 px) and unrelated frames reach rho <= 0.18. Measured on valid conditions: rho >= 0.67 (5 % contrast),
# >= 0.76 (scale 1.02, rotation 1 deg, +3 DOF defocus). The coefficient is contrast-normalised, so unlike a
# coefficient on the raw frames (where the shared vignette inflates it for unrelated frames and noise-level
# contrast deflates it for real ones) it does not depend on illumination.
SNR_MIN_CORRELATION = 0.5
# Validity envelope of a translation-only result: the rotation and scale change between the frames, measured by an
# affine ECC fit, must stay within the limits register_pair already applies to similarity fits. Beyond them the
# translation-only estimate is biased by the unmodelled deformation (measured at the image centre, translation
# model, 96 pairs each: median 0.15 px at 1 deg, 50/96 pairs within 0.5 px at 2 deg, median 1.3 px and up to
# 3.6 px at 3 deg). The affine fit itself is accurate to ~0.01 deg and ~0.001 in scale on these pairs.
MAX_ROT_DEG = 2.0
MAX_SCALE_DEV = 0.03


@dataclass
class PairRegistration:
    warp: np.ndarray  # 2x3, template coords -> moving coords, full resolution
    ecc: float  # ECC correlation (or phase-correlation response on fallback)
    ok: bool  # False when ECC failed or was rejected and the translation-only estimate was used
    method: str = "ecc"  # "ecc" | "phase" (qualified fallback) | "none" (motion not observable)

    @property
    def observable(self) -> bool:
        return self.method != "none"


def _hann(h: int, w: int) -> np.ndarray:
    return cv2.createHanningWindow((w, h), cv2.CV_32F)


def phase_shift(a: np.ndarray, b: np.ndarray, window: np.ndarray | None = None) -> tuple[float, float, float]:
    """Phase correlation that never modifies its inputs.

    OpenCV 5.0's phaseCorrelate applies the window to the input arrays in place, so the window is applied
    to copies here and phaseCorrelate is called without one.
    """
    win = _hann(*a.shape) if window is None else window
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    # zero mean before windowing: otherwise the window's own spectrum (identical in both images) adds a
    # spurious zero-shift peak that biases the estimate toward (0, 0)
    (dx, dy), resp = cv2.phaseCorrelate((a - a.mean()) * win, (b - b.mean()) * win)
    return float(dx), float(dy), float(resp)


def remove_shading(img: np.ndarray) -> np.ndarray:
    """Frame minus its frame-scale illumination: high-pass with three box filters of total variance sigma^2, sigma =
    SHADING_SIGMA_FRACTION x short side (a Gaussian to within a few %, at a cost independent of sigma). Like any
    symmetric normalised kernel, it maps a quadratic q to q + const, so quadratic shading is removed exactly."""
    img = img.astype(np.float32)
    s = SHADING_SIGMA_FRACTION * min(img.shape[:2])
    k = max(1, int(round(math.sqrt(4 * s * s + 1)))) | 1  # (k^2 - 1) / 12 per pass, three passes
    low = img
    for _ in range(3):
        low = cv2.blur(low, (k, k), borderType=cv2.BORDER_REFLECT)
    return img - low


def correlation_surface(a: np.ndarray, b: np.ndarray, window: np.ndarray | None = None) -> np.ndarray:
    """Windowed cross-correlation c(s) = sum_x a(x) b(x + s) of the zero-mean frames, fft-shifted so that shift
    s = (sx, sy) is at index (h // 2 + sy, w // 2 + sx). With b(x) = a(x - d) it peaks at s = d (phase_shift's
    convention). Plain (not phase-only) correlation: it is what ECC and NCC maximise, so its competing peaks are
    the shifts those checks could be fooled by."""
    h, w = a.shape
    win = _hann(h, w) if window is None else window
    A = np.fft.rfft2((a - a.mean()) * win)
    B = np.fft.rfft2((b - b.mean()) * win)
    return np.fft.fftshift(np.fft.irfft2(np.conj(A) * B, s=(h, w)))


@dataclass
class Peaks:
    shift: tuple[float, float]  # integer shift of the global maximum (content shift, phase_shift convention)
    separation: float  # (best - best competing peak) / robust sigma of the surface; 0 when not measurable
    sigma: float  # robust standard deviation of the surface outside the main lobe
    candidates: list  # [(sx, sy, z)] every local maximum (main peak first), z = (value - best) / sigma <= 0


def correlation_peaks(c: np.ndarray, comparable: float = PEAK_SEPARATION_MIN) -> Peaks:
    """The global maximum of a correlation surface and how clearly it beats the competing peaks.

    The main lobe is the connected region around the maximum above half its height (relative to the surface
    median), so its width follows the texture's own correlation length; local maxima outside it are the competing
    candidates. Their distance from the best is expressed in robust standard deviations of the surface outside
    the lobe; only candidates within `comparable` of the best are listed."""
    h, w = c.shape
    c = c.astype(np.float32, copy=False)
    py, px = np.unravel_index(int(np.argmax(c)), c.shape)
    c1 = float(c[py, px])
    shift = (float(px - w // 2), float(py - h // 2))
    med = float(np.median(c))
    if not np.isfinite(c1) or c1 <= med:  # flat surface (no texture): nothing is measurable
        return Peaks(shift, 0.0, 0.0, [(shift[0], shift[1], 0.0)])
    _, lab = cv2.connectedComponents((c > med + 0.5 * (c1 - med)).astype(np.uint8), connectivity=8)
    lobe = lab == lab[py, px]
    rest = c[~lobe]
    if rest.size < 16:
        return Peaks(shift, 0.0, 0.0, [(shift[0], shift[1], 0.0)])
    m = float(np.median(rest))
    sigma = 1.4826 * float(np.median(np.abs(rest - m)))
    if sigma <= 0:
        return Peaks(shift, 0.0, 0.0, [(shift[0], shift[1], 0.0)])
    loc = (c >= cv2.dilate(c, np.ones((3, 3), np.uint8))) & ~lobe
    vals_all = c[loc]
    second = float(vals_all.max()) if vals_all.size else m
    ys, xs = np.nonzero(loc & (c > c1 - comparable * sigma))
    vals = c[ys, xs]
    order = np.argsort(-vals)
    cands = [(shift[0], shift[1], 0.0)] + [(float(xs[i] - w // 2), float(ys[i] - h // 2), (float(vals[i]) - c1) / sigma)
                                           for i in order]
    return Peaks(shift, (c1 - second) / sigma, sigma, cands)


def deformation(a: np.ndarray, b: np.ndarray, dx: float, dy: float, max_side: int = 160) -> tuple[float, float] | None:
    """(scale, rotation in degrees) of b relative to a from an affine ECC fit started at the translation (dx, dy).

    Run on a copy reduced to at most max_side pixels (rotation and scale do not depend on resolution; measured on
    160 x 120 band-pass pairs: rotation 1 deg -> 0.97-1.02 deg, scale 1.02 -> 0.020). None when the fit fails."""
    f = max(1, math.ceil(max(a.shape[:2]) / max_side))
    if f > 1:
        size = (a.shape[1] // f, a.shape[0] // f)
        a = cv2.resize(a, size, interpolation=cv2.INTER_AREA)
        b = cv2.resize(b, size, interpolation=cv2.INTER_AREA)
    init = np.array([[1, 0, dx / f], [0, 1, dy / f]], np.float32)
    try:
        crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 1e-4)
        _, A = cv2.findTransformECC(a.astype(np.float32, copy=False), b.astype(np.float32, copy=False), init,
                                    cv2.MOTION_AFFINE, crit, None, 3)
    except cv2.error:
        return None
    if not np.isfinite(A).all():
        return None
    _, s, rot = _to_similarity(A)
    return s, rot


def within_envelope(d: tuple[float, float] | None, max_scale_dev: float = MAX_SCALE_DEV,
                    max_rot_deg: float = MAX_ROT_DEG) -> bool:
    return d is not None and abs(d[0] - 1) < max_scale_dev and abs(d[1]) < max_rot_deg


def _to_similarity(A: np.ndarray) -> tuple[np.ndarray, float, float]:
    lin = A[:, :2]
    theta = math.atan2(lin[1, 0] - lin[0, 1], lin[0, 0] + lin[1, 1])
    s = math.sqrt(max(np.linalg.det(lin), 1e-12))
    c, si = s * math.cos(theta), s * math.sin(theta)
    out = A.copy()
    out[:, :2] = [[c, -si], [si, c]]
    return out, s, math.degrees(theta)


def register_pair(template: np.ndarray, moving: np.ndarray, factor: int = 2, model: str = "translation",
                  max_scale_dev: float = MAX_SCALE_DEV, max_rot_deg: float = MAX_ROT_DEG) -> PairRegistration:
    """Translation ("translation") or similarity ("similarity") transform between two frames.

    Estimated at 1/factor resolution and returned at full resolution. ok is True only for an ECC result on the
    shading-free frames that (1) sits on a correlation peak that clearly beats every competing peak
    (PEAK_SEPARATION_MIN: periodic texture and unrelated frames fail this), (2) converged near that peak, (3) has
    SNR >= 0 dB (SNR_MIN_CORRELATION: noise-level contrast and unrelated frames fail this), and (4) lies inside the
    rotation / scale envelope (for "translation" the deformation is measured with an affine fit, because a
    translation-only fit cannot see it).
    """
    if factor > 1:
        size = (template.shape[1] // factor, template.shape[0] // factor)
        t = cv2.resize(template, size, interpolation=cv2.INTER_AREA)
        m = cv2.resize(moving, size, interpolation=cv2.INTER_AREA)
    else:
        t, m = template.astype(np.float32), moving.astype(np.float32)
    dx, dy, resp = phase_shift(t, m)
    W0 = np.array([[1, 0, dx], [0, 1, dy]], np.float32)
    h_s, w_s = t.shape[:2]
    # a fallback is qualified only with a real peak at a plausible shift; otherwise motion is unobservable
    # (identical flat frames return response 0 at half the frame size, which is not a measurement)
    qualified = (np.isfinite(resp) and resp >= PHASE_MIN_RESPONSE
                 and math.hypot(dx, dy) <= PHASE_MAX_FRACTION * min(h_s, w_s))
    W, cc, ok, method = (W0, resp, False, "phase") if qualified else \
        (np.array([[1, 0, 0], [0, 1, 0]], np.float32), resp, False, "none")
    # the decision is made on shading-free frames: the shared vignette correlates at zero shift for any content
    th, mh = remove_shading(t), remove_shading(m)
    # the peak test runs on copies of at most ~320 px (the frame size its threshold was measured at); ECC then
    # refines at full analysis resolution from the scaled peak
    r = max(1, math.ceil(max(h_s, w_s) / 320))
    small = (w_s // r, h_s // r)
    pk = correlation_peaks(correlation_surface(th, mh) if r == 1 else correlation_surface(
        cv2.resize(th, small, interpolation=cv2.INTER_AREA), cv2.resize(mh, small, interpolation=cv2.INTER_AREA)))
    pk.shift = (r * pk.shift[0], r * pk.shift[1])
    significant = (pk.separation >= PEAK_SEPARATION_MIN
                   and math.hypot(*pk.shift) <= PHASE_MAX_FRACTION * min(h_s, w_s))
    if significant:
        try:
            crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 60, 1e-5)
            motion = cv2.MOTION_TRANSLATION if model == "translation" else cv2.MOTION_AFFINE
            # ECC is a local optimiser of the correlation: start it on the surface's global maximum. It writes its
            # result into the warp it is given (OpenCV returns the same array), so it gets a copy: on main the
            # "near" test below compared ECC's result with itself and never rejected anything.
            init = np.array([[1, 0, pk.shift[0]], [0, 1, pk.shift[1]]], np.float32)
            cc_e, We = cv2.findTransformECC(th, mh, init.copy(), motion, crit, None, 3)
            Ws, sc, rot = _to_similarity(We)
            near = math.hypot(We[0, 2] - init[0, 2], We[1, 2] - init[1, 2]) < 3.0
            if near and np.isfinite(Ws).all() and float(cc_e) >= SNR_MIN_CORRELATION:
                # last, as it costs a second (affine) ECC run: is a translation-only result valid for this pair?
                d = deformation(th, mh, float(We[0, 2]), float(We[1, 2])) if model == "translation" else (sc, rot)
                if within_envelope(d, max_scale_dev, max_rot_deg):
                    W, cc, ok, method = Ws, float(cc_e), True, "ecc"
        except cv2.error:
            pass
    # rescale to full resolution, honouring INTER_AREA pixel centres: x_full = f*x + (f-1)/2
    if factor > 1:
        A = W[:, :2].astype(np.float64)
        c = np.full(2, (factor - 1) / 2.0)
        tr = factor * W[:, 2].astype(np.float64) + (np.eye(2) - A) @ c
        W = np.hstack([A, tr[:, None]]).astype(np.float32)
    return PairRegistration(W.astype(np.float32), cc, ok, method)


class RegistrationCancelled(Exception):
    pass


def register_sweep(frames: list[np.ndarray], ref: int | None = None, factor: int = 2, workers: int = 4,
                   progress=None, cancelled=None) -> tuple[list[np.ndarray], list[PairRegistration]]:
    """Register a focus sweep to its reference frame.

    Consecutive pairs (similar defocus) are registered with translation only, in parallel, and chained;
    a pair whose ECC fails inherits the median motion of its ECC-valid neighbours. A pair whose motion is not
    observable at all (no texture, response at noise level) and has no valid neighbours contributes zero motion
    instead of an invented shift; `unsupported_pairs` counts them. Allowing scale in every pair lets
    defocus-induced scale errors accumulate along the chain, so the model is translation only.
    Returns warps W_k (reference coords -> frame k coords) and the pair results (pair k = frames k, k+1).
    """
    n = len(frames)
    ref = n // 2 if ref is None else ref
    done = [0]

    def job(k):
        if cancelled and cancelled():
            raise RegistrationCancelled()
        res = register_pair(frames[k], frames[k + 1], factor, "translation")
        done[0] += 1
        if progress:
            progress(0.8 * done[0] / max(1, n - 1))
        return res

    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        pairs = list(ex.map(job, range(n - 1)))
    steps = np.array([p.warp[:, 2] for p in pairs], np.float64)
    for k, p in enumerate(pairs):
        if not p.ok:
            nb = [steps[j] for j in range(max(0, k - 3), min(n - 1, k + 4)) if j != k and pairs[j].ok]
            if nb:
                steps[k] = np.median(nb, axis=0)
            elif not p.observable:
                steps[k] = 0.0  # no evidence of motion either way: do not invent one
    T = np.zeros((n, 2))
    for k in range(ref + 1, n):
        T[k] = T[k - 1] + steps[k - 1]
    for k in range(ref - 1, -1, -1):
        T[k] = T[k + 1] - steps[k]

    # Focus breathing (magnification change with focus) is not corrected: both reference-based and
    # pair-based similarity fits were biased by defocus by more than the breathing itself (see self-test).
    scale = np.ones(n)
    if progress:
        progress(1.0)
    h, w = frames[0].shape
    c = np.array([(w - 1) / 2.0, (h - 1) / 2.0])
    warps = []
    for k in range(n):
        A = np.eye(2) * scale[k]
        t = (np.eye(2) - A) @ c + T[k]
        warps.append(np.hstack([A, t[:, None]]).astype(np.float32))
    return warps, pairs


def warp_to_reference(frame: np.ndarray, W: np.ndarray, out: np.ndarray | None = None) -> np.ndarray:
    """Aligned frame. Borders are replicated so focus filters see no artificial edge; those pixels are not
    observations, so quantitative use must also apply `warp_support`."""
    h, w = frame.shape[:2]
    return cv2.warpAffine(frame, W, (w, h), dst=out, flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                          borderMode=cv2.BORDER_REPLICATE)


def warp_support(shape: tuple[int, int], W: np.ndarray) -> np.ndarray:
    """Boolean mask of reference pixels that the warped frame actually observed (no replicated border)."""
    h, w = shape
    ones = np.ones((h, w), np.uint8)
    s = cv2.warpAffine(ones, W, (w, h), flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP,
                       borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return s.astype(bool)


def warp_scale_shift(W: np.ndarray, shape: tuple[int, int]) -> tuple[float, float, float]:
    """Scale and the displacement of the image centre implied by warp W (for diagnostics and tests)."""
    h, w = shape
    c = np.array([(w - 1) / 2.0, (h - 1) / 2.0])
    p = W[:, :2] @ c + W[:, 2]
    s = math.sqrt(max(float(np.linalg.det(W[:, :2])), 1e-12))
    return s, float(p[0] - c[0]), float(p[1] - c[1])
