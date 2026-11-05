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


def _to_similarity(A: np.ndarray) -> tuple[np.ndarray, float, float]:
    lin = A[:, :2]
    theta = math.atan2(lin[1, 0] - lin[0, 1], lin[0, 0] + lin[1, 1])
    s = math.sqrt(max(np.linalg.det(lin), 1e-12))
    c, si = s * math.cos(theta), s * math.sin(theta)
    out = A.copy()
    out[:, :2] = [[c, -si], [si, c]]
    return out, s, math.degrees(theta)


def register_pair(template: np.ndarray, moving: np.ndarray, factor: int = 2, model: str = "translation",
                  max_scale_dev: float = 0.03, max_rot_deg: float = 2.0) -> PairRegistration:
    """Translation ("translation") or similarity ("similarity") transform between two frames.

    Estimated at 1/factor resolution and returned at full resolution.
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
    try:
        crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 60, 1e-5)
        motion = cv2.MOTION_TRANSLATION if model == "translation" else cv2.MOTION_AFFINE
        init = W0.copy() if qualified else np.array([[1, 0, 0], [0, 1, 0]], np.float32)
        cc_e, We = cv2.findTransformECC(t, m, init, motion, crit, None, 3)
        Ws, sc, rot = _to_similarity(We)
        shape_ok = np.isfinite(Ws).all() and abs(sc - 1) < max_scale_dev and abs(rot) < max_rot_deg
        near = math.hypot(We[0, 2] - init[0, 2], We[1, 2] - init[1, 2]) < 3.0
        # without a qualified phase peak, ECC is trusted only with a strong correlation near zero motion
        if shape_ok and near and (qualified or float(cc_e) >= 0.6):
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
