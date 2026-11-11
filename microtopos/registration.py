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


@dataclass
class PairRegistration:
    warp: np.ndarray  # 2x3, template coords -> moving coords, full resolution
    ecc: float  # ECC correlation (or phase-correlation response on fallback)
    ok: bool  # False when ECC failed or was rejected and the translation-only estimate was used


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
    W, cc, ok = W0, resp, False
    try:
        crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 60, 1e-5)
        motion = cv2.MOTION_TRANSLATION if model == "translation" else cv2.MOTION_AFFINE
        cc_e, We = cv2.findTransformECC(t, m, W0.copy(), motion, crit, None, 3)
        Ws, sc, rot = _to_similarity(We)
        if np.isfinite(Ws).all() and abs(sc - 1) < max_scale_dev and abs(rot) < max_rot_deg \
                and math.hypot(We[0, 2] - dx, We[1, 2] - dy) < 3.0:
            W, cc, ok = Ws, float(cc_e), True
    except cv2.error:
        pass
    # rescale to full resolution, honouring INTER_AREA pixel centres: x_full = f*x + (f-1)/2
    if factor > 1:
        A = W[:, :2].astype(np.float64)
        c = np.full(2, (factor - 1) / 2.0)
        tr = factor * W[:, 2].astype(np.float64) + (np.eye(2) - A) @ c
        W = np.hstack([A, tr[:, None]]).astype(np.float32)
    return PairRegistration(W.astype(np.float32), cc, ok)


def register_sweep(frames: list[np.ndarray], ref: int | None = None, factor: int = 2, workers: int = 4,
                   progress=None) -> tuple[list[np.ndarray], list[PairRegistration]]:
    """Register a focus sweep to its reference frame.

    Consecutive pairs (similar defocus) are registered with translation only, in parallel, and chained;
    a pair whose ECC fails inherits the median motion of its neighbours. Allowing scale in every pair lets
    defocus-induced scale errors accumulate along the chain, so the model is translation only.
    Returns warps W_k (reference coords -> frame k coords) and the pair results (pair k = frames k, k+1).
    """
    n = len(frames)
    ref = n // 2 if ref is None else ref
    done = [0]

    def job(k):
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


def warp_to_reference(frame: np.ndarray, W: np.ndarray) -> np.ndarray:
    h, w = frame.shape[:2]
    return cv2.warpAffine(frame, W, (w, h), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                          borderMode=cv2.BORDER_REPLICATE)


def warp_scale_shift(W: np.ndarray, shape: tuple[int, int]) -> tuple[float, float, float]:
    """Scale and the displacement of the image centre implied by warp W (for diagnostics and tests)."""
    h, w = shape
    c = np.array([(w - 1) / 2.0, (h - 1) / 2.0])
    p = W[:, :2] @ c + W[:, 2]
    s = math.sqrt(max(float(np.linalg.det(W[:, :2])), 1e-12))
    return s, float(p[0] - c[0]), float(p[1] - c[1])


def register_joint_intensity_height(
    template_img: np.ndarray,
    template_height: np.ndarray | None,
    moving_img: np.ndarray,
    moving_height: np.ndarray | None,
    weight_height: float = 0.5,
) -> tuple[float, float, float]:
    """Jointly register two observations using optical intensity and 2.5D topographic surface gradient.

    When optical contrast is low or specular, topographic gradient information provides robust spatial lock.

    Returns:
        (dx, dy, correlation)
    """
    t_img = template_img.astype(np.float32)
    m_img = moving_img.astype(np.float32)

    t_norm = (t_img - np.nanmean(t_img)) / (np.nanstd(t_img) + 1e-6)
    m_norm = (m_img - np.nanmean(m_img)) / (np.nanstd(m_img) + 1e-6)

    has_height = (
        template_height is not None
        and moving_height is not None
        and np.isfinite(template_height).any()
        and np.isfinite(moving_height).any()
    )

    if has_height:
        th = np.nan_to_num(template_height, nan=float(np.nanmedian(template_height)))
        mh = np.nan_to_num(moving_height, nan=float(np.nanmedian(moving_height)))

        gx_t = cv2.Sobel(th.astype(np.float32), cv2.CV_32F, 1, 0)
        gy_t = cv2.Sobel(th.astype(np.float32), cv2.CV_32F, 0, 1)
        grad_t = np.hypot(gx_t, gy_t)
        grad_t = (grad_t - np.mean(grad_t)) / (np.std(grad_t) + 1e-6)

        gx_m = cv2.Sobel(mh.astype(np.float32), cv2.CV_32F, 1, 0)
        gy_m = cv2.Sobel(mh.astype(np.float32), cv2.CV_32F, 0, 1)
        grad_m = np.hypot(gx_m, gy_m)
        grad_m = (grad_m - np.mean(grad_m)) / (np.std(grad_m) + 1e-6)

        wh = np.clip(weight_height, 0.0, 1.0)
        t_joint = (1.0 - wh) * t_norm + wh * grad_t
        m_joint = (1.0 - wh) * m_norm + wh * grad_m
    else:
        t_joint = t_norm
        m_joint = m_norm

    h, w = t_joint.shape[:2]
    win = cv2.createHanningWindow((w, h), cv2.CV_32F)
    dx, dy, resp = phase_shift(t_joint, m_joint, win)

    try:
        crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 1e-4)
        _, W = cv2.findTransformECC(
            t_joint,
            m_joint,
            np.float32([[1, 0, dx], [0, 1, dy]]),
            cv2.MOTION_TRANSLATION,
            crit,
            None,
            1,
        )
        if np.isfinite(W).all() and math.hypot(W[0, 2] - dx, W[1, 2] - dy) < 2.0:
            dx, dy = float(W[0, 2]), float(W[1, 2])
    except cv2.error:
        pass

    return dx, dy, resp

