"""Rolling-shutter readout fraction from registered frame pairs, and deskewing of frames and points.

Model. A rolling-shutter sensor exposes row r at t0 + rho * T * r / H (T the frame interval, H the frame height in
rows, rho the readout fraction, 0..1). While the image content moves at a constant v = (vx, vy) px/frame, the row a
scene point lands in is displaced, relative to the middle row c = (H - 1) / 2, by v * k * (y - c) with k = rho / H,
where y is the row the point is actually recorded in. With (X, Y) the point's position in a global-shutter frame
taken at the middle row's time, origin on the middle row:

    x = X + vx * k * y'            y' = Y' + vy * k * y'        (y' = y - c, Y' = Y - c)

so the frame is the global-shutter frame mapped by the linear operator

    S(v) = [[1, vx k / (1 - vy k)],
            [0, 1 / (1 - vy k)  ]]

about the middle row: a horizontal pan shears the frame, a vertical pan stretches it (content moving down,
vy > 0, is stretched) or compresses it (vy < 0). The model is exact for a velocity that is constant during the
frame's readout; motion blur within one row's exposure is a separate effect and is not modelled here.

Velocity convention. `registration.register_pair(template=a, moving=b)` returns a warp W with b(W x) = a(x): a
translation (dx, dy) means content at x in a is at x + (dx, dy) in b. A frame's velocity is that translation from
the previous frame to it, in px/frame. Two consecutive frames at the same velocity have the same S, so their
relative transform is a pure translation (exactly v, the middle row's displacement); skew shows only against a frame
taken at another velocity.

Pair relation. For frames a and b with velocities va and vb, the linear part of the warp a -> b is
A = S(vb) S(va)^-1, which gives two relations that are EXACTLY linear in k once A is measured:

    A[0,1]     = k * (A[1,1] * vxb - vxa)     (shear)
    A[1,1] - 1 = k * (A[1,1] * vyb - vya)     (stretch)

and A[0,0] = 1, A[1,0] = 0. To first order these are shear = k (vxb - vxa), stretch = k (vyb - vya): rho is
observable only from pairs whose velocities differ.

Not verified: every number here was checked against a synthetic rolling-shutter camera built from the specimen
simulator (tests/test_rollingshutter.py). Nothing has run on the OV2710 camera, whose readout fraction is unknown.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import cv2
import numpy as np

from .calibration import bgr_to_luma
from .registration import SNR_MIN_CORRELATION, register_pair, remove_shading

# A pair whose velocities differ by less than this (px/frame) is left out of the fit: its skew is below rho px from top
# to bottom of the frame, and a velocity difference this small is comparable with the frame-to-frame jitter of a
# hand-held pan (the specimen simulator draws 0.1-0.45 px of tremor per frame). Frames at equal velocity have equal
# skew, so pairs of them never enter the fit.
MIN_PAIR_DV_PX = 1.0
# The estimate is refused when its standard error exceeds this: the pairs do not determine rho (velocity differences
# too small for the fit's noise, or skews inconsistent with the stated velocities). On the synthetic camera, 180-row
# frames with 4-11 px/frame differences gave standard errors of 3e-4 to 9e-4.
RHO_SE_MAX = 0.05
# The affine fit runs coarse-to-fine from the translation; the coarsest level has at most this many pixels per side.
AFFINE_COARSE_SIDE = 80


@dataclass
class PairSkew:
    """Measured linear part of the warp a -> b (register_pair convention) for a pair with known velocities."""
    A: np.ndarray  # 2x2
    velocity_a: tuple[float, float]
    velocity_b: tuple[float, float]
    ecc: float


def skew_matrix(velocity: Sequence[float], rho: float, height: int) -> np.ndarray:
    """2x3 affine map from global-shutter (deskewed) pixel coordinates to rolling-shutter frame coordinates.

    S(v) of the module docstring about the middle row c = (height - 1) / 2. Raises ValueError when the vertical
    speed reaches the readout speed (1 - vy rho / H <= 0), where rows are recorded out of order."""
    vx, vy = float(velocity[0]), float(velocity[1])
    k = float(rho) / float(height)
    den = 1.0 - vy * k
    if den <= 0:
        raise ValueError(f"vertical speed {vy} px/frame is not below the readout speed {height / max(rho, 1e-12):.1f}")
    c = (height - 1) / 2.0
    a01, a11 = vx * k / den, 1.0 / den
    return np.array([[1.0, a01, -a01 * c], [0.0, a11, c - a11 * c]], np.float64)


def deskew(frame: np.ndarray, velocity: Sequence[float], rho: float) -> np.ndarray:
    """The frame as a global shutter at the middle row's time would have recorded it (exact for constant velocity).

    Rows near the top and bottom of a fast pan may sample outside the frame; those pixels replicate the border and
    are not observations."""
    h, w = frame.shape[:2]
    M = skew_matrix(velocity, rho, h).astype(np.float32)
    return cv2.warpAffine(frame, M, (w, h), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                          borderMode=cv2.BORDER_REPLICATE)


def deskew_points(points: np.ndarray, velocity: Sequence[float], rho: float, height: int) -> np.ndarray:
    """Global-shutter coordinates of points (N x 2, x then y) measured in a rolling-shutter frame."""
    p = np.asarray(points, np.float64).reshape(-1, 2)
    M = skew_matrix(velocity, rho, height)
    inv = cv2.invertAffineTransform(M)
    return p @ inv[:, :2].T + inv[:, 2]


def _gray(img: np.ndarray) -> np.ndarray:
    return bgr_to_luma(img).astype(np.float32, copy=False)


def measure_pair(frame_a: np.ndarray, frame_b: np.ndarray) -> tuple[np.ndarray, float] | None:
    """(2x3 affine warp a -> b in register_pair's convention, ECC correlation), or None when not measurable.

    The translation and the decision that motion is observable come from `register_pair`; registration returns
    translation or similarity only, so the shear and stretch terms are refined here with an affine ECC fit on the
    shading-free frames, coarse to fine from that translation. The fit is accepted when its correlation reaches
    registration's SNR_MIN_CORRELATION."""
    a, b = _gray(frame_a), _gray(frame_b)
    reg = register_pair(a, b, factor=1, model="translation")
    if not reg.observable:
        return None
    ta, tb = remove_shading(a), remove_shading(b)
    h, w = ta.shape
    levels = max(0, math.ceil(math.log2(max(h, w) / AFFINE_COARSE_SIDE)))
    W = reg.warp.astype(np.float64)
    crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 1e-6)
    cc = float("nan")
    for lev in range(levels, -1, -1):
        f = 2 ** lev
        A, t = W[:, :2], W[:, 2]
        c = np.full(2, (f - 1) / 2.0)
        if f > 1:  # INTER_AREA pixel centres: x_full = f * x + (f - 1) / 2
            size = (w // f, h // f)
            la = cv2.resize(ta, size, interpolation=cv2.INTER_AREA)
            lb = cv2.resize(tb, size, interpolation=cv2.INTER_AREA)
        else:
            la, lb = ta, tb
        init = np.hstack([A, ((A @ c + t - c) / f)[:, None]]).astype(np.float32)
        try:
            cc, Wl = cv2.findTransformECC(la, lb, init, cv2.MOTION_AFFINE, crit, None, 3)
        except cv2.error:
            return None
        if not np.isfinite(Wl).all():
            return None
        Al, tl = Wl[:, :2].astype(np.float64), Wl[:, 2].astype(np.float64)
        W = np.hstack([Al, (f * tl + (np.eye(2) - Al) @ c)[:, None]])
    if not float(cc) >= SNR_MIN_CORRELATION:
        return None
    return W, float(cc)


def estimate_readout(pairs: Iterable[tuple[np.ndarray, Sequence[float], np.ndarray, Sequence[float]]]) -> dict:
    """Least-squares readout fraction rho from (frame_a, velocity_a, frame_b, velocity_b) pairs.

    Velocities are in px/frame in register_pair's convention (the translation from the frame before). Each pair is
    registered b to a with an affine model and contributes the four relations of the module docstring: shear and
    stretch with regressors (A11 vxb - vxa) / H and (A11 vyb - vya) / H, and A00 - 1 = 0, A10 = 0 (no regressor; they
    measure the fit's noise). rho is the least-squares slope through the origin; its standard error is the residual
    RMS over sqrt(sum of squared regressors).

    Returns a dict: observable (bool), rho, rho_se (standard error), residual_rms (of the affine terms), n_pairs
    (pairs used), n_rejected (pairs whose registration failed), n_low_dv (pairs whose velocity difference is below
    MIN_PAIR_DV_PX), reason (why rho is not observable, else ""). rho and rho_se are None when not observable.
    """
    rows, rhs, n_rej, n_low, height = [], [], 0, 0, None
    for fa, va, fb, vb in pairs:
        va = (float(va[0]), float(va[1]))
        vb = (float(vb[0]), float(vb[1]))
        if math.hypot(vb[0] - va[0], vb[1] - va[1]) < MIN_PAIR_DV_PX:
            n_low += 1
            continue
        H = fa.shape[0]
        if fb.shape[:2] != fa.shape[:2] or (height is not None and H != height):
            raise ValueError("all frames must have the same size")
        height = H
        m = measure_pair(fa, fb)
        if m is None:
            n_rej += 1
            continue
        A = m[0][:, :2]
        a11 = A[1, 1]
        rows += [(a11 * vb[0] - va[0]) / H, (a11 * vb[1] - va[1]) / H, 0.0, 0.0]
        rhs += [A[0, 1], a11 - 1.0, A[0, 0] - 1.0, A[1, 0]]
    out = {"observable": False, "rho": None, "rho_se": None, "residual_rms": None,
           "n_pairs": len(rows) // 4, "n_rejected": n_rej, "n_low_dv": n_low, "reason": ""}
    if not rows:
        out["reason"] = (f"no pair with a velocity difference of at least {MIN_PAIR_DV_PX} px/frame registered: "
                         "frames at equal velocity have equal skew, so rho does not show between them")
        return out
    g, y = np.asarray(rows), np.asarray(rhs)
    sgg = float(g @ g)
    rho = float(g @ y) / sgg if sgg > 0 else 0.0
    res = y - rho * g
    rms = math.sqrt(float(res @ res) / max(len(y) - 1, 1))
    se = rms / math.sqrt(sgg) if sgg > 0 else math.inf
    out.update(residual_rms=rms)
    if not se <= RHO_SE_MAX:
        out["reason"] = (f"standard error of rho {se:.3g} exceeds {RHO_SE_MAX}: velocity differences too small for "
                         "the fit's noise, or skews inconsistent with the stated velocities")
        return out
    out.update(observable=True, rho=rho, rho_se=se)
    return out
