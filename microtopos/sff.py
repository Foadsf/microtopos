"""Shape from focus: sweep segmentation, sub-frame depth fitting in z, confidence, edge-aware refinement,
all-in-focus fusion and the experimental depth-of-field ruler."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

# ---------------------------------------------------------------------------------------------- segmentation


def monotonic_segments_from_z(z: np.ndarray, min_len: int = 5) -> list[tuple[int, int]]:
    """Split known z labels at direction reversals; pauses (dz == 0) inherit the previous direction."""
    dz = np.sign(np.diff(z))
    for i in range(1, len(dz)):
        if dz[i] == 0:
            dz[i] = dz[i - 1]
    cuts = [0] + [i + 1 for i in range(1, len(dz)) if dz[i] * dz[i - 1] < 0] + [len(z)]
    segs = [(a, b) for a, b in zip(cuts[:-1], cuts[1:]) if b - a >= min_len]
    return sorted(segs, key=lambda s: s[1] - s[0], reverse=True) or [(0, len(z))]


def find_optical_reversal(tiles: np.ndarray, min_arm: int = 4) -> int | None:
    """Turnaround index of a sweep with unknown z, from per-tile focus vectors (n_frames x n_tiles).

    At a reversal at T, frame T-m and T+m see the same focus plane, so their tile vectors match while
    frames T-m and T differ. A pause makes both distances small and is therefore not mistaken for one.
    """
    n = len(tiles)
    if n < 2 * min_arm + 1:
        return None
    v = np.log(tiles + 1e-6)
    v = (v - v.mean(0)) / (v.std(0) + 1e-6)
    noise = np.median(np.linalg.norm(np.diff(v, axis=0), axis=1)) + 1e-6
    best, best_score = None, 0.35
    for t in range(min_arm, n - min_arm):
        arm = min(t, n - 1 - t, 12)
        m = np.arange(1, arm + 1)
        mirror = np.linalg.norm(v[t - m] - v[t + m], axis=1).mean()
        travel = np.linalg.norm(v[t - m] - v[t], axis=1).mean()
        if travel < 3 * noise:
            continue
        score = mirror / travel
        if score < best_score:
            best, best_score = t, score
    return best


def find_reversal(tiles: np.ndarray, min_prominence: float = 0.5) -> int | None:
    """Turnaround index from tiles that come into focus twice (forward pass and return pass).

    For each tile with two clear focus peaks at frames a < b, the turnaround lies near (a + b) / 2.
    A tile with two height levels can also show two peaks, so a reversal is accepted only when such
    midpoints agree across many tiles; the estimate is then refined with the mirror score.
    """
    from scipy.signal import find_peaks

    n = len(tiles)
    lt = np.log(tiles + 1e-6)
    mids = []
    for j in range(lt.shape[1]):
        tr = np.convolve(lt[:, j], np.ones(3) / 3, mode="same")
        pk, _ = find_peaks(tr, prominence=min_prominence, distance=4)
        if len(pk) >= 2:
            mids.append(0.5 * (pk[0] + pk[-1]))
    if len(mids) < max(3, 0.25 * lt.shape[1]):
        return find_optical_reversal(tiles)
    mids = np.array(mids)
    med = float(np.median(mids))
    agree = np.abs(mids - med) <= max(3.0, 0.06 * n)
    if agree.mean() < 0.6:
        return find_optical_reversal(tiles)
    v = (lt - lt.mean(0)) / (lt.std(0) + 1e-6)
    best, best_s = int(round(med)), np.inf
    for t in range(max(2, int(med) - 6), min(n - 2, int(med) + 7)):
        arm = min(t, n - 1 - t, 15)
        m = np.arange(1, arm + 1)
        s = np.linalg.norm(v[t - m] - v[t + m], axis=1).mean()
        if s < best_s:
            best, best_s = t, s
    return best


def drop_pauses(tiles: np.ndarray, rel: float = 0.25) -> np.ndarray:
    """Indices to keep when z is unknown: consecutive frames whose tile vectors barely change are a pause."""
    v = np.log(tiles + 1e-6)
    v = (v - v.mean(0)) / (v.std(0) + 1e-6)
    step = np.linalg.norm(np.diff(v, axis=0), axis=1)
    thr = rel * np.median(step)
    keep = [0]
    for k in range(1, len(v)):
        if np.linalg.norm(v[k] - v[keep[-1]]) > thr:
            keep.append(k)
    return np.array(keep)


# ---------------------------------------------------------------------------------------------- depth fit


@dataclass
class DepthFit:
    z: np.ndarray  # fitted depth (same unit as the z labels), NaN where unreliable
    z_raw: np.ndarray  # before masking/refinement
    confidence: np.ndarray  # 0..1
    peak_index: np.ndarray
    fwhm_z_median: float  # median focus-curve FWHM in z units (from the Gaussian fit)


def _gather(fv, idx):
    return np.take_along_axis(fv, idx[None], axis=0)[0]


def fit_depth(fv: np.ndarray, z: np.ndarray, conf_threshold: float = 0.25) -> DepthFit:
    """Per-pixel peak of the focus curve, fitted as a Gaussian (parabola in log FV) on the actual z values.

    Uses 5 samples around the peak (least squares) when available, otherwise 3. Confidence combines the
    peak's prominence over the curve's floor (in log units) with its robust SNR, and is zero at the sweep ends.
    """
    n = fv.shape[0]
    z = np.asarray(z, np.float64)
    lf = np.log(np.maximum(fv, 1e-12)).astype(np.float32)
    k = np.argmax(lf, axis=0)
    interior = (k >= 1) & (k <= n - 2)
    kc = np.clip(k, 2, n - 3) if n >= 5 else np.clip(k, 1, n - 2)
    offs = (-2, -1, 0, 1, 2) if n >= 5 else (-1, 0, 1)
    z0 = z[k]
    # weighted LS fit y = a dz^2 + b dz + c, weights 0 for samples outside the valid range
    S = {p: 0.0 for p in range(5)}
    T = {p: 0.0 for p in range(3)}
    for o in offs:
        j = kc + o
        use = (j >= 0) & (j < n) & (np.abs(j - k) <= 2)
        jj = np.clip(j, 0, n - 1)
        x = (z[jj] - z0).astype(np.float32)
        y = _gather(lf, jj)
        wgt = use.astype(np.float32)
        xp = np.ones_like(x)
        for p in range(5):
            S[p] = S[p] + wgt * xp
            if p < 3:
                T[p] = T[p] + wgt * xp * y
            xp = xp * x
    # normal equations [[S4 S3 S2],[S3 S2 S1],[S2 S1 S0]] [a b c] = [T2 T1 T0]
    m11, m12, m13 = S[4], S[3], S[2]
    m22, m23, m33 = S[2], S[1], S[0]
    det = m11 * (m22 * m33 - m23 * m23) - m12 * (m12 * m33 - m23 * m13) + m13 * (m12 * m23 - m22 * m13)
    det = np.where(np.abs(det) < 1e-20, np.nan, det)
    a = (T[2] * (m22 * m33 - m23 * m23) - m12 * (T[1] * m33 - m23 * T[0]) + m13 * (T[1] * m23 - m22 * T[0])) / det
    b = (m11 * (T[1] * m33 - m23 * T[0]) - T[2] * (m12 * m33 - m23 * m13) + m13 * (m12 * T[0] - T[1] * m13)) / det
    peak_ok = interior & np.isfinite(a) & (a < 0)
    dz_hi = np.abs(z[np.clip(k + 1, 0, n - 1)] - z0)
    dz_lo = np.abs(z[np.clip(k - 1, 0, n - 1)] - z0)
    lim = np.maximum(np.maximum(dz_hi, dz_lo), 1e-9)
    delta = np.where(peak_ok, -b / (2 * np.where(peak_ok, a, -1.0)), 0.0)
    delta = np.clip(delta, -lim, lim)
    z_raw = (z0 + delta).astype(np.float32)

    floor = np.percentile(lf, 10, axis=0)
    peak = _gather(lf, k)
    prominence = peak - floor  # natural-log units: 0.7 = peak twice the floor
    # measurement noise of the curve from its second differences (a smooth focus curve has ~0 there)
    d2 = lf[2:] - 2 * lf[1:-1] + lf[:-2] if n >= 3 else np.zeros_like(lf)
    noise = np.median(np.abs(d2), axis=0) * 1.4826 / np.sqrt(6.0) + 1e-3
    snr = prominence / noise
    conf = np.clip((prominence - 0.2) / 0.8, 0, 1) * np.clip((snr - 3.0) / 5.0, 0, 1)
    conf = (conf * peak_ok).astype(np.float32)

    sig = np.sqrt(-1.0 / (2 * a[peak_ok & (conf > 0.5)])) if np.any(peak_ok & (conf > 0.5)) else np.array([np.nan])
    fwhm = float(np.nanmedian(2.3548 * sig)) if sig.size else float("nan")
    zm = np.where(conf >= conf_threshold, z_raw, np.nan).astype(np.float32)
    return DepthFit(zm, z_raw, conf, k.astype(np.int32), fwhm)


# ---------------------------------------------------------------------------------------------- refinement


def weighted_guided_filter(p: np.ndarray, w: np.ndarray, guide: np.ndarray, radius: int, eps: float) -> np.ndarray:
    """Guided filter (He et al.) with per-pixel data weights w; NaN in p counts as weight 0."""
    w = np.where(np.isfinite(p), w, 0.0).astype(np.float32)
    p0 = np.where(np.isfinite(p), p, 0.0).astype(np.float32)
    I = guide.astype(np.float32)
    k = (2 * radius + 1, 2 * radius + 1)

    def box(x):
        return cv2.boxFilter(x, cv2.CV_32F, k, normalize=False, borderType=cv2.BORDER_REFLECT)

    sw = box(w) + 1e-6
    mI = box(w * I) / sw
    mp = box(w * p0) / sw
    vI = box(w * I * I) / sw - mI * mI
    cIp = box(w * I * p0) / sw - mI * mp
    A = cIp / (vI + eps)
    B = mp - A * mI
    n = box(np.ones_like(I))
    support = box((w > 0).astype(np.float32)) / n
    A = box(A) / n
    B = box(B) / n
    out = A * I + B
    return np.where(support > 0.15, out, np.nan).astype(np.float32)


def refine_depth(z: np.ndarray, conf: np.ndarray, guide: np.ndarray, radius: int) -> np.ndarray:
    """Edge-aware smoothing guided by the all-in-focus image, with one round of outlier rejection."""
    g = guide.astype(np.float32)
    g = (g - g.mean()) / (g.std() + 1e-6)
    zr = np.nanstd(z) if np.any(np.isfinite(z)) else 1.0
    eps = (0.1) ** 2
    zn = z / (zr + 1e-9)
    sm = weighted_guided_filter(zn, conf, g, radius, eps)
    resid = np.abs(zn - sm)
    r_ok = resid[np.isfinite(resid)]
    if r_ok.size:
        thr = 4.0 * 1.4826 * np.median(r_ok) + 1e-6
        zn = np.where(resid > thr, np.nan, zn)
        sm = weighted_guided_filter(zn, conf, g, radius, eps)
    out = np.where(np.isfinite(zn), sm, np.nan) * (zr + 1e-9)
    return out.astype(np.float32)


# ---------------------------------------------------------------------------------------------- fusion


class EDFAccumulator:
    """Streaming Laplacian-pyramid fusion: add (frame, weight) pairs one at a time, normalised at the end.

    Weights should already be normalised across the stack per pixel; they are low-pass filtered by the
    Gaussian pyramid, which avoids seams between frames.
    """

    def __init__(self, shape: tuple[int, int], levels: int = 5):
        self.shape = shape
        self.levels = max(1, min(levels, int(np.log2(min(shape))) - 3))
        self.acc = None
        self.wsum = None

    def _gauss(self, a):
        pyr = [a]
        for _ in range(self.levels):
            pyr.append(cv2.pyrDown(pyr[-1]))
        return pyr

    def add(self, frame: np.ndarray, weight: np.ndarray) -> None:
        g = self._gauss(frame.astype(np.float32))
        lap = [g[i] - cv2.pyrUp(g[i + 1], dstsize=(g[i].shape[1], g[i].shape[0])) for i in range(self.levels)]
        lap.append(g[-1])
        wp = self._gauss(weight.astype(np.float32))
        if self.acc is None:
            self.acc = [np.zeros_like(x) for x in lap]
            self.wsum = [np.zeros_like(x) for x in wp]
        for i in range(len(lap)):
            self.acc[i] += wp[i] * lap[i]
            self.wsum[i] += wp[i]

    def result(self) -> np.ndarray:
        parts = [a / np.maximum(w, 1e-6) for a, w in zip(self.acc, self.wsum)]
        out = parts[-1]
        for i in range(self.levels - 1, -1, -1):
            out = cv2.pyrUp(out, dstsize=(parts[i].shape[1], parts[i].shape[0])) + parts[i]
        return out


def softmax_weights(fv: np.ndarray, temperature: float = 0.25) -> np.ndarray:
    """Per-pixel softmax over log focus energy; temperature is in natural-log units."""
    lf = np.log(np.maximum(fv, 1e-12))
    lf = lf - lf.max(axis=0, keepdims=True)
    e = np.exp(lf / temperature)
    return (e / e.sum(axis=0, keepdims=True)).astype(np.float32)


# ---------------------------------------------------------------------------------------------- DOF ruler


def theoretical_fwhm_um(objective, um_per_px: float, measure_fn, texture: np.ndarray) -> float:
    """FWHM (in um of defocus) of the focus measure's response for this objective and this specimen.

    The all-in-focus image is blurred with the optical defocus model at known defocus steps, so the
    ruler has no hand-tuned factor. Assumes the all-in-focus image is close to the true in-focus texture.
    """
    h, w = texture.shape
    c = min(h, w, 256) // 2
    tex = texture[h // 2 - c:h // 2 + c, w // 2 - c:w // 2 + c].astype(np.float32)
    dof = objective.dof_um
    dz = np.linspace(-4 * dof, 4 * dof, 81)
    resp = []
    for d in dz:
        # extra blur on top of the in-focus PSF already present in the texture
        s = float(np.sqrt(max(objective.defocus_sigma_um(d) ** 2 - objective.airy_sigma_um**2, 0))) / um_per_px
        img = cv2.GaussianBlur(tex, (0, 0), s) if s > 0.05 else tex
        resp.append(float(np.mean(measure_fn(img)[20:-20, 20:-20])))
    r_arr = np.asarray(resp, dtype=np.float64)
    r_max = float(np.max(r_arr))
    r_min = float(np.min(r_arr))
    if not np.isfinite(r_max) or r_max <= 1e-12 or r_max <= r_min * (1.0 + 1e-4):
        return float("nan")
    lr = np.log(np.maximum(r_arr, 1e-12))
    half = lr.max() - np.log(2)
    above = dz[lr >= half]
    return float(above.max() - above.min())
