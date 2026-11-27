"""Shape from focus: sweep segmentation, sub-frame depth fitting in z, confidence, edge-aware refinement,
all-in-focus fusion and the experimental depth-of-field ruler."""

from __future__ import annotations

import itertools
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
    segs = [(a, b) for a, b in itertools.pairwise(cuts) if b - a >= min_len]
    return sorted(segs, key=lambda s: s[1] - s[0], reverse=True) or [(0, len(z))]


# Largest mirror / travel ratio (see _mirror_travel) at which a turnaround located from focus peaks that include a
# maximum at the first or last frame is accepted. Such a maximum is also what a single pass that starts or ends
# sharp on another layer (dust, a cover slip) shows in every tile, so it needs the mirror evidence a peak inside the
# sweep does not. Measured on simulated hand sweeps (seeds 10-25, 10x / 0.25 and 40x / 0.85, tremor on): true
# turnarounds 0.08-0.39; single passes with a decaying layer at either end (tests/test_reported_bugs.py B8,
# 30-300 % of the surface's contrast) 0.62-0.80. find_optical_reversal searches every frame without peak evidence and
# keeps its stricter 0.35.
BOUNDARY_PEAK_MIRROR_MAX = 0.5


def _mirror_travel(v: np.ndarray, t: int, cap: int) -> tuple[float, float]:
    """Mean distance between the standardised tile vectors of frames t-m and t+m (mirror) and of t-m and t
    (travel), over m = 1 .. min(t, n-1-t, cap). At a turnaround mirror is small while travel is not."""
    arm = min(t, len(v) - 1 - t, cap)
    m = np.arange(1, arm + 1)
    return (float(np.linalg.norm(v[t - m] - v[t + m], axis=1).mean()),
            float(np.linalg.norm(v[t - m] - v[t], axis=1).mean()))


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
        mirror, travel = _mirror_travel(v, t, 12)
        if travel < 3 * noise:
            continue
        score = mirror / travel
        if score < best_score:
            best, best_score = t, score
    return best


def _agreeing_median(mids: list[float], n: int, n_tiles: int) -> float | None:
    """Median turnaround of the per-tile midpoints when enough tiles give one and most of them agree."""
    if len(mids) < max(3, 0.25 * n_tiles):
        return None
    mids_a = np.array(mids)
    med = float(np.median(mids_a))
    if (np.abs(mids_a - med) <= max(3.0, 0.06 * n)).mean() < 0.6:
        return None
    return med


def find_reversal(tiles: np.ndarray, min_prominence: float = 0.5, min_arm: int = 4) -> int | None:
    """Turnaround index from tiles that come into focus twice (forward pass and return pass).

    For each tile with two clear focus peaks at frames a < b, the turnaround lies near (a + b) / 2.
    A tile with two height levels can also show two peaks, so a reversal is accepted only when such
    midpoints agree across many tiles; the estimate is then refined with the mirror score.

    When the user turns back and stops near focus, the return pass's peak is the last frame: the curve is still
    rising there. Such an end maximum (or a start maximum, for a sweep that starts near focus) counts as a peak
    when it is prominent, but only if peaks inside the sweep did not already decide, and only if the frames
    around the turnaround mirror each other (BOUNDARY_PEAK_MIRROR_MAX), because a single pass that starts or ends
    sharp on another layer has the same maximum in every tile. Its midpoint (a + last) / 2 is a lower bound of the
    turnaround (the return peak lies at or beyond the last frame), so the search extends towards that end.
    """
    from scipy.signal import find_peaks

    n = len(tiles)
    lt = np.log(tiles + 1e-6)
    mids, mids_ends, sides = [], [], [0, 0]  # sides: tiles whose second peak is the first / the last frame
    for j in range(lt.shape[1]):
        tr = np.convolve(np.pad(lt[:, j], 1, mode="edge"), np.ones(3) / 3, mode="valid")
        pk, _ = find_peaks(tr, prominence=min_prominence, distance=4)
        if len(pk) >= 2:
            mids.append(0.5 * (pk[0] + pk[-1]))
            mids_ends.append(mids[-1])
            continue
        # a maximum at either end of the sweep is a peak of the curve mirrored about that end (full reflection,
        # so its prominence is measured against the trough on its one real side)
        pe, _ = find_peaks(np.pad(tr, n - 1, mode="reflect"), prominence=min_prominence, distance=4)
        pe = np.unique(pe[(pe >= n - 1) & (pe <= 2 * n - 2)] - (n - 1))
        if len(pe) == 2 and len(pk) == 1 and (pe[0] == 0) != (pe[1] == n - 1):  # one inner peak, one end
            mids_ends.append(0.5 * (pe[0] + pe[1]))
            sides[int(pe[1] == n - 1)] += 1
    v = (lt - lt.mean(0)) / (lt.std(0) + 1e-6)
    med = _agreeing_median(mids, n, lt.shape[1])
    if med is not None:
        best, best_s = round(med), np.inf
        for t in range(max(2, int(med) - 6), min(n - 2, int(med) + 7)):
            s, _ = _mirror_travel(v, t, 15)
            if s < best_s:
                best, best_s = t, s
        return best
    med = _agreeing_median(mids_ends, n, lt.shape[1])
    if med is not None and n >= 2 * min_arm + 1:
        lo = min_arm if sides[0] > sides[1] else max(min_arm, int(med) - 6)
        hi = n - min_arm if sides[1] > sides[0] else min(n - min_arm, int(med) + 7)
        best, best_score = None, BOUNDARY_PEAK_MIRROR_MAX
        for t in range(lo, hi):
            mirror, travel = _mirror_travel(v, t, 12)
            score = mirror / max(travel, 1e-12)
            if score < best_score:
                best, best_score = t, score
        if best is not None:
            return best
    return find_optical_reversal(tiles)


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
    fwhm_z_samples: np.ndarray | None = None  # the per-pixel FWHMs that median is taken over (strip-wise stitching)


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
    fwhm_samples = (2.3548 * sig).astype(np.float64)
    fwhm = float(np.nanmedian(fwhm_samples)) if sig.size else float("nan")
    zm = np.where(conf >= conf_threshold, z_raw, np.nan).astype(np.float32)
    return DepthFit(zm, z_raw, conf, k.astype(np.int32), fwhm, fwhm_samples)


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


def refine_depth(z: np.ndarray, conf: np.ndarray, guide: np.ndarray, radius: int,
                 *, backend: str = "cpu") -> tuple[np.ndarray, set[str]]:
    """Edge-aware smoothing guided by the all-in-focus image, with one round of outlier rejection.

    backend selects "cpu" (reference) or "opencl" for the guided filter (CPU fallback if OpenCL is
    unavailable or fails); see backend.guided_filter_backend. Returns (refined depth, the set of backends the
    guided filter actually ran on), so the caller can record what ran.
    """
    from .backend import guided_filter_backend
    g = guide.astype(np.float32)
    g = (g - g.mean()) / (g.std() + 1e-6)
    zr = np.nanstd(z) if np.any(np.isfinite(z)) else 1.0
    eps = (0.1) ** 2
    zn = z / (zr + 1e-9)
    sm, used = guided_filter_backend(zn, conf, g, radius, eps, backend)
    ran = {used}
    resid = np.abs(zn - sm)
    r_ok = resid[np.isfinite(resid)]
    if r_ok.size:
        thr = 4.0 * 1.4826 * np.median(r_ok) + 1e-6
        zn = np.where(resid > thr, np.nan, zn)
        sm, used = guided_filter_backend(zn, conf, g, radius, eps, backend)
        ran.add(used)
    out = np.where(np.isfinite(zn), sm, np.nan) * (zr + 1e-9)
    return out.astype(np.float32), ran


# ---------------------------------------------------------------------------------------------- fusion


class EDFAccumulator:
    """Streaming Laplacian-pyramid fusion: add (frame, weight) pairs one at a time, normalised at the end.

    Weights should already be normalised across the stack per pixel; they are low-pass filtered by the
    Gaussian pyramid, which avoids seams between frames. `shape` is (H, W) for grey frames or (H, W, channels)
    for colour frames fused with a (H, W) weight; the pyramid depth depends on (H, W) only, so a colour fusion
    uses the same levels as the grey one.
    """

    def __init__(self, shape: tuple[int, ...], levels: int = 5):
        self.shape = shape
        self.levels = max(1, min(levels, int(np.log2(min(shape[:2]))) - 3))  # (H, W) or (H, W, channels)
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
        if frame.ndim == 3:  # colour frame fused with a luma weight: each pyramid level (H,W) -> (H,W,1)
            wp = [level[..., None] for level in wp]
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
    return softmax_weights_inplace(np.array(fv, np.float32, copy=True), temperature)


def softmax_weights_inplace(fv: np.ndarray, temperature: float = 0.25) -> np.ndarray:
    """Same as softmax_weights, but overwrites the float32 volume fv (no extra full-size volume)."""
    if fv.dtype != np.float32:
        raise TypeError("softmax_weights_inplace needs a float32 volume")
    np.maximum(fv, 1e-12, out=fv)
    np.log(fv, out=fv)
    peak = fv.max(axis=0)
    fv -= peak[None]
    fv /= temperature
    np.exp(fv, out=fv)
    fv /= fv.sum(axis=0, keepdims=True)
    return fv


# ---------------------------------------------------------------------------------------------- DOF ruler


def theoretical_fwhm_um(objective, um_per_px: float, measure_fn, texture: np.ndarray) -> float | None:
    """A THEORETICAL ESTIMATE of the FWHM (um of defocus) of the focus measure's response curve for this
    objective and this specimen. It is a model prediction, not a calibration: nothing is measured against a
    reference, and it carries the model's own error (the DOF ruler built on it was off by -34 % to +13 % on
    simulated sweeps).

    Method: the (already in-focus-ish) all-in-focus texture is blurred with the objective's defocus model at
    defocus steps of dof/3 over +-4 dof, `measure_fn` is applied to every blurred copy, and each half-maximum
    crossing is located between its two bracketing samples by bisection on the same model response (to 1/256 of a
    step), so a response narrower than two steps (100x / 1.25 oil) still has a width. Hand-checkable limit: for a
    pure sinusoid of spatial
    frequency f (cycles/px) and a measure that is its peak-to-peak amplitude, the response is the Gaussian
    exp(-2 pi^2 f^2 s^2) with s = |dz| tan(theta) / (2 um_per_px), so FWHM = 2 sqrt(2 ln 2) um_per_px /
    (pi f tan(theta)) (tests/test_feat_zstack_backend.py checks this to 1 % for two objectives).

    Heuristics (not derived from the optics, and each moves the answer): the defocused PSF is a Gaussian
    (Airy sigma = 0.21 lambda / NA; geometric blur disc of radius |dz| tan(theta) approximated by a Gaussian of
    sigma = R / 2, added in quadrature); the texture is taken as the true in-focus image (its own PSF blur is
    only subtracted in quadrature); the centre 256 x 256 px is used and 20 px are cut from each border of the
    response; the 75th percentile of the response map stands for the field; the sampling step is dof/3 and the
    range +-4 dof; NA below 0.24 is refused (flat response). Returns None when the response carries no focus
    information (objective not usable, zero or non-finite energy, or a half-maximum not bracketed inside the
    search range): a blank crop must not yield the search width.
    """
    if objective is None or not np.isfinite(objective.na) or objective.na <= 0:
        return None
    dof = objective.dof_um
    if not np.isfinite(dof) or dof <= 0:
        return None
    # Below 0.24 NA the search range is huge and the response is flat: no meaningful ruler.
    if objective.na < 0.24:
        return None
    step = dof / 3.0
    search = np.arange(-4 * dof, 4 * dof + step, step)
    if len(search) < 3:
        return None
    h, w = texture.shape
    c = min(h, w, 256) // 2
    tex = texture[h // 2 - c:h // 2 + c, w // 2 - c:w // 2 + c].astype(np.float32)
    airy = objective.airy_sigma_um

    def response(d: float) -> float:
        # extra blur on top of the in-focus PSF already present in the texture
        s = float(np.sqrt(max(objective.defocus_sigma_um(d) ** 2 - airy * airy, 0.0))) / um_per_px
        img = cv2.GaussianBlur(tex, (0, 0), s) if s > 0.05 else tex
        return float(np.percentile(measure_fn(img)[20:-20, 20:-20], 75))

    resp = np.array([response(d) for d in search], np.float64)
    if not np.all(np.isfinite(resp)) or resp.max() <= 1e-9:
        return None
    half = float(resp.max()) / 2.0
    above = np.nonzero(resp >= half)[0]
    # one sample above half is enough (a high-NA response can be narrower than two steps); the half maximum must
    # still be crossed inside the search range on both sides
    if above.size == 0 or above[0] == 0 or above[-1] == len(resp) - 1:  # not bracketed: no defined width
        return None

    def cross(i: int) -> float:
        """Crossing of `half` between samples i and i+1 in defocus units: the bracket is narrowed by bisection on
        the model response to 1/256 of a step, then interpolated linearly."""
        a, b, ra, rb = float(search[i]), float(search[i + 1]), resp[i], resp[i + 1]
        for _ in range(8):
            m = 0.5 * (a + b)
            rm = response(m)
            if (rm >= half) == (ra >= half):
                a, ra = m, rm
            else:
                b, rb = m, rm
        den = rb - ra
        t = (half - ra) / den if abs(den) > 1e-12 * half else 0.5
        return a + t * (b - a)

    # above[0] is the first sample >= half: the upward crossing lies between above[0]-1 and above[0];
    # above[-1] is the last sample >= half: the downward crossing lies between above[-1] and above[-1]+1.
    fwhm = abs(cross(above[-1]) - cross(above[0] - 1))
    return float(fwhm) if np.isfinite(fwhm) and fwhm > 0 else None
