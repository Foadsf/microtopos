"""ISO 25178-style areal surface texture parameters (not a certified implementation), texture and grains.

Pipeline: F-operator (polynomial form removal) -> S-filter (Gaussian low-pass, nesting index lambda_s)
-> optional L-filter (Gaussian high-pass, nesting index lambda_c) -> parameters on valid pixels only.
Gaussian weighting per ISO 16610-61: s(x) = exp(-pi (x / (alpha lambda))^2), alpha = sqrt(ln2 / pi),
equivalent to sigma = alpha * lambda / sqrt(2 pi) = 0.1874 lambda (50% transmission at lambda).
"""

from __future__ import annotations

import math

import cv2
import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.distance import pdist
from skimage.feature import graycomatrix, graycoprops, peak_local_max
from skimage.measure import regionprops
from skimage.segmentation import watershed

ALPHA = math.sqrt(math.log(2) / math.pi)
SIGMA_PER_LAMBDA = ALPHA / math.sqrt(2 * math.pi)  # 0.18739
MIN_VALID_FRACTION = 0.30


def remove_form(z: np.ndarray, order: int = 1) -> np.ndarray:
    valid = np.isfinite(z)
    if valid.sum() < 16:
        return z.copy()
    h, w = z.shape
    yy, xx = np.mgrid[-1:1:complex(0, h), -1:1:complex(0, w)]
    terms = [np.ones_like(xx), xx, yy]
    if order >= 2:
        terms += [xx * xx, xx * yy, yy * yy]
    if order >= 3:
        terms += [xx**3, xx * xx * yy, xx * yy * yy, yy**3]
    A = np.column_stack([t[valid] for t in terms])
    coef, *_ = np.linalg.lstsq(A, z[valid].astype(np.float64), rcond=None)
    form = sum(c * t for c, t in zip(coef, terms))
    return np.where(valid, z - form, np.nan).astype(np.float32)


def gaussian_filter_nan(z: np.ndarray, lam_um: float, um_per_px: float) -> np.ndarray:
    """ISO 16610-61 Gaussian low-pass with cutoff lam_um, NaN-aware (normalised convolution)."""
    sigma = SIGMA_PER_LAMBDA * lam_um / um_per_px
    valid = np.isfinite(z)
    if sigma < 0.3:
        return z.copy()
    v = valid.astype(np.float32)
    num = cv2.GaussianBlur(np.where(valid, z, 0).astype(np.float32), (0, 0), sigma, borderType=cv2.BORDER_REFLECT)
    den = cv2.GaussianBlur(v, (0, 0), sigma, borderType=cv2.BORDER_REFLECT)
    return np.where(valid & (den > 1e-3), num / np.maximum(den, 1e-6), np.nan).astype(np.float32)


def sl_surface(z: np.ndarray, um_per_px: float, form_order: int, lam_s: float, lam_c: float) -> np.ndarray:
    s = remove_form(z, form_order)
    if lam_s > 0:
        s = gaussian_filter_nan(s, lam_s, um_per_px)
    if lam_c > 0:
        s = s - gaussian_filter_nan(s, lam_c, um_per_px)
    return s


def height_parameters(s: np.ndarray, um_per_px: float) -> dict:
    v = s[np.isfinite(s)].astype(np.float64)
    if v.size < 16:
        return {}
    v = v - v.mean()
    sq = float(np.sqrt(np.mean(v * v)))
    out = {"Sa": float(np.mean(np.abs(v))), "Sq": sq, "Sp": float(v.max()), "Sv": float(-v.min()),
           "Sz": float(v.max() - v.min()),
           "Ssk": float(np.mean(v**3) / sq**3) if sq > 0 else float("nan"),
           "Sku": float(np.mean(v**4) / sq**4) if sq > 0 else float("nan")}
    valid = np.isfinite(s)
    inner = cv2.erode(valid.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    f = np.where(valid, s, 0).astype(np.float32)
    gx = cv2.Sobel(f, cv2.CV_32F, 1, 0, ksize=3) / (8 * um_per_px)
    gy = cv2.Sobel(f, cv2.CV_32F, 0, 1, ksize=3) / (8 * um_per_px)
    if inner.any():
        out["Sdr_pct"] = float(np.mean(np.sqrt(1 + gx[inner] ** 2 + gy[inner] ** 2) - 1) * 100)
    return out


def radial_psd(s: np.ndarray, um_per_px: float, f_lo: float, f_hi: float, n_bins: int = 24) -> dict:
    """Isotropic PSD in physical frequency (1/um) and its log-log slope over [f_lo, f_hi]."""
    valid = np.isfinite(s)
    z = np.where(valid, s - np.nanmean(s), 0).astype(np.float64)
    h, w = z.shape
    win = np.outer(np.hanning(h), np.hanning(w))
    P = np.abs(np.fft.fft2(z * win)) ** 2 * um_per_px**2 / (np.sum(win**2))
    fy = np.fft.fftfreq(h, d=um_per_px)
    fx = np.fft.fftfreq(w, d=um_per_px)
    fr = np.hypot(*np.meshgrid(fx, fy))
    edges = np.geomspace(f_lo, f_hi, n_bins + 1)
    idx = np.digitize(fr.ravel(), edges) - 1
    ok = (idx >= 0) & (idx < n_bins)
    sums = np.bincount(idx[ok], P.ravel()[ok], minlength=n_bins)
    cnt = np.bincount(idx[ok], minlength=n_bins)
    good = cnt > 0
    fc = np.sqrt(edges[:-1] * edges[1:])[good]
    pm = sums[good] / cnt[good]
    res = {"psd_f": fc, "psd_p": pm, "psd_band": (f_lo, f_hi)}
    if good.sum() >= 5 and np.all(pm > 0):
        slope = np.polyfit(np.log10(fc), np.log10(pm), 1)[0]
        beta = -float(slope)
        res["PSD_beta"] = beta
        # FD = (8 - beta) / 2 for an isotropic self-affine surface, meaningful only for 2 < beta < 4
        res["fractal_dimension"] = (8 - beta) / 2 if 2 < beta < 4 else None
    return res


def glcm_features(img: np.ndarray, levels: int = 64) -> dict:
    q = np.clip(img / 256.0 * levels, 0, levels - 1).astype(np.uint8)
    g = graycomatrix(q, [1, 2, 4], [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4], levels=levels, symmetric=True, normed=True)
    p = g[g > 0]
    return {"GLCM_contrast": float(graycoprops(g, "contrast").mean()),
            "GLCM_homogeneity": float(graycoprops(g, "homogeneity").mean()),
            "GLCM_correlation": float(graycoprops(g, "correlation").mean()),
            "GLCM_energy": float(graycoprops(g, "energy").mean()),
            "GLCM_entropy_bits": float(-np.sum(p * np.log2(p)) / g.shape[2] / g.shape[3])}


def grains(img: np.ndarray, um_per_px: float, min_distance_px: int = 8) -> dict:
    """Watershed on the gradient of the all-in-focus image; border-touching regions are excluded."""
    sm = cv2.GaussianBlur(img.astype(np.float32), (0, 0), 1.0)
    grad = cv2.magnitude(cv2.Sobel(sm, cv2.CV_32F, 1, 0), cv2.Sobel(sm, cv2.CV_32F, 0, 1))
    seeds = peak_local_max(-cv2.GaussianBlur(grad, (0, 0), 2.0), min_distance=min_distance_px, threshold_rel=0.0,
                           exclude_border=False)
    markers = np.zeros(grad.shape, np.int32)
    markers[tuple(seeds.T)] = np.arange(1, len(seeds) + 1)
    labels = watershed(grad, markers)
    h, w = labels.shape
    areas, circ, feret = [], [], []
    for rp in regionprops(labels):
        r0, c0, r1, c1 = rp.bbox
        if r0 == 0 or c0 == 0 or r1 == h or c1 == w or rp.area < 20:
            continue
        areas.append(rp.area * um_per_px**2)
        circ.append(min(1.0, 4 * math.pi * rp.area / max(rp.perimeter, 1e-6) ** 2))
        pts = rp.coords
        try:
            hull = pts[ConvexHull(pts).vertices]
            feret.append(float(pdist(hull).max()) * um_per_px)
        except Exception:  # degenerate (collinear) region
            feret.append(float(rp.feret_diameter_max) * um_per_px)
    out = {"grain_count": len(areas), "watershed_labels": labels}
    if areas:
        out.update(mean_grain_area_um2=float(np.mean(areas)), median_grain_area_um2=float(np.median(areas)),
                   mean_circularity=float(np.mean(circ)), mean_max_feret_um=float(np.mean(feret)),
                   grain_areas_um2=np.array(areas))
    return out


def analyze(height: np.ndarray, edf: np.ndarray, um_per_px: float, z_unit: str, lateral_cutoff_um: float,
            form_order: int = 1, lam_s_um: float | None = None, lam_c_um: float = 0.0) -> dict:
    """All surface metrics. Height parameters are withheld when < 30% of pixels have a reliable depth."""
    valid_frac = float(np.mean(np.isfinite(height)))
    lam_s = lateral_cutoff_um if lam_s_um is None else lam_s_um
    res = {"valid_fraction": valid_frac, "um_per_px": um_per_px, "z_unit": z_unit, "lambda_s_um": lam_s,
           "lambda_c_um": lam_c_um, "form_order": form_order, "warnings": []}
    s = sl_surface(height, um_per_px, form_order, lam_s, lam_c_um)
    res["sl_surface"] = s
    if valid_frac < MIN_VALID_FRACTION:
        res["warnings"].append(f"Only {valid_frac:.0%} of pixels have a reliable depth (< {MIN_VALID_FRACTION:.0%}); "
                               "height parameters withheld. Needs more texture, light or a wider focus sweep.")
    else:
        res.update(height_parameters(s, um_per_px))
        h, w = s.shape
        f_lo = 2.0 / (min(h, w) * um_per_px)
        f_hi = min(0.8 / max(lateral_cutoff_um, 2 * um_per_px), 0.4 / um_per_px)
        if f_hi > 2 * f_lo:
            res.update(radial_psd(s, um_per_px, f_lo, f_hi))
    res.update(glcm_features(edf))
    res.update(grains(edf, um_per_px))
    return res


def repeatability(z1: np.ndarray, z2: np.ndarray) -> float | None:
    """sigma_inst = std(z1 - z2) / sqrt(2) over pixels valid in both (two independent sweeps, same spot)."""
    both = np.isfinite(z1) & np.isfinite(z2)
    if both.sum() < 100:
        return None
    d = (z1 - z2)[both]
    return float(np.std(d - d.mean()) / math.sqrt(2))
