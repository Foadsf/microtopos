"""ISO 25178-style areal surface texture parameters, ISO 21920 1D profile filtering,
interactive 3-point levelling / mask / step-height metrology tools, texture and 2D/3D grains.

Pipeline: F-operator (polynomial form removal) -> S-filter (Gaussian low-pass, nesting index lambda_s)
-> optional L-filter (Gaussian high-pass, nesting index lambda_c) -> parameters on valid pixels only.
Gaussian weighting per ISO 16610-61: s(x) = exp(-pi (x / (alpha lambda))^2), alpha = sqrt(ln2 / pi),
equivalent to sigma = alpha * lambda / sqrt(2 pi) = 0.1874 lambda (50% transmission at lambda).
"""

from __future__ import annotations

import math

import cv2
import numpy as np
import scipy.ndimage as ndi
from scipy.spatial import ConvexHull
from scipy.spatial.distance import pdist
from skimage.feature import graycomatrix, graycoprops
from skimage.measure import label, regionprops
from skimage.morphology import h_maxima, h_minima
from skimage.segmentation import watershed

from . import units

ALPHA = math.sqrt(math.log(2) / math.pi)
SIGMA_PER_LAMBDA = ALPHA / math.sqrt(2 * math.pi)  # 0.18739
MIN_VALID_FRACTION = 0.30
# A sampled Gaussian reproduces the ISO 16610-61 50 % transmission at the cutoff only when the cutoff spans
# enough pixels: measured 0.892 at 2 px, 0.560 at 3 px, 0.502 at 4 px, 0.500 at 8 px. Shorter cutoffs are refused.
MIN_CUTOFF_PX = 4.0
ACF_MIN_PAIRS = 200  # valid pixel pairs needed before an autocorrelation lag counts as measured
FORM_NAMES = {0: "mean only", 1: "plane", 2: "quadratic", 3: "cubic"}
PARAMETER_KEYS = (
    "Sa", "Sq", "Sp", "Sv", "Sz", "Ssk", "Sku",
    "Sdq", "Sdr_pct",
    "PSD_beta", "fractal_dimension", "psd_f", "psd_p", "psd_band",
    "Sk", "Spk", "Svk", "Smr1_pct", "Smr2_pct",
    "Vmp", "Vmc", "Vvc", "Vvv",
    "Sds_per_mm2", "Ssc_per_um", "Spd_per_mm2", "Spc_per_um", "S5p", "S5v", "S10z",
    "Sal_um", "Str", "Std_deg",
    "topo_grain_count", "topo_mean_grain_area_um2", "topo_watershed_labels", "features_done",
    "eligibility", "sl_surface", "valid_fraction", "filter_notes",
    "lambda_s_requested_um", "lambda_c_requested_um",
)
_METRO_WARN = ("reliable depth", "S-filter", "L-filter", "Sdr withheld")


def is_metrology_warning(text: str) -> bool:
    """True for warnings produced by analyze() (replaced when metrology is recomputed)."""
    return any(k in text for k in _METRO_WARN)


def remove_form(z: np.ndarray, order: int = 1) -> np.ndarray:
    """F-operator: least-squares polynomial removal. 0 = mean only, 1 = plane, 2 = quadratic, 3 = cubic."""
    if order not in FORM_NAMES:
        raise ValueError(f"form order must be one of {sorted(FORM_NAMES)}")
    valid = np.isfinite(z)
    if valid.sum() < 16:
        return z.copy()
    h, w = z.shape
    yy, xx = np.mgrid[-1:1:complex(0, h), -1:1:complex(0, w)]
    terms = [np.ones_like(xx)]
    if order >= 1:
        terms += [xx, yy]
    if order >= 2:
        terms += [xx * xx, xx * yy, yy * yy]
    if order >= 3:
        terms += [xx**3, xx * xx * yy, xx * yy * yy, yy**3]
    A = np.column_stack([t[valid] for t in terms])
    coef, *_ = np.linalg.lstsq(A, z[valid].astype(np.float64), rcond=None)
    form = sum(c * t for c, t in zip(coef, terms))
    return np.where(valid, z - form, np.nan).astype(np.float32)


def level_three_points(z: np.ndarray, pts_xy_px: list[tuple[float, float]], radius_px: int = 3) -> np.ndarray:
    """Interactive 3-point levelling: subtract the plane passing through 3 user-selected points (or local patches)."""
    if len(pts_xy_px) != 3:
        raise ValueError("3-point levelling requires exactly 3 points")
    h, w = z.shape
    samples = []
    for x, y in pts_xy_px:
        ix, iy = round(x), round(y)
        x0, x1 = max(0, ix - radius_px), min(w, ix + radius_px + 1)
        y0, y1 = max(0, iy - radius_px), min(h, iy + radius_px + 1)
        patch = z[y0:y1, x0:x1]
        fin = patch[np.isfinite(patch)]
        if not fin.size:
            raise ValueError(f"point ({x:.1f}, {y:.1f}) has no valid height")
        samples.append((float(x), float(y), float(np.median(fin))))
    A = np.array([[1.0, p[0], p[1]] for p in samples], np.float64)
    b = np.array([p[2] for p in samples], np.float64)
    if abs(np.linalg.det(A)) < 1e-9:
        raise ValueError("the 3 points are collinear")
    c0, cx, cy = np.linalg.solve(A, b)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    plane = c0 + cx * xx + cy * yy
    return np.where(np.isfinite(z), z - plane, np.nan).astype(np.float32)


def apply_exclusion_mask(z: np.ndarray, rect_px: tuple[int, int, int, int], invert: bool = False) -> np.ndarray:
    """Mask out a rectangular region `(x0, y0, x1, y1)` in pixel coordinates (or keep only that region if `invert`)."""
    h, w = z.shape
    x0, y0, x1, y1 = rect_px
    x0, x1 = max(0, min(x0, x1)), min(w, max(x0, x1))
    y0, y1 = max(0, min(y0, y1)), min(h, max(y0, y1))
    mask = np.zeros((h, w), bool)
    mask[y0:y1, x0:x1] = True
    out = z.copy()
    out[~mask if invert else mask] = np.nan
    return out


def step_height_regions(z: np.ndarray, rect_a_px: tuple[int, int, int, int],
                        rect_b_px: tuple[int, int, int, int]) -> dict:
    """ISO 5436-style two-region step height measurement with common-plane tilt removal."""
    h, w = z.shape

    def mask_of(r):
        x0, y0, x1, y1 = r
        m = np.zeros((h, w), bool)
        m[max(0, min(y0, y1)):min(h, max(y0, y1)), max(0, min(x0, x1)):min(w, max(x0, x1))] = True
        return m & np.isfinite(z)

    ma, mb = mask_of(rect_a_px), mask_of(rect_b_px)
    if ma.sum() < 9 or mb.sum() < 9:
        raise ValueError("each step-height region needs at least 9 valid pixels")
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    both = ma | mb
    # Fit z = c_a * I_a + c_b * I_b + gx * x + gy * y so a shared tilt across both terraces is removed
    X = np.column_stack([ma[both].astype(float), mb[both].astype(float), xx[both], yy[both]])
    coef, *_ = np.linalg.lstsq(X, z[both].astype(np.float64), rcond=None)
    ca, cb, gx, gy = coef
    step = float(cb - ca)
    ra = z[ma] - (ca + gx * xx[ma] + gy * yy[ma])
    rb = z[mb] - (cb + gx * xx[mb] + gy * yy[mb])
    se = float(math.sqrt(np.var(ra) / ma.sum() + np.var(rb) / mb.sum()))
    return {
        "step_height": step,
        "abs_step_height": abs(step),
        "std_a": float(np.std(ra)),
        "std_b": float(np.std(rb)),
        "standard_error": se,
        "pixels_a": int(ma.sum()),
        "pixels_b": int(mb.sum()),
    }


def cutoff_resolvable(lam_um: float, um_per_px: float) -> bool:
    """Whether a Gaussian filter with this cutoff is qualified at this sampling (see MIN_CUTOFF_PX)."""
    return lam_um / um_per_px >= MIN_CUTOFF_PX


def gaussian_filter_nan(z: np.ndarray, lam_um: float, um_per_px: float) -> np.ndarray:
    """ISO 16610-61 Gaussian low-pass with cutoff lam_um, NaN-aware (normalised convolution).

    Raises ValueError for a cutoff shorter than MIN_CUTOFF_PX pixels: the sampled kernel would not have the
    standard's transmission there, so the caller must refuse or report it rather than filter silently.
    """
    if not cutoff_resolvable(lam_um, um_per_px):
        raise ValueError(f"cutoff {lam_um:g} µm is {lam_um / um_per_px:.1f} px, below {MIN_CUTOFF_PX:g} px")
    sigma = SIGMA_PER_LAMBDA * lam_um / um_per_px
    valid = np.isfinite(z)
    v = valid.astype(np.float32)
    num = cv2.GaussianBlur(np.where(valid, z, 0).astype(np.float32), (0, 0), sigma, borderType=cv2.BORDER_REFLECT)
    den = cv2.GaussianBlur(v, (0, 0), sigma, borderType=cv2.BORDER_REFLECT)
    return np.where(valid & (den > 1e-3), num / np.maximum(den, 1e-6), np.nan).astype(np.float32)


def gaussian_filter_1d_nan(prof: np.ndarray, lam_um: float, step_um: float) -> np.ndarray:
    """ISO 16610-21 / ISO 21920 1D Gaussian low-pass filter on a line profile with NaN handling.

    A cutoff below MIN_CUTOFF_PX samples is not resolvable: the unfiltered profile is returned so the
    caller can still report metrics rather than crashing.
    """
    if lam_um <= 0 or lam_um / max(step_um, 1e-9) < MIN_CUTOFF_PX:
        return prof.copy()
    sigma = SIGMA_PER_LAMBDA * lam_um / step_um
    valid = np.isfinite(prof)
    if not valid.any():
        return prof.copy()
    v = valid.astype(np.float32).reshape(1, -1)
    p0 = np.where(valid, prof, 0.0).astype(np.float32).reshape(1, -1)
    ksize = math.ceil(sigma * 6.0) | 1
    num = cv2.GaussianBlur(p0, (ksize, 1), sigmaX=sigma, sigmaY=0, borderType=cv2.BORDER_REFLECT).ravel()
    den = cv2.GaussianBlur(v, (ksize, 1), sigmaX=sigma, sigmaY=0, borderType=cv2.BORDER_REFLECT).ravel()
    return np.where(valid & (den > 1e-3), num / np.maximum(den, 1e-6), np.nan).astype(np.float32)


PROFILE_EVAL_CUTOFFS = 5.0  # ISO 21920-2 evaluation length for roughness: 5 sampling lengths (= cut-offs lambda_c)


def profile_cutoff_problem(n_samples: int, step_um: float, lam_c_um: float) -> str | None:
    """Why a profile of `n_samples` points at `step_um` cannot give roughness at cut-off `lam_c_um`, or None.

    The cut-off must span at least MIN_CUTOFF_PX samples (below that the sampled Gaussian is not the ISO 16610-21
    filter) and the profile must hold PROFILE_EVAL_CUTOFFS cut-offs (the ISO 21920-2 evaluation length).
    """
    if lam_c_um / step_um < MIN_CUTOFF_PX:
        return (f"λc = {lam_c_um:.3g} µm is {lam_c_um / step_um:.1f} px; the Gaussian filter needs at least "
                f"{MIN_CUTOFF_PX:g} px per cut-off")
    n_cut = (n_samples - 1) * step_um / lam_c_um
    if n_cut < PROFILE_EVAL_CUTOFFS:
        return (f"the profile is {n_cut:.1f} cut-offs long at λc = {lam_c_um:.3g} µm; ISO 21920-2 evaluates "
                f"{PROFILE_EVAL_CUTOFFS:g}")
    return None


def profile_parameters(prof: np.ndarray, step_um: float, lam_s_um: float = 0.0, lam_c_um: float = 0.0) -> dict:
    """ISO 21920 1D profile filtering (F-operator line levelling -> S-filter -> L-filter) and Ra/Rq/Rz/Pt metrics.

    Returns the primary profile (line levelled), waviness (L-filtered) and roughness profiles together with
    Pa/Pq/Pt on the primary profile and Ra/Rq/Rp/Rv/Rz/Rsk/Rku on the roughness profile; Wa/Wq are added when
    the waviness profile has enough samples.
    """
    valid = np.isfinite(prof)
    if valid.sum() < 5:
        return {}
    x = np.arange(len(prof), dtype=np.float64) * step_um
    # Least-squares line removal (F-operator)
    slope, intercept = np.polyfit(x[valid], prof[valid].astype(np.float64), 1)
    primary = np.where(valid, prof - (slope * x + intercept), np.nan).astype(np.float32)
    lam_s_done = lam_c_done = 0.0
    if lam_s_um > 0 and lam_s_um / max(step_um, 1e-9) >= MIN_CUTOFF_PX:
        primary = gaussian_filter_1d_nan(primary, lam_s_um, step_um)
        lam_s_done = float(lam_s_um)
    if lam_c_um > 0 and lam_c_um / max(step_um, 1e-9) >= MIN_CUTOFF_PX:
        waviness = gaussian_filter_1d_nan(primary, lam_c_um, step_um)
        roughness = np.where(np.isfinite(primary) & np.isfinite(waviness), primary - waviness, np.nan)
        lam_c_done = float(lam_c_um)
    else:
        waviness = np.full_like(primary, np.nan)
        roughness = primary
    pv = primary[np.isfinite(primary)].astype(np.float64)
    rv = roughness[np.isfinite(roughness)].astype(np.float64)
    if rv.size < 4:
        return {}
    rv = rv - rv.mean()
    pv = pv - pv.mean()
    rq = float(np.sqrt(np.mean(rv * rv)))
    # Rz over up to 5 sampling sections
    n_sec = min(5, max(1, rv.size // 8))
    chunks = np.array_split(rv, n_sec)
    rz = float(np.mean([c.max() - c.min() for c in chunks if c.size]))
    out = {
        "lambda_s_applied_um": lam_s_done,  # 0.0 = that filter was not applied (off, or below MIN_CUTOFF_PX)
        "lambda_c_applied_um": lam_c_done,
        "primary_profile": primary,
        "waviness_profile": waviness,
        "roughness_profile": roughness,
        "Pa": float(np.mean(np.abs(pv))),
        "Pq": float(np.sqrt(np.mean(pv * pv))),
        "Pt": float(pv.max() - pv.min()),
        "Ra": float(np.mean(np.abs(rv))),
        "Rq": rq,
        "Rp": float(rv.max()),
        "Rv": float(-rv.min()),
        "Rz": rz,
        "Rsk": float(np.mean(rv**3) / rq**3) if rq > 0 else 0.0,
        "Rku": float(np.mean(rv**4) / rq**4) if rq > 0 else 0.0,
    }
    wv = waviness[np.isfinite(waviness)].astype(np.float64)
    if wv.size >= 4:
        wv = wv - wv.mean()
        out["Wa"] = float(np.mean(np.abs(wv)))
        out["Wq"] = float(np.sqrt(np.mean(wv * wv)))
    return out


def sl_surface(z: np.ndarray, um_per_px: float, form_order: int, lam_s: float, lam_c: float,
               notes: list | None = None) -> np.ndarray:
    """Form removal, then S- and L-filters. A filter whose cutoff is not resolvable is skipped and noted."""
    s = remove_form(z, form_order)
    if lam_s > 0:
        if cutoff_resolvable(lam_s, um_per_px):
            s = gaussian_filter_nan(s, lam_s, um_per_px)
        elif notes is not None:
            notes.append(f"S-filter not applied: λs {lam_s:.3g} µm is only {lam_s / um_per_px:.1f} px at this "
                         f"sampling (needs ≥ {MIN_CUTOFF_PX:g} px for the ISO 16610-61 response). Values include "
                         "wavelengths down to the pixel size.")
    if lam_c > 0:
        if cutoff_resolvable(lam_c, um_per_px):
            s = s - gaussian_filter_nan(s, lam_c, um_per_px)
        elif notes is not None:
            notes.append(f"L-filter not applied: λc {lam_c:.3g} µm is below {MIN_CUTOFF_PX:g} px.")
    return s


def root_mean_square_gradient(s: np.ndarray, d: float) -> float | None:
    """Sdq (dimensionless root-mean-square surface slope) per ISO 25178-2."""
    gx = (s[:, 1:] - s[:, :-1])[:-1, :] / d
    gy = (s[1:, :] - s[:-1, :])[:, :-1] / d
    ok = np.isfinite(gx) & np.isfinite(gy)
    if ok.sum() < 16:
        return None
    return float(np.sqrt(np.mean(gx[ok] ** 2 + gy[ok] ** 2)))


def developed_area_ratio(s: np.ndarray, d: float) -> float | None:
    """Sdr (%) per ISO 25178-2: each 2x2 cell split into two triangles; mean of (true area / projected) - 1.

    Requires heights and lateral spacing in the same unit (um).
    """
    z00, z10, z01, z11 = s[:-1, :-1], s[:-1, 1:], s[1:, :-1], s[1:, 1:]
    ok = np.isfinite(z00) & np.isfinite(z10) & np.isfinite(z01) & np.isfinite(z11)
    if ok.sum() < 16:
        return None
    a = z10[ok] - z00[ok]
    b = z01[ok] - z00[ok]
    c = z11[ok] - z10[ok]
    e = z11[ok] - z01[ok]
    d2 = d * d
    area = 0.5 * np.sqrt(d2 * (a * a + b * b) + d2 * d2) + 0.5 * np.sqrt(d2 * (c * c + e * e) + d2 * d2)
    return float((np.mean(area) / d2 - 1.0) * 100.0)


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
    sdq = root_mean_square_gradient(s, um_per_px)
    if sdq is not None:
        out["Sdq"] = sdq
    sdr = developed_area_ratio(s, um_per_px)
    if sdr is not None:
        out["Sdr_pct"] = sdr
    return out


def _material_void_volume(z_desc: np.ndarray, mr: float) -> tuple[float, float]:
    """(Vm, Vv) at material ratio `mr` (fraction 0..1) per unit area, from heights sorted high to low.

    Vm(mr) is the material volume above the height c(mr) that mr of the area exceeds; Vv(mr) is the void
    volume below that height (ISO 25178-2, 4.4.x), both in height units (volume per unit area).
    """
    n = z_desc.size
    k = min(n - 1, max(0, round(mr * n)))
    c = z_desc[k]
    return float((z_desc[:k] - c).sum() / n), float((c - z_desc[k:]).sum() / n)


def functional_parameters(s: np.ndarray) -> dict:
    """Material-ratio parameters Sk, Spk, Svk, Smr1, Smr2 and volume parameters Vmp, Vmc, Vvc, Vvv (ISO 25178-2).

    The equivalent straight line is the 40 % secant of the material-ratio curve with the smallest height
    difference; Spk and Svk are the heights of triangles with the same area as the peak and valley regions.
    Volumes follow ISO 25178-2 with p = 10 % and q = 80 %, from Vm(mr) (material volume above the height at
    material ratio mr) and Vv(mr) (void volume below it): Vmp = Vm(p), Vmc = Vm(q) - Vm(p),
    Vvc = Vv(p) - Vv(q), Vvv = Vv(q). They are volumes per unit area, i.e. in the height unit of `s`
    (µm³/µm² = µm = mL/m² when `s` is in µm), and independent of the pixel size.
    """
    v = s[np.isfinite(s)].astype(np.float64)
    if v.size < 200:
        return {}
    if v.size > 200_000:
        v = np.random.default_rng(0).choice(v, 200_000, replace=False)
    z = np.sort(v)[::-1]  # heights from highest to lowest
    n = z.size
    mr = (np.arange(n) + 0.5) / n * 100.0  # material ratio (%) at each sorted height
    w = round(0.4 * n)
    i = int(np.argmin(z[: n - w] - z[w:]))
    m1, z1, z2 = mr[i], z[i], z[i + w]
    slope = (z2 - z1) / (mr[i + w] - m1)
    z_at0 = z1 - slope * m1
    z_at100 = z1 + slope * (100.0 - m1)
    asc = z[::-1]
    i1 = n - int(np.searchsorted(asc, z_at0, side="right"))  # samples strictly above the upper limit
    i2 = n - int(np.searchsorted(asc, z_at100, side="left"))  # samples at or above the lower limit
    smr1 = float(i1 / n * 100.0)
    smr2 = float(i2 / n * 100.0)
    a1 = (z[z > z_at0] - z_at0).sum() / n * 100.0  # peak-region area (height x %)
    a2 = (z_at100 - z[z < z_at100]).sum() / n * 100.0
    spk = 2 * a1 / smr1 if smr1 > 0 else 0.0
    svk = 2 * a2 / (100.0 - smr2) if smr2 < 100 else 0.0
    # ISO 25178-2 volume parameters from the two primitives Vm(mr) and Vv(mr), p = 10 %, q = 80 %.
    vm_p, vv_p = _material_void_volume(z, 0.10)
    vm_q, vv_q = _material_void_volume(z, 0.80)
    vmp, vmc, vvc, vvv = vm_p, vm_q - vm_p, vv_p - vv_q, vv_q
    return {
        "Sk": float(z_at0 - z_at100), "Spk": float(spk), "Svk": float(svk),
        "Smr1_pct": smr1, "Smr2_pct": smr2,
        "Vmp": float(vmp), "Vmc": float(vmc), "Vvc": float(vvc), "Vvv": float(vvv),
    }


WOLF_PRUNE = 0.05  # ISO 25178-2 default pruning: features with a dynamic below 5 % of Sz are merged away
_NEIGHBOURS8 = np.ones((3, 3), bool)


def _centred(s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    valid = np.isfinite(s)
    return valid, (s - np.nanmean(s)).astype(np.float32)


def _mean_curvature(z: np.ndarray, um_per_px: float) -> np.ndarray:
    """Arithmetic mean of the two principal curvatures, -(z_xx + z_yy) / 2, on the interior pixels.

    Positive at a peak. Units: height unit per µm² (1/µm when z is in µm). The second derivative is the
    three-point central difference, which is about 1.6 % low for a Gaussian peak of σ = 4 px.
    """
    c = z[1:-1, 1:-1]
    d2x = (z[1:-1, 2:] - 2 * c + z[1:-1, :-2]) / (um_per_px ** 2)
    d2y = (z[2:, 1:-1] - 2 * c + z[:-2, 1:-1]) / (um_per_px ** 2)
    return -0.5 * (d2x + d2y)


def _window_valid(valid: np.ndarray) -> np.ndarray:
    """Interior pixels whose whole 3 x 3 neighbourhood is valid."""
    return cv2.erode(valid.astype(np.uint8), np.ones((3, 3), np.uint8))[1:-1, 1:-1] > 0


def summit_parameters(s: np.ndarray, um_per_px: float) -> dict:
    """Sds and Ssc: density and mean curvature of ALL summits (cheap; no pruning).

    A summit is a pixel strictly higher than its 8 neighbours (EUR 15178N 'summit'; no minimum height or
    separation criterion is applied). Sds = number of summits per mm² of valid area; Ssc = arithmetic mean of
    the principal curvatures at the summits, -(z_xx + z_yy)/2, in 1/µm when z is in µm.
    Because nothing is pruned, Sds counts every noise maximum: on a noisy surface it is far larger than
    Spd (the density of Wolf-pruned significant peaks, see feature_parameters) and it depends on the pixel size
    and on the S-filter. Use Spd for a noise-robust density.
    """
    valid, z = _centred(s)
    if valid.sum() < 64:
        return {}
    c = z[1:-1, 1:-1]
    nbr_max = _neighbour_max(z)
    summits = _window_valid(valid) & (c > nbr_max)
    area_mm2 = float(valid.sum()) * (um_per_px * 1e-3) ** 2
    out = {"Sds_per_mm2": int(summits.sum()) / area_mm2}
    if summits.any():
        out["Ssc_per_um"] = float(np.mean(_mean_curvature(z, um_per_px)[summits]))
    return out


def _neighbour_max(z: np.ndarray) -> np.ndarray:
    """Interior maximum over the 8 neighbours of every pixel."""
    h, w = z.shape
    out = np.full((h - 2, w - 2), -np.inf, np.float32)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dx or dy:
                np.maximum(out, z[1 + dy:h - 1 + dy, 1 + dx:w - 1 + dx], out=out)
    return out


def _significant_peak_mask(z: np.ndarray, valid: np.ndarray) -> np.ndarray | None:
    """Boolean mask of the regional maxima whose dynamic (height above the saddle to any higher peak) is at least
    WOLF_PRUNE x Sz: the h-maxima, which is what Wolf pruning of the hill/dale segmentation leaves. None if flat."""
    fin = z[valid]
    sz = float(fin.max() - fin.min())
    if sz <= 1e-9:
        return None
    return h_maxima(np.where(valid, z, fin.min()), WOLF_PRUNE * sz)


def _peak_positions(mask: np.ndarray, z: np.ndarray, win_valid: np.ndarray) -> np.ndarray:
    """(row, col) of the highest pixel of every region of `mask`, dropping those on the border or next to
    invalid pixels (their curvature and height would not be measured)."""
    lab, n = ndi.label(mask, structure=_NEIGHBOURS8)
    if n == 0:
        return np.empty((0, 2), int)
    pos = np.array(ndi.maximum_position(z, lab, np.arange(1, n + 1)), dtype=int).reshape(-1, 2)
    h, w = z.shape
    inner = (pos[:, 0] >= 1) & (pos[:, 0] < h - 1) & (pos[:, 1] >= 1) & (pos[:, 1] < w - 1)
    pos = pos[inner]
    return pos[win_valid[pos[:, 0] - 1, pos[:, 1] - 1]]


def feature_parameters(s: np.ndarray, um_per_px: float, _peaks: np.ndarray | None = None) -> dict:
    """ISO 25178-2 feature parameters Spd, Spc, S5p, S5v, S10z plus the summit parameters Sds and Ssc.

    Heights are measured from the mean plane. 'Significant' peaks and pits are the regional maxima / minima
    whose dynamic is at least 5 % of Sz (Wolf pruning, the ISO 25178-2 default).
    Spd  density of significant peaks, per mm² of valid area (scales as 1 / pixel size²).
    Spc  arithmetic mean of the principal curvatures at the significant peaks, 1/µm when z is in µm.
    S5p  mean height of the five highest significant peaks above the mean plane (height unit).
    S5v  mean depth of the five deepest significant pits below the mean plane (height unit).
    S10z = S5p + S5v.
    Peaks and pits on the 1-px border or next to invalid pixels are not used. Sds and Ssc: see summit_parameters.
    Costs about 1.5 s per megapixel (two h-maxima reconstructions), so analyze() leaves it to feature_set().
    """
    out = summit_parameters(s, um_per_px)
    if not out:
        return {}
    valid, z = _centred(s)
    win_valid = _window_valid(valid)
    area_mm2 = float(valid.sum()) * (um_per_px * 1e-3) ** 2
    peak_mask = _peaks if _peaks is not None else _significant_peak_mask(z, valid)
    if peak_mask is None:
        return out
    fin = z[valid]
    zp = np.where(valid, z, fin.min())
    pos = _peak_positions(peak_mask, zp, win_valid)
    out["Spd_per_mm2"] = len(pos) / area_mm2
    if len(pos):
        curv = _mean_curvature(z, um_per_px)
        out["Spc_per_um"] = float(np.mean(curv[pos[:, 0] - 1, pos[:, 1] - 1]))
        out["S5p"] = float(np.mean(np.sort(z[pos[:, 0], pos[:, 1]])[-5:]))
    zv = np.where(valid, z, fin.max())
    pit_pos = _peak_positions(h_maxima(-zv, WOLF_PRUNE * float(fin.max() - fin.min())), -zv, win_valid)
    if len(pit_pos):
        out["S5v"] = float(np.mean(np.sort(-z[pit_pos[:, 0], pit_pos[:, 1]])[-5:]))
    if "S5p" in out and "S5v" in out:
        out["S10z"] = out["S5p"] + out["S5v"]
    return out


def spatial_parameters(s: np.ndarray, um_per_px: float, threshold: float = 0.2, notes: list | None = None) -> dict:
    """Sal, Str and Std."""
    notes = [] if notes is None else notes
    valid = np.isfinite(s)
    if valid.mean() < MIN_VALID_FRACTION:
        return {}
    z = np.where(valid, s - np.nanmean(s), 0.0)
    if float(np.sum(z * z)) <= 1e-18 * max(1, valid.sum()):
        notes.append("Sal, Str and Std withheld: the surface has no height variation.")
        return {}
    h, w = z.shape
    shape = (2 * h, 2 * w)
    acf = np.fft.fftshift(np.real(np.fft.ifft2(np.abs(np.fft.fft2(z, s=shape)) ** 2)))
    pairs = np.fft.fftshift(np.real(np.fft.ifft2(np.abs(np.fft.fft2(valid.astype(float), s=shape)) ** 2)))
    supported = pairs >= ACF_MIN_PAIRS
    acf = np.where(supported, acf / np.maximum(pairs, 1.0), np.nan)
    cy, cx = h, w
    acf = acf / acf[cy, cx]
    r = np.arange(1, min(h, w) // 2)
    lengths = []
    for th in np.deg2rad(np.arange(0, 180, 2)):
        ys = np.clip(np.round(cy + r * np.sin(th)).astype(int), 0, 2 * h - 1)
        xs = np.clip(np.round(cx + r * np.cos(th)).astype(int), 0, 2 * w - 1)
        a = acf[ys, xs]
        below = np.nonzero(a < threshold)[0]
        unknown = np.nonzero(~np.isfinite(a))[0]
        first_unknown = unknown[0] if unknown.size else len(a)
        if below.size and below[0] < first_unknown:
            lengths.append(float(r[below[0]]))
        elif unknown.size:
            lengths.append(np.nan)
        else:
            lengths.append(np.inf)
    lengths = np.array(lengths, float)
    out = {}
    n_unknown = int(np.sum(np.isnan(lengths)))
    if n_unknown:
        notes.append(f"Sal and Str withheld: the autocorrelation is unsupported in {n_unknown} of {len(lengths)} "
                     "directions (too few valid pixel pairs; clustered or patterned invalid pixels).")
    else:
        if np.isfinite(lengths.min()):
            out["Sal_um"] = float(lengths.min() * um_per_px)
        if np.all(np.isfinite(lengths)):
            out["Str"] = float(lengths.min() / lengths.max())
    P = np.fft.fftshift(np.abs(np.fft.fft2(z * np.outer(np.hanning(h), np.hanning(w)))) ** 2)
    fy, fx = np.mgrid[-(h // 2):h - h // 2, -(w // 2):w - w // 2]
    ang = (np.degrees(np.arctan2(fy / h, fx / w)) + 180.0) % 180.0
    rad = np.hypot(fy / h, fx / w)
    band = (rad > 2.0 / min(h, w)) & (rad < 0.25)
    hist = np.bincount(np.minimum((ang[band] // 2).astype(int), 89), P[band], minlength=90)
    if hist.max() > 0 and hist.max() >= 2.0 * float(np.median(hist)):
        out["Std_deg"] = float((np.argmax(hist) * 2 + 1 + 90) % 180)
    else:
        notes.append("Std withheld: the angular spectrum has no dominant direction.")
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


def grains(img: np.ndarray, um_per_px: float, h_rel: float = 0.15) -> dict:
    """Marker-controlled watershed on the gradient of the all-in-focus image.

    Markers are the h-minima of the smoothed gradient (regional minima deeper than h = h_rel x the 95th
    percentile gradient): no missing seeds, and no seed per noise dip. Border-touching regions are excluded
    from the statistics. The ASTM E112 grain size number uses Heyn's mean lineal intercept.
    """
    sm = cv2.GaussianBlur(img.astype(np.float32), (0, 0), 1.0)
    grad = cv2.magnitude(cv2.Sobel(sm, cv2.CV_32F, 1, 0), cv2.Sobel(sm, cv2.CV_32F, 0, 1))
    gs = cv2.GaussianBlur(grad, (0, 0), 2.0)
    hval = h_rel * float(np.percentile(gs, 95))
    markers = label(h_minima(gs, hval)) if hval > 0 else np.zeros(gs.shape, np.int32)
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
        except Exception:  # degenerate (collinear) region  # noqa: BLE001
            feret.append(float(rp.feret_diameter_max) * um_per_px)
    out = {"grain_count": len(areas), "watershed_labels": labels}
    # Heyn mean lineal intercept on every 8th row and column: boundary crossings per unit line length
    rows, cols = labels[::8, :], labels[:, ::8]
    crossings = int(np.count_nonzero(np.diff(rows, axis=1)) + np.count_nonzero(np.diff(cols, axis=0)))
    line_um = (rows.size + cols.size) * um_per_px
    if crossings >= 10:
        l_mm = line_um / crossings / 1000.0
        out["mean_intercept_um"] = l_mm * 1000.0
        out["ASTM_E112_G"] = float(-3.2877 - 6.6439 * math.log10(l_mm))
    if areas:
        out.update(mean_grain_area_um2=float(np.mean(areas)), median_grain_area_um2=float(np.median(areas)),
                   mean_circularity=float(np.mean(circ)), mean_max_feret_um=float(np.mean(feret)),
                   grain_areas_um2=np.array(areas))
    return out


def grains_topographic(s: np.ndarray, um_per_px: float, _peaks: np.ndarray | None = None) -> dict:
    """Topographic hill segmentation of the height map (`sl_surface`): ISO 25178-2 hills by watershed.

    Markers are the Wolf-pruned peaks (regional maxima with a dynamic of at least 5 % of Sz, the same peaks that
    Spd counts); every valid pixel is then assigned to a hill by a watershed of the inverted height. Hills
    touching the border, or smaller than 20 px, are excluded from the count. Returns `topo_grain_count`,
    `topo_mean_grain_area_um2` (mean hill area, µm²) and `topo_watershed_labels` (all catchments, int32).
    {} when fewer than 30 % of the pixels are valid; {"topo_grain_count": 0} for a flat surface.
    """
    valid = np.isfinite(s)
    if valid.mean() < MIN_VALID_FRACTION:
        return {}
    _, z = _centred(s)
    peak_mask = _peaks if _peaks is not None else _significant_peak_mask(z, valid)
    if peak_mask is None:
        return {"topo_grain_count": 0}
    markers, _n = ndi.label(peak_mask, structure=_NEIGHBOURS8)
    labels = watershed(-np.where(valid, z, z[valid].min()), markers, mask=valid)
    area_px = np.bincount(labels.ravel())
    border = np.unique(np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]]))
    keep = area_px >= 20
    keep[0] = False
    keep[border] = False
    out: dict = {"topo_grain_count": int(keep.sum()), "topo_watershed_labels": labels.astype(np.int32)}
    if keep.any():
        out["topo_mean_grain_area_um2"] = float(area_px[keep].mean() * um_per_px ** 2)
    return out


def feature_set(s: np.ndarray, um_per_px: float, z_unit: str) -> dict:
    """The expensive, on-demand surface features: Spd, Spc, S5p, S5v, S10z (feature_parameters) and the
    topographic hills (grains_topographic), sharing one peak search. Curvatures are withheld unless z is in µm."""
    valid, z = _centred(s)
    mask = _significant_peak_mask(z, valid) if valid.sum() >= 64 else None
    out = feature_parameters(s, um_per_px, _peaks=mask)
    if mask is not None:
        out.update(grains_topographic(s, um_per_px, _peaks=mask))
    else:
        out.update(grains_topographic(s, um_per_px))
    if not units.is_um(z_unit):
        out.pop("Ssc_per_um", None)
        out.pop("Spc_per_um", None)
    return out


def analyze(height: np.ndarray, edf: np.ndarray, um_per_px: float, z_unit: str, lateral_cutoff_um: float,
            form_order: int = 1, lam_s_um: float | None = None, lam_c_um: float = 0.0,
            features: bool = False) -> dict:
    """All surface metrics, with the eligibility of each group stated.

    `features=False` (default) computes only the cheap summit parameters Sds/Ssc; `features=True` also runs the
    Wolf-pruned feature set and the topographic hills (feature_set, ~3 s per megapixel).

    Height, functional, spatial and spectral parameters need >= 30 % valid depth pixels; Sdr also needs heights
    in µm. Texture (GLCM) and grains are computed from the all-in-focus IMAGE, not from heights, and are always
    reported as image-based.
    """
    z_unit = units.canon(z_unit)
    valid_frac = float(np.mean(np.isfinite(height)))
    notes: list[str] = []
    default_note = None
    if lam_s_um is None:  # default: the optical cutoff, when the sampling can represent it
        lam_s = lateral_cutoff_um
        if not cutoff_resolvable(lam_s, um_per_px):
            default_note = (f"No S-filter: the default λs (optical cutoff {lam_s:.3g} µm) is "
                            f"{lam_s / um_per_px:.1f} px, below {MIN_CUTOFF_PX:g} px; the pixel size limits the "
                            "bandwidth instead.")
            lam_s = 0.0
    else:
        lam_s = lam_s_um
    res = {"valid_fraction": valid_frac, "um_per_px": um_per_px, "z_unit": z_unit, "lambda_s_um": lam_s,
           "lambda_c_um": lam_c_um, "form_order": form_order, "form_name": FORM_NAMES.get(form_order, "?"),
           "lambda_s_default": lam_s_um is None, "warnings": []}
    s = sl_surface(height, um_per_px, form_order, lam_s, lam_c_um, notes)
    res["sl_surface"] = s
    # record what was APPLIED; a refused cut-off is kept separately so the report cannot claim it
    if lam_s > 0 and not cutoff_resolvable(lam_s, um_per_px):
        res["lambda_s_requested_um"], res["lambda_s_um"] = lam_s, 0.0
    if lam_c_um > 0 and not cutoff_resolvable(lam_c_um, um_per_px):
        res["lambda_c_requested_um"], res["lambda_c_um"] = lam_c_um, 0.0
    elig = {}
    if valid_frac < MIN_VALID_FRACTION:
        reason = (f"only {valid_frac:.0%} of pixels have a reliable depth (needs {MIN_VALID_FRACTION:.0%})")
        res["warnings"].append(f"Only {valid_frac:.0%} of pixels have a reliable depth (< {MIN_VALID_FRACTION:.0%}); "
                               "height parameters withheld. Needs more texture, light or a wider focus sweep.")
        for g in ("height", "functional", "spatial", "spectral", "Sdr", "feature"):
            elig[g] = reason
    else:
        hp = height_parameters(s, um_per_px)
        res.update(hp)
        elig["height"] = "ok" if hp else "too few valid pixels"
        fp = functional_parameters(s)
        res.update(fp)
        elig["functional"] = "ok" if fp else "too few valid pixels (needs 200)"
        feat = feature_set(s, um_per_px, z_unit) if features else summit_parameters(s, um_per_px)
        res.update(feat)
        if not feat:
            elig["feature"] = "too few valid pixels"
        elif features:
            elig["feature"] = "ok"
        else:
            elig["feature"] = ("only the summit density Sds and curvature Ssc are computed automatically; Spd, Spc, "
                               "S5p, S5v, S10z and the topographic hills are computed on demand (Height tab)")
        sp_notes: list[str] = []
        res.update(spatial_parameters(s, um_per_px, notes=sp_notes))
        notes += sp_notes
        elig["spatial"] = "ok" if not sp_notes else "; ".join(sp_notes)
        if z_unit != units.UM:  # Sdr and slope-derived quantities need heights and lateral spacing in the same unit
            res.pop("Sdr_pct", None)
            res.pop("Sdq", None)
            res.pop("Ssc_per_um", None)
            res.pop("Spc_per_um", None)
            elig["Sdr"] = "withheld: heights are not in µm"
        else:
            elig["Sdr"] = "ok" if "Sdr_pct" in res else "too few valid 2x2 cells"
        h, w = s.shape
        f_lo = 2.0 / (min(h, w) * um_per_px)
        f_hi = min(0.8 / max(lateral_cutoff_um, 2 * um_per_px), 0.4 / um_per_px)
        if f_hi > 2 * f_lo:
            res.update(radial_psd(s, um_per_px, f_lo, f_hi))
            elig["spectral"] = "ok"
        else:
            elig["spectral"] = "field too small for a spectral band"
    elig["texture"] = "image-based (all-in-focus image, not heights)"
    elig["grains"] = "image-based watershed of the all-in-focus image; topographic 3D regions also reported"
    res["eligibility"] = elig
    res["filter_notes"] = ([default_note] if default_note else []) + [n for n in notes if "filter" in n]
    res["warnings"] += [n for n in notes if "filter" in n]
    res.update(glcm_features(edf))
    res.update(grains(edf, um_per_px))
    return res


def repeatability(z1: np.ndarray, z2: np.ndarray, min_pixels: int = 100) -> dict | None:
    """Relative shape repeatability of two sweeps of the same spot, over pixels valid in both.

    sigma_inst = std(z1 - z2 - mean) / sqrt(2): the mean difference (datum bias between sweeps) is removed and
    reported separately. Returns None with fewer than min_pixels common pixels. The caller must make sure both
    maps share unit, scale, orientation, objective and position; this function cannot check that.
    """
    both = np.isfinite(z1) & np.isfinite(z2)
    if both.sum() < min_pixels:
        return None
    d = (z1 - z2)[both]
    return {"sigma_inst": float(np.std(d - d.mean()) / math.sqrt(2)), "mean_offset": float(d.mean()),
            "overlap_fraction": float(both.mean())}
