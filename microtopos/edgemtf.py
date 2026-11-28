"""Sharpness of the imaging chain measured from one straight edge (the slanted-edge method).

A straight edge tilted a few degrees from the pixel grid is seen by every row at a different sub-pixel phase.
Projecting all pixels onto the edge normal therefore samples the edge spread function (ESF) much more finely
than the pixel pitch. Steps, as implemented here:

1. Orientation: rows are analysed for a near-vertical edge, columns (by transposing) for a near-horizontal one.
2. Edge line: per row, the centroid of the row derivative inside a Hamming window around the current estimate;
   a straight line x = a + b*y is fitted to those centroids (least squares, 3-sigma outlier rejection) and the
   window is re-centred on it, four times.
3. ESF: every pixel of the rows used is placed at its signed distance d = (x - a - b*y) cos(theta) from the line,
   positive towards the bright side, and averaged in bins 1/4 pixel wide (4x oversampling). A bin's value is
   interpolated from the mean position and mean value of its samples to the bin centre.
4. LSF: forward difference of the ESF, multiplied by a Hamming window centred on the LSF centroid over the ESF
   half-span D (20 LSF RMS widths w when the ROI allows). The window lowers the noise floor of the MTF to 0.63x
   the unwindowed value; it also narrows the LSF slightly, which raises MTF50 by about 2.3 (w/D)^2 (0.6 % at the
   default span; a warning gives the figure when the ROI forces a shorter span).
5. MTF: |FFT| of the windowed LSF normalised at zero frequency, divided by sinc(f*bin)^2 (the transfer function of
   the bin average and of the forward difference, both a box one bin wide), in cycles per pixel along the edge
   normal, reported from 0 to 1 cycle/pixel (twice Nyquist). MTF50 is interpolated linearly.
6. Gaussian width: the (unwindowed) LSF is fitted by a Gaussian of sigma psf_sigma_px convolved with the
   projection of one square pixel onto the normal (a box of width cos(theta) convolved with one of width
   sin(theta)) and with the two one-bin boxes of steps 3-4. psf_sigma_px is therefore the optical blur with the
   pixel aperture removed, IF the optical PSF is Gaussian. lsf_sigma_px = sqrt(psf_sigma_px^2 + 1/12) is the RMS
   width of the camera-level LSF (optics and pixel aperture): a unit square pixel projected onto any direction has
   variance (cos^2 + sin^2)/12 = 1/12 px^2.

The MTF reported is the MTF along the edge normal. Its pixel-aperture part is sinc(f cos theta) sinc(f sin theta),
which equals the axis value sinc(f) only at theta = 0 (at 25 degrees and f = 0.5 it is 0.644 instead of 0.637).

Allowed edge tilt, from the nearest pixel axis:
* not more than MAX_TILT_DEG = 40 degrees. At 45 degrees every row has the same sub-pixel phase (no super-sampling),
  and close to it the choice between row and column analysis is ambiguous.
* at least enough tilt that the edge crosses MIN_SWEEP_PX = 2 pixel columns over the rows used, so that every
  1/4-pixel bin receives samples from several rows. An exactly vertical or horizontal edge never qualifies. The
  minimum angle therefore depends on the ROI length: atan(2 / rows), 1.1 degrees for 100 rows.
* in addition no ESF bin may be empty, which refuses slopes too close to a small-integer ratio such as 1:2 or 1:3
  for which only a few sub-pixel phases occur.

This is the slanted-edge method as described above; it is not an implementation of, nor tested for conformance
with, ISO 12233. Verified only against synthetic edges (see tests/test_edgemtf.py), never on a real optical edge.

Relation to superres.fit_edge_psf_sigma: that function fits an erf to a profile resampled by bilinear
interpolation across the single strongest gradient pixel, without super-sampling, so its sigma contains the pixel
aperture (+1/12 px^2) and the interpolation blur, and it falls back to the nominal sigma silently. It cannot serve
as a reportable measurement; this module's psf_sigma_px is the quantity it approximates, and superres can call
measure_edge(...)["psf_sigma_px"] in its place (falling back to the nominal sigma on ValueError).
"""

from __future__ import annotations

import math

import numpy as np
from scipy.optimize import least_squares
from scipy.special import ndtr

OVERSAMPLE = 4
BIN_PX = 1.0 / OVERSAMPLE
MAX_TILT_DEG = 40.0
MIN_SWEEP_PX = 2.0
MIN_ROWS = 20
MAX_SAGITTA_PX = 0.5      # quadratic departure from the fitted line over the edge length
MAX_ROW_SCATTER_PX = 1.0  # RMS of per-row edge positions about the fitted curve
MIN_SNR = 10.0            # edge contrast / pixel noise
LOW_SNR = 25.0
SPAN_LSF_WIDTHS = 20.0    # ESF half-span used, in LSF RMS widths (Hamming bias on MTF50 ~ 2.3 / ratio^2)
MIN_SPAN_LSF_WIDTHS = 6.0
MIN_SPAN_PX = 8.0
MTF_MAX_FREQ = 1.0
_ITERATIONS = 4


def mtf50_to_lp_per_mm(mtf50_cyc_per_px: float, um_per_px: float) -> float:
    """Convert an MTF50 in cycles per pixel to line pairs per millimetre in the object plane."""
    return 1000.0 * mtf50_cyc_per_px / um_per_px


def _hamming(x: np.ndarray, half: float) -> np.ndarray:
    w = 0.54 + 0.46 * np.cos(np.pi * x / half)
    return np.where(np.abs(x) <= half, w, 0.0)


def _psi(u: np.ndarray) -> np.ndarray:
    """Integral of the standard normal CDF: d/du psi = Phi."""
    return u * ndtr(u) + np.exp(-0.5 * u * u) / math.sqrt(2.0 * math.pi)


def lsf_model(x: np.ndarray, sigma: float, c: float, s: float) -> np.ndarray:
    """Unit-area Gaussian(sigma) convolved with a box of width c and a box of width s (s may be ~0)."""
    sigma = max(sigma, 1e-6)
    if s < 1e-4:
        return (ndtr((x + c / 2) / sigma) - ndtr((x - c / 2) / sigma)) / c
    hp, hm = (c + s) / 2, (c - s) / 2
    return sigma / (c * s) * (_psi((x + hp) / sigma) - _psi((x + hm) / sigma)
                              - _psi((x - hm) / sigma) + _psi((x - hp) / sigma))


def _as_float(img: np.ndarray, roi, saturation):
    a = np.asarray(img)
    if a.ndim != 2:
        raise ValueError("measure_edge needs a single-channel (grey) image")
    off = (0, 0)
    if roi is not None:
        x, y, w, h = (int(v) for v in roi)
        if w <= 0 or h <= 0 or x < 0 or y < 0 or x + w > a.shape[1] or y + h > a.shape[0]:
            raise ValueError(f"ROI {roi} is not inside the {a.shape[1]}x{a.shape[0]} image")
        a = a[y:y + h, x:x + w]
        off = (x, y)
    lo = hi = None
    if np.issubdtype(a.dtype, np.integer):
        info = np.iinfo(a.dtype)
        lo, hi = info.min, info.max
    if saturation is not None:
        hi = saturation
    clip_frac = 0.0
    if hi is not None:
        clip_frac = float(np.mean(a >= hi))
    if lo is not None:
        clip_frac = max(clip_frac, float(np.mean(a <= lo)))
    return a.astype(np.float64), off, clip_frac, hi is not None


def _line_fit(y: np.ndarray, x: np.ndarray):
    """Least-squares line with 3-sigma (MAD) outlier rejection; returns (a, b, keep mask)."""
    keep = np.ones(len(y), bool)
    for _ in range(3):
        b, a = np.polyfit(y[keep], x[keep], 1)
        r = x - (a + b * y)
        mad = 1.4826 * np.median(np.abs(r[keep] - np.median(r[keep])))
        new = np.abs(r) <= max(3.0 * mad, 0.5)
        if new.sum() < 3 or np.array_equal(new, keep):
            break
        keep = new
    b, a = np.polyfit(y[keep], x[keep], 1)
    return float(a), float(b), keep


def _row_centroids(d: np.ndarray, a: float, b: float, hw: float):
    """Hamming-windowed centroid of the (polarity-corrected) row derivative around the line x = a + b*y."""
    h, w = d.shape
    ys, xs, steps = [], [], []
    for y in range(h):
        xc = a + b * y
        lo, hi = math.ceil(xc - hw), math.floor(xc + hw)
        if lo < 1 or hi > w - 2:
            continue
        xx = np.arange(lo, hi + 1, dtype=np.float64)
        seg = d[y, lo:hi + 1] * _hamming(xx - xc, hw)
        tot = seg.sum()
        if tot <= 0:
            continue
        ys.append(y)
        xs.append(float((xx * seg).sum() / tot))
        steps.append(float(d[y, lo:hi + 1].sum()))
    return np.array(ys, np.float64), np.array(xs), np.array(steps)


def _esf(dist: np.ndarray, val: np.ndarray, half: float):
    """Bin (dist, val) into 1/OVERSAMPLE-px bins over [-half, half]; value interpolated to the bin centres."""
    nb = math.floor(half / BIN_PX)
    edges = np.arange(-nb, nb + 1) * BIN_PX
    centres = (edges[:-1] + edges[1:]) / 2
    sel = (dist >= edges[0]) & (dist < edges[-1])
    idx = np.floor((dist[sel] - edges[0]) / BIN_PX).astype(np.int64)
    n = np.bincount(idx, minlength=len(centres))
    sv = np.bincount(idx, weights=val[sel], minlength=len(centres))
    sd = np.bincount(idx, weights=dist[sel], minlength=len(centres))
    if np.any(n == 0):
        return centres, None, n
    esf = np.interp(centres, sd / n, sv / n)
    return centres, esf, n


def _fit_sigma(x: np.ndarray, lsf: np.ndarray, c: float, s: float, guess: float):
    """Least-squares fit of A * model(x - x0) + base; returns (sigma, x0)."""
    extra = 2.0 * BIN_PX ** 2 / 12.0  # bin average + forward difference, each a box one bin wide

    def res(q):
        amp, x0, sg, base = q
        return amp * lsf_model(x - x0, math.sqrt(sg * sg + extra), c, s) + base - lsf

    x0g = float((x * lsf).sum() / lsf.sum())
    q0 = [float(lsf.sum() * BIN_PX), x0g, max(guess, 0.3), 0.0]
    r = least_squares(res, q0, bounds=([0.0, x0g - 3, 0.0, -np.inf], [np.inf, x0g + 3, 100.0, np.inf]))
    return float(r.x[2]), float(r.x[1])


def measure_edge(img: np.ndarray, roi=None, saturation: float | None = None) -> dict:
    """Measure ESF, LSF, MTF, MTF50 and Gaussian blur from the one dominant straight edge in `roi`.

    img: 2-D grey image (any real dtype). roi: (x, y, width, height) in pixels, default the whole image.
    saturation: grey level at which the sensor clips; defaults to the dtype maximum for integer images and is
    unknown (not checked, a warning says so) for float images.

    Raises ValueError with a plain reason when there is no edge, it is too short, curved, tilted outside the
    allowed range, clipped, too noisy, or too close to the ROI border for the measured blur.

    Returns a dict:
      angle_deg       tilt from the nearest pixel axis (sign of dx/dy for a vertical edge, dy/dx for horizontal)
      orientation     "vertical" or "horizontal"
      polarity        +1 if intensity rises with increasing x (vertical) / y (horizontal), else -1
      edge_xy         (x, y) image coordinates of the edge at the middle of the rows used
      mtf             {"freq_cyc_per_px": array, "value": array}, 0..1 cycle/pixel along the edge normal
      mtf50_cyc_per_px  first frequency where the MTF falls to 0.5 (nan with a warning if it does not)
      psf_sigma_px    Gaussian optical sigma with the pixel aperture removed (Gaussian-PSF model)
      lsf_sigma_px    sqrt(psf_sigma_px^2 + 1/12): RMS width of the LSF including the square pixel aperture
      edge_contrast, dark_level, bright_level   from the ESF plateaux (image units)
      noise_sigma     robust std of pixels about the ESF on the plateaux; snr = edge_contrast / noise_sigma
      n_rows          rows (or columns) used; span_px: ESF half-span used
      esf_x_px, esf, lsf_x_px, lsf   the curves (distance along the normal, bright side positive)
      warnings        list of plain-language caveats
    """
    a, off, clip_frac, clip_checked = _as_float(img, roi, saturation)
    warnings: list[str] = []
    if not clip_checked:
        warnings.append("float image without a saturation level: clipping was not checked")
    if clip_frac > 0.001:
        raise ValueError(f"the edge is clipped: {100 * clip_frac:.1f}% of the ROI pixels are at the sensor "
                         "limit; reduce exposure or gain")
    if min(a.shape) < 16:
        raise ValueError(f"the ROI ({a.shape[1]}x{a.shape[0]} px) is too small; it needs at least 16 px each way")

    gx = np.abs(np.diff(a, axis=1)).sum()
    gy = np.abs(np.diff(a, axis=0)).sum()
    transposed = bool(gy > gx)
    if transposed:
        a = a.T
    h, w = a.shape

    d = np.zeros_like(a)
    d[:, 1:-1] = (a[:, 2:] - a[:, :-2]) / 2.0
    noise0 = 1.4826 * float(np.median(np.abs(d - np.median(d)))) * math.sqrt(2.0)
    row_step = (a[:, -2:].mean(axis=1) - a[:, :2].mean(axis=1))
    step = float(np.mean(row_step))
    if abs(step) <= 1e-9 * max(1.0, float(np.abs(a).max())) or abs(step) < 3.0 * noise0:
        raise ValueError("no edge: the ROI has no intensity step that stands out of the noise")
    pol = 1.0 if step > 0 else -1.0
    d *= pol

    # coarse per-row position: peak of the smoothed derivative, then a robust line
    ker = np.ones(5) / 5.0
    sm = np.apply_along_axis(lambda r: np.convolve(r, ker, mode="same"), 1, d)
    sm[:, :3] = sm[:, -3:] = -np.inf
    ya = np.arange(h, dtype=np.float64)
    xa = np.argmax(sm, axis=1).astype(np.float64)
    peaks = sm.max(axis=1)
    has = peaks > 0.3 * np.percentile(peaks, 90)  # rows that hold the edge at all
    if has.sum() < MIN_ROWS:
        raise ValueError(f"the edge is too short: only {int(has.sum())} rows hold it (needs {MIN_ROWS})")
    la, lb, keep = _line_fit(ya[has], xa[has])
    if keep.sum() < MIN_ROWS:
        raise ValueError(f"the edge is too short: only {int(keep.sum())} rows hold it (needs {MIN_ROWS})")

    hw = 8.0
    ys = xs = steps = np.empty(0)
    for _ in range(_ITERATIONS):
        ys, xs, steps = _row_centroids(d, la, lb, hw)
        if len(ys) >= MIN_ROWS:
            good = steps > 0.5 * np.median(steps)
            ys, xs = ys[good], xs[good]
        if len(ys) < MIN_ROWS:
            raise ValueError(f"the edge is too short or too close to the ROI border: {len(ys)} usable rows "
                             f"(needs {MIN_ROWS})")
        la, lb, keep = _line_fit(ys, xs)
        ys, xs = ys[keep], xs[keep]
        theta = math.atan(lb)
        cth = math.cos(theta)
        xe = la + lb * ys
        dist = (pol * (np.arange(w)[None, :] - xe[:, None]) * cth).ravel()
        val = a[ys.astype(np.int64)].ravel()
        avail = float(min(xe.min(), (w - 1 - xe).min()) * cth)
        _, esf0, _ = _esf(dist, val, min(avail, 24.0))
        if esf0 is None:
            break
        lsf0 = np.diff(esf0)
        xx = np.arange(len(lsf0)) * BIN_PX
        tot = lsf0.sum()
        if tot <= 0:
            break
        mu = (xx * lsf0).sum() / tot
        core = np.abs(xx - mu) <= 8.0
        var = max(float(((xx - mu) ** 2 * lsf0)[core].sum() / lsf0[core].sum()), 0.1)
        hw = float(np.clip(3.0 * math.sqrt(var) / cth + 2.0, 4.0, 0.45 * w))

    # straightness
    cq = np.polyfit(ys, xs, 2)
    ym = (ys.max() + ys.min()) / 2
    half_len = (ys.max() - ys.min()) / 2
    quad_res = xs - np.polyval(cq, ys)
    scatter = float(np.sqrt(np.mean(quad_res ** 2)))
    sag = abs(cq[0]) * half_len ** 2
    t = (ys - ym) / max(half_len, 1.0)
    se_c = scatter / max(math.sqrt(float(np.sum((t * t - np.mean(t * t)) ** 2))), 1e-9)
    if sag > MAX_SAGITTA_PX and sag > 4.0 * se_c:
        raise ValueError(f"the edge is curved: it departs {sag:.2f} px from a straight line over its length "
                         f"(allowed {MAX_SAGITTA_PX} px)")
    if scatter > MAX_ROW_SCATTER_PX:
        raise ValueError(f"the edge is too noisy or ragged: per-row edge positions scatter {scatter:.2f} px RMS "
                         f"about a smooth curve (allowed {MAX_ROW_SCATTER_PX} px)")

    n_rows = len(ys)
    tilt = abs(math.degrees(math.atan(lb)))
    if tilt > MAX_TILT_DEG:
        raise ValueError(f"the edge is tilted {tilt:.1f} deg from the pixel axis; the slanted-edge method here "
                         f"needs at most {MAX_TILT_DEG:.0f} deg (at 45 deg all rows have the same sub-pixel phase)")
    sweep = n_rows * abs(lb)
    if sweep < MIN_SWEEP_PX:
        need = math.degrees(math.atan(MIN_SWEEP_PX / n_rows))
        raise ValueError(f"the edge is too close to vertical/horizontal ({tilt:.2f} deg): over {n_rows} rows it "
                         f"crosses only {sweep:.2f} px, at least {MIN_SWEEP_PX:.0f} px are needed for 4x "
                         f"super-sampling (tilt it to at least {need:.1f} deg or use a longer ROI)")

    theta = math.atan(lb)
    cth, sth = math.cos(theta), abs(math.sin(theta))
    xe = la + lb * ys
    avail = float(min(xe.min(), (w - 1 - xe).min()) * cth)
    dist = (pol * (np.arange(w)[None, :] - xe[:, None]) * cth).ravel()
    val = a[ys.astype(np.int64)].ravel()

    def pipeline(half):
        centres, esf, _ = _esf(dist, val, half)
        if esf is None:
            raise ValueError(f"some 1/{OVERSAMPLE}-px bins of the edge spread function are empty: the tilt "
                             f"({tilt:.2f} deg) is too close to 0, 45 deg or a small-integer slope such as 1:2 "
                             "for this ROI")
        lsf = np.diff(esf) / BIN_PX
        lx = centres[:-1] + BIN_PX / 2
        tot = lsf.sum()
        if tot <= 0:
            raise ValueError("no edge: the edge spread function does not rise")
        mu = float((lx * lsf).sum() / tot)
        core = np.abs(lx - mu) <= max(6.0, half / 3)
        var = max(float(((lx - mu) ** 2 * lsf)[core].sum() / lsf[core].sum()), BIN_PX ** 2)
        sig, x0 = _fit_sigma(lx[core], lsf[core], cth, sth, math.sqrt(var))
        return centres, esf, lx, lsf, sig, x0

    first = min(avail, 24.0)
    if first < MIN_SPAN_PX:
        raise ValueError(f"the edge is too close to the ROI border: only {avail:.1f} px on its narrower side "
                         f"(needs {MIN_SPAN_PX:.0f})")
    _, _, _, _, sig, _ = pipeline(first)
    width = math.sqrt(sig * sig + 1.0 / 12.0)
    need_min = max(MIN_SPAN_PX, MIN_SPAN_LSF_WIDTHS * width)
    if avail < need_min:
        raise ValueError(f"the edge is too close to the ROI border for its blur: {avail:.1f} px on the narrower "
                         f"side, {need_min:.1f} px needed (6 LSF widths)")
    half = min(avail, max(MIN_SPAN_PX, SPAN_LSF_WIDTHS * width))
    if half < SPAN_LSF_WIDTHS * width:
        bias = 2.3 * (width / half) ** 2 * 100
        warnings.append(f"ESF span limited by the ROI to {half:.1f} px ({half / width:.1f} LSF widths): the "
                        f"LSF window may raise MTF50 by about {bias:.1f}%")
    centres, esf, lx, lsf, sig, x0 = pipeline(half)
    width = math.sqrt(sig * sig + 1.0 / 12.0)

    # MTF
    win = _hamming(lx - x0, half)
    n = len(lsf)
    npad = max(8192, 1 << math.ceil(math.log2(n)))
    spec = np.abs(np.fft.rfft(lsf * win, npad))
    freq = np.fft.rfftfreq(npad, d=BIN_PX)
    mtf = spec / spec[0] / np.sinc(freq * BIN_PX) ** 2
    keep_f = freq <= MTF_MAX_FREQ + 1e-12
    freq, mtf = freq[keep_f], mtf[keep_f]
    below = np.nonzero(mtf < 0.5)[0]
    if len(below) == 0:
        mtf50 = float("nan")
        warnings.append(f"the MTF stays above 0.5 up to {MTF_MAX_FREQ} cycle/pixel; MTF50 is not defined")
    else:
        k = int(below[0])
        f0, f1, m0, m1 = freq[k - 1], freq[k], mtf[k - 1], mtf[k]
        mtf50 = float(f0 + (m0 - 0.5) * (f1 - f0) / (m0 - m1))

    # levels and noise from the plateaux
    plateau = 3.0 * width + 1.0
    lo_bins = centres < x0 - plateau
    hi_bins = centres > x0 + plateau
    dark = float(np.median(esf[lo_bins])) if lo_bins.any() else float(esf[0])
    bright = float(np.median(esf[hi_bins])) if hi_bins.any() else float(esf[-1])
    contrast = bright - dark
    in_pl = (np.abs(dist - x0) > plateau) & (np.abs(dist) < half)
    resid = val[in_pl] - np.interp(dist[in_pl], centres, esf)
    noise = 1.4826 * float(np.median(np.abs(resid - np.median(resid)))) if resid.size else 0.0
    snr = contrast / noise if noise > 0 else float("inf")
    if contrast <= 0 or snr < MIN_SNR:
        raise ValueError(f"the edge is too noisy: contrast/noise = {snr:.1f} (needs at least {MIN_SNR:.0f})")
    if snr < LOW_SNR:
        warnings.append(f"low contrast/noise ({snr:.1f}): MTF values above ~0.5 cycle/pixel are noise-dominated")

    yc = float(np.median(ys))
    xcen = la + lb * yc
    ex, ey = (yc, xcen) if transposed else (xcen, yc)
    return {
        "angle_deg": math.degrees(theta),
        "orientation": "horizontal" if transposed else "vertical",
        "polarity": int(pol),
        "edge_xy": (ex + off[0], ey + off[1]),
        "mtf": {"freq_cyc_per_px": freq, "value": mtf},
        "mtf50_cyc_per_px": mtf50,
        "psf_sigma_px": sig,
        "lsf_sigma_px": width,
        "edge_contrast": contrast,
        "dark_level": dark,
        "bright_level": bright,
        "noise_sigma": noise,
        "snr": snr,
        "n_rows": n_rows,
        "span_px": half,
        "esf_x_px": centres, "esf": esf, "lsf_x_px": lx, "lsf": lsf,
        "warnings": warnings,
    }
