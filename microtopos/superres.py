"""Experimental multi-frame super-resolution for a static (DWELL) stack at one focus position.

Drizzle (Fruchter & Hook 2002): every input pixel is shrunk to a square "drop" of side pixfrac (in input
pixels), shifted by the frame's sub-pixel offset, and its value is spread over the output pixels it overlaps,
weighted by overlap area. Then Richardson-Lucy deconvolution with a discrepancy-principle stop.
Gains are only possible where the optics resolve more than one camera pixel can sample (low magnification).
"""

from __future__ import annotations

import math

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.special import erf

from .registration import register_pair


def immerkaer_sigma(img: np.ndarray) -> float:
    """Immerkaer (1996) noise sigma = sqrt(pi/2) * mean|I * N| / 6, on the flattest 50% of the image."""
    k = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], np.float32)
    r = np.abs(cv2.filter2D(img.astype(np.float32), cv2.CV_32F, k))[1:-1, 1:-1]
    m = cv2.boxFilter(img.astype(np.float32), cv2.CV_32F, (9, 9))
    v = cv2.boxFilter(img.astype(np.float32) ** 2, cv2.CV_32F, (9, 9)) - m * m
    v = v[1:-1, 1:-1]
    flat = v <= np.percentile(v, 50)
    return float(math.sqrt(math.pi / 2) * r[flat].mean() / 6.0)


def fit_edge_psf_sigma(img: np.ndarray, nominal: float) -> tuple[float, str]:
    """PSF sigma from an erf fit across the strongest straight edge; falls back to the nominal value."""
    img = img.astype(np.float32)
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.hypot(gx, gy)
    m = 12
    if min(img.shape) < 4 * m:
        return nominal, "nominal (image too small)"
    roi = mag[m:-m, m:-m]
    y, x = np.unravel_index(int(np.argmax(roi)), roi.shape)
    y, x = y + m, x + m
    g = float(mag[y, x])
    if g < 20:
        return nominal, "nominal (no strong edge)"
    ux, uy = float(gx[y, x]) / g, float(gy[y, x]) / g
    s = np.linspace(-8, 8, 33).astype(np.float32)
    # average 7 parallel profiles along the edge to beat noise
    profs = []
    for t in np.linspace(-3, 3, 7):
        mx = (x + s * ux - t * uy).astype(np.float32).reshape(1, -1)
        my = (y + s * uy + t * ux).astype(np.float32).reshape(1, -1)
        profs.append(cv2.remap(img, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT).ravel())
    p = np.mean(profs, axis=0)

    def res(q):
        a, b, x0, sg = q
        return a + b * erf((s - x0) / (math.sqrt(2) * sg)) - p

    try:
        q0 = [float(p.mean()), float(p[-1] - p[0]) / 2, 0.0, max(nominal, 0.6)]
        r = least_squares(res, q0, bounds=([-1e3, -1e3, -3, 0.3], [1e3, 1e3, 3, 8]))
        r2 = 1 - np.var(r.fun) / (np.var(p) + 1e-9)
        if r.success and r2 > 0.95:
            return float(r.x[3]), f"measured edge fit (R2={r2:.3f})"
    except (ValueError, RuntimeError):
        pass
    return nominal, "nominal (edge fit rejected)"


def drizzle(frames: list[np.ndarray], shifts: list[tuple[float, float]], scale: int = 2,
            pixfrac: float = 0.7) -> tuple[np.ndarray, np.ndarray]:
    """Drizzle frames onto a grid `scale` times finer. shifts[k] = content shift of frame k vs the reference.

    Returns (image, weight map). Input pixel (x, y) of frame k covers reference coordinates centred at
    (x - dx_k, y - dy_k); output pixel j covers reference interval [(j - 0.5)/scale - 0.5 + 0.5/scale ...].
    """
    h, w = frames[0].shape
    H, W = h * scale, w * scale
    num = np.zeros(H * W, np.float64)
    den = np.zeros(H * W, np.float64)
    half = pixfrac / 2.0
    ys, xs = np.mgrid[0:h, 0:w]
    for f, (dx, dy) in zip(frames, shifts):
        cx = (xs - dx + 0.5) * scale - 0.5  # drop centre in output pixel coordinates
        cy = (ys - dy + 0.5) * scale - 0.5
        hx = half * scale
        x0, x1 = cx - hx, cx + hx
        y0, y1 = cy - hx, cy + hx
        jx0 = np.floor(x0 + 0.5).astype(int)
        jy0 = np.floor(y0 + 0.5).astype(int)
        span = math.ceil(2 * hx) + 1
        vals = f.astype(np.float64).ravel()
        for oy in range(span):
            jy = jy0 + oy
            ov_y = np.clip(np.minimum(y1, jy + 0.5) - np.maximum(y0, jy - 0.5), 0, None)
            for ox in range(span):
                jx = jx0 + ox
                ov_x = np.clip(np.minimum(x1, jx + 0.5) - np.maximum(x0, jx - 0.5), 0, None)
                a = (ov_x * ov_y).ravel()
                ok = (a > 0) & (jx.ravel() >= 0) & (jx.ravel() < W) & (jy.ravel() >= 0) & (jy.ravel() < H)
                idx = jy.ravel()[ok] * W + jx.ravel()[ok]
                num += np.bincount(idx, a[ok] * vals[ok], minlength=H * W)
                den += np.bincount(idx, a[ok], minlength=H * W)
    img = (num / np.maximum(den, 1e-9)).reshape(H, W)
    den = den.reshape(H, W)
    hole = den < 1e-6
    if np.any(hole):  # fill rare uncovered pixels from a bilinear upsample of the reference
        up = cv2.resize(frames[0].astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR)
        img[hole] = up[hole]
    return img.astype(np.float32), den.astype(np.float32)


def richardson_lucy(obs: np.ndarray, sigma_px: float, noise_sigma: float, max_iter: int = 30) -> tuple[np.ndarray, int]:
    """Gaussian-PSF Richardson-Lucy, stopped when the residual RMS reaches the noise level (discrepancy)."""
    b = np.maximum(obs.astype(np.float32), 1e-3)
    x = b.copy()
    it = 0
    for it in range(1, max_iter + 1):
        conv = cv2.GaussianBlur(x, (0, 0), sigma_px)
        rms = float(np.sqrt(np.mean((conv - b) ** 2)))
        if rms <= noise_sigma:
            break
        x = x * cv2.GaussianBlur(b / np.maximum(conv, 1e-3), (0, 0), sigma_px)
    return x, it


def dwell_superres(frames: list[np.ndarray], psf_sigma_lr: float, scale: int = 2, pixfrac: float = 0.7,
                   deconvolve: bool = True) -> dict:
    ref = frames[0].astype(np.float32)
    # ECC translation (sub-pixel, unbiased) initialised by phase correlation; keep only valid registrations (F05)
    used_frames: list[np.ndarray] = [frames[0]]
    shifts: list[tuple[float, float]] = [(0.0, 0.0)]
    for f in frames[1:]:
        reg = register_pair(ref, f.astype(np.float32), 1)
        if reg.ok:
            used_frames.append(f)
            shifts.append((float(reg.warp[0, 2]), float(reg.warp[1, 2])))
    driz, cover = drizzle(used_frames, shifts, scale, pixfrac)
    sigma_lr, src = fit_edge_psf_sigma(np.mean(used_frames, axis=0), psf_sigma_lr)
    out, iters = driz, 0
    noise = immerkaer_sigma(driz)
    if deconvolve:
        # the drizzled image is blurred by the optics (sigma_lr*scale) and by the drop (pixfrac*scale box)
        sig_hr = math.sqrt((sigma_lr * scale) ** 2 + (pixfrac * scale) ** 2 / 12.0)
        out, iters = richardson_lucy(driz, sig_hr, noise)  # stops at the noise level or after 30 iterations
    return {"image": out, "drizzled": driz, "shifts": shifts, "frames_used": len(used_frames),
            "psf_sigma_lr": sigma_lr, "psf_source": src,
            "rl_iterations": iters, "noise_sigma": noise, "coverage_min": float(cover.min())}
