"""Aligned, outlier-robust stacking of still (DWELL) frames at one focus position.

Every frame is registered to a reference frame with `registration.register_pair` (translation model, full
resolution, the same call `superres.dwell_superres` makes), frames whose registration is not a valid ECC match are
dropped, the rest are resampled onto the reference grid by bilinear interpolation and merged per pixel with a
sigma-clipped mean. The result keeps the reference frame's pixel grid and size; no pixel is invented: a pixel near
the border uses only the frames whose interpolation taps all lie inside that frame, and `count` / `valid` say how
many frames each pixel holds and where every used frame contributed.

What stacking gains: the random part of the error (sensor noise, and JPEG artefacts that differ from frame to
frame because the view moved) falls roughly as 1/sqrt(frames used). What it does not remove: anything common to
all frames (defocus, the optics' blur, fixed-pattern noise, a JPEG artefact that repeats because the view did not
move), and bilinear resampling smooths every frame except the reference a little.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from .registration import register_pair
from .superres import MIN_SR_FRAMES, immerkaer_sigma, richardson_lucy

REFERENCES = ("median", "first")
# How many frames are tried as the frame everything is first registered to before giving up. One or two bad frames
# at the start of the buffer (a bump, a particle) are passed over; a buffer whose first three frames each fail to
# register with more than half of the others is not a still view of one scene.
PRELIMINARY_TRIES = 3
MAD_TO_SIGMA = 1.4826  # median absolute deviation of a Gaussian sample times this estimates its sigma
# A resampled pixel counts as observed only when its bilinear taps all fall inside the frame. OpenCV interpolates
# on a 1/32-pixel grid, so a tap outside the frame carries a weight of at least 1/32 and an all-ones frame warped with
# a zero border reads at most 1 - 1/32 there; this threshold separates the two cases with margin.
FULL_SUPPORT = 1.0 - 1.0 / 64.0
GAIN_LOW_DN, GAIN_HIGH_DN = 5.0, 250.0  # pixels outside this range do not set a frame's brightness gain
_ROWS_PER_CHUNK = 64  # the per-pixel sort runs on bands of rows to bound memory on full-resolution frames


def _median_sorted(xs: np.ndarray, n: np.ndarray) -> np.ndarray:
    """Median along axis 0 of `xs`, sorted with NaN last, where n holds the number of finite values per pixel."""
    lo = np.take_along_axis(xs, ((n - 1) // 2)[None], 0)[0]
    hi = np.take_along_axis(xs, (n // 2)[None], 0)[0]
    return 0.5 * (lo + hi)


def _translation(d: tuple[float, float]) -> np.ndarray:
    return np.array([[1.0, 0.0, d[0]], [0.0, 1.0, d[1]]], np.float32)


def _warp(img: np.ndarray, W: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """aligned(x) = img(W x), registration.py's convention; outside the frame the value is 0 and is masked off."""
    return cv2.warpAffine(img, W, size, flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def stack_frames(frames: list[np.ndarray], colour: list[np.ndarray] | None = None, reference: str = "median",
                 clip_sigma: float = 3.0, deconvolve_psf_sigma: float | None = None) -> dict:
    """Register, resample and merge still frames into one image on the reference frame's grid.

    frames: luma frames (H x W, uint8 or float32) of one still view. colour: optional matching H x W x 3 frames;
    they get the same warps and the same per-sample rejection as the luma, so the colour stack and the luma stack
    are made from the same samples.

    reference: "median" (default) registers every frame to frame 0 (or, when frame 0 does not register with more
    than half of the frames, to frame 1, then 2) and then uses the frame whose shift is nearest the median shift as
    the reference. The median ignores a minority of outlying frames, and the central frame keeps the area that all
    frames cover as large as possible. Shifts are re-expressed relative to it by subtraction, which is exact for the
    translation model used. "first" uses frame 0 and fails if frame 0 is not usable.

    clip_sigma: per pixel, samples farther than clip_sigma * scale from the median of that pixel's samples are
    rejected and the rest are averaged. scale = max(1.4826 * MAD of that pixel's samples, the noise of one
    unresampled sample (the larger of the median MAD scale and Immerkaer's estimate)): the MAD follows local excess spread (residual mis-registration at
    edges, JPEG ringing), and the noise floor keeps a pixel whose few samples happen to agree closely from rejecting good data. A transient
    (a passing particle) or a frame that registered wrongly is rejected where it differs; where it does not differ
    it does no harm.

    deconvolve_psf_sigma: when given, the stack is deconvolved with `superres.richardson_lucy` for a Gaussian PSF of
    this sigma (pixels), stopped when the residual reaches `noise_sigma_stack`. Only the luma is deconvolved.

    Raises ValueError when fewer than `superres.MIN_SR_FRAMES` frames register (the same minimum
    super-resolution applies to the same buffer).

    Returns a dict:
      image: float32 H x W, the stack (deconvolved when requested); stacked: the stack before deconvolution.
      colour: float32 H x W x 3 or None.
      count: int H x W, frames that observed each pixel (before rejection); valid: bool H x W, count == frames_used
        (the area every used frame covers); common_box: (y0, y1, x0, x1), the bounding box of `valid`
        (half-open). Outside `valid` a pixel is the merge of fewer frames and is noisier, never border fill.
      frames_used, frame_indices (input indices used, ascending), frames_rejected, reference (input index).
      shifts: per used frame, the content shift (dx, dy) in pixels relative to the reference: frame k shows the
        reference content moved by +shift (the translation of its registration warp).
      noise_sigma_single: frame-to-frame noise of one resampled sample, RMS over `valid` of the per-pixel standard
        deviation of the kept samples (a temporal estimate, right for spatially correlated camera noise; for a still
        view it is the camera's noise, with sub-pixel shifts bilinear resampling lowers it).
      gains: per used frame, its brightness relative to the reference (lamp flicker, automatic exposure), divided
        out before the merge.
      noise_sigma_stack: RMS over `valid` of each pixel's standard error, sqrt(sample variance / samples kept), from
        the scatter of the kept samples about their mean. It measures the frame-to-frame (random) part of the error
        only; blur and any error common to all frames are not in it.
      pixels_clipped_fraction: rejected samples / observed samples.
      rl_iterations: Richardson-Lucy iterations run (0 without deconvolution).
    """
    if reference not in REFERENCES:
        raise ValueError(f"reference must be one of {REFERENCES}, not {reference!r}")
    if not clip_sigma >= 1.0:
        # below 1 the clip could reject every sample of a pixel (at least half lie within one MAD of the median)
        raise ValueError(f"clip_sigma must be at least 1, not {clip_sigma}")
    if deconvolve_psf_sigma is not None and not deconvolve_psf_sigma > 0:
        raise ValueError(f"deconvolve_psf_sigma must be positive, not {deconvolve_psf_sigma}")
    n = len(frames)
    if n < MIN_SR_FRAMES:
        raise ValueError(f"{n} frames given; stacking needs at least {MIN_SR_FRAMES}")
    lum = [np.asarray(f).astype(np.float32) for f in frames]
    h, w = lum[0].shape[:2]
    if any(f.ndim != 2 or f.shape != (h, w) for f in lum):
        raise ValueError("frames must all be single-channel images of the same size")
    if colour is not None and (len(colour) != n or any(np.asarray(c).shape != (h, w, 3) for c in colour)):
        raise ValueError("colour must hold one H x W x 3 frame per luma frame, of the same size")

    # ---- registration: to a preliminary frame, then re-expressed relative to the reference
    best = None
    tries = [0] if reference == "first" else list(range(min(PRELIMINARY_TRIES, n)))
    for p in tries:
        shifts_p = {p: (0.0, 0.0)}
        for k in range(n):
            if k != p:
                r = register_pair(lum[p], lum[k], 1)
                if r.ok:
                    shifts_p[k] = (float(r.warp[0, 2]), float(r.warp[1, 2]))
        if best is None or len(shifts_p) > len(best[1]):
            best = (p, shifts_p)
        if 2 * len(shifts_p) > n:  # more than half of the frames agree with p: p belongs to the main group
            break
    p, shifts_p = best
    keep = sorted(shifts_p)
    if len(keep) < MIN_SR_FRAMES:
        raise ValueError(f"only {len(keep)} of {n} frames registered reliably (needs {MIN_SR_FRAMES}); "
                         "the view needs texture, must stay in focus and must not move by more than a quarter frame")
    if reference == "median":
        s = np.array([shifts_p[k] for k in keep])
        m = np.median(s, axis=0)
        ref = keep[int(np.argmin(np.hypot(s[:, 0] - m[0], s[:, 1] - m[1])))]
    else:
        ref = p
    shifts = [(shifts_p[k][0] - shifts_p[ref][0], shifts_p[k][1] - shifts_p[ref][1]) for k in keep]
    warps = [_translation(d) for d in shifts]

    # ---- resample onto the reference grid; samples outside a frame's full bilinear support are NaN
    size = (w, h)
    ones = np.ones((h, w), np.float32)
    nu = len(keep)
    S = np.empty((nu, h, w), np.float32)
    for i, (k, W) in enumerate(zip(keep, warps)):
        S[i] = _warp(lum[k], W, size)
        S[i][_warp(ones, W, size) < FULL_SUPPORT] = np.nan

    # ---- brightness: lamp flicker and automatic exposure scale whole frames; each frame's gain relative to the
    # reference (ratio of their means over the pixels both observe, clipped and near-black pixels excluded) is
    # divided out
    iref = keep.index(ref)
    sref = S[iref]
    gains = np.ones(nu)
    for i in range(nu):
        if i != iref:
            m = np.isfinite(S[i]) & (S[i] > GAIN_LOW_DN) & (S[i] < GAIN_HIGH_DN) & (sref > GAIN_LOW_DN) \
                & (sref < GAIN_HIGH_DN)
            if m.sum() > 0.1 * h * w:
                # ratio of means, not least squares: noise and the bilinear smoothing of a resampled frame are
                # zero-mean, but they lower a least-squares slope against the noisy reference (regression dilution)
                gains[i] = float(S[i][m].astype(np.float64).sum() / sref[m].astype(np.float64).sum())
                S[i] /= np.float32(gains[i])

    # ---- the clip floor: the noise of one unresampled sample, bounded from below by two estimates that each read
    # low in one case: the spread of every pixel's samples (temporal; low when sub-pixel shifts smooth the resampled
    # samples) and Immerkaer's single-frame estimate (spatial; low for the spatially correlated noise that
    # demosaicing and JPEG give a real camera). The larger keeps good samples of the noisiest frame.
    sub = S[:, ::4, ::4]
    full = np.all(np.isfinite(sub), axis=0)
    xs = sub[:, full]
    temporal = float(np.median(MAD_TO_SIGMA * np.median(np.abs(xs - np.median(xs, axis=0)), axis=0))) \
        if xs.shape[1] else 0.0
    floor = max(temporal, float(np.median([immerkaer_sigma(lum[k]) for k in keep])), 1e-6)

    # ---- robust merge, in bands of rows
    image = np.empty((h, w), np.float32)
    count = np.empty((h, w), np.int32)
    kept_n = np.empty((h, w), np.int32)
    se2 = np.zeros((h, w), np.float64)
    kept = np.empty((nu, h, w), bool)
    for r0 in range(0, h, _ROWS_PER_CHUNK):
        x = S[:, r0:r0 + _ROWS_PER_CHUNK]
        cnt = np.sum(~np.isnan(x), axis=0)  # >= 1: the reference observes every pixel of its own grid
        med = _median_sorted(np.sort(x, axis=0), cnt)
        dev = np.abs(x - med)
        mad = _median_sorted(np.sort(dev, axis=0), cnt)
        scale = np.maximum(MAD_TO_SIGMA * mad, floor)
        k_mask = dev <= clip_sigma * scale  # NaN compares False, so unobserved samples are never kept
        nk = k_mask.sum(axis=0)
        xv = np.where(k_mask, x, 0.0).astype(np.float64)
        mean = xv.sum(axis=0) / nk
        ss = (np.where(k_mask, x - mean, 0.0).astype(np.float64) ** 2).sum(axis=0)
        se2[r0:r0 + _ROWS_PER_CHUNK] = np.where(nk > 1, ss / np.maximum(nk - 1, 1) / nk, np.nan)
        image[r0:r0 + _ROWS_PER_CHUNK] = mean
        count[r0:r0 + _ROWS_PER_CHUNK] = cnt
        kept_n[r0:r0 + _ROWS_PER_CHUNK] = nk
        kept[:, r0:r0 + _ROWS_PER_CHUNK] = k_mask

    valid = count == nu
    rows, cols = np.flatnonzero(valid.any(axis=1)), np.flatnonzero(valid.any(axis=0))
    common_box = (int(rows[0]), int(rows[-1]) + 1, int(cols[0]), int(cols[-1]) + 1) if rows.size else (0, 0, 0, 0)
    se_valid = se2[valid]
    nk_valid = kept_n[valid]
    fin = np.isfinite(se_valid)
    noise_stack = float(math.sqrt(se_valid[fin].mean())) if fin.any() else float("nan")
    noise_single = float(math.sqrt((se_valid[fin] * nk_valid[fin]).mean())) if fin.any() else float("nan")
    clipped = 1.0 - float(kept_n.sum()) / float(count.sum())

    col_out = None
    if colour is not None:
        col_out = np.zeros((h, w, 3), np.float64)
        for i, (k, W) in enumerate(zip(keep, warps)):
            c = _warp(np.asarray(colour[k]).astype(np.float32), W, size) / np.float32(gains[i])
            col_out += np.where(kept[i][..., None], c, 0.0)
        col_out = (col_out / kept_n[..., None]).astype(np.float32)

    out, iters = image, 0
    if deconvolve_psf_sigma is not None:
        out, iters = richardson_lucy(image, float(deconvolve_psf_sigma), noise_stack)
    return {"image": out.astype(np.float32), "stacked": image, "colour": col_out, "count": count, "valid": valid,
            "common_box": common_box, "frames_used": nu, "frame_indices": keep, "frames_rejected": n - nu,
            "reference": ref, "shifts": shifts, "noise_sigma_single": noise_single, "noise_sigma_stack": noise_stack,
            "gains": [float(g) for g in gains],
            "pixels_clipped_fraction": clipped, "rl_iterations": iters}
