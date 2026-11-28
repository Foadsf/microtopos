"""Lateral chromatic aberration estimate and correction for colour (BGR) frames.

Cheap objectives and the eyepiece camera relay image red and blue at a slightly different magnification (and
possibly position) than green, so edges carry coloured fringes that grow towards the field edge. This module
measures that as an image-plane model per channel and resamples the red and blue planes onto green.

Model (per channel C in {R, B}, relative to G), in pixel coordinates of the frame it was measured on:

    x_C = c + s_C (x_G - c) + t_C,        c = ((w - 1) / 2, (h - 1) / 2)

A point imaged at x_G in green is imaged at x_C in channel C. s_C is the isotropic magnification of C relative to G
and t_C (px) the displacement of C relative to G at the frame centre. The centre is fixed at the frame centre and is
not a constraint: for an isotropic scale plus translation, a scale about any other centre c' is exactly the same
map with t replaced by t + (1 - s)(c' - c), so a free centre would only make the parameters degenerate. Placing it
at the frame centre (the centroid of the measured tiles) also decorrelates the scale and translation estimates. The
point the aberration is centred on, if wanted, is c + t / (1 - s), which is ill-conditioned as s -> 1.

Rotation is not part of the model (lateral colour of a rotationally symmetric system is radial). The estimator fits
it as a diagnostic (`rotation_mrad`); a value well outside its uncertainty means the model does not describe the
optics (for example a tilted relay), not something `apply` corrects.

Estimator: each channel is band-passed (frame-scale shading removed with `registration.remove_shading`, then a
Gaussian low-pass) so that brightness, colour balance and the channels' different noise do not bias it. R and B are
resampled onto G with the current model (Lanczos-4), the residual displacement of every textured tile is measured
with ECC (`cv2.findTransformECC`, translation, the registration module's conventions), and a linear least-squares
fit of the model to the tile displacements updates the parameters; this repeats until the update is negligible.
Measuring only a residual that is already near zero keeps ECC's bilinear sub-pixel bias (periodic in the
displacement, see registration.BREATHING_MAX_SIDE) out of the estimate. Several frames are averaged.

This is verified against synthetic frames with a known aberration only (tests/test_chromatic.py); it has not been
run on the real camera.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass

import cv2
import numpy as np

from .registration import SNR_MIN_CORRELATION, remove_shading

# Low-pass of the band-pass, applied after shading removal; it suppresses noise and Bayer demosaicing artefacts at
# Nyquist. It is kept small because the main error of the estimate is the specimen's own colour: where features of
# different colour overlap within the low-pass footprint, a channel is not a shifted copy of green and each tile's
# displacement is biased (randomly from tile to tile). Measured on the synthetic colour texture of
# tests/test_chromatic.py (384 x 288 px, 3 frames, scale 0.35-0.4 %): scale error of the 3-frame estimate
# 0.3-0.4e-4 with sigma 0.6-0.8 px and 32 px tiles, 0.9e-4 at sigma 1.0, 1.4e-4 at 1.5; 48 px tiles (fewer of them)
# 1.1-2.8e-4. On achromatic content the error is <= 0.6e-4 per single frame at sigma 1.5. ECC's bilinear sub-pixel
# bias, which a small sigma would expose, is avoided by measuring only residual displacements (module docstring).
LOWPASS_SIGMA = 0.8
ECC_MARGIN = 8  # px of the moving plane around a tile handed to ECC (residual shifts are < 3 px)
TILE = 32  # side of the tiles whose displacements the model is fitted to (px)
BORDER = 8  # tiles keep this far from the frame edge, where the resampled planes are replicated border pixels
# A tile is textured when the band-passed green plane has at least this RMS (grey levels on the 0..255 scale).
# 8-bit quantisation is 0.29 RMS and about 0.1 after the low-pass, so this is 5x the quantisation floor; whether the
# texture also beats sensor noise is decided by the correlation test below.
TEXTURE_MIN_RMS = 0.5
MIN_TILES = 6  # textured, correlated tiles needed per frame: 2 equations per tile, 3 parameters
MAX_TILES_PER_FRAME = 300  # a 1920x1080 frame has ~1900 tiles; 300 spread over the field fit the 3 parameters ample
MAX_ITER = 6
CONVERGED_PX = 1e-4  # an iteration whose update moves no frame corner by more than this ends the loop
# A channel's aberration is significant when its largest corner displacement exceeds three standard uncertainties
# and this floor. Measured on the synthetic colour texture of tests/test_chromatic.py (384 x 288 px, corner 239 px
# from the centre): without aberration a single frame reads a corner fringe of 0.03-0.10 px (5 specimens) with
# 3 standard uncertainties of 0.11-0.13 px, three frames 0.02 px, so the uncertainty test is what rejects them; the
# floor keeps a precisely measured but invisible fringe (a twentieth of a pixel) from being reported as one.
MIN_FRINGE_PX = 0.05
SIGNIFICANCE_K = 3.0


@dataclass(frozen=True)
class LateralCA:
    """Lateral chromatic aberration of R and B relative to G, measured on frames of `size` = (width, height).

    `scale_*` is s, `shift_*` is t (px, at the frame centre) of the module's model. `u_*` are standard
    uncertainties (statistical only: the scatter of tile displacements about the model and between frames; on
    synthetic colour frames the actual errors were within 3 of them, but interpolation and demosaicing errors that
    are the same in every tile would not show in that scatter). `rotation_mrad`
    is the fitted rotation (diagnostic, not applied), `frames` the number of frames averaged, `tiles` the number of
    tile measurements used, `significant` whether either channel's aberration exceeds its uncertainty and
    MIN_FRINGE_PX."""

    size: tuple[int, int]
    scale_r: float
    shift_r: tuple[float, float]
    scale_b: float
    shift_b: tuple[float, float]
    u_scale_r: float = 0.0
    u_shift_r: tuple[float, float] = (0.0, 0.0)
    u_scale_b: float = 0.0
    u_shift_b: tuple[float, float] = (0.0, 0.0)
    rotation_mrad: tuple[float, float] = (0.0, 0.0)  # (R, B)
    u_rotation_mrad: tuple[float, float] = (0.0, 0.0)
    frames: int = 0
    tiles: int = 0
    significant: bool = False

    # -- model ----------------------------------------------------------------------------------------------------
    def warp(self, channel: str) -> np.ndarray:
        """2x3 warp W of channel "r" or "b": maps G (output) coordinates to that channel's coordinates, so that
        cv2.warpAffine(plane, W, size, flags=WARP_INVERSE_MAP) puts the plane onto G (registration's convention)."""
        if channel == "r":
            return _model_warp(self.scale_r, self.shift_r, self.size)
        if channel == "b":
            return _model_warp(self.scale_b, self.shift_b, self.size)
        raise ValueError(f"channel must be 'r' or 'b', not {channel!r}")

    def fringe_px_at_corner(self) -> float:
        """Largest displacement of R or B relative to G at a corner of the frame (px), for display."""
        return max(_corner_fringe(self.scale_r, self.shift_r, self.size),
                   _corner_fringe(self.scale_b, self.shift_b, self.size))

    def apply(self, bgr: np.ndarray) -> np.ndarray:
        """Frame with R and B resampled onto G. Input: H x W x 3 BGR, uint8 or float (0..255 scale), of exactly
        `size`. Output: float32 BGR on the same 0..255 scale; G is the input's green plane unchanged (converted to
        float32), R and B are resampled with Lanczos-4 and clipped to 0..255 (Lanczos overshoots at sharp edges).
        Pixels within about one fringe of the border come from replicated border pixels, not observations.
        A frame of another size is refused: use `rescaled` when it is known to show the same field."""
        img = _as_bgr_float(bgr)
        if (img.shape[1], img.shape[0]) != tuple(self.size):
            raise ValueError(f"frame is {img.shape[1]}x{img.shape[0]} px but the aberration was measured at "
                             f"{self.size[0]}x{self.size[1]} px; use rescaled() if it shows the same field")
        out = np.empty_like(img)
        out[..., 1] = img[..., 1]
        for idx, ch in ((2, "r"), (0, "b")):
            out[..., idx] = np.clip(_warp_plane(img[..., idx], self.warp(ch)), 0.0, 255.0)
        return out

    def rescaled(self, width: int, height: int) -> LateralCA:
        """The same aberration for a capture of (width, height) that shows the same field of view resampled (a
        binned or scaled camera mode). Only the caller knows that; a cropped mode (region of interest) is not the
        same field and must be measured again. A different aspect ratio is refused, as is a size that is not the
        same field scaled (more than half a pixel off on the short side). Scales are unchanged, shifts scale by
        width / measured width (pixel centres: x' = k (x + 1/2) - 1/2, so the frame centre maps to the centre)."""
        w0, h0 = self.size
        k = width / w0
        if width <= 0 or height <= 0 or abs(h0 * k - height) > 0.5 or abs(height / h0 * w0 - width) > 0.5:
            raise ValueError(f"{width}x{height} px is not {w0}x{h0} px scaled (aspect ratio differs); "
                             "measure the aberration in that capture mode")
        def sc(v):
            return (v[0] * k, v[1] * k)
        return LateralCA((int(width), int(height)), self.scale_r, sc(self.shift_r), self.scale_b, sc(self.shift_b),
                         self.u_scale_r, sc(self.u_shift_r), self.u_scale_b, sc(self.u_shift_b),
                         self.rotation_mrad, self.u_rotation_mrad, self.frames, self.tiles, self.significant)

    # -- persistence ----------------------------------------------------------------------------------------------
    def to_json(self) -> str:
        d = asdict(self)
        d["model"] = "x_C = c + s_C (x_G - c) + t_C, c = frame centre, pixels of `size`"
        return json.dumps(d, indent=2)

    @classmethod
    def from_json(cls, text: str) -> LateralCA:
        d = json.loads(text)
        d.pop("model", None)
        pairs = ("size", "shift_r", "shift_b", "u_shift_r", "u_shift_b", "rotation_mrad", "u_rotation_mrad")
        for k in pairs:
            d[k] = tuple(d[k])
        d["size"] = (int(d["size"][0]), int(d["size"][1]))
        return cls(**d)


def _centre(size: tuple[int, int]) -> np.ndarray:
    return np.array([(size[0] - 1) / 2.0, (size[1] - 1) / 2.0])


def _model_warp(s: float, t: tuple[float, float], size: tuple[int, int]) -> np.ndarray:
    c = _centre(size)
    tr = (1.0 - s) * c + np.asarray(t, np.float64)
    return np.array([[s, 0.0, tr[0]], [0.0, s, tr[1]]], np.float64)


def _corners(size: tuple[int, int]) -> np.ndarray:
    w, h = size
    return np.array([[0, 0], [w - 1, 0], [0, h - 1], [w - 1, h - 1]], np.float64)


def _corner_fringe(s: float, t: tuple[float, float], size: tuple[int, int]) -> float:
    d = (s - 1.0) * (_corners(size) - _centre(size)) + np.asarray(t, np.float64)
    return float(np.hypot(d[:, 0], d[:, 1]).max())


def _warp_plane(plane: np.ndarray, W: np.ndarray) -> np.ndarray:
    h, w = plane.shape
    return cv2.warpAffine(plane, W.astype(np.float64), (w, h), flags=cv2.INTER_LANCZOS4 | cv2.WARP_INVERSE_MAP,
                          borderMode=cv2.BORDER_REPLICATE)


def _as_bgr_float(bgr: np.ndarray) -> np.ndarray:
    a = np.asarray(bgr)
    if a.ndim != 3 or a.shape[2] != 3:
        raise ValueError(f"expected an H x W x 3 BGR frame, got shape {a.shape}")
    if a.dtype != np.uint8 and not np.issubdtype(a.dtype, np.floating):
        # a 10/12/16-bit frame has no known 0..255 scale here; converting it would be a guess
        raise ValueError(f"expected uint8 or float BGR on the 0..255 scale, got {a.dtype}")
    return a.astype(np.float32)


def _band_pass(plane: np.ndarray) -> np.ndarray:
    return cv2.GaussianBlur(remove_shading(plane), (0, 0), LOWPASS_SIGMA, borderType=cv2.BORDER_REFLECT)


def _tile_origins(size: tuple[int, int]) -> list[tuple[int, int]]:
    w, h = size
    nx, ny = (w - 2 * BORDER) // TILE, (h - 2 * BORDER) // TILE
    x0 = (w - nx * TILE) // 2
    y0 = (h - ny * TILE) // 2
    return [(x0 + i * TILE, y0 + j * TILE) for j in range(ny) for i in range(nx)]


def _tile_shift(g: np.ndarray, moving: np.ndarray, x0: int, y0: int) -> tuple[float, float, float] | None:
    """(dx, dy, rho) such that moving(x + d) ~ g(x) on the tile at (x0, y0); None when ECC fails."""
    tmpl = np.ascontiguousarray(g[y0:y0 + TILE, x0:x0 + TILE])
    # ECC computes the gradients of the whole image it is given: a crop of the tile plus a margin keeps that
    # proportional to the tile (on 1920x1080 frames the full image made the estimate take minutes)
    h, w = moving.shape
    cx0, cy0 = max(0, x0 - ECC_MARGIN), max(0, y0 - ECC_MARGIN)
    crop = np.ascontiguousarray(moving[cy0:min(h, y0 + TILE + ECC_MARGIN), cx0:min(w, x0 + TILE + ECC_MARGIN)])
    init = np.array([[1, 0, x0 - cx0], [0, 1, y0 - cy0]], np.float32)
    try:
        crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 1e-6)
        rho, W = cv2.findTransformECC(tmpl, crop, init.copy(), cv2.MOTION_TRANSLATION, crit, None, 1)
    except cv2.error:
        return None
    dx, dy = float(W[0, 2]) - (x0 - cx0), float(W[1, 2]) - (y0 - cy0)
    if not (np.isfinite(rho) and math.isfinite(dx) and math.isfinite(dy)) or math.hypot(dx, dy) > 3.0:
        return None
    return dx, dy, float(rho)


def _fit(points: np.ndarray, d: np.ndarray, rotation: bool) -> tuple[np.ndarray, np.ndarray]:
    """Least-squares fit of d = a X + b (+ theta perp(X)) to displacements d at centred positions X.
    Returns (params, standard uncertainties): (a, bx, by[, theta])."""
    n = len(points)
    X, Y = points[:, 0], points[:, 1]
    one, zero = np.ones(n), np.zeros(n)
    cols_x = [X, one, zero] + ([-Y] if rotation else [])
    cols_y = [Y, zero, one] + ([X] if rotation else [])
    A = np.vstack([np.stack(cols_x, 1), np.stack(cols_y, 1)])
    rhs = np.concatenate([d[:, 0], d[:, 1]])
    p, *_ = np.linalg.lstsq(A, rhs, rcond=None)
    dof = max(1, len(rhs) - A.shape[1])
    sigma2 = float(np.sum((rhs - A @ p) ** 2)) / dof
    cov = sigma2 * np.linalg.inv(A.T @ A)
    return p, np.sqrt(np.diag(cov))


def _frames_list(bgr_frames) -> list[np.ndarray]:
    if isinstance(bgr_frames, np.ndarray):
        frames = [bgr_frames] if bgr_frames.ndim == 3 else list(bgr_frames)
    else:
        frames = list(bgr_frames)
    if not frames:
        raise ValueError("no frames")
    out = [_as_bgr_float(f) for f in frames]
    if any(f.shape != out[0].shape for f in out):
        raise ValueError("all frames must have the same size")
    return out


def estimate_lateral_ca(bgr_frames, require_significant: bool = True) -> LateralCA:
    """Lateral chromatic aberration of R and B relative to G from one or more BGR frames of the same size
    (uint8 or float, 0..255; an N x H x W x 3 array or a list). The frames may show the same or different views.

    Raises ValueError when a frame has too little texture (fewer than MIN_TILES tiles with band-passed RMS >=
    TEXTURE_MIN_RMS), when the channels are too dissimilar (or noisy) to register (fewer than MIN_TILES textured
    tiles whose ECC correlation reaches registration.SNR_MIN_CORRELATION, i.e. SNR >= 0 dB), and, unless
    require_significant is False, when neither channel's aberration is significant (see MIN_FRINGE_PX).

    Uncertainty: per frame, from the scatter of the tile displacements about the model; across frames the larger
    of the between-frame scatter / sqrt(n) and the median single-frame uncertainty (frames of the same view share
    their texture, so averaging them is not credited with independent information)."""
    frames = _frames_list(bgr_frames)
    h, w = frames[0].shape[:2]
    size = (w, h)
    c = _centre(size)
    origins = _tile_origins(size)
    centres = np.array([(x0 + (TILE - 1) / 2.0, y0 + (TILE - 1) / 2.0) for x0, y0 in origins]) - c
    if len(origins) < MIN_TILES:
        raise ValueError(f"frame {w}x{h} px is too small for {MIN_TILES} tiles of {TILE} px")

    bands = [[_band_pass(f[..., i]) for i in range(3)] for f in frames]  # B, G, R per frame
    textured = []
    for bp in bands:
        g = bp[1]
        keep = [k for k, (x0, y0) in enumerate(origins) if float(g[y0:y0 + TILE, x0:x0 + TILE].std()) >= TEXTURE_MIN_RMS]
        if len(keep) < MIN_TILES:
            raise ValueError(f"too little texture to measure lateral chromatic aberration: {len(keep)} of "
                             f"{len(origins)} tiles reach {TEXTURE_MIN_RMS} grey levels RMS (need {MIN_TILES})")
        if len(keep) > MAX_TILES_PER_FRAME:  # evenly spread over the field: the fit needs coverage, not every tile
            keep = [keep[i] for i in np.linspace(0, len(keep) - 1, MAX_TILES_PER_FRAME).round().astype(int)]
        textured.append(keep)

    result = {}
    total_tiles = 0
    for name, idx in (("r", 2), ("b", 0)):
        s, t = 1.0, np.zeros(2)
        per_frame = []
        for _ in range(MAX_ITER):
            W = _model_warp(s, t, size)
            per_frame = []
            for bp, keep in zip(bands, textured):
                moving = _warp_plane(bp[idx], W)
                pts, dis = [], []
                for k in keep:
                    r = _tile_shift(bp[1], moving, *origins[k])
                    if r is not None and r[2] >= SNR_MIN_CORRELATION:
                        pts.append(centres[k])
                        dis.append(r[:2])
                if len(pts) < MIN_TILES:
                    raise ValueError(f"{'red' if name == 'r' else 'blue'} and green are too dissimilar (or too noisy) "
                                     f"to register: {len(pts)} of {len(keep)} textured tiles correlate at >= "
                                     f"{SNR_MIN_CORRELATION} (need {MIN_TILES})")
                pts, dis = np.array(pts), np.array(dis)
                p, u = _fit(pts, dis, rotation=False)
                pr, ur = _fit(pts, dis, rotation=True)
                per_frame.append((p, u, pr[3], ur[3], len(pts)))
            a, bx, by = np.mean([pf[0] for pf in per_frame], axis=0)
            # compose: the residual map x -> x + a (x - c) + b is applied before the current model
            s_new, t_new = s * (1.0 + a), t + s * np.array([bx, by])
            moved = np.abs((s_new - s) * (_corners(size) - c) + (t_new - t)).max()
            s, t = float(s_new), t_new
            if moved < CONVERGED_PX:
                break
        P = np.array([pf[0] for pf in per_frame])
        U = np.array([pf[1] for pf in per_frame])
        n = len(per_frame)
        u = np.median(U, axis=0)
        if n > 1:
            u = np.maximum(u, P.std(axis=0, ddof=1) / math.sqrt(n))
        rot = float(np.mean([pf[2] for pf in per_frame]))
        u_rot = float(np.median([pf[3] for pf in per_frame]))
        if n > 1:
            u_rot = max(u_rot, float(np.std([pf[2] for pf in per_frame], ddof=1)) / math.sqrt(n))
        total_tiles += sum(pf[4] for pf in per_frame)
        # uncertainties of the residual fit are those of (s, t) scaled by s (s = s_prev (1 + a), t = t_prev + s b)
        result[name] = {"s": s, "t": (float(t[0]), float(t[1])), "u_s": float(s * u[0]),
                        "u_t": (float(s * u[1]), float(s * u[2])), "rot": 1e3 * rot, "u_rot": 1e3 * u_rot}

    def significant(ch) -> bool:
        d = (ch["s"] - 1.0) * (_corners(size) - c) + np.asarray(ch["t"])
        mag = np.hypot(d[:, 0], d[:, 1])
        R = np.hypot(*(_corners(size) - c).T)
        u_d = np.sqrt((ch["u_s"] * R) ** 2 + ch["u_t"][0] ** 2 + ch["u_t"][1] ** 2)
        k = int(np.argmax(mag))
        return bool(mag[k] > max(SIGNIFICANCE_K * u_d[k], MIN_FRINGE_PX))

    r, b = result["r"], result["b"]
    ca = LateralCA(size, r["s"], r["t"], b["s"], b["t"], r["u_s"], r["u_t"], b["u_s"], b["u_t"],
                   (r["rot"], b["rot"]), (r["u_rot"], b["u_rot"]), len(frames), total_tiles,
                   significant(r) or significant(b))
    if require_significant and not ca.significant:
        raise ValueError(f"lateral chromatic aberration not significant: corner fringe "
                         f"{ca.fringe_px_at_corner():.3f} px (needs > {MIN_FRINGE_PX} px and > "
                         f"{SIGNIFICANCE_K:g} standard uncertainties)")
    return ca
