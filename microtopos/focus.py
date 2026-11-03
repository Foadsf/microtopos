"""Focus measures: per-pixel maps (for shape-from-focus) and global scores (for live state detection)."""

from __future__ import annotations

import cv2
import numpy as np

MEASURES = ("DoG band-pass", "Tenengrad", "SML", "Vollath F4", "Vollath F5")


def _box(a: np.ndarray, win: int) -> np.ndarray:
    return cv2.boxFilter(a, cv2.CV_32F, (win, win), borderType=cv2.BORDER_REFLECT)


def focus_map(img: np.ndarray, measure: str = "DoG band-pass", window: int = 9, psf_sigma_px: float = 1.0) -> np.ndarray:
    """Local focus energy, averaged over a window x window box (no subsampling, no spatial offset)."""
    img = img.astype(np.float32, copy=False)
    if measure == "DoG band-pass":
        s1 = max(0.8, 0.8 * psf_sigma_px)
        bp = cv2.GaussianBlur(img, (0, 0), s1) - cv2.GaussianBlur(img, (0, 0), 2.5 * s1)
        return _box(bp * bp, window)
    if measure == "Tenengrad":
        gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
        return _box(gx * gx + gy * gy, window)
    if measure == "SML":
        k = np.array([[-1, 2, -1]], np.float32)
        lx = np.abs(cv2.filter2D(img, cv2.CV_32F, k, borderType=cv2.BORDER_REFLECT))
        ly = np.abs(cv2.filter2D(img, cv2.CV_32F, k.T, borderType=cv2.BORDER_REFLECT))
        return _box(lx + ly, window)
    if measure in ("Vollath F4", "Vollath F5"):
        i1 = np.roll(img, -1, axis=1)
        p1 = _box(img * i1, window)
        if measure == "Vollath F4":
            i2 = np.roll(img, -2, axis=1)
            return np.maximum(p1 - _box(img * i2, window), 0.0)
        # F5 = sum I(x)I(x+1) - N * mean(I)^2  (local version)
        m = _box(img, window)
        return np.maximum(p1 - m * m, 0.0)
    raise ValueError(f"unknown focus measure {measure!r}")


def brenner(img: np.ndarray) -> float:
    """Mean squared 2-pixel horizontal difference (Brenner), a fast global sharpness score."""
    d = img[:, 2:] - img[:, :-2]
    return float(np.mean(d * d))


def tile_scores(img: np.ndarray, grid: int = 4, measure_sigma: float = 1.0) -> np.ndarray:
    """Band-pass energy per tile of a grid x grid layout (used for sweep segmentation and coverage)."""
    bp = cv2.GaussianBlur(img, (0, 0), measure_sigma) - cv2.GaussianBlur(img, (0, 0), 2.5 * measure_sigma)
    e = bp * bp
    h, w = e.shape
    th, tw = h // grid, w // grid
    return e[: th * grid, : tw * grid].reshape(grid, th, grid, tw).mean(axis=(1, 3)).ravel()
