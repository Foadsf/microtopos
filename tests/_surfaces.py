"""Analytic surfaces rendered as focus sweeps for shape-from-focus generalisation tests.

The blur law (sigma = 0.6 + 1.5 |defocus um| px) is deliberately NOT the reconstruction's optical model, as in the
self-test's terrace check, so the test is independent of the shared optics code. Truth roughness (Sa, Sq) is
computed here with plain NumPy (least-squares plane removal), not with microtopos.metrology.
"""

from __future__ import annotations

import zlib

import cv2
import numpy as np

from microtopos.simulator import _variable_blur

W, H = 320, 180
UM_PER_PX = 0.5


def surface(kind: str, rng: np.random.Generator) -> np.ndarray:
    """Height map (um) of one analytic surface; geometry is randomised from rng within each kind."""
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    if kind == "slope":  # a tilted plane: 1.2-2.0 um rise across the field, random azimuth
        ang = rng.uniform(0, 2 * np.pi)
        rise = rng.uniform(1.2, 2.0)
        u = (xx - W / 2) * np.cos(ang) + (yy - H / 2) * np.sin(ang)
        return rise * u / np.abs(u).max() / 2
    if kind == "step":  # two terraces, random step height and edge position
        step = rng.uniform(1.2, 2.0)
        edge = int(rng.uniform(0.4, 0.6) * W)
        return np.where(xx >= edge, step, 0.0)
    if kind == "sphere":  # spherical cap, 1.2-2.0 um sag over a 70-90 px radius, random centre
        sag = rng.uniform(1.2, 2.0)
        a = rng.uniform(70, 90)
        R = (a * a * UM_PER_PX**2 + sag * sag) / (2 * sag)  # sphere radius (um) with chord radius a, sag `sag`
        cx, cy = W / 2 + rng.uniform(-20, 20), H / 2 + rng.uniform(-10, 10)
        r2 = ((xx - cx) ** 2 + (yy - cy) ** 2) * UM_PER_PX**2
        return np.maximum(np.sqrt(np.maximum(R * R - r2, 0)) - (R - sag), 0.0)  # flat base outside the cap
    if kind == "sinusoid":  # 2D sinusoid, 64-96 px period along x, 0.6-0.9 um amplitude
        amp = rng.uniform(0.6, 0.9)
        per = rng.uniform(64, 96)
        return amp * np.sin(2 * np.pi * xx / per + rng.uniform(0, 2 * np.pi))
    raise ValueError(kind)


def sweep(kind: str, seed: int, noise: float = 1.0, dz: float = 0.1):
    """(frames uint8, z labels um, true height um)."""
    rng = np.random.default_rng(zlib.crc32(repr((kind, seed, noise, dz)).encode()))
    relief = surface(kind, rng)
    tex = cv2.GaussianBlur(rng.uniform(40, 215, (H, W)).astype(np.float32), (0, 0), 1.0)
    zs = np.arange(relief.min() - 2.5, relief.max() + 2.5 + 1e-9, dz)
    frames = [np.clip(_variable_blur(tex, (0.6 + 1.5 * np.abs(relief - z)).astype(np.float32))
                      + rng.normal(0, noise, (H, W)), 0, 255).astype(np.uint8) for z in zs]
    return frames, zs, relief.astype(np.float32)


def plane_removed(z: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """z minus its least-squares plane over mask (independent of metrology.remove_form)."""
    yy, xx = np.mgrid[0:z.shape[0], 0:z.shape[1]].astype(np.float64)
    A = np.column_stack([np.ones(mask.sum()), xx[mask], yy[mask]])
    coef, *_ = np.linalg.lstsq(A, z[mask].astype(np.float64), rcond=None)
    return z - (coef[0] + coef[1] * xx + coef[2] * yy)


def sa_sq(z: np.ndarray, mask: np.ndarray) -> tuple[float, float]:
    v = plane_removed(z, mask)[mask]
    v = v - v.mean()
    return float(np.mean(np.abs(v))), float(np.sqrt(np.mean(v * v)))
