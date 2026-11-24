"""FAILING tests for two REAL defects on this stack whose fixes live in microtopos/mosaic.py (owned by another slice).

Not collected by default (file name does not start with test_). Run: pytest tests/pending_neon_patches.py
When the patches in the docstrings below are applied, rename this file to test_neon_pending.py.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from microtopos.mosaic import Mosaic


# ----------------------------------------------------------------------------------------------- N6
def _texture_mosaic() -> Mosaic:
    rng = np.random.default_rng(1)
    big = cv2.GaussianBlur(rng.uniform(0, 255, (600, 1400)).astype(np.float32), (0, 0), 2.0)
    big = (big - big.min()) / (big.max() - big.min()) * 200 + 20
    m = Mosaic()
    for x in range(0, 481, 20):  # pan 480 px right in 20 px steps: 5 keyframes chained by edges
        m.track(big[100:340, x:x + 320].astype(np.uint8))
    return m


def test_relax_moves_the_recovery_reference_with_the_graph():
    """A loop edge that disagrees by 12 px makes _relax shift the whole current chain by ~9.6 px. The position of the
    last good frame (the reference for re-acquiring a lost track) must move with it: before the relax it equals the
    camera position, so it must still equal it afterwards. Observed on the stack: 480.09 vs 489.69.
    Patch: at the end of Mosaic._relax, next to the prev_xy update, add
        if self.last_good is not None:
            gf, gb, gxy = self.last_good
            self.last_good = (gf, gb, (gxy[0] + dxc, gxy[1] + dyc))"""
    m = _texture_mosaic()
    assert len(m.keyframes) == 5
    assert m.last_good[2] == pytest.approx((m.x, m.y))
    j = len(m.keyframes) - 1
    k0, kj = m.keyframes[0], m.keyframes[j]
    m.edges.append((0, j, kj.x - k0.x + 12.0, kj.y - k0.y, 1.0))
    before = (m.x, m.y)
    m._relax()
    assert abs(m.x - before[0]) > 5.0  # the relaxation really moved the camera
    assert m.last_good[2] == pytest.approx((m.x, m.y), abs=1e-6)


# ----------------------------------------------------------------------------------------------- N7
def _blob() -> np.ndarray:
    yy, xx = np.mgrid[0:101, 0:101].astype(np.float32)
    return (30 + 200 * np.exp(-((xx - 50) ** 2 + (yy - 50) ** 2) / (2 * 8.0**2))).astype(np.uint8)


def _centroid_error(fx: float, fy: float) -> tuple[float, float]:
    m = Mosaic()
    m._splat(_blob(), 100.0 + fx, 200.0 + fy)
    out, (ox, oy) = m.image()
    w = np.clip(np.nan_to_num(out) - 30.0, 0.0, None)
    yy, xx = np.mgrid[0 : out.shape[0], 0 : out.shape[1]]
    cx = (w * xx).sum() / w.sum() + ox
    cy = (w * yy).sum() / w.sum() + oy
    return cx - (150.0 + fx), cy - (250.0 + fy)


@pytest.mark.parametrize("frac", [0.25, 0.5, 0.75])
def test_splat_places_a_keyframe_at_its_subpixel_position(frac):
    """_splat rounds the position to a whole pixel (round(x)): a blob placed at +0.25/+0.5/+0.75 px has its mosaic
    centroid at 0 / 0 / +1 px, i.e. errors of 0.25, 0.5 and 0.25 px. Patch: resample the image, feather and mask
    by the fractional part (bilinear or cubic cv2.warpAffine) before accumulating, deterministically in (img, x, y)
    so that the weight=-1 un-splat in _relax stays exact; costs a little softening, so it needs a sharpness check."""
    ex, ey = _centroid_error(frac, frac)
    assert abs(ex) <= 0.1 and abs(ey) <= 0.1, (ex, ey)
