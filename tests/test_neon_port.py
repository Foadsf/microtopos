"""Findings of the neon bug hunt that matter on this stack.

N5: empty or mismatched sweeps are rejected with a clear ValueError (pinned here).
N6: a graph relaxation moves the recovery reference together with the camera (pinned here).
N7: keyframes are placed on whole pixels; the sub-pixel error is bounded and documented in the README (pinned here
as a characterization of the current limit, not as a fix).
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from microtopos import objectives, zstack
from microtopos.mosaic import Mosaic

OBJ = objectives.resolve("10x / 0.25", objset=objectives.default_set())


# ----------------------------------------------------------------------------------------------- N5
@pytest.mark.characterization("zstack.reconstruct already refused empty and mismatched sweeps with a ValueError")
@pytest.mark.parametrize(
    ("frames", "labels", "match"),
    [
        ([], None, "empty sweep"),
        ([np.zeros((48, 64), np.float32)], np.array([0.0]), "Too few usable frames"),
        ([np.zeros((48, 64), np.float32), np.zeros((40, 64), np.float32)], np.array([0.0, 1.0]), "same size"),
        ([np.zeros((48, 64), np.float32)] * 2, np.array([0.0]), "z labels"),
    ],
)
def test_reconstruct_rejects_degenerate_sweeps_with_a_clear_error(frames, labels, match):
    with pytest.raises(ValueError, match=match):
        zstack.reconstruct(frames, labels, OBJ, 1.0, "um")


# ----------------------------------------------------------------------------------------------- N6
def _texture_mosaic() -> Mosaic:
    rng = np.random.default_rng(1)
    big = cv2.GaussianBlur(rng.uniform(0, 255, (600, 1400)).astype(np.float32), (0, 0), 2.0)
    big = (big - big.min()) / (big.max() - big.min()) * 200 + 20
    m = Mosaic()
    for x in range(0, 481, 20):  # pan 480 px right in 20 px steps: 5 keyframes chained by edges
        m.track(big[100:340, x:x + 320].astype(np.uint8))
    return m


@pytest.mark.characterization("the mosaic of this stack already moves last_good with the relaxation; pinned so it stays")
def test_relax_moves_the_recovery_reference_with_the_graph():
    """A loop edge that disagrees by 12 px makes _relax shift the whole current chain. The position of the last good
    frame (the reference for re-acquiring a lost track) equals the camera position before the relax and must still
    equal it afterwards."""
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


@pytest.mark.characterization("whole-pixel keyframe placement is a documented limit; this bounds it, it does not fix it")
@pytest.mark.parametrize("frac", [0.25, 0.5, 0.75])
def test_a_keyframe_lands_within_half_a_pixel_of_its_position(frac):
    """_splat rounds the position to a whole pixel, so a blob placed at +0.25 / +0.5 / +0.75 px has its mosaic centroid
    off by 0.25 / 0.5 / 0.25 px. A resampling fix would have to keep the weight=-1 un-splat in _relax exact."""
    ex, ey = _centroid_error(frac, frac)
    assert abs(ex) <= 0.5 + 1e-6 and abs(ey) <= 0.5 + 1e-6, (ex, ey)
