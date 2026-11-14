"""Striped specimens (machined or lapped surfaces: parallel lines with grain between them), as seen on the real
camera on 2026-11-14. The stripes alone repeat; the grain does not. Mosaic must track a hand pan that starts from
rest, and recover when the view comes back over a mapped area, from the grain. With no grain at all the position
along the stripes is unknowable: then it must refuse, never report a confident wrong position.

Measured on main 4ea87a7 (before the fix), period 45 px, grain 2 % of the stripe amplitude, angle 80 deg: a pan
from rest at 10 px/frame lost all 25 frames, at 14 and 20 px/frame it tracked 140 and 412 px wrong, and a return
after a loss started a separate area instead of recovering."""

import math

import cv2
import numpy as np
import pytest

from microtopos.mosaic import Mosaic

VIEW_H, VIEW_W = 360, 640  # the live preview size Mosaic is fed


def _world(seed, period, angle_deg, grain, h=900, w=2000):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    a = math.radians(angle_deg)
    u = xx * math.cos(a) + yy * math.sin(a)
    t = cv2.GaussianBlur(rng.normal(0, 1, (h, w)).astype(np.float32), (0, 0), 1.5)
    return 110 + 45 * np.sin(2 * np.pi * u / period) + grain * 45 * t / t.std(), rng


def _view(world, rng, x, y):
    return world[int(y):int(y) + VIEW_H, int(x):int(x) + VIEW_W] + rng.normal(0, 2, (VIEW_H, VIEW_W)).astype(
        np.float32)


def _pan_from_rest(world, rng, step, n=25, x0=100.0, y0=100.0, direction=(1.0, 0.0)):
    m = Mosaic()
    for _ in range(8):  # dwell: the mosaic starts from a still view
        m.track(_view(world, rng, x0, y0))
    lost, errs = 0, []
    for k in range(1, n + 1):
        x, y = x0 + direction[0] * step * k, y0 + direction[1] * step * k
        m.track(_view(world, rng, x, y))
        if m.lost:
            lost += 1
        else:
            errs.append(math.hypot(m.x - (int(x) - int(x0)), m.y - (int(y) - int(y0))))
    return m, lost, errs


CASES = [(seed, period, angle, grain) for seed, period, angle, grain in
         [(1, 45, 80, 0.02), (2, 45, 80, 0.07), (3, 30, 20, 0.02), (4, 60, 135, 0.04), (5, 24, 60, 0.03)]]


@pytest.mark.parametrize("step", [6, 10, 14, 20])
@pytest.mark.parametrize("seed,period,angle,grain", CASES, ids=lambda v: str(v))
def test_pan_from_rest_over_stripes_with_grain(seed, period, angle, grain, step):
    world, rng = _world(seed, period, angle, grain)
    m, lost, errs = _pan_from_rest(world, rng, step, direction=(0.8, 0.6))
    assert lost == 0, f"lost {lost}/25 frames"
    assert max(errs) < 1.0, f"max error {max(errs):.2f} px"
    assert m.areas == 1


@pytest.mark.parametrize("seed,period,angle,grain", CASES, ids=lambda v: str(v))
def test_return_after_loss_recovers_over_stripes_with_grain(seed, period, angle, grain):
    world, rng = _world(seed, period, angle, grain)
    m, _, _ = _pan_from_rest(world, rng, 5, n=20)
    x_end = 100 + 5 * 20
    other = 110 + 30 * cv2.GaussianBlur(rng.normal(0, 1, (VIEW_H, VIEW_W)).astype(np.float32), (0, 0), 2) / 0.1
    for _ in range(5):  # an unrelated view: tracking is lost
        m.track(other + rng.normal(0, 2, other.shape).astype(np.float32))
    assert m.lost
    for k in range(10):  # back over the mapped strip
        m.track(_view(world, rng, x_end - 30, 100))
        if not m.lost:
            break
    assert not m.lost, "did not recover within 10 frames over a mapped area"
    assert m.areas == 1
    assert abs(m.x - (x_end - 30 - 100)) < 1.0 and abs(m.y) < 1.0, (m.x, m.y)


@pytest.mark.parametrize("step", [10, 14, 20])
@pytest.mark.parametrize("seed,angle", [(11, 80), (12, 30)])
def test_pure_stripes_are_refused_not_misplaced(seed, angle, step):
    """Negative control: without grain the shift along the stripes is not in the data. Any frame the mosaic does
    place must be placed right; refusing (lost) is the correct answer. (Before main 4ea87a7 such pans were tracked
    150-500 px wrong.)"""
    world, rng = _world(seed, 45, angle, 0.0)
    _, _, errs = _pan_from_rest(world, rng, step, direction=(0.8, 0.6))
    assert not errs or max(errs) < 2.0, f"placed frames up to {max(errs):.1f} px wrong"
