"""Exact-value and dtype-invariance tests for the global focus scores (integer frames must not wrap)."""

from __future__ import annotations

import numpy as np
import pytest

from microtopos.focus import MEASURES, brenner, focus_map, tile_scores


def _step_u8() -> np.ndarray:
    a = np.zeros((4, 8), np.uint8)
    a[:, 4:] = 255
    return a


def test_brenner_exact_step_uint8():
    # columns 0..3 = 0, 4..7 = 255: d = I[x+2]-I[x] is 255 for x = 2,3 and 0 elsewhere (6 diffs per row)
    assert brenner(_step_u8()) == pytest.approx(2 * 255.0**2 / 6, rel=1e-6)


def test_brenner_exact_descending_uint8_no_underflow():
    # a falling edge makes the uint8 difference negative (wraps to +1 under modulo 256): 3 diffs of -200, rest 0
    a = np.full((2, 5), 10, np.uint8)
    a[:, :2] = 210
    # columns [210,210,10,10,10]: d = [-200,-200,0] -> mean(d^2) = 2*200^2/3
    assert brenner(a) == pytest.approx(2 * 200.0**2 / 3, rel=1e-6)


@pytest.mark.parametrize("fn", [brenner, lambda x: float(tile_scores(x, grid=2).sum())])
def test_global_scores_dtype_invariant(fn):
    rng = np.random.default_rng(3)
    u8 = rng.integers(0, 256, (48, 48)).astype(np.uint8)
    f32 = u8.astype(np.float32)
    u16 = u8.astype(np.uint16) * 257  # full-range 16-bit storage of the same image: scale 257, energy 257**2
    ref = fn(f32)
    assert fn(u8) == pytest.approx(ref, rel=1e-5)
    assert fn(u16) == pytest.approx(ref * 257.0**2, rel=1e-5)


def test_tile_scores_exact_uint8_vs_float():
    rng = np.random.default_rng(4)
    u8 = rng.integers(0, 256, (48, 48)).astype(np.uint8)
    np.testing.assert_allclose(tile_scores(u8, 2), tile_scores(u8.astype(np.float64), 2), rtol=1e-5)


@pytest.mark.characterization("focus_map already converted to float32 before differencing; brenner and tile_scores did not")
@pytest.mark.parametrize("measure", MEASURES)
def test_focus_map_dtype_invariant(measure):
    rng = np.random.default_rng(5)
    u8 = rng.integers(0, 256, (40, 40)).astype(np.uint8)
    ref = focus_map(u8.astype(np.float32), measure)
    np.testing.assert_allclose(focus_map(u8, measure), ref, rtol=1e-4, atol=1e-3)
    u16 = focus_map(u8.astype(np.uint16) * 257, measure)
    scale = 257.0 if measure == "SML" else 257.0**2
    np.testing.assert_allclose(u16, ref * scale, rtol=1e-3, atol=1e-2 * scale)
