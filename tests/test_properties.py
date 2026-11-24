"""Seeded property tests for the global focus scores: random inputs from numpy Generators, no hypothesis."""

from __future__ import annotations

import numpy as np
import pytest

from microtopos.focus import brenner, tile_scores

SEEDS = range(25)


def _u8(seed: int, shape=(24, 24), hi: int = 255) -> np.ndarray:
    rng = np.random.default_rng(seed)
    h, w = rng.integers(8, 41, 2) if shape is None else shape
    return rng.integers(0, hi + 1, (h, w)).astype(np.uint8)


@pytest.mark.parametrize("seed", SEEDS)
def test_brenner_same_image_any_storage_dtype(seed):
    """The same pixel values stored as uint8, float or (x257) uint16 give the same score up to the stated scale."""
    img = _u8(seed, None)
    ref = brenner(img.astype(np.float64))
    assert brenner(img) == pytest.approx(ref, rel=1e-5)
    assert brenner(img.astype(np.uint16) * 257) == pytest.approx(ref * 257.0**2, rel=1e-5)


@pytest.mark.parametrize("seed", SEEDS)
def test_brenner_energy_scales_with_gain_squared(seed):
    gain = (0.25, 0.5, 2.0, 4.0)[seed % 4]
    base = _u8(seed, None).astype(np.float32)
    assert brenner(base * gain) == pytest.approx(brenner(base) * gain**2, rel=1e-5)


@pytest.mark.parametrize("seed", SEEDS)
def test_brenner_and_tiles_ignore_a_dc_offset(seed):
    img = _u8(seed, hi=200)
    offset = int(np.random.default_rng(1000 + seed).integers(0, 56))
    shifted = img + np.uint8(offset)  # stays <= 255, no clipping
    assert brenner(shifted) == pytest.approx(brenner(img), rel=1e-5)
    np.testing.assert_allclose(tile_scores(shifted, 2), tile_scores(img, 2), rtol=1e-3, atol=1e-2)


@pytest.mark.parametrize("seed", SEEDS)
def test_tile_scores_storage_dtype_invariant(seed):
    img = _u8(seed)
    ref = tile_scores(img.astype(np.float64), 2)
    # float32 working precision: absolute tolerance relative to the largest tile energy (a wrapped integer
    # difference is off by orders of magnitude more)
    atol = 1e-4 * max(float(ref.max()), 1.0)
    np.testing.assert_allclose(tile_scores(img, 2), ref, rtol=1e-3, atol=atol)
    np.testing.assert_allclose(tile_scores(img.astype(np.uint16) * 257, 2), ref * 257.0**2, rtol=1e-3,
                               atol=atol * 257.0**2)