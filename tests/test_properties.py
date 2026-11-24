"""Property-based tests (hypothesis) for product code that has a non-trivial invariant to violate."""

from __future__ import annotations

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp

from microtopos.focus import brenner, tile_scores

_u8_images = hnp.arrays(np.uint8, st.tuples(st.integers(8, 40), st.integers(8, 40)), elements=st.integers(0, 255))


@given(img=_u8_images)
@settings(max_examples=60, deadline=None)
def test_brenner_same_image_any_storage_dtype(img):
    """The same pixel values stored as uint8, float32 or (x257) uint16 give the same score up to the stated scale."""
    ref = brenner(img.astype(np.float64))
    assert np.isclose(brenner(img), ref, rtol=1e-5, atol=1e-6)
    assert np.isclose(brenner(img.astype(np.uint16) * 257), ref * 257.0**2, rtol=1e-5, atol=1e-6)


@given(img=_u8_images, gain=st.sampled_from([0.25, 0.5, 2.0, 4.0]))
@settings(max_examples=60, deadline=None)
def test_brenner_energy_scales_with_gain_squared(img, gain):
    base = img.astype(np.float32)
    assert np.isclose(brenner(base * gain), brenner(base) * gain**2, rtol=1e-5, atol=1e-6)


@given(img=hnp.arrays(np.uint8, (24, 24), elements=st.integers(0, 200)), offset=st.integers(0, 55))
@settings(max_examples=60, deadline=None)
def test_brenner_and_tiles_ignore_a_dc_offset(img, offset):
    shifted = img + np.uint8(offset)  # stays <= 255, no clipping
    assert np.isclose(brenner(shifted), brenner(img), rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(tile_scores(shifted, 2), tile_scores(img, 2), rtol=1e-3, atol=1e-2)


@given(img=hnp.arrays(np.uint8, (24, 24), elements=st.integers(0, 255)))
@settings(max_examples=40, deadline=None)
def test_tile_scores_storage_dtype_invariant(img):
    ref = tile_scores(img.astype(np.float64), 2)
    # float32 working precision: a near-constant image has tiny band-pass energy next to a large DC level, so the
    # absolute tolerance is relative to the largest tile energy (a wrapped integer difference is off by orders more)
    atol = 1e-4 * max(float(ref.max()), 1.0)
    np.testing.assert_allclose(tile_scores(img, 2), ref, rtol=1e-3, atol=atol)
    np.testing.assert_allclose(tile_scores(img.astype(np.uint16) * 257, 2), ref * 257.0**2, rtol=1e-3, atol=atol * 257.0**2)
