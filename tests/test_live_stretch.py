"""The live view's optional brightness stretch (display only)."""

import numpy as np

from microtopos.gui.widgets import STRETCH_MIN_SPAN, frame_percentile, stretch_levels


def _dim_frame(seed=0, base=1.0, amp=3.0, shape=(540, 960)):
    """A frame like the dark camera's (mean ~1 DN, max a few DN) with real texture underneath."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[: shape[0], : shape[1]]
    texture = amp * (0.5 + 0.5 * np.sin((xx + 0.4 * yy) / 6.0))
    return np.clip(np.round(base + texture + rng.normal(0, 0.3, shape)), 0, 255).astype(np.float32)


def test_dim_frame_is_spread_over_its_own_range():
    f = _dim_frame()
    lo, hi = stretch_levels(f)
    assert 0 <= lo <= 2 and 4 <= hi <= 6, (lo, hi)
    # the texture now covers about half the grey scale instead of about 1 % of it
    shown = np.clip((f - lo) / (hi - lo), 0, 1)
    spread = np.percentile(shown, 95) - np.percentile(shown, 5)
    plain = (np.percentile(f, 95) - np.percentile(f, 5)) / 255
    assert spread >= 0.5 and spread > 30 * plain, (spread, plain)


def test_bright_frame_is_left_close_to_full_range():
    rng = np.random.default_rng(1)
    f = rng.uniform(0, 255, (540, 960)).astype(np.float32)
    lo, hi = stretch_levels(f)
    assert lo < 5 and hi > 250


def test_flat_frame_keeps_a_minimum_span():
    """A uniform frame must not have its sensor noise blown up to full contrast."""
    f = np.full((540, 960), 1.0, np.float32)
    lo, hi = stretch_levels(f)
    assert hi - lo == STRETCH_MIN_SPAN and lo >= 0


def test_levels_are_smoothed_between_frames():
    a, b = _dim_frame(base=1.0), _dim_frame(base=40.0)
    la = stretch_levels(a)
    lb = stretch_levels(b)
    mixed = stretch_levels(b, la)
    assert la[0] < mixed[0] < lb[0] and la[1] < mixed[1] < lb[1]


def test_does_not_modify_the_frame():
    f = _dim_frame()
    before = f.copy()
    stretch_levels(f)
    assert np.array_equal(f, before)


def test_percentile_ignores_nan():
    f = _dim_frame()
    f[:100] = np.nan
    assert np.isfinite(frame_percentile(f, 99))
