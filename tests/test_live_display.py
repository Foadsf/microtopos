"""The live view's still average is aligned on the newest frame, so hand tremor does not blur it (display only)."""
import numpy as np
import pytest

from microtopos.calibration import CalibrationStore
from microtopos.live import LiveEngine, Override
from microtopos.objectives import resolve

H, W = 240, 320


def _texture(seed=4):
    r = np.random.default_rng(seed)
    fy, fx = np.fft.fftfreq(H)[:, None], np.fft.rfftfreq(W)[None, :]
    spec = np.fft.rfft2(r.normal(size=(H, W))) * np.exp(-2 * (np.pi * 1.5) ** 2 * (fx ** 2 + fy ** 2))

    def at(dx, dy):
        img = np.fft.irfft2(spec * np.exp(-2j * np.pi * (fx * dx + fy * dy)), s=(H, W))
        return 120 + 40 * img / img.std()
    return at


def _run(tremor_px, n=12, seed=4):
    """Still view with hand tremor (random walk of tremor_px per frame) and noise 4 DN; returns the engine's
    displayed preview and the noiseless newest frame."""
    at = _texture(seed)
    rng = np.random.default_rng(seed + 1)
    eng = LiveEngine(CalibrationStore(None), resolve("10x / 0.25"))
    eng.sm.override = Override.PAUSE  # DWELL: the averaging state
    pos = np.zeros(2)
    for i in range(n):
        pos += rng.normal(0, tremor_px, 2)
        clean = at(*pos)
        frame = np.clip(clean + rng.normal(0, 4.0, clean.shape) + 0.5, 0, 255).astype(np.uint8)
        eng.process_frame(np.dstack([frame] * 3), i * 0.04, i + 1)
    return eng.snapshot().preview, clean


def _rms(a, b):
    c = (slice(12, -12), slice(12, -12))
    return float(np.sqrt(np.mean((a[c] - b[c]) ** 2)))


@pytest.mark.parametrize("tremor", [0.4, 0.8])
def test_the_displayed_average_follows_a_trembling_view(tremor):
    """8 frames averaged: noise 4 DN falls towards 4/sqrt(8) = 1.4 DN; bilinear resampling of this sharp texture adds
    a little blur. Measured RMS against the noiseless newest frame: 2.19 DN (0.4 px tremor) and 2.26 DN (0.8 px);
    without alignment, or with the shift negated, 4.2-5.5 DN, worse than a single frame."""
    shown, clean = _run(tremor)
    assert _rms(shown, clean) < 2.5
    assert _rms(shown, clean) < 0.6 * 4.0  # better than any single frame


def test_a_still_view_averages_exactly_as_before():
    """Measured 1.54 DN against 4 / sqrt(8) = 1.41 DN (the first frames of the buffer are shifted by registration
    noise of ~0.01 px only)."""
    shown, clean = _run(0.0)
    assert _rms(shown, clean) == pytest.approx(4.0 / np.sqrt(8), rel=0.15)
