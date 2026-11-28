"""Super-resolution is refused where the camera already samples finer than the optics resolve.

The MikrOkular puts 3.0 um pixels behind the eyepiece: 0.734 um/px measured at 4x, so every objective is sampled
about twice as finely as its incoherent cutoff 2 NA / lambda needs. A 2x drizzle of such frames holds interpolated
values only; dwell_superres must say so instead of returning them as a finer image.
"""
import numpy as np
import pytest

from microtopos.calibration import bgr_to_luma
from microtopos.objectives import resolve
from microtopos.optics import WAVELENGTH_UM
from microtopos.simulator import SpecimenSimulator
from microtopos.superres import dwell_superres, max_useful_scale


def _frames(tremor_px: float, seed: int = 3, n: int = 16):
    """16 in-focus simulator frames at 4x / 0.10, 320x180 (4.5 um/px), with hand tremor of tremor_px."""
    obj = resolve("4x / 0.10")
    sim = SpecimenSimulator(seed, obj, 320, 180)
    sim.tremor_px = tremor_px
    _, hz = sim.truth_patch()
    sim.focus_z_um = float(np.median(hz))
    return [bgr_to_luma(sim.render(breathing=False)[0]) for _ in range(n)], obj.psf_sigma_px(sim.um_per_px)


@pytest.mark.parametrize("name,na", [("4x / 0.10", 0.10), ("10x / 0.25", 0.25), ("40x / 0.85", 0.85)])
def test_nyquist_pixel_is_lambda_over_4na(name, na):
    obj = resolve(name)
    assert obj.nyquist_pixel_um == pytest.approx(WAVELENGTH_UM / (4 * na), rel=1e-12)
    assert obj.optical_oversampling(obj.nyquist_pixel_um / 2) == pytest.approx(2.0, rel=1e-12)


def test_the_mikrokular_at_4x_oversamples_the_optics():
    """0.734 um/px measured at 4x / 0.10 (stage travel, 2026-11-14): lambda / 4NA = 1.375 um, 1.87x oversampled."""
    assert resolve("4x / 0.10").optical_oversampling(0.734) == pytest.approx(1.375 / 0.734, rel=1e-9)


@pytest.mark.parametrize("osr,scale", [(1.87, 1), (1.0, 1), (0.99, 2), (0.5, 2), (0.49, 3), (1 / 3, 3), (0.31, 4),
                                       (0.1, 4)])
def test_max_useful_scale_is_the_nyquist_grid(osr, scale):
    assert max_useful_scale(osr) == scale


def test_an_oversampled_camera_is_refused_super_resolution():
    """The frames carry sub-pixel tremor (the motion gate would pass them), but the optics are already resolved."""
    frames, psf = _frames(0.6)
    with pytest.raises(ValueError, match="adds no detail"):
        dwell_superres(frames, psf, scale=2, oversampling=1.87)


def test_an_undersampled_camera_keeps_super_resolution():
    """The simulator's 320 px frames at 4x are 4.5 um/px, oversampling 0.31: 2x is below the useful 4x and runs."""
    frames, psf = _frames(0.6)
    osr = resolve("4x / 0.10").optical_oversampling(4.5)
    r = dwell_superres(frames, psf, scale=2, oversampling=osr)
    assert r["scale_linear"] == 2 and r["subpixel_bins"] == 4
