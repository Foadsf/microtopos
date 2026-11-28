"""Super-resolution's motion gate: frames without sub-pixel motion between them are refused, whatever the noise.

Found on the real MikrOkular (2026-11-28): 16 frames of a microscope nobody touched registered with shifts of
0.00-0.01 px, yet the gate counted 3 of 4 sub-pixel phases and returned a resampled image as super-resolved.
"""
import numpy as np
import pytest

from microtopos.calibration import bgr_to_luma
from microtopos.objectives import resolve
from microtopos.simulator import SpecimenSimulator
from microtopos.superres import dwell_superres, subpixel_bins


@pytest.mark.parametrize("scale", [2, 3, 4])
def test_shifts_that_scatter_around_one_grid_position_are_one_phase(scale):
    """Registration noise of a few thousandths of a pixel on either side of zero is no motion: one phase."""
    shifts = [(0.0, 0.0), (0.004, -0.003), (-0.004, 0.002), (-0.002, -0.006), (0.006, 0.001)]
    assert subpixel_bins(shifts, scale) == 1
    # a view that jumped by whole pixels between frames gains no sub-pixel information either
    jumped = shifts + [(dx + 1, dy - 1) for dx, dy in shifts] + [(dx - 3, dy + 2) for dx, dy in shifts]
    assert subpixel_bins(jumped, scale) == 1


@pytest.mark.characterization("control for the rounding: shifts on every grid point were already counted correctly")
@pytest.mark.parametrize("scale", [2, 3, 4])
def test_shifts_on_every_grid_phase_cover_every_phase(scale):
    """One frame on each point of the 1/scale grid, each a little off it: scale*scale phases."""
    shifts = [(i / scale + 0.01, j / scale - 0.01) for i in range(scale) for j in range(scale)]
    assert subpixel_bins(shifts, scale) == scale * scale


def _frames(tremor_px: float, seed: int = 3, n: int = 16):
    obj = resolve("4x / 0.10")
    sim = SpecimenSimulator(seed, obj, 320, 180)
    sim.tremor_px = tremor_px
    _, hz = sim.truth_patch()
    sim.focus_z_um = float(np.median(hz))
    frames = [bgr_to_luma(sim.render(breathing=False)[0]) for _ in range(n)]
    return frames, obj.psf_sigma_px(sim.um_per_px)


@pytest.mark.parametrize("seed", [3, 11, 29])
def test_a_still_camera_is_refused(seed):
    """No tremor: the frames differ only by sensor noise, so the true motion is zero and nothing can be gained."""
    frames, psf = _frames(0.0, seed)
    with pytest.raises(ValueError, match="sub-pixel phases"):
        dwell_superres(frames, psf, scale=2)


@pytest.mark.characterization("control: hand tremor was already accepted; the stricter count must not refuse it")
def test_a_trembling_hand_is_accepted_and_covers_all_four_phases():
    """The control: 0.6 px of hand tremor spreads the frames over the 2x grid, and the gate lets them through."""
    frames, psf = _frames(0.6)
    r = dwell_superres(frames, psf, scale=2)
    assert r["subpixel_bins"] == 4 and r["frames_used"] == 16
