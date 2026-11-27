"""Regressions in reconstruction found by the v0.10.1 review (compared with v0.9.0), each against ground truth."""

import functools

import cv2
import numpy as np
import pytest

from microtopos import registration, sff, zstack
from microtopos.focus import tile_scores
from microtopos.objectives import resolve
from microtopos.selftest import BUILTIN, _frames
from microtopos.simulator import SpecimenSimulator

OBJ40 = resolve("40x / 0.85", objset=BUILTIN)


# ------------------------------------------------------------------------------------------------ reversal
def _render(seed: int, reversal: bool):
    """A simulated hand sweep at 40x as the GUI records it (tremor, noise, breathing on) and the true turnaround
    frame (None without a reversal)."""
    sim = SpecimenSimulator(seed, OBJ40)
    zs = sim.sweep_trajectory(reversal=reversal)
    frames, _ = _frames(sim, zs)
    turn = int(np.argmax(zs) if zs[1] > zs[0] else np.argmin(zs)) if reversal else None
    return sim, zs, frames, turn


@functools.cache
def _hand_sweep(seed: int, reversal: bool):
    """Tile vectors of that sweep, as zstack.reconstruct computes them, and the true turnaround."""
    *_, frames, turn = _render(seed, reversal)
    tiles = np.array([tile_scores(cv2.resize(f, (320, 180), interpolation=cv2.INTER_AREA).astype(np.float32), grid=6)
                      for f in frames])
    return tiles, turn


@pytest.mark.parametrize("seed", [14, 22, 24])
def test_reversed_40x_hand_sweep_that_stops_near_focus_is_detected(seed):
    """The user turns back and stops near focus: the return pass's focus peak is the last frame. v0.10.1 smoothed
    the tile curves with edge padding, so a curve still rising at the end has no peak there; too few tiles had two
    peaks and the turnaround was missed (v0.9.0 found it within 2 frames on these seeds)."""
    tiles, turn = _hand_sweep(seed, True)
    t = sff.find_reversal(tiles)
    assert t is not None and abs(t - turn) <= 2, (t, turn)


@pytest.mark.characterization("guards the false-reversal side of the fix: single passes already gave no reversal")
@pytest.mark.parametrize("seed", [11, 13, 15])
def test_single_40x_hand_sweep_reports_no_reversal(seed):
    tiles, _ = _hand_sweep(seed, False)
    assert sff.find_reversal(tiles) is None


def _one_pass(n=30):
    x = np.linspace(-2.5, 2.5, n)[:, None]
    offsets = np.linspace(-0.6, 0.6, 36)[None, :]
    return 100.0 * np.exp(-(x - offsets) ** 2) + 20.0  # 36 tiles in focus at frames 11..18


@pytest.mark.parametrize("back", [6, 7, 8])
@pytest.mark.parametrize("played_backwards", [False, True])
def test_return_pass_stopped_before_the_focus_planes_is_a_reversal(back, played_backwards):
    """Forward pass over frames 0..29 (tiles in focus at 11..18), then back over positions 28, 27, ... for `back`
    frames, stopping before the focus planes: every tile's curve is still rising at the last frame. The turnaround
    is frame 29. Played backwards, the sweep starts near focus and the turnaround is at len - 1 - 29."""
    one = _one_pass()
    tiles = np.vstack([one, one[::-1][1:back + 1]])
    turn = 29
    if played_backwards:
        tiles, turn = tiles[::-1], len(tiles) - 1 - 29
    t = sff.find_reversal(tiles)
    assert t is not None and abs(t - turn) <= 1, (t, turn)


@pytest.mark.characterization("v0.10.1 already rejected these; the boundary maximum the fix admits must not bring "
                              "back the false turnaround tests/test_reported_bugs.py B8 removed")
@pytest.mark.parametrize("amplitude", [30.0, 60.0, 300.0])
@pytest.mark.parametrize("at_end", [False, True])
def test_single_pass_with_a_sharp_layer_at_one_end_is_not_a_reversal(amplitude, at_end):
    """A single pass that starts (or ends) sharp on another layer has a prominent maximum at that end in every
    tile, like a return pass stopped near focus; the frames around the would-be turnaround do not mirror."""
    one = _one_pass()
    tiles = one + amplitude * np.exp(-np.arange(len(one), dtype=float)[:, None] / 1.5)
    assert sff.find_reversal(tiles[::-1] if at_end else tiles) is None


def test_reversed_40x_hand_sweep_reconstruction_matches_v090():
    """Seed 22 end to end, scored like the review: v0.9.0 valid 94.0 %, r 0.984, RMSE after a linear fit to truth
    0.133 um; v0.10.1 (turnaround missed, return pass folded into the depth fit) 63.5 %, 0.850, 0.302 um."""
    sim, zs, frames, turn = _render(22, True)
    _, gt = sim.truth_patch()
    sign = 1 if zs[1] > zs[0] else -1
    res = zstack.reconstruct(frames, None, OBJ40, OBJ40.nominal_um_per_px(frames[0].shape[1]), "frames",
                             settings=zstack.ZSettings(workers=2), sample_pos=np.arange(len(frames), dtype=float),
                             z_sign=sign)
    assert res["reversal_at"] is not None and abs(res["reversal_at"] - turn) <= 2
    h = res["height"]
    g = cv2.resize(gt, (h.shape[1], h.shape[0]), interpolation=cv2.INTER_AREA)
    c = (slice(16, -16), slice(16, -16))
    v = np.isfinite(h[c])
    hh, gg = h[c][v], g[c][v]
    a, b = np.polyfit(hh, gg, 1)
    rmse = float(np.sqrt(np.mean((a * hh + b - gg) ** 2)))
    assert v.mean() >= 0.93, v.mean()
    assert float(np.corrcoef(hh, gg)[0, 1]) >= 0.98
    assert rmse <= 0.14, rmse


# ------------------------------------------------------------------------------------------------ registration
def _crops(offsets, shape=(240, 320), seed=3):
    """Frames cut from one smooth random texture at integer offsets (x, y): frame k shows the texture moved by
    -offsets[k], so the warp of frame k relative to frame 0 is a translation of -(offsets[k] - offsets[0])."""
    rng = np.random.default_rng(seed)
    big = cv2.GaussianBlur(rng.normal(0, 1, (shape[0] + 120, shape[1] + 120)).astype(np.float32), (0, 0), 2.0)
    big = (128 + 40 * big / big.std()).astype(np.float32)
    return [big[60 + oy:60 + oy + shape[0], 60 + ox:60 + ox + shape[1]].copy() for ox, oy in offsets]


def test_register_pair_falls_back_to_phase_when_the_deformation_fit_fails(monkeypatch):
    """deformation() returns None by design when its affine ECC fit fails; v0.9.0 then rejected the ECC result and
    kept the phase-correlation shift. v0.10.1 called float(None) and raised TypeError out of the reconstruction."""
    t, m = _crops([(0, 0), (6, -4)])
    good = registration.register_pair(t, m)
    assert good.method == "ecc" and good.ok  # the pair reaches the deformation test (the gate under test)

    monkeypatch.setattr(registration, "deformation", lambda *a, **k: None)
    res = registration.register_pair(t, m)
    assert res.method == "phase" and not res.ok
    assert res.scale_step == 1.0  # not observed
    np.testing.assert_allclose(res.warp[:, :2], np.eye(2), atol=1e-6)
    np.testing.assert_allclose(res.warp[:, 2], [-6.0, 4.0], atol=0.25)


def test_register_sweep_completes_when_every_deformation_fit_fails(monkeypatch):
    offsets = [(4 * k, -2 * k) for k in range(6)]  # even: whole pixels at the half resolution phase runs at
    frames = _crops(offsets)
    monkeypatch.setattr(registration, "deformation", lambda *a, **k: None)
    warps, pairs = registration.register_sweep(frames, ref=0, workers=1)
    assert [p.method for p in pairs] == ["phase"] * 5
    got = np.array([w[:, 2] for w in warps])
    truth = -np.array(offsets, np.float64)
    np.testing.assert_allclose(got, truth, atol=0.5)


# ------------------------------------------------------------------------------------------------ DOF ruler
@pytest.mark.parametrize(("name", "um_per_px", "cycles_per_px"), [
    ("100x / 1.25 oil", 0.03, 1 / 16),  # FWHM 0.246 um: below two dof/3 steps (0.380 um), one sample above half
    ("100x / 1.25 oil", 0.03, 1 / 8),   # FWHM 0.123 um: about two thirds of one step
    ("40x / 0.85", 0.075, 1 / 8),       # FWHM 0.279 um, dof/3 = 0.283 um
])
def test_theoretical_fwhm_of_a_response_narrower_than_two_samples(name, um_per_px, cycles_per_px):
    """A pure sinusoid and a peak-to-peak measure respond as exp(-2 pi^2 f^2 s^2) to the extra blur
    s = |dz| tan(theta) / (2 um_per_px), so FWHM = 2 sqrt(2 ln 2) um_per_px / (pi f tan(theta)), tan(theta) =
    tan(asin(NA / n)). v0.10.1 samples defocus at dof/3 and required two samples above half maximum, so a narrow
    (high-NA) response returned None although its half maximum was bracketed."""
    import math

    from microtopos import objectives
    from microtopos.sff import theoretical_fwhm_um

    obj = objectives.resolve(name, objset=objectives.default_set())
    tan = math.tan(math.asin(obj.na / obj.immersion_n))
    hand = 2 * math.sqrt(2 * math.log(2)) * um_per_px / (math.pi * cycles_per_px * tan)
    assert hand < 2 * obj.dof_um / 3  # the case under test: narrower than two sampling steps
    x = np.arange(256)
    tex = np.tile((128 + 60 * np.cos(2 * np.pi * cycles_per_px * x)).astype(np.float32), (256, 1))
    got = theoretical_fwhm_um(obj, um_per_px, lambda im: np.full(im.shape, float(im.max() - im.min()), np.float32), tex)
    assert got is not None
    assert got == pytest.approx(hand, rel=0.01), (got, hand)
