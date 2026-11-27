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


def _single_pass_with_end_layer(seed, n, w, gain, layer, decay, at_end, noise):
    """Tile focus curves of ONE monotonic pass over frames 0..n-1 (no turnaround): 36 tiles in focus at random frames
    inside the sweep, plus a layer that is sharp at one end of the sweep and decays over `decay` frames."""
    rng = np.random.default_rng(seed)
    z = np.arange(n, dtype=float)
    focus = rng.uniform(6 * w, n - 6 * w, 36)
    base = rng.uniform(0.5, 2.0, 36)
    tiles = base * (1 + gain * np.exp(-(z[:, None] - focus[None]) ** 2 / (2 * w * w)))
    d = (n - 1 - z) if at_end else z
    tiles = tiles + (layer * gain * np.exp(-d / decay))[:, None] * rng.uniform(0.5, 1.5, 36)[None]
    return tiles * np.exp(noise * rng.normal(size=tiles.shape))


# Single passes that find_reversal still takes for a turnaround (strict: a fix turns them red, so the entry goes with
# the fix). Both score under BOUNDARY_PEAK_MIRROR_MAX at or past the end midpoint; see its comment.
KNOWN_FALSE_TURNAROUNDS = {
    "n39-layer1.7": "accepted at frame 30 (end midpoint 29.5), mirror/travel 0.46",
    "n40-layer0.5": "accepted at frame 31 (end midpoint 30.0), mirror/travel 0.43",
}
SINGLE_PASSES = {
    # found by shrinking a random failure: accepted at frame 12, six frames before the end midpoint 18, which is a
    # lower bound of any turnaround that ends near focus; the search now stays within the midpoint tolerance of it
    "n25-layer0.31": (8, 25, 1.5, 5.0, 0.3125, 1.5, True, 0.0),
    # two of 20000 random draws whose verdict the bound changed: v0.10.1 accepted them at frames 12 and 16
    "n26-layer0.31": (10138, 26, 1.8084925014223954, 11.123667131887967, 0.3076532972307618, 4.351642135858976, True,
                      0.02937259458998135),
    "n31-layer0.46": (18726, 31, 2.241102290886313, 40.78915220600517, 0.45764315037898645, 1.9724137525035477, True,
                      0.046566725055764036),
    "n39-layer1.7": (273, 39, 2.785107701806346, 33.537211358410744, 1.7089067383076242, 1.6067971457612413, True,
                     0.0022289952694011296),
    "n40-layer0.5": (379, 40, 2.9679130100310047, 49.57424831708108, 0.49950011737170696, 2.5129747873108554, True,
                     0.016678346370918203),
}


@pytest.mark.parametrize("case", [pytest.param(k, marks=[pytest.mark.xfail(strict=True, reason=KNOWN_FALSE_TURNAROUNDS[k])]
                                               if k in KNOWN_FALSE_TURNAROUNDS else []) for k in SINGLE_PASSES])
def test_single_pass_with_a_decaying_end_layer_is_not_a_reversal(case):
    """Ground truth: one monotonic pass, so there is no turnaround to report."""
    assert sff.find_reversal(_single_pass_with_end_layer(*SINGLE_PASSES[case])) is None


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


# ------------------------------------------------------------------------------------------------ colour fusion
def _two_frame_stack(h: int, w: int):
    """Two integer-valued frames, each sharp on one half, and the complementary hard weight step."""
    rng = np.random.default_rng(0)
    tex = np.rint(rng.random((h, w)) * 255).astype(np.float32)
    left = np.arange(w)[None, :] < w // 2
    f0 = np.where(left, tex, 128.0).astype(np.float32)
    f1 = np.where(~left, tex, 128.0).astype(np.float32)
    w0 = np.broadcast_to(left, (h, w)).astype(np.float32)
    return (f0, f1), (w0, 1.0 - w0)


def test_colour_fusion_uses_the_same_pyramid_as_the_grey_fusion():
    """EDFAccumulator took its pyramid depth from min(shape), which is 3 for (H, W, 3): the colour all-in-focus image
    got 1 level instead of 5, so it was not the fusion the luma image got. Ground truth: frames whose three channels
    equal a grey frame must fuse to the grey fusion, channel by channel."""
    frames, weights = _two_frame_stack(256, 320)
    grey, colour = sff.EDFAccumulator((256, 320)), sff.EDFAccumulator((256, 320, 3))
    for f, wt in zip(frames, weights):
        grey.add(f, wt)
        colour.add(np.dstack([f] * 3), wt)
    rg, rc = grey.result(), colour.result()
    assert np.max(np.abs(rc - rg[..., None])) < 1e-3


OBJ10 = resolve("10x / 0.25", objset=BUILTIN)


def _grey_sweep(seed: int = 3):
    """A 9-frame simulated 10x sweep, 320 x 240, as uint8 grey frames, and its true focus positions."""
    sim = SpecimenSimulator(seed, OBJ10, out_w=320, out_h=240)
    frames, zs = [], []
    for z_k in sim.sweep_trajectory(n_frames=9):
        sim.focus_z_um = float(z_k)
        bgr, truth = sim.render()
        frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY))
        zs.append(truth.focus_z_um)
    return frames, np.array(zs)


def test_reconstructed_colour_edf_of_grey_frames_equals_the_grey_edf():
    """End to end through zstack.reconstruct: colour frames that are the luma frames replicated into three channels
    must give edf_rgb equal to the rounded luma edf in every channel."""
    frames, zs = _grey_sweep()
    r = zstack.reconstruct(frames, zs, OBJ10, 1.0, "um", color_frames=[np.dstack([f] * 3) for f in frames],
                           settings=zstack.ZSettings(allow_tiling=False))
    for c in range(3):
        assert np.max(np.abs(r["edf_rgb"][..., c].astype(np.float32) - r["edf"])) <= 0.5 + 1e-3


# ------------------------------------------------------------------------------------------------ compute backend
def test_refine_depth_reports_the_backend_its_guided_filter_ran_on():
    """refine_depth discarded the backend guided_filter_backend returned; it must hand it to the caller."""
    rng = np.random.default_rng(0)
    z = rng.normal(0, 1, (48, 64)).astype(np.float32)
    guide = rng.normal(0, 1, (48, 64)).astype(np.float32)
    out, ran = sff.refine_depth(z, np.ones_like(z), guide, 3, backend="cpu")
    assert ran == {"cpu"} and out.shape == z.shape


def test_refine_depth_on_opencl_reports_opencl_only(opencl):
    """Both guided-filter passes (before and after outlier rejection) ran on OpenCL: the set is exactly {'opencl'}."""
    rng = np.random.default_rng(0)
    z = rng.normal(0, 1, (48, 64)).astype(np.float32)
    guide = rng.normal(0, 1, (48, 64)).astype(np.float32)
    _out, ran = sff.refine_depth(z, np.ones_like(z), guide, 3, backend="opencl")
    assert ran == {"opencl"}


def test_result_backend_names_the_opencl_guided_filter(opencl, monkeypatch):
    """With OpenCL selected and a focus measure that has no OpenCL version (Vollath F4) the focus maps run on the CPU
    but the guided filter of the depth refinement runs on OpenCL; the result said 'cpu'. It must name both."""
    from microtopos import backend

    calls = []
    real = backend._guided_filter_umat
    monkeypatch.setattr(backend, "_guided_filter_umat", lambda *a, **k: (calls.append(1), real(*a, **k))[1])
    frames, zs = _grey_sweep()
    r = zstack.reconstruct(frames, zs, OBJ10, 1.0, "um",
                           settings=zstack.ZSettings(backend="opencl", measure="Vollath F4", workers=1))
    assert calls, "the guided filter did not run on OpenCL"
    assert r["backend"] == "cpu+opencl"
