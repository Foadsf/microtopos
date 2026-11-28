"""Measured camera response (microtopos.response) against a synthetic camera whose response is known exactly.

The synthetic camera: scene radiance E times exposure t goes through a known tone curve f (sRGB, or a power law
1/1.8 with a toe), gets Gaussian read noise in DN, is rounded to 8 bit and clipped at 0 and 255. The exposure series
uses this camera's real UVC settings (exposure_time_absolute in 100 us units, 78..738), so the span is the 9.5x the
hardware allows. Ground truth for the measured response is f^-1, normalised the same way (1 at the curve's
reference DN).
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from microtopos import response
from microtopos.response import ResponseCurve, estimate_response, merge_hdr, tonemap_for_display

UVC_UNITS = np.array([78, 110, 156, 220, 312, 440, 620, 738])
EXPOSURES_S = UVC_UNITS * 1e-4
H, W = 240, 320
DN = np.arange(256, dtype=np.float64)


def srgb_encode(x):
    x = np.clip(x, 0.0, None)
    return np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.power(x, 1 / 2.4) - 0.055)


def srgb_decode(v):
    return np.where(v <= 0.04045, v / 12.92, np.power((v + 0.055) / 1.055, 2.4))


_A = 0.01  # toe of the second camera: v = ((x + a)^(1/1.8) - a^(1/1.8)) / ((1 + a)^(1/1.8) - a^(1/1.8))
_A0, _A1 = _A ** (1 / 1.8), (1 + _A) ** (1 / 1.8)


def toe_encode(x):
    return (np.power(np.clip(x, 0.0, None) + _A, 1 / 1.8) - _A0) / (_A1 - _A0)


def toe_decode(v):
    return np.power(v * (_A1 - _A0) + _A0, 1.8) - _A


def gamma18_encode(x):
    return np.power(np.clip(x, 0.0, None), 1 / 1.8)


def gamma18_decode(v):
    return np.power(v, 1.8)


CAMERAS = {"srgb": (srgb_encode, srgb_decode), "gamma1.8-toe": (toe_encode, toe_decode)}


def shoot(encode, radiance, t, rng, sigma=0.7):
    v = 255.0 * encode(radiance * t) + rng.normal(0.0, sigma, radiance.shape)
    return np.clip(np.round(v), 0, 255).astype(np.uint8)


def wide_scene(rng, span=1000.0, top=40.0):
    """Log gradient over `span` along x with a 30 % texture; at the longest exposure the bright end is clipped."""
    y, x = np.mgrid[0:H, 0:W]
    e = np.exp(np.log(1.0 / span) * (1.0 - x / (W - 1))) * (1.0 + 0.3 * np.sin(y / 5.0) * np.sin(x / 7.0))
    return e * top * (1.0 + 0.02 * rng.standard_normal((H, W)))


def true_inverse(decode, reference_dn):
    g = decode(DN / 255.0)
    return g / g[reference_dn]


def max_rel_error(lut, truth, lo=10, hi=245):
    return float(np.max(np.abs(lut[lo:hi + 1] / truth[lo:hi + 1] - 1.0)))


def gamma22_lut(reference_dn):
    g = np.power(DN / 255.0, 2.2)
    return g / g[reference_dn]


@pytest.fixture(scope="module")
def srgb_series():
    rng = np.random.default_rng(7)
    scene = wide_scene(rng)
    frames = [shoot(srgb_encode, scene, t, rng) for t in EXPOSURES_S]
    return scene, frames, estimate_response(frames, EXPOSURES_S)


@pytest.mark.parametrize("camera", sorted(CAMERAS))
def test_recovers_the_inverse_response_and_gamma22_does_not(camera):
    """Max relative error < 3 % between 10 and 245 DN after normalisation; a gamma-2.2 curve fails the same check."""
    encode, decode = CAMERAS[camera]
    rng = np.random.default_rng(11)
    frames = [shoot(encode, wide_scene(rng), t, rng) for t in EXPOSURES_S]
    curve = estimate_response(frames, EXPOSURES_S)
    assert curve.channels == "luma" and curve.lut.shape == (1, 256)
    lo, hi = curve.observed_range[0]
    assert lo <= 10 and hi >= 245, curve.observed_range
    assert curve.reference_dn == 128 and curve.lut[0, 128] == 1.0
    truth = true_inverse(decode, curve.reference_dn)
    err = max_rel_error(curve.lut[0].astype(np.float64), truth)
    assert err < 0.03, f"{camera}: measured response is {err:.2%} off f^-1"
    assert max_rel_error(gamma22_lut(curve.reference_dn), truth) > 0.10  # the assumption being replaced
    assert np.all(np.diff(curve.lut[0]) >= 0)


def test_bgr_curves_recover_each_channel_with_different_channel_levels():
    rng = np.random.default_rng(5)
    scene = wide_scene(rng)
    gains = (0.55, 1.0, 0.75)  # blue, green, red scene levels differ (white balance)
    frames = [np.dstack([shoot(srgb_encode, scene * k, t, rng) for k in gains]) for t in EXPOSURES_S]
    curve = estimate_response(frames, EXPOSURES_S)
    assert curve.channels == "bgr" and curve.lut.shape == (3, 256)
    truth = true_inverse(srgb_decode, curve.reference_dn)
    for c in range(3):
        assert max_rel_error(curve.lut[c].astype(np.float64), truth) < 0.03, c


def test_gamma_equivalent_of_a_pure_power_law_camera():
    rng = np.random.default_rng(3)
    frames = [shoot(gamma18_encode, wide_scene(rng), t, rng) for t in EXPOSURES_S]
    curve = estimate_response(frames, EXPOSURES_S)
    assert curve.gamma_equivalent() == pytest.approx(1.8, abs=0.02)


def test_linearized_exposures_give_the_true_exposure_ratio(srgb_series):
    _, frames, curve = srgb_series
    for a, b in ((1, 6), (0, 7), (2, 4)):
        lin_a, lin_b = curve.linearize(frames[a]), curve.linearize(frames[b])
        both = (frames[a] >= 20) & (frames[a] <= 240) & (frames[b] >= 20) & (frames[b] <= 240)
        assert both.sum() > 2000
        ratio = float(np.median(lin_b[both] / lin_a[both]))
        true = EXPOSURES_S[b] / EXPOSURES_S[a]
        assert ratio == pytest.approx(true, rel=0.02), (a, b, ratio, true)


def test_linearize_float_input_interpolates_the_lut(srgb_series):
    _, frames, curve = srgb_series
    img = frames[4]
    np.testing.assert_array_equal(curve.linearize(img.astype(np.float32)), curve.linearize(img))
    half = curve.linearize(np.array([100.5, 200.25], np.float64))
    lut = curve.lut[0].astype(np.float64)
    np.testing.assert_allclose(half, [0.5 * (lut[100] + lut[101]), 0.75 * lut[200] + 0.25 * lut[201]], rtol=1e-6)


def _step_wedge(levels, size=24):
    """Square patches of constant radiance, one per level, row-major."""
    cols = 8
    rows = int(np.ceil(len(levels) / cols))
    img = np.zeros((rows * size, cols * size))
    ids = -np.ones(img.shape, int)
    for k, v in enumerate(levels):
        r, c = divmod(k, cols)
        img[r * size:(r + 1) * size, c * size:(c + 1) * size] = v
        ids[r * size:(r + 1) * size, c * size:(c + 1) * size] = k
    return img, ids


def _true_curve():
    """The synthetic camera's exact inverse response, declared known at every level (0 and 255 included), so that
    only the merge's hat weight, not the curve's measured mask, keeps the clipped values out."""
    g = srgb_decode(DN / 255.0)
    return ResponseCurve(lut=(g / g[128])[None], channels="luma", exposures_s=tuple(EXPOSURES_S),
                         observed_range=((0, 255),), reference_dn=128, method="ground truth", inconsistency_dn=0.0)


def _wedge_bracket():
    """A 1000:1 step wedge plus a specular patch; every exposure clips somewhere, the brightest patch clips in all."""
    rng = np.random.default_rng(21)
    levels = list(np.geomspace(0.04, 40.0, 31)) + [400.0]  # 1000:1 measurable range plus one specular highlight
    scene, ids = _step_wedge(levels)
    frames = [shoot(srgb_encode, scene, t, rng) for t in EXPOSURES_S]
    return np.array(levels), scene, ids, frames


def _patch_errors(radiance, ids, levels, scale):
    errs = np.full(len(levels), np.nan)
    for k, v in enumerate(levels):
        vals = radiance[ids == k]
        if np.all(np.isfinite(vals)):
            errs[k] = np.mean(vals) * scale / v - 1.0
    return errs


def test_merge_hdr_recovers_a_1000_to_1_wedge_and_ignores_clipped_pixels(monkeypatch):
    levels, scene, ids, frames = _wedge_bracket()
    for f in frames:
        assert (f == 255).any()  # every exposure clips somewhere
    x128 = float(srgb_decode(128 / 255.0))  # g(128) = 1, so radiance * x128 is scene radiance
    curve = _true_curve()
    out = merge_hdr(frames, EXPOSURES_S, curve)
    assert out["radiance"].dtype == np.float32
    # Usable exactly where the shortest exposure is below the clip level: the specular patch only is lost.
    expect_unusable = 255.0 * srgb_encode(scene * EXPOSURES_S.min()) >= response.CLIP_DN
    np.testing.assert_array_equal(out["unusable"], expect_unusable)
    assert expect_unusable.any() and np.all(np.isnan(out["radiance"][expect_unusable]))
    assert np.all(out["weight"][expect_unusable] == 0) and np.all(out["weight"][~expect_unusable] > 0)
    errs = _patch_errors(out["radiance"], ids, levels, x128)
    usable = np.isfinite(errs)
    assert usable.sum() == 31
    assert np.max(np.abs(errs[usable])) < 0.01, errs
    # Negative control: without the hat weight the clipped values of the long exposures bias the bright patches.
    clipped_long = np.array([255.0 * srgb_encode(v * EXPOSURES_S.max()) >= 255 for v in levels]) & usable
    assert clipped_long.sum() >= 5
    monkeypatch.setattr(response, "_hat", lambda black_dn, clip_dn: np.ones(256))
    flat = _patch_errors(merge_hdr(frames, EXPOSURES_S, curve)["radiance"], ids, levels, x128)
    assert np.max(np.abs(flat[clipped_long])) > 0.05, flat


def test_merge_hdr_with_the_measured_curve(srgb_series):
    _, _, curve = srgb_series
    levels, _, ids, frames = _wedge_bracket()
    x_ref = float(srgb_decode(curve.reference_dn / 255.0))
    errs = _patch_errors(merge_hdr(frames, EXPOSURES_S, curve)["radiance"], ids, levels, x_ref)
    assert np.isfinite(errs).sum() == 31
    assert np.nanmax(np.abs(errs)) < 0.03, errs


def test_unobserved_levels_are_not_presented_as_measured():
    """A dim series (as this camera delivers) measures only the low levels; the rest is marked and not merged."""
    rng = np.random.default_rng(13)
    scene = wide_scene(rng, span=200.0, top=0.6)
    frames = [shoot(srgb_encode, scene, t, rng) for t in EXPOSURES_S]
    top_dn = max(int(f.max()) for f in frames)
    assert top_dn < 120
    curve = estimate_response(frames, EXPOSURES_S)
    lo, hi = curve.observed_range[0]
    assert hi <= top_dn
    assert lo <= curve.reference_dn <= hi < 128  # 128 was not observed, so it is not the reference
    assert curve.lut[0, curve.reference_dn] == 1.0
    assert not curve.measured[0, hi + 1:].any() and curve.measured[0, lo:hi + 1].all()
    truth = true_inverse(srgb_decode, curve.reference_dn)
    assert max_rel_error(curve.lut[0].astype(np.float64), truth, lo=10, hi=hi - 5) < 0.03
    # A pixel brighter than anything measured gets no merge weight, so it is reported unusable, not guessed.
    bright = np.full((4, 4), 200, np.uint8)
    out = merge_hdr([bright] * 3, EXPOSURES_S[:3], curve)
    assert out["unusable"].all()


def test_refuses_two_exposures_and_a_narrow_span():
    rng = np.random.default_rng(1)
    scene = wide_scene(rng)
    frames = [shoot(srgb_encode, scene, t, rng) for t in EXPOSURES_S]
    with pytest.raises(ValueError, match="distinct exposure"):
        estimate_response([frames[0], frames[7], frames[0], frames[7]], EXPOSURES_S[[0, 7, 0, 7]])
    with pytest.raises(ValueError, match="span"):
        estimate_response(frames[:4], EXPOSURES_S[:4])  # 220 / 78 = 2.8x


def test_refuses_a_scene_that_drifts_between_exposures():
    rng = np.random.default_rng(2)
    scene = wide_scene(rng) * (1.0 + 0.5 * (rng.random((H, W)) > 0.5))
    frames = [shoot(srgb_encode, np.roll(scene, 6 * k, axis=1), t, rng) for k, t in enumerate(EXPOSURES_S)]
    with pytest.raises(ValueError):
        estimate_response(frames, EXPOSURES_S)


def test_refuses_one_moved_frame_among_still_ones():
    """The still-scene check is per exposure: one frame of another view is not outvoted by seven still ones."""
    rng = np.random.default_rng(4)
    scene = wide_scene(rng) * (1.0 + 0.5 * (rng.random((H, W)) > 0.5))
    frames = [shoot(srgb_encode, scene[::-1] if k == 4 else scene, t, rng) for k, t in enumerate(EXPOSURES_S)]
    with pytest.raises(ValueError, match="DN \\(median\\)"):
        estimate_response(frames, EXPOSURES_S)
    still = [shoot(srgb_encode, scene, t, rng) for t in EXPOSURES_S]
    assert estimate_response(still, EXPOSURES_S).inconsistency_dn < 1.0


def test_json_round_trip_is_exact(srgb_series):
    _, frames, curve = srgb_series
    back = ResponseCurve.from_json(json.loads(json.dumps(curve.to_json())))
    assert back.lut.dtype == np.float32
    np.testing.assert_array_equal(back.lut, curve.lut)
    assert back.lut.tobytes() == curve.lut.tobytes()
    assert (back.channels, back.exposures_s, back.observed_range, back.reference_dn, back.method) == (
        curve.channels, curve.exposures_s, curve.observed_range, curve.reference_dn, curve.method)
    assert back.inconsistency_dn == curve.inconsistency_dn
    np.testing.assert_array_equal(back.linearize(frames[3]), curve.linearize(frames[3]))
    with pytest.raises(ValueError, match="measured camera response"):
        ResponseCurve.from_json({**curve.to_json(), "kind": "gamma"})


def test_tonemap_is_monotonic_and_blanks_unusable_pixels():
    r = np.geomspace(1e-3, 1.0, 1000).astype(np.float32).reshape(10, 100)
    r[0, :3] = np.nan
    out = tonemap_for_display(r)
    assert out.dtype == np.uint8
    assert np.all(out[0, :3] == 0)
    flat = out.ravel()[3:]
    assert np.all(np.diff(flat.astype(int)) >= 0) and flat[0] == 0 and flat[-1] == 255
    # log scaling: the middle of the log range maps to the middle of the display range
    assert abs(int(out.ravel()[500]) - 128) <= 2
