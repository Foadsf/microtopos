"""Lateral chromatic aberration estimate (microtopos.chromatic) against synthetic frames with a known aberration.

Ground truth: a colour specimen (band-limited luminance texture plus sharp-edged disks of independent random
colour) is defined on a 4x finer grid. Each colour plane of a frame is resampled there with the aberration's
geometry (bicubic on the fine grid) and then area-averaged 4x to the frame's pixels, so the aberration is imposed by
pixel integration of a finer image, not by the Lanczos resampling the module corrects with. The CA-free truth is the
same specimen area-averaged without the geometry. Frames get 0.5 grey-level noise per channel and are quantised to
uint8.
"""

from __future__ import annotations

import json

import cv2
import numpy as np
import pytest

from microtopos.chromatic import MIN_FRINGE_PX, LateralCA, estimate_lateral_ca

W, H, F = 384, 288, 4  # frame size and supersampling of the ground-truth specimen
CENTRE = np.array([(W - 1) / 2.0, (H - 1) / 2.0])
TRUE_R = (1.004, (0.3, -0.2))
TRUE_B = (0.9965, (-0.15, 0.25))
MARGIN = 6  # border pixels left out of image comparisons (resampled from replicated border pixels)


def specimen(seed: int) -> list[np.ndarray]:
    """B, G, R planes (0..255 scale) on the F x finer grid."""
    rng = np.random.default_rng(seed)
    hh, wh = H * F, W * F

    def noise(sigma):
        n = cv2.GaussianBlur(rng.standard_normal((hh, wh)).astype(np.float32), (0, 0), sigma)
        return n / n.std()

    lum = 0.7 * noise(1.5 * F) + 0.5 * noise(5 * F)
    planes = [128 + 30 * lum * g for g in (0.8, 1.0, 1.1)]
    for _ in range(40):
        cx, cy, r = rng.uniform(0, wh), rng.uniform(0, hh), rng.uniform(6 * F, 30 * F)
        colour = rng.uniform(-35, 35, 3)
        x0, x1 = max(0, int(cx - r)), min(wh, int(cx + r) + 1)
        y0, y1 = max(0, int(cy - r)), min(hh, int(cy + r) + 1)
        yy, xx = np.mgrid[y0:y1, x0:x1]
        disk = (xx - cx) ** 2 + (yy - cy) ** 2 < r * r
        for i in range(3):
            planes[i][y0:y1, x0:x1][disk] += colour[i]
    return [p.astype(np.float32) for p in planes]


def render(planes, r=None, b=None, noise=0.5, seed=0, quantise=True) -> np.ndarray:
    """Frame of the specimen with channel R imaged at x_R = c + s (x - c) + t (r = (s, t)), B likewise."""
    yy, xx = np.mgrid[0:H * F, 0:W * F].astype(np.float64)
    x, y = (xx - (F - 1) / 2) / F, (yy - (F - 1) / 2) / F  # frame coordinates of the fine pixels (area centres)
    out = []
    for i, p in enumerate(planes):
        geo = {2: r, 0: b}.get(i)
        if geo is not None:
            s, t = geo
            px = CENTRE[0] + (x - CENTRE[0] - t[0]) / s  # the specimen point imaged at x
            py = CENTRE[1] + (y - CENTRE[1] - t[1]) / s
            p = cv2.remap(p, (F * px + (F - 1) / 2).astype(np.float32), (F * py + (F - 1) / 2).astype(np.float32),
                          cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
        out.append(cv2.resize(p, (W, H), interpolation=cv2.INTER_AREA))
    img = np.stack(out, -1).astype(np.float64)
    if noise:
        img += np.random.default_rng(1000 + seed).normal(0, noise, img.shape)
    return np.clip(np.round(img), 0, 255).astype(np.uint8) if quantise else img.astype(np.float32)


@pytest.fixture(scope="module")
def scene():
    planes = [specimen(seed) for seed in (11, 12, 13)]
    frames = [render(p, TRUE_R, TRUE_B, seed=k) for k, p in enumerate(planes)]
    truth = render(planes[0], noise=0, quantise=False)
    return planes, frames, truth, estimate_lateral_ca(frames)


def fringe_rms(img: np.ndarray, truth: np.ndarray) -> tuple[float, float]:
    """RMS difference from the CA-free truth of the R and B planes (grey levels), border excluded."""
    m = MARGIN
    d = np.asarray(img, np.float64)[m:-m, m:-m] - truth[m:-m, m:-m]
    return float(np.sqrt(np.mean(d[..., 2] ** 2))), float(np.sqrt(np.mean(d[..., 0] ** 2)))


def test_parameters_recovered(scene):
    est = scene[3]
    assert est.size == (W, H)
    assert abs(est.scale_r - TRUE_R[0]) < 2e-4
    assert abs(est.scale_b - TRUE_B[0]) < 2e-4
    np.testing.assert_allclose(est.shift_r, TRUE_R[1], atol=0.03)
    np.testing.assert_allclose(est.shift_b, TRUE_B[1], atol=0.03)
    assert est.significant
    # the stated uncertainties cover the error and are not vacuous
    assert 0 < est.u_scale_r < 2e-4 and 0 < est.u_scale_b < 2e-4
    assert abs(est.scale_r - TRUE_R[0]) < 3 * est.u_scale_r and abs(est.scale_b - TRUE_B[0]) < 3 * est.u_scale_b
    # the aberration is rotation-free and the diagnostic fit says so
    assert all(abs(r) < 1.0 for r in est.rotation_mrad)
    # corner fringe of the truth: R at the corner moves by 0.004 * 239.4 px plus the centre shift
    corners = np.array([[0, 0], [W - 1, 0], [0, H - 1], [W - 1, H - 1]], float) - CENTRE
    truth_fringe = max(np.hypot(*((s - 1) * corners + np.array(t)).T).max() for s, t in (TRUE_R, TRUE_B))
    assert abs(est.fringe_px_at_corner() - truth_fringe) < 0.05


def test_single_frame_estimate(scene):
    """One frame is enough; its uncertainty is larger but it still finds the aberration."""
    est = estimate_lateral_ca(scene[1][0])
    assert est.frames == 1
    assert abs(est.scale_r - TRUE_R[0]) < 5e-4 and abs(est.scale_b - TRUE_B[0]) < 5e-4
    np.testing.assert_allclose(est.shift_r, TRUE_R[1], atol=0.06)
    np.testing.assert_allclose(est.shift_b, TRUE_B[1], atol=0.06)


def test_apply_removes_fringes(scene):
    _, frames, truth, est = scene
    before = fringe_rms(frames[0], truth)
    corrected = est.apply(frames[0])
    after = fringe_rms(corrected, truth)
    assert corrected.dtype == np.float32 and corrected.shape == frames[0].shape
    # green is untouched
    np.testing.assert_array_equal(corrected[..., 1], frames[0][..., 1].astype(np.float32))
    for b, a in zip(before, after):
        assert a * 5 <= b
    # and the corrected planes match the CA-free truth to within the frame's own noise + quantisation (0.58) and
    # the resampling's smoothing
    assert max(after) < 1.0
    # float input gives the same result as uint8 input
    np.testing.assert_allclose(est.apply(frames[0].astype(np.float32)), corrected, atol=1e-3)


def test_inverse_correction_makes_fringes_worse(scene):
    """Sign: applying the negated aberration doubles the displacement instead of removing it."""
    _, frames, truth, est = scene
    inverse = LateralCA(est.size, 2 - est.scale_r, (-est.shift_r[0], -est.shift_r[1]),
                        2 - est.scale_b, (-est.shift_b[0], -est.shift_b[1]))
    before = fringe_rms(frames[0], truth)
    worse = fringe_rms(inverse.apply(frames[0]), truth)
    for b, w in zip(before, worse):
        assert w > 1.5 * b


def test_grey_frame_is_identity_and_not_significant(scene):
    planes = scene[0]
    g = render(planes[1], noise=0.5, seed=7)[..., 1]
    grey = np.dstack([g, g, g])
    with pytest.raises(ValueError, match="not significant"):
        estimate_lateral_ca(grey)
    est = estimate_lateral_ca(grey, require_significant=False)
    assert not est.significant
    assert abs(est.scale_r - 1) < 1e-5 and abs(est.scale_b - 1) < 1e-5
    np.testing.assert_allclose(est.shift_r + est.shift_b, 0, atol=1e-3)
    assert est.fringe_px_at_corner() < 0.2 * MIN_FRINGE_PX


def test_colour_frame_without_aberration_is_not_significant(scene):
    """The negative control that matters: a colour specimen (channels differ in content) without aberration. Its
    estimate is not exactly zero (the specimen's colour biases tiles), so the uncertainty test must reject it; this
    frame reads a corner fringe above MIN_FRINGE_PX, so the floor alone would not."""
    frame = render(scene[0][0], seed=3)
    with pytest.raises(ValueError, match="not significant"):
        estimate_lateral_ca(frame)
    est = estimate_lateral_ca(frame, require_significant=False)
    assert not est.significant
    assert est.fringe_px_at_corner() < 0.15
    assert abs(est.scale_r - 1) < 3e-4 and abs(est.scale_b - 1) < 3e-4


def test_refuses_low_texture():
    rng = np.random.default_rng(5)
    flat = np.clip(np.round(128 + rng.normal(0, 0.3, (H, W, 3))), 0, 255).astype(np.uint8)
    with pytest.raises(ValueError, match="too little texture"):
        estimate_lateral_ca(flat)


def test_refuses_dissimilar_channels(scene):
    planes = scene[0]
    frame = render(planes[0], seed=1)
    frame[..., 2] = render(planes[2], seed=2)[..., 2]  # red from an unrelated view
    with pytest.raises(ValueError, match="dissimilar"):
        estimate_lateral_ca(frame)


def test_refuses_unknown_bit_depth(scene):
    with pytest.raises(ValueError, match="uint8 or float"):
        estimate_lateral_ca(scene[1][0].astype(np.uint16) * 256)


def test_json_round_trip_is_exact(scene):
    est = scene[3]
    text = est.to_json()
    back = LateralCA.from_json(text)
    assert back == est
    assert json.loads(text)["scale_r"] == est.scale_r
    assert back.to_json() == text


def test_other_capture_size_is_refused_or_rescaled_explicitly(scene):
    _, frames, _, est = scene
    half = cv2.resize(frames[0], (W // 2, H // 2), interpolation=cv2.INTER_AREA)
    with pytest.raises(ValueError, match="rescaled"):
        est.apply(half)
    small = est.rescaled(W // 2, H // 2)
    assert small.size == (W // 2, H // 2)
    assert small.scale_r == est.scale_r and small.scale_b == est.scale_b
    np.testing.assert_allclose(small.shift_r, np.array(est.shift_r) / 2, rtol=1e-12)
    np.testing.assert_allclose(small.shift_b, np.array(est.shift_b) / 2, rtol=1e-12)
    # the corner fringe is a length in pixels of the new frame: halved up to the corner's half-pixel offset
    assert abs(small.fringe_px_at_corner() - est.fringe_px_at_corner() / 2) < 0.01
    assert small.apply(half).shape == half.shape
    with pytest.raises(ValueError, match="aspect ratio"):
        est.rescaled(W // 2, H // 2 + 20)
