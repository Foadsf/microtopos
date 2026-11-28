"""Slanted-edge MTF measurement (microtopos.edgemtf) against closed forms.

Ground truth: a straight edge blurred by a Gaussian of known sigma (in output pixels) is evaluated on a 16x16
sub-pixel grid per output pixel (step convolved with a Gaussian = the normal CDF of the distance to the edge, exact)
and the 16x16 samples are averaged, so every output pixel integrates over its full square area. Along the edge
normal, at tilt theta, the system MTF is then

    MTF(f) = exp(-2 pi^2 sigma^2 f^2) * |sinc(f cos theta) * sinc(f sin theta)|

(Gaussian optics times the Fourier transform of a unit square pixel projected onto the normal; it reduces to the
textbook Gaussian * sinc(f) at theta = 0), and the RMS width of the line spread function is exactly
sqrt(sigma^2 + 1/12): the projected square pixel has variance (cos^2 + sin^2) / 12 = 1/12 px^2 at every tilt.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.optimize import brentq
from scipy.special import ndtr

from microtopos.edgemtf import measure_edge, mtf50_to_lp_per_mm

SS = 16
DARK, BRIGHT = 40.0, 200.0


def render_edge(h, w, sigma, tilt_deg, dark=DARK, bright=BRIGHT, cx=None, cy=None):
    """Straight edge through (cx, cy), tilted tilt_deg from vertical, Gaussian blur sigma, square pixel aperture.

    Pixel (i, j) covers [j - 0.5, j + 0.5] x [i - 0.5, i + 0.5]. Intensity rises from `dark` to `bright` towards +x.
    """
    th = math.radians(tilt_deg)
    cx = w / 2 if cx is None else cx
    cy = h / 2 if cy is None else cy
    sub = (np.arange(SS) + 0.5) / SS - 0.5
    xs = (np.arange(w)[:, None] + sub[None, :]).ravel()
    out = np.empty((h, w))
    for i in range(h):
        yy = i + sub
        dist = (xs[None, :] - cx) * math.cos(th) - (yy[:, None] - cy) * math.sin(th)
        out[i] = ndtr(dist / sigma).reshape(SS, w, SS).mean(axis=(0, 2))
    return dark + (bright - dark) * out


def render_arc(h, w, sigma, radius, tilt_deg=8.0, dark=DARK, bright=BRIGHT):
    """Rim of a disc of the given radius crossing the image centre with its tangent tilted tilt_deg from vertical.

    Same pixel model as render_edge; the disc is the dark side.
    """
    sub = (np.arange(SS) + 0.5) / SS - 0.5
    xs = (np.arange(w)[:, None] + sub[None, :]).ravel()
    th = math.radians(tilt_deg)
    cx, cy = w / 2 - radius * math.cos(th), h / 2 + radius * math.sin(th)
    out = np.empty((h, w))
    for i in range(h):
        yy = i + sub
        r = np.hypot(xs[None, :] - cx, yy[:, None] - cy)
        out[i] = ndtr((r - radius) / sigma).reshape(SS, w, SS).mean(axis=(0, 2))
    return dark + (bright - dark) * out


def mtf_true(f, sigma, tilt_deg):
    th = math.radians(tilt_deg)
    return np.exp(-2 * math.pi ** 2 * sigma ** 2 * f ** 2) * np.abs(np.sinc(f * math.cos(th)) *
                                                                   np.sinc(f * abs(math.sin(th))))


def mtf50_true(sigma, tilt_deg):
    return brentq(lambda f: mtf_true(f, sigma, tilt_deg) - 0.5, 1e-6, 1.0)


# 0.3 px: a pixel-limited system (MTF still 0.41 at Nyquist), where the bin/derivative correction matters most
GRID = [(s, a) for s in (0.3, 0.8, 1.5, 3.0) for a in (4.0, 8.0, 25.0)]


@pytest.mark.parametrize("sigma,tilt", GRID)
def test_noiseless_edge_matches_gaussian_times_pixel_aperture(sigma, tilt):
    img = render_edge(128, 192, sigma, tilt)
    r = measure_edge(img)
    assert r["orientation"] == "vertical" and r["polarity"] == 1
    assert r["angle_deg"] == pytest.approx(tilt, abs=0.01)
    # sub-pixel edge position: the true line is x = cx + tan(theta) (y - cy) with (cx, cy) = (96, 64)
    ex, ey = r["edge_xy"]
    assert ex == pytest.approx(96 + math.tan(math.radians(tilt)) * (ey - 64), abs=0.02)
    # MTF50 (measured +0.6 %, the Hamming window's narrowing of the LSF at a 20-LSF-width span)
    assert r["mtf50_cyc_per_px"] == pytest.approx(mtf50_true(sigma, tilt), rel=0.03)
    # Gaussian sigma of the optics (pixel aperture removed) and RMS width of the camera LSF
    assert r["psf_sigma_px"] == pytest.approx(sigma, rel=0.01)
    assert r["lsf_sigma_px"] == pytest.approx(math.sqrt(sigma ** 2 + 1 / 12), rel=0.01)
    # whole MTF curve to Nyquist and on to the reported 1 cycle/pixel (measured max deviation 0.0048)
    f, m = r["mtf"]["freq_cyc_per_px"], r["mtf"]["value"]
    assert f[0] == 0 and m[0] == pytest.approx(1.0) and f[-1] >= 0.99
    sel = f <= 0.5
    assert np.max(np.abs(m[sel] - mtf_true(f[sel], sigma, tilt))) < 0.01
    assert np.max(np.abs(m - mtf_true(f, sigma, tilt))) < 0.01
    # levels
    assert r["dark_level"] == pytest.approx(DARK, abs=0.5)
    assert r["bright_level"] == pytest.approx(BRIGHT, abs=0.5)
    assert r["edge_contrast"] == pytest.approx(BRIGHT - DARK, rel=0.005)
    assert r["n_rows"] >= 100


def test_pixel_aperture_term_is_needed_to_explain_the_result():
    """The closed form without the pixel aperture is off by far more than the tolerance: the test can tell."""
    sigma, tilt = 0.8, 8.0
    r = measure_edge(render_edge(128, 192, sigma, tilt))
    gauss_only = brentq(lambda f: math.exp(-2 * math.pi ** 2 * sigma ** 2 * f ** 2) - 0.5, 1e-6, 1.0)
    assert abs(r["mtf50_cyc_per_px"] / gauss_only - 1) > 0.05
    assert abs(r["lsf_sigma_px"] / sigma - 1) > 0.05


@pytest.mark.parametrize("sigma", [0.8, 1.5, 3.0])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_noisy_edge_snr_50(sigma, seed):
    """Gaussian noise of std (step / 50): MTF50 within 5 %, noise and SNR estimates close to the truth."""
    rng = np.random.default_rng(seed)
    noise = (BRIGHT - DARK) / 50
    img = render_edge(128, 192, sigma, 8.0) + rng.normal(0.0, noise, (128, 192))
    r = measure_edge(img)
    assert r["mtf50_cyc_per_px"] == pytest.approx(mtf50_true(sigma, 8.0), rel=0.05)
    assert r["psf_sigma_px"] == pytest.approx(sigma, rel=0.05)
    assert r["noise_sigma"] == pytest.approx(noise, rel=0.15)
    assert r["snr"] == pytest.approx(50, rel=0.15)


def test_mtf_noise_floor_matches_the_hamming_windowed_prediction():
    """Beyond the passband the reported MTF is noise; its mean square follows from white pixel noise.

    sigma = 3 px: the true MTF above 0.3 cycle/pixel is below 1e-7. Each ESF bin averages n_k = rows * bin / cos(theta)
    pixels of noise std s, the forward difference multiplies the bin noise spectrum by 2 sin(pi f bin) / bin, the
    window by sqrt(sum w_k^2) (sum of Hamming^2 = 0.3974 per bin), the normalisation divides by the contrast C and
    the bin/derivative correction by sinc(f bin)^2:
        E[MTF(f)^2] = (s^2 / n_k) * 4 sin^2(pi f bin) * 0.3974 * n_bins / (C^2 * sinc(f bin)^4).
    Measured/predicted = 1.12 with the window (3.0 without it).
    """
    sigma, tilt, rows, binw = 3.0, 8.0, 128, 0.25
    clean = render_edge(rows, 192, sigma, tilt)
    s = (BRIGHT - DARK) / 50
    got, want = [], []
    for seed in range(4):
        rng = np.random.default_rng(10 + seed)
        r = measure_edge(clean + rng.normal(0.0, s, clean.shape))
        f, m = r["mtf"]["freq_cyc_per_px"], r["mtf"]["value"]
        sel = (f >= 0.3) & (f <= 0.6)
        n_bins = 2 * math.floor(r["span_px"] / binw) - 1
        n_k = rows * binw / math.cos(math.radians(tilt))
        pred = (s ** 2 / n_k) * 4 * np.sin(np.pi * f[sel] * binw) ** 2 * 0.3974 * n_bins
        pred /= (BRIGHT - DARK) ** 2 * np.sinc(f[sel] * binw) ** 4
        got.append(np.mean(m[sel] ** 2))
        want.append(np.mean(pred))
    assert 0.7 < np.mean(got) / np.mean(want) < 1.6


def test_contrast_inversion_gives_the_same_mtf():
    up = render_edge(128, 192, 1.5, 8.0)
    down = render_edge(128, 192, 1.5, 8.0, dark=BRIGHT, bright=DARK)  # bright on the left
    ru, rd = measure_edge(up), measure_edge(down)
    assert ru["polarity"] == 1 and rd["polarity"] == -1
    assert rd["mtf50_cyc_per_px"] == pytest.approx(ru["mtf50_cyc_per_px"], rel=1e-3)
    assert rd["psf_sigma_px"] == pytest.approx(ru["psf_sigma_px"], rel=1e-3)
    assert np.max(np.abs(rd["mtf"]["value"] - ru["mtf"]["value"])) < 1e-3
    assert rd["edge_contrast"] == pytest.approx(BRIGHT - DARK, rel=0.005)
    assert rd["dark_level"] == pytest.approx(DARK, abs=0.5)


def test_near_horizontal_edge_is_measured_along_columns():
    img = render_edge(128, 192, 1.5, 8.0)
    r = measure_edge(np.ascontiguousarray(img.T))
    assert r["orientation"] == "horizontal"
    assert r["angle_deg"] == pytest.approx(8.0, abs=0.01)
    assert r["mtf50_cyc_per_px"] == pytest.approx(mtf50_true(1.5, 8.0), rel=0.03)
    ex, ey = r["edge_xy"]
    assert ey == pytest.approx(96 + math.tan(math.radians(8.0)) * (ex - 64), abs=0.02)


def test_roi_selects_the_edge_and_reports_image_coordinates():
    img = render_edge(128, 192, 1.5, 8.0)
    big = np.full((200, 300), 120.0)
    big[40:168, 70:262] = img
    big[:, :50] = np.where(np.arange(50) < 25, 30.0, 220.0)[None, :]  # a second, sharp edge outside the ROI
    r = measure_edge(big, roi=(70, 40, 192, 128))
    ex, ey = r["edge_xy"]
    assert ex - 70 == pytest.approx(96 + math.tan(math.radians(8.0)) * (ey - 40 - 64), abs=0.02)
    assert r["mtf50_cyc_per_px"] == pytest.approx(mtf50_true(1.5, 8.0), rel=0.03)


def test_uint8_edge_is_measured_and_clipped_one_is_refused():
    ok = np.round(render_edge(128, 192, 1.5, 8.0)).astype(np.uint8)
    r = measure_edge(ok)
    assert not any("clipping" in w for w in r["warnings"])
    assert r["mtf50_cyc_per_px"] == pytest.approx(mtf50_true(1.5, 8.0), rel=0.03)
    clipped = np.clip(np.round(render_edge(128, 192, 1.5, 8.0, bright=320.0)), 0, 255).astype(np.uint8)
    with pytest.raises(ValueError, match="clipped"):
        measure_edge(clipped)
    # a float image with a declared saturation level is checked too
    with pytest.raises(ValueError, match="clipped"):
        measure_edge(clipped.astype(np.float32), saturation=255.0)


def test_flat_images_are_refused():
    with pytest.raises(ValueError, match="no edge"):
        measure_edge(np.full((128, 192), 100.0))
    rng = np.random.default_rng(4)
    with pytest.raises(ValueError, match="no edge"):
        measure_edge(100.0 + rng.normal(0.0, 3.0, (128, 192)))


def test_curved_edge_is_refused():
    # radius 1000 px: the rim departs 2 px from its chord over 128 rows
    with pytest.raises(ValueError, match="curved"):
        measure_edge(render_arc(128, 192, 1.0, 1000.0))


def test_straight_control_for_the_curvature_gate():
    """A huge radius (sagitta 0.08 px over the ROI) is straight enough and is measured."""
    r = measure_edge(render_arc(128, 192, 1.0, 25000.0))
    assert r["psf_sigma_px"] == pytest.approx(1.0, rel=0.02)


@pytest.mark.parametrize("tilt", [0.0, 0.5])
def test_edge_too_close_to_the_axis_is_refused(tilt):
    # 0.5 deg over 128 rows crosses 1.1 px, fewer than the 2 px needed for 4x bins
    with pytest.raises(ValueError, match="too close to vertical"):
        measure_edge(render_edge(128, 192, 1.0, tilt))


def test_edge_near_45_degrees_is_refused():
    with pytest.raises(ValueError, match="tilted"):
        measure_edge(render_edge(128, 192, 1.0, 44.0))


def test_slope_one_to_two_is_refused_for_empty_bins():
    """tan(theta) = 1/2 exactly: only two sub-pixel phases occur, so half of the 1/4-px bins stay empty."""
    with pytest.raises(ValueError, match="empty"):
        measure_edge(render_edge(128, 192, 1.0, math.degrees(math.atan(0.5))))


def test_too_noisy_edge_is_refused():
    rng = np.random.default_rng(5)
    img = render_edge(128, 192, 1.5, 8.0) + rng.normal(0.0, (BRIGHT - DARK) / 6, (128, 192))
    with pytest.raises(ValueError, match="noisy"):
        measure_edge(img)


def test_short_edge_is_refused():
    img = render_edge(128, 192, 1.0, 8.0)
    with pytest.raises(ValueError, match="too short"):
        measure_edge(img, roi=(0, 0, 192, 16))  # 16 px of edge: fewer than the 20 rows required
    assert measure_edge(img, roi=(0, 0, 192, 24))["n_rows"] == 24  # control: 24 rows are enough


def test_lp_per_mm_conversion():
    # 0.25 cycle/pixel at 2 um/pixel = 0.125 cycle/um = 125 lp/mm
    assert mtf50_to_lp_per_mm(0.25, 2.0) == pytest.approx(125.0)
