"""Metrology extensions (ISO 21920 1D profile filtering, ISO 25178 Sdq / volumes / feature parameters,
interactive 3-point levelling / exclusion mask / step height, topographic watershed hills).

Every numerical claim is checked against a closed form or an independently constructed ground truth.
"""

import math

import numpy as np
import pytest
import scipy.ndimage as ndi

from microtopos import metrology


def _sinusoid(n: int, step_um: float, amplitude_um: float, period_um: float) -> np.ndarray:
    x = np.arange(n, dtype=np.float64) * step_um
    return amplitude_um * np.sin(2.0 * np.pi * x / period_um)


def _add_nans(prof: np.ndarray, rng: np.random.Generator, fraction: float) -> np.ndarray:
    p = prof.copy()
    p[rng.choice(p.size, round(fraction * p.size), replace=False)] = np.nan
    return p


def _gauss_transmission(period_um: float, cutoff_um: float) -> float:
    """ISO 16610-21 Gaussian: amplitude transmission of a sine of this period, exp(-pi (alpha lc / lambda)^2)."""
    return math.exp(-math.pi * (metrology.ALPHA * cutoff_um / period_um) ** 2)


# ---------------------------------------------------------------- 1. ISO 21920 1D profile filtering

def test_gaussian_filter_1d_transmission_is_50_percent_at_the_cutoff():
    """ISO 16610-21: a sine whose period equals the cutoff keeps 50 % of its amplitude (+/- 2 %)."""
    step, lam_c = 0.5, 10.0
    p = _sinusoid(4000, step, 1.0, lam_c)
    out = metrology.gaussian_filter_1d_nan(p, lam_c, step)
    core = out[400:-400]  # away from the reflected borders
    amp = float(np.sqrt(2.0 * np.mean(core ** 2)))
    assert amp == pytest.approx(0.5, abs=0.01), amp
    # and the closed form at another wavelength, lambda = 2 lambda_c: exp(-pi (alpha/2)^2) = 0.9576
    p2 = _sinusoid(4000, step, 1.0, 2 * lam_c)
    out2 = metrology.gaussian_filter_1d_nan(p2, lam_c, step)[400:-400]
    assert float(np.sqrt(2.0 * np.mean(out2 ** 2))) == pytest.approx(_gauss_transmission(2 * lam_c, lam_c), rel=0.02)


def test_profile_parameters_roughness_of_a_two_frequency_sine_matches_the_analytic_value():
    """Roughness = x - Gaussian(x): each sine keeps a_i (1 - T_i); Ra is the mean |.| of that exact sum (2 %)."""
    step_um, n = 0.5, 4000  # 2000 um: 250 periods of 8 um and 20 periods of 100 um, so no net line slope
    lam_c = 25.0
    a1, l1, a2, l2 = 1.0, 8.0, 3.0, 100.0
    prof = _sinusoid(n, step_um, a1, l1) + _sinusoid(n, step_um, a2, l2)
    res = metrology.profile_parameters(prof, step_um, lam_s_um=0.0, lam_c_um=lam_c)
    r1 = a1 * (1 - _gauss_transmission(l1, lam_c))
    r2 = a2 * (1 - _gauss_transmission(l2, lam_c))
    x = np.linspace(0.0, 2000.0, 400001)[:-1]
    ideal = r1 * np.sin(2 * np.pi * x / l1) + r2 * np.sin(2 * np.pi * x / l2)
    ra_true, rq_true = float(np.mean(np.abs(ideal))), float(np.sqrt(np.mean(ideal ** 2)))
    assert res["Ra"] == pytest.approx(ra_true, rel=0.02), (res["Ra"], ra_true)
    assert res["Rq"] == pytest.approx(rq_true, rel=0.02), (res["Rq"], rq_true)
    # waviness is the complement: a_i T_i
    w1, w2 = a1 * _gauss_transmission(l1, lam_c), a2 * _gauss_transmission(l2, lam_c)
    wav = w1 * np.sin(2 * np.pi * x / l1) + w2 * np.sin(2 * np.pi * x / l2)
    assert res["Wa"] == pytest.approx(float(np.mean(np.abs(wav))), rel=0.02)
    assert res["lambda_c_applied_um"] == lam_c


def test_profile_parameters_with_dropouts_still_matches():
    rng = np.random.default_rng(3)
    step_um, n = 0.5, 4000
    prof = _add_nans(_sinusoid(n, step_um, 1.0, 20.0) + 0.5 * _sinusoid(n, step_um, 1.0, 200.0), rng, 0.02)
    res = metrology.profile_parameters(prof, step_um, lam_s_um=0.0, lam_c_um=60.0)
    r1 = 1.0 * (1 - _gauss_transmission(20.0, 60.0))
    r2 = 0.5 * (1 - _gauss_transmission(200.0, 60.0))
    x = np.linspace(0.0, 2000.0, 400001)[:-1]
    ra_true = float(np.mean(np.abs(r1 * np.sin(2 * np.pi * x / 20.0) + r2 * np.sin(2 * np.pi * x / 200.0))))
    assert res["Ra"] == pytest.approx(ra_true, rel=0.03), (res["Ra"], ra_true)  # 2 % of samples are missing


def test_profile_parameters_all_nan_and_too_short():
    """Negative control: an all-NaN or too-short profile must return empty metrics, never crash."""
    assert metrology.profile_parameters(np.full(100, np.nan), 0.5) == {}
    assert metrology.profile_parameters(np.array([1.0, 2.0, np.nan]), 0.5) == {}
    assert metrology.profile_parameters(np.array([1.0, np.nan, 3.0, 4.0]), 0.5) == {}


def test_profile_cutoff_check_reports_why_a_profile_is_too_short():
    ok = metrology.profile_cutoff_problem(n_samples=1000, step_um=0.5, lam_c_um=20.0)  # 500 um = 25 cutoffs
    assert ok is None
    px = metrology.profile_cutoff_problem(1000, 0.5, 1.0)  # 2 px
    assert px is not None and "2.0 px" in px and "4" in px
    short = metrology.profile_cutoff_problem(100, 0.5, 20.0)  # 50 um is 2.5 cutoffs, needs 5
    assert short is not None and "2.5" in short and "5" in short


def test_profile_parameters_unresolvable_cutoff_is_reported_not_applied():
    """A 1D cutoff below 4 px is skipped; the result says which cutoff was actually applied (none)."""
    p = _sinusoid(300, 0.5, 1.0, 10.0)  # lam_c = 1.0 um = 2 px < 4 px -> skipped
    res = metrology.profile_parameters(p, 0.5, lam_c_um=1.0)
    assert res["Ra"] == pytest.approx(2.0 / math.pi, rel=0.01)  # unfiltered primary profile of a sine, A = 1
    assert res["lambda_c_applied_um"] == 0.0


def test_gaussian_filter_1d_nan_keeps_dropouts_and_passes_long_waves():
    p = _sinusoid(600, 0.2, 2.0, 25.0)
    p[120:130] = np.nan
    out = metrology.gaussian_filter_1d_nan(p, 5.0, 0.2)
    assert out.shape == p.shape and out.dtype == np.float32
    assert np.isnan(out[120:130]).all()
    finite = np.isfinite(out)
    assert finite.sum() == p.size - 10
    # lambda_c = 5 um on a 25 um sine: closed-form transmission exp(-pi (alpha 5/25)^2) = 0.9726
    amp = float(np.sqrt(2 * np.mean((out[finite][100:-100] - np.mean(out[finite])) ** 2)))
    assert amp == pytest.approx(2.0 * _gauss_transmission(25.0, 5.0), rel=0.02)


# ---------------------------------------------------------------- 2. Sdq and functional / volume parameters

def test_root_mean_square_gradient_matches_analytical_slope():
    xx = np.arange(80)[None, :].repeat(60, 0).astype(np.float32)
    yy = np.arange(60)[:, None].repeat(80, 1).astype(np.float32)
    # Plane 0.37 um/px in x: Sdq == slope
    sdq = metrology.root_mean_square_gradient((0.37 * xx).astype(np.float32), 1.0)
    assert sdq == pytest.approx(0.37, rel=1e-5)
    # the lateral unit matters: 0.3 / px in x and 0.4 / px in y at 2 um/px is a slope of 0.25 (a 5 % error fails)
    sdq = metrology.root_mean_square_gradient((0.3 * xx + 0.4 * yy).astype(np.float32), 2.0)
    assert sdq == pytest.approx(0.25, rel=1e-4)
    # sinusoid A sin(2 pi x / L): rms slope = A 2 pi / L / sqrt(2) (sampled forward difference: 0.1 % low)
    sdq = metrology.root_mean_square_gradient(np.sin(2.0 * np.pi * xx / 40.0).astype(np.float32), 1.0)
    assert sdq == pytest.approx((2.0 * np.pi / 40.0) / math.sqrt(2), rel=0.01)
    assert metrology.root_mean_square_gradient(np.zeros((10, 10), np.float32), 1.0) == 0.0


_P, _Q = 0.10, 0.80  # ISO 25178-2 default material-ratio levels for Vmp/Vmc and Vvc/Vvv


def _volumes_expected(vm, vv):
    return {"Vmp": vm(_P), "Vmc": vm(_Q) - vm(_P), "Vvc": vv(_P) - vv(_Q), "Vvv": vv(_Q)}


def test_volume_parameters_of_a_uniform_height_ramp_match_the_iso_closed_form():
    """Uniform heights on [0, 1]: c(p) = 1 - p, Vm(m) = m^2/2, Vv(m) = (1-m)^2/2 (per unit area, height units)."""
    n = 100_000
    z = ((np.arange(n) + 0.5) / n).reshape(250, 400)
    want = _volumes_expected(lambda m: m * m / 2, lambda m: (1 - m) ** 2 / 2)
    assert want == pytest.approx({"Vmp": 0.005, "Vmc": 0.315, "Vvc": 0.385, "Vvv": 0.02})
    fp = metrology.functional_parameters(z)
    for k, v in want.items():
        assert fp[k] == pytest.approx(v, rel=0.01), (k, fp[k], v)
    # the equivalent line of a straight ramp is the ramp itself: Sk = full range, no peak or valley regions
    assert fp["Sk"] == pytest.approx(1.0, rel=0.01)
    assert fp["Spk"] == pytest.approx(0.0, abs=0.01) and fp["Svk"] == pytest.approx(0.0, abs=0.01)
    assert fp["Smr1_pct"] == pytest.approx(0.0, abs=0.5) and fp["Smr2_pct"] == pytest.approx(100.0, abs=0.5)


def test_volume_parameters_of_a_sinusoid_match_the_iso_closed_form():
    """z = sin(phase), phase uniform: c(p) = cos(pi p); Vm(m) = sin(pi m)/pi - m cos(pi m),
    Vv(m) = (1-m) cos(pi m) + sin(pi m)/pi. An independent surface from the ramp, with a symmetric distribution."""
    n = 100_000
    z = np.sin(2 * np.pi * (np.arange(n) + 0.5) / n).reshape(250, 400)
    def vm(m):
        return math.sin(math.pi * m) / math.pi - m * math.cos(math.pi * m)

    def vv(m):
        return (1 - m) * math.cos(math.pi * m) + math.sin(math.pi * m) / math.pi

    want = _volumes_expected(vm, vv)
    assert want["Vmc"] == pytest.approx(0.83106, rel=1e-3) and want["Vvc"] == pytest.approx(0.92929, rel=1e-3)
    fp = metrology.functional_parameters(z)
    for k, v in want.items():
        assert fp[k] == pytest.approx(v, rel=0.01), (k, fp[k], v)
    # the volumes are per unit area in the height unit: twice the height is twice the volume
    fp2 = metrology.functional_parameters(2.0 * z)
    for k, v in want.items():
        assert fp2[k] == pytest.approx(2.0 * v, rel=0.01), k


def test_functional_parameters_too_few_pixels_is_empty():
    assert metrology.functional_parameters(np.zeros((10, 10), np.float32)) == {}


# ---------------------------------------------------------------- 3. Feature parameters

_D, _SIG, _UM = 24, 4.0, 0.5  # lattice pitch (px), Gaussian sigma (px), pixel size (um)
_PEAK_AMP = 1.0 + 0.04 * np.arange(25)  # 25 distinct peak heights, 1.00 ... 1.96
_PIT_AMP = 0.5 + 0.03 * np.arange(16)  # 16 distinct pit depths, 0.50 ... 0.95


def _lattice(noise_sigma: float = 0.0, seed: int = 11):
    """5 x 5 peaks at the cell centres and 4 x 4 pits at the interior cell corners of a 120 x 120 px field."""
    n = 5 * _D
    yy, xx = np.mgrid[0:n, 0:n].astype(np.float64)
    rng = np.random.default_rng(seed)
    peaks = rng.permutation(_PEAK_AMP)
    pits = rng.permutation(_PIT_AMP)
    z = np.zeros((n, n))
    k = 0
    for j in range(5):
        for i in range(5):
            cx, cy = (i + 0.5) * _D, (j + 0.5) * _D
            z += peaks[k] * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * _SIG ** 2))
            k += 1
    k = 0
    for j in range(1, 5):
        for i in range(1, 5):
            z -= pits[k] * np.exp(-((xx - i * _D) ** 2 + (yy - j * _D) ** 2) / (2 * _SIG ** 2))
            k += 1
    clean = z.copy()
    if noise_sigma:
        z = z + rng.normal(0.0, noise_sigma, z.shape)
    return z.astype(np.float32), clean


def _brute_force_summit_count(z: np.ndarray) -> int:
    """Independent count of strict 8-neighbour maxima (scipy maximum_filter with the centre excluded)."""
    fp = np.ones((3, 3), bool)
    fp[1, 1] = False
    nb = ndi.maximum_filter(z, footprint=fp, mode="nearest")
    inner = np.zeros(z.shape, bool)
    inner[1:-1, 1:-1] = True
    return int(np.count_nonzero((z > nb) & inner))


def test_noise_free_lattice_summit_density_and_curvature_match_the_analytic_values():
    z, _ = _lattice()
    area_um2 = (z.shape[0] * _UM) ** 2
    density_mm2 = 25 / area_um2 * 1e6  # 25 peaks in 25 lattice cells
    f = metrology.feature_parameters(z, _UM)
    assert f["Spd_per_mm2"] == pytest.approx(density_mm2, rel=0.01)
    assert f["Sds_per_mm2"] == pytest.approx(density_mm2, rel=0.01)  # noise free: every summit is significant
    # a Gaussian peak A exp(-r^2/2s^2) has mean curvature -(1/2)(z_xx + z_yy) = A / s^2 at the top (s in um);
    # the 3-point second difference is 1.6 % low at s = 4 px
    s_um = _SIG * _UM
    want = float(np.mean(_PEAK_AMP)) / s_um ** 2
    assert f["Spc_per_um"] == pytest.approx(want, rel=0.05)
    assert f["Ssc_per_um"] == pytest.approx(want, rel=0.05)
    assert f["Spc_per_um"] < want  # the discretisation bias has a sign; a sign flip or a doubling fails both


def test_noisy_lattice_significant_peak_density_height_parameters_and_unpruned_summit_count():
    z, clean = _lattice(noise_sigma=0.01)
    mean_plane = float(np.mean(clean))
    area_um2 = (z.shape[0] * _UM) ** 2
    f = metrology.feature_parameters(z, _UM)
    # Spd counts the Wolf-pruned peaks: the lattice density, whatever the noise
    assert f["Spd_per_mm2"] == pytest.approx(25 / area_um2 * 1e6, rel=0.01)
    # Sds counts every strict 8-neighbour maximum; it must equal an independent brute-force count and, on a noisy
    # surface, exceed Spd by a large factor (that is the definition, not a bug: it is unpruned)
    n_summits = _brute_force_summit_count(z)
    assert n_summits > 60
    assert f["Sds_per_mm2"] == pytest.approx(n_summits / area_um2 * 1e6, rel=1e-6)
    assert f["Sds_per_mm2"] > 2.5 * f["Spd_per_mm2"]
    # S5p: five highest peaks above the mean plane; S5v: five deepest pits below it; S10z = sum
    top5 = float(np.mean(np.sort(_PEAK_AMP)[-5:]))
    deep5 = float(np.mean(np.sort(_PIT_AMP)[-5:]))
    assert f["S5p"] == pytest.approx(top5 - mean_plane, rel=0.01)
    assert f["S5v"] == pytest.approx(deep5 + mean_plane, rel=0.01)
    assert f["S10z"] == pytest.approx(top5 + deep5, rel=0.01)
    assert f["S10z"] == pytest.approx(f["S5p"] + f["S5v"], rel=1e-9)


def test_feature_parameters_scale_with_the_pixel_size_as_documented():
    z, _ = _lattice(noise_sigma=0.01)
    a = metrology.feature_parameters(z, 0.5)
    b = metrology.feature_parameters(z, 1.0)
    # densities are per area: doubling the pixel size divides them by 4 (the pixel grid is identical)
    assert a["Spd_per_mm2"] / b["Spd_per_mm2"] == pytest.approx(4.0, rel=1e-9)
    assert a["Sds_per_mm2"] / b["Sds_per_mm2"] == pytest.approx(4.0, rel=1e-9)
    # curvature is height per length^2: divided by 4 as well
    assert a["Spc_per_um"] / b["Spc_per_um"] == pytest.approx(4.0, rel=1e-6)
    assert a["Ssc_per_um"] / b["Ssc_per_um"] == pytest.approx(4.0, rel=1e-6)
    # heights do not depend on the lateral scale
    assert a["S5p"] == pytest.approx(b["S5p"], rel=1e-9) and a["S10z"] == pytest.approx(b["S10z"], rel=1e-9)


def test_feature_parameters_invalid_pixels_do_not_create_peaks():
    z, _ = _lattice(noise_sigma=0.01)
    z[0:30, 0:30] = np.nan  # removes the peak centred at (12, 12) and borders its neighbours
    f = metrology.feature_parameters(z, _UM)
    valid_area_um2 = float(np.isfinite(z).sum()) * _UM ** 2
    assert f["Spd_per_mm2"] == pytest.approx(24 / valid_area_um2 * 1e6, rel=0.01)


def test_feature_parameters_flat_plane_is_zero():
    """Negative control: a flat plane has no summits, pits or curvatures."""
    z = np.full((80, 80), 2.5, np.float32)
    feat = metrology.feature_parameters(z, 1.0)
    assert feat and feat["Sds_per_mm2"] == 0.0
    assert "Ssc_per_um" not in feat and "Spd_per_mm2" not in feat
    assert "S5p" not in feat and "S5v" not in feat and "S10z" not in feat


def test_feature_parameters_are_fast_on_a_full_hd_map():
    """analyze() computes only the cheap summit parameters; the Wolf-pruned set is on demand (see feature_set)."""
    import time
    rng = np.random.default_rng(0)
    z = ndi.gaussian_filter(rng.normal(size=(1080, 1920)), 6).astype(np.float32)
    t0 = time.perf_counter()
    r = metrology.analyze(z, np.full(z.shape, 128, np.float32), 0.5, "um", 2.0, 1, 0.0, 0.0)
    dt = time.perf_counter() - t0
    assert "Sds_per_mm2" in r and "Spd_per_mm2" not in r and "topo_grain_count" not in r
    assert "on demand" in r["eligibility"]["feature"]
    assert dt < 6.0, f"analyze() on 1920x1080 took {dt:.1f} s"  # was 8.4 s on 960x540 white noise with both calls


def test_feature_set_matches_the_separate_calls_and_reports_units():
    z, _ = _lattice(noise_sigma=0.01)
    full = metrology.feature_set(z, _UM, "um")
    assert full["Spd_per_mm2"] == metrology.feature_parameters(z, _UM)["Spd_per_mm2"]
    assert full["topo_grain_count"] == metrology.grains_topographic(z, _UM)["topo_grain_count"]
    frames = metrology.feature_set(z, _UM, "frames")  # curvature needs z and x in the same unit
    assert "Spc_per_um" not in frames and "Ssc_per_um" not in frames and "S5p" in frames


# ---------------------------------------------------------------- 4. Interactive tools

def test_level_three_points_removes_tilt_exactly():
    ny, nx = 50, 70
    yy, xx = np.mgrid[0:ny, 0:nx].astype(np.float64)
    z = (3.0 + 0.04 * xx - 0.07 * yy).astype(np.float32)
    z[5:8, 5:8] = np.nan  # NaN handling must not break the fit
    leveled = metrology.level_three_points(z, [(10.0, 10.0), (60.0, 10.0), (10.0, 40.0)], radius_px=2)
    assert np.all(leveled[np.isfinite(z)] == pytest.approx(0.0, abs=1e-6))
    assert np.all(np.isnan(leveled[~np.isfinite(z)]))
    hp = metrology.height_parameters(leveled, 1.0)
    assert hp and hp["Sq"] < 1e-6


def test_level_three_points_collinear_raises():
    z = np.arange(25.0, dtype=np.float32).reshape(5, 5)
    with pytest.raises(ValueError, match="collinear"):
        metrology.level_three_points(z, [(1.0, 1.0), (2.0, 2.0), (3.0, 3.0)])
    with pytest.raises(ValueError, match="exactly 3 points"):
        metrology.level_three_points(z, [(1.0, 1.0), (2.0, 2.0)])


def test_apply_exclusion_mask():
    z = np.arange(25.0, dtype=np.float32).reshape(5, 5)
    out = metrology.apply_exclusion_mask(z, (1, 1, 4, 4))
    assert np.all(np.isnan(out[1:4, 1:4]))
    assert np.all(np.isfinite(out[[0, 4], :])) and np.all(np.isfinite(out[:, [0, 4]]))
    assert np.array_equal(out[[0, 4], :], z[[0, 4], :])  # outside the rectangle nothing changes
    out2 = metrology.apply_exclusion_mask(z, (1, 1, 4, 4), invert=True)
    assert np.array_equal(out2[1:4, 1:4], z[1:4, 1:4])
    assert np.all(np.isnan(out2[[0, 4], :])) and np.all(np.isnan(out2[:, [0, 4]]))


def test_step_height_regions_recovers_planted_step_and_standard_error():
    ny, nx = 60, 80
    rng = np.random.default_rng(5)
    xx = np.arange(nx)[None, :].repeat(ny, 0).astype(np.float64)
    noise = 0.1
    z = (0.05 * xx + np.where(xx > 40.0, 3.5, 0.0) + rng.normal(0, noise, (ny, nx))).astype(np.float32)
    res = metrology.step_height_regions(z, (0, 0, 30, 60), (50, 0, 80, 60))
    assert res["step_height"] == pytest.approx(3.5, abs=0.02)
    assert res["abs_step_height"] == pytest.approx(3.5, abs=0.02)
    assert res["std_a"] == pytest.approx(noise, rel=0.1) and res["std_b"] == pytest.approx(noise, rel=0.1)
    # standard error of the difference of two region means: noise * sqrt(1/n_a + 1/n_b)
    assert res["standard_error"] == pytest.approx(noise * math.sqrt(1 / 1800 + 1 / 1800), rel=0.1)
    assert (res["pixels_a"], res["pixels_b"]) == (1800, 1800)


def test_step_height_regions_controls():
    z = np.arange(25.0, dtype=np.float32).reshape(5, 5)
    with pytest.raises(ValueError, match="at least 9 valid pixels"):
        metrology.step_height_regions(z, (0, 0, 2, 2), (2, 2, 4, 4))


# ---------------------------------------------------------------- 5. Topographic watershed hills

def _hill_lattice(rows=5, cols=6, d=24, sig=4.0, tiny_bump=True):
    """rows x cols Gaussian hills of height 1 on a d-pitch lattice, plus (optionally) one 0.02 high bump that the
    5 % Sz Wolf pruning must discard. Returns the surface and the number of hills not touching the border."""
    ny, nx = rows * d, cols * d
    yy, xx = np.mgrid[0:ny, 0:nx].astype(np.float64)
    z = np.zeros((ny, nx))
    for j in range(rows):
        for i in range(cols):
            z += np.exp(-((xx - (i + 0.5) * d) ** 2 + (yy - (j + 0.5) * d) ** 2) / (2 * sig ** 2))
    if tiny_bump:
        z += 0.02 * np.exp(-((xx - 2 * d) ** 2 + (yy - 2 * d) ** 2) / (2 * sig ** 2))  # a cell corner
    return z.astype(np.float32), (rows - 2) * (cols - 2)


def test_grains_topographic_counts_separated_hills_exactly_and_sizes_them():
    z, expect = _hill_lattice()
    g = metrology.grains_topographic(z, 0.5)
    assert expect == 12
    assert g["topo_grain_count"] == expect  # the 0.02 bump is pruned, the border-touching hills are excluded
    # every hill's catchment is its lattice cell, d^2 px^2 (areas within 10 %)
    assert g["topo_mean_grain_area_um2"] == pytest.approx(24 * 24 * 0.5 ** 2, rel=0.10)
    lab = g["topo_watershed_labels"]
    assert lab.shape == z.shape
    assert len(np.unique(lab)) == 30  # 30 hills = 30 catchments (the bump belongs to a neighbour)


def test_grains_topographic_count_follows_the_number_of_hills_and_the_pixel_size():
    z, _ = _hill_lattice(rows=4, cols=4, tiny_bump=False)
    a = metrology.grains_topographic(z, 1.0)
    assert a["topo_grain_count"] == 4  # 2 x 2 interior hills
    b = metrology.grains_topographic(z, 2.0)
    assert b["topo_grain_count"] == 4
    assert b["topo_mean_grain_area_um2"] == pytest.approx(4 * a["topo_mean_grain_area_um2"], rel=1e-9)


def test_grains_topographic_flat_surface_and_mostly_invalid_surface():
    z = np.full((40, 40), 2.5, np.float32)
    assert metrology.grains_topographic(z, 1.0) == {"topo_grain_count": 0}
    z2 = np.full((40, 40), np.nan, np.float32)
    z2[:5] = 1.0
    assert metrology.grains_topographic(z2, 1.0) == {}


# ---------------------------------------------------------------- integration

def test_analyze_reports_volume_and_summit_keys_and_withholds_slopes_in_frames():
    z, _ = _lattice(noise_sigma=0.01)
    edf = np.full(z.shape, 128, np.float32)
    r = metrology.analyze(z, edf, 0.5, "um", 2.0, 0, 0.0, 0.0)
    assert r["Sdq"] == pytest.approx(metrology.root_mean_square_gradient(r["sl_surface"], 0.5))
    n = np.isfinite(r["sl_surface"]).sum()
    assert r["Sds_per_mm2"] == pytest.approx(_brute_force_summit_count(r["sl_surface"]) / (n * 0.25) * 1e6,
                                             rel=1e-6)
    want = metrology.functional_parameters(r["sl_surface"])
    for k in ("Vmp", "Vmc", "Vvc", "Vvv"):
        assert r[k] == pytest.approx(want[k])
    r2 = metrology.analyze(z, edf, 0.5, "frames", 2.0, 0, 0.0, 0.0)
    assert "Sdr_pct" not in r2 and "Sdq" not in r2 and "Ssc_per_um" not in r2
    r3 = metrology.analyze(z, edf, 0.5, "um", 2.0, 0, 0.0, 0.0, features=True)
    assert r3["Spd_per_mm2"] == pytest.approx(25 / (60.0 ** 2) * 1e6, rel=0.02)
    assert r3["eligibility"]["feature"] == "ok"
