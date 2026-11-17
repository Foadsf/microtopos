"""Slice 4 metrology extensions (ISO 21920 1D profiling, ISO 25178 Sdq / volumes / feature params,
interactive 3-point levelling / exclusion mask / step height, topographic watershed grains).

Fast, deterministic tests on analytic surfaces with negative controls. The full slow suite is not run here.
"""

import math

import numpy as np
import pytest

from microtopos import metrology


def _sinusoid(n: int, step_um: float, amplitude_um: float, period_um: float) -> np.ndarray:
    x = np.arange(n, dtype=np.float64) * step_um
    return amplitude_um * np.sin(2.0 * np.pi * x / period_um)


def _add_nans(prof: np.ndarray, rng: np.random.Generator, fraction: float) -> np.ndarray:
    p = prof.copy()
    p[rng.choice(p.size, round(fraction * p.size), replace=False)] = np.nan
    return p


# ---------------------------------------------------------------- 1. ISO 21920 1D profile filtering

def test_profile_parameters_separates_roughness_from_waviness():
    """Lambda_c nesting index must separate the long wave (waviness) from the short wave (roughness)."""
    rng = np.random.default_rng(3)
    step_um = 0.5
    n = 4000
    prof = _sinusoid(n, step_um, 1.0, 20.0) + 0.5 * _sinusoid(n, step_um, 1.0, 200.0)
    prof = _add_nans(prof, rng, 0.02)
    # lam_c between the two periods: the L-filter passes the 200 um wave and rejects the 20 um wave
    res = metrology.profile_parameters(prof, step_um, lam_s_um=0.0, lam_c_um=60.0)
    assert res, "profile_parameters should return metrics for a valid profile"
    assert "roughness_profile" in res and "waviness_profile" in res and "primary_profile" in res
    assert np.isfinite(res["roughness_profile"]).sum() > 100
    # Roughness ~= the short sinusoid: Ra -> 2A/pi, Rq -> A/sqrt(2)
    assert res["Ra"] == pytest.approx(2.0 / math.pi, rel=0.06), res["Ra"]
    assert res["Rq"] == pytest.approx(1.0 / math.sqrt(2), rel=0.06), res["Rq"]
    assert res["Rz"] == pytest.approx(2.0, rel=0.12), res["Rz"]
    # Waviness ~= the long sinusoid (94% of its amplitude survives the Gaussian at lambda_c = 60 um)
    assert "Wa" in res and "Wq" in res
    assert res["Wa"] == pytest.approx(2.0 * 0.5 / math.pi, rel=0.20), res["Wa"]


def test_profile_parameters_all_nan_and_too_short():
    """Negative control: an all-NaN or too-short profile must return empty metrics, never crash."""
    assert metrology.profile_parameters(np.full(100, np.nan), 0.5) == {}
    assert metrology.profile_parameters(np.array([1.0, 2.0, np.nan]), 0.5) == {}
    assert metrology.profile_parameters(np.array([1.0, np.nan, 3.0, 4.0]), 0.5) == {}


def test_profile_parameters_unresolvable_cutoff_degrades_gracefully():
    """A 1D cutoff below 4 px is skipped (no raise) and the primary-profile metrics survive."""
    p = _sinusoid(300, 0.5, 1.0, 10.0)  # lam_c = 1.0 um = 2 px < 4 px -> skipped
    res = metrology.profile_parameters(p, 0.5, lam_c_um=1.0)
    assert res and res["Ra"] == pytest.approx(2.0 / math.pi, rel=0.05)


def test_gaussian_filter_1d_nan_properties():
    p = _sinusoid(600, 0.2, 2.0, 25.0)
    p[120:130] = np.nan
    out = metrology.gaussian_filter_1d_nan(p, 5.0, 0.2)
    assert out.shape == p.shape and out.dtype == np.float32
    assert np.isnan(out[120:130]).all()
    finite = np.isfinite(out)
    assert finite.sum() > 500
    # Low-pass at lambda_c=5 um barely attenuates the 25 um period sine (transmission ~0.973)
    amp = float(np.sqrt(2 * np.mean((out[finite] - np.mean(out[finite])) ** 2)))
    assert amp == pytest.approx(2.0, rel=0.05)


# ---------------------------------------------------------------- 2. Sdq and volume parameters

def test_root_mean_square_gradient_matches_analytical_slope():
    xx = np.arange(80)[None, :].repeat(60, 0).astype(np.float32)
    # Plane of slope 0.37 um/px in x: Sdq == |slope| (dimensionless)
    z = (0.37 * xx).astype(np.float32)
    sdq = metrology.root_mean_square_gradient(z, 1.0)
    assert sdq is not None and abs(sdq - 0.37) < 1e-6
    # Sinusoid A sin(2 pi x / L): rms slope = A * 2 pi / L / sqrt(2)
    z = np.sin(2.0 * np.pi * xx / 40.0).astype(np.float32)
    sdq = metrology.root_mean_square_gradient(z, 1.0)
    assert sdq is not None and abs(sdq - (2.0 * np.pi / 40.0) / math.sqrt(2)) < 0.005
    # Flat surface: zero slope
    assert metrology.root_mean_square_gradient(np.zeros((10, 10), np.float32), 1.0) == 0.0


def test_functional_parameters_volumes_non_negative():
    rng = np.random.default_rng(1)
    z = rng.normal(0.0, 1.0, (100, 100)).astype(np.float32)
    fp = metrology.functional_parameters(z)
    assert fp, "functional_parameters needs >= 200 valid pixels"
    for key in ("Sk", "Spk", "Svk", "Smr1_pct", "Smr2_pct", "Vmp", "Vmc", "Vvc", "Vvv"):
        assert key in fp and np.isfinite(fp[key]), key
    assert all(fp[k] >= 0.0 for k in ("Vmp", "Vmc", "Vvc", "Vvv"))
    # A spiky surface has material volume above the 10% level and none below the 80% level
    spike = np.zeros((100, 100), np.float32)
    spike[45:55, 45:55] = 5.0
    fp2 = metrology.functional_parameters(spike)
    assert fp2["Vmp"] > 0.0 and fp2["Vvv"] == 0.0


# ---------------------------------------------------------------- 3. Feature parameters

def test_feature_parameters_gaussian_bumps_and_pits():
    rng = np.random.default_rng(2)
    ny, nx = 120, 160
    yy, xx = np.mgrid[0:ny, 0:nx].astype(np.float32)
    z = np.zeros((ny, nx), np.float32)
    for cy, cx, amp, sig in [(30.0, 40.0, 3.0, 7.0), (90.0, 110.0, 4.0, 9.0),
                             (60.0, 20.0, -2.5, 6.0), (100.0, 60.0, -3.0, 8.0)]:
        z += amp * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2.0 * sig ** 2))
    z += (rng.normal(0.0, 0.02, (ny, nx))).astype(np.float32)
    feat = metrology.feature_parameters(z, 1.0)
    assert feat, "feature_parameters needs >= 64 valid pixels"
    assert feat["Sds_per_mm2"] > 0.0 and feat["Spd_per_mm2"] > 0.0
    assert "Ssc_per_um" in feat and feat["Ssc_per_um"] > 0.0
    assert feat["S10z"] >= feat["S5p"] + feat["S5v"] - 1e-6
    assert feat["S10z"] >= 0.0 and feat["S5p"] > 0.0 and feat["S5v"] > 0.0


def test_feature_parameters_flat_plane_is_zero():
    """Negative control: a flat plane has no summits, pits or curvatures."""
    z = np.full((80, 80), 2.5, np.float32)
    feat = metrology.feature_parameters(z, 1.0)
    assert feat and feat["Sds_per_mm2"] == 0.0
    assert "Ssc_per_um" not in feat and "Spd_per_mm2" not in feat
    assert "S5p" not in feat and "S5v" not in feat and "S10z" not in feat


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
    out2 = metrology.apply_exclusion_mask(z, (1, 1, 4, 4), invert=True)
    assert np.all(np.isfinite(out2[1:4, 1:4]))
    assert np.all(np.isnan(out2[[0, 4], :])) and np.all(np.isnan(out2[:, [0, 4]]))


def test_step_height_regions_recovers_planted_step():
    ny, nx = 60, 80
    xx = np.arange(nx)[None, :].repeat(ny, 0).astype(np.float64)
    z = (0.05 * xx + np.where(xx > 40.0, 3.5, 0.0)).astype(np.float32)
    res = metrology.step_height_regions(z, (0, 0, 30, 60), (50, 0, 80, 60))
    assert res is not None
    assert abs(res["step_height"] - 3.5) < 0.05, res
    assert abs(res["abs_step_height"] - 3.5) < 0.05
    assert res["standard_error"] >= 0.0
    assert res["pixels_a"] >= 9 and res["pixels_b"] >= 9


def test_step_height_regions_controls():
    z = np.arange(25.0, dtype=np.float32).reshape(5, 5)
    with pytest.raises(ValueError, match="at least 9 valid pixels"):
        metrology.step_height_regions(z, (0, 0, 2, 2), (2, 2, 4, 4))


# ---------------------------------------------------------------- 5. Topographic watershed grains

def test_grains_topographic_segments_planted_domes():
    ny, nx = 100, 120
    yy, xx = np.mgrid[0:ny, 0:nx].astype(np.float32)
    z = np.zeros((ny, nx), np.float32)
    for cy, cx, hgt, sig in [(30.0, 40.0, 4.0, 10.0), (75.0, 85.0, 5.0, 12.0)]:
        z += hgt * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2.0 * sig ** 2))
    g = metrology.grains_topographic(z, 1.0, h_rel=0.15)
    assert g and g["topo_grain_count"] >= 1, g
    assert g["topo_watershed_labels"].shape == z.shape
    assert g["topo_mean_grain_area_um2"] > 0.0


def test_grains_topographic_flat_surface():
    """Negative control: a flat surface has no gradient structure, hence zero grains."""
    z = np.full((40, 40), 2.5, np.float32)
    g = metrology.grains_topographic(z, 1.0)
    assert g == {"topo_grain_count": 0}, g


# ---------------------------------------------------------------- integration

def test_analyze_reports_slice4_keys():
    rng = np.random.default_rng(7)
    h = rng.normal(0.0, 1.0, (80, 100)).astype(np.float32)
    edf = np.full(h.shape, 128, np.float32)
    r = metrology.analyze(h, edf, 1.0, "um", 2.0)
    assert "Sdq" in r and r["Sdq"] is not None
    for key in ("Vmp", "Vmc", "Vvc", "Vvv", "Sds_per_mm2", "topo_grain_count"):
        assert key in r, key
    assert all(r[k] >= 0.0 for k in ("Vmp", "Vmc", "Vvc", "Vvv"))
    assert r["eligibility"]["feature"] == "ok"
    # withheld in non-um units
    r2 = metrology.analyze(h, edf, 1.0, "frames", 2.0)
    assert "Sdr_pct" not in r2 and "Sdq" not in r2
