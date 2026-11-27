"""Regression tests for metrology defects found in review of PR #14 (profile roughness, volume parameters, Sds).

Each value is compared with a closed form or an independently built ground truth (numpy, not the cv2 filter the
module uses), with tolerances that a 5 % error fails.
"""

import math

import numpy as np
import pytest

from microtopos import metrology

# ---------------------------------------------------------------- ISO 21920 profile: cut-off count, end effect, Rz


def _iso_weighting(step_um: float, lam_c_um: float) -> np.ndarray:
    """ISO 16610-21 Gaussian weighting function s(x) = exp(-pi (x / (alpha lc))^2) / (alpha lc), sampled to +/- 2 lc."""
    alpha = math.sqrt(math.log(2) / math.pi)
    half = round(2 * lam_c_um / step_um)
    x = np.arange(-half, half + 1) * step_um
    s = np.exp(-np.pi * (x / (alpha * lam_c_um)) ** 2)
    return s / s.sum()


def _sections(step_um: float, n: int, lam_c_um: float, n_sc: int) -> list[np.ndarray]:
    """Sample indices of n_sc sections of length lc centred on a profile of n samples (independent of the module)."""
    x = np.arange(n) * step_um
    x0 = 0.5 * ((n - 1) * step_um - n_sc * lam_c_um)
    edges = x0 + lam_c_um * np.arange(n_sc + 1)
    return [np.nonzero((x >= edges[k] - 1e-9) & ((x < edges[k + 1] - 1e-9) if k < n_sc - 1 else (x <= edges[k + 1] + 1e-9)))[0]
            for k in range(n_sc)]


def test_default_cutoff_of_a_sixth_of_the_line_always_gives_five_sections():
    """5 sections of λc are evaluated after λc/2 run-in and run-out, so a line needs 6 λc and the GUI's default λc is
    the line length / 6. Rounding of the float step must never make that 5.9999999 cut-offs: v0.10 refused about
    3.6 % of lines with its default (n=1517, length 3231.28 µm was one). Every random line must be accepted and give
    Rq = A / sqrt(2) of a sine well inside the pass band over exactly 5 sections."""
    n, length = 1517, 3231.28  # the case found in review, a binary-inexact line length
    # a line shorter than 6 cut-offs is refused (v0.10 accepted it and evaluated the filter's end effect)
    why = metrology.profile_cutoff_problem(n, length / (n - 1), length / 5.5)
    assert why is not None and "5.5 cut-offs" in why and "6" in why
    assert metrology.profile_cutoff_problem(n, length / (n - 1), length / 6.0) is None
    rng = np.random.default_rng(0)
    for _ in range(3000):
        n = int(rng.integers(300, 3000))  # λc >= 50 samples, so the sine below has >= 12 samples per period
        length = float(rng.uniform(5.0, 5000.0))
        step = length / (n - 1)  # as the GUI computes it from the line
        lam_c = length / 6.0
        assert metrology.profile_cutoff_problem(n, step, lam_c) is None, (n, length)
        p = np.sin(2 * np.pi * np.arange(n) * step / (lam_c / 4))  # transmission 1 - 1.5e-5 through the L-filter
        res = metrology.profile_parameters(p, step, lam_c_um=lam_c)
        assert res["Rq"] == pytest.approx(1 / math.sqrt(2), rel=0.01), (n, length)
        assert res["n_sections"] == 5, (n, length)
        assert res["evaluation_length_um"] == pytest.approx(5 * lam_c, rel=1e-12)


def test_roughness_excludes_the_gaussian_end_effect():
    """1 µm sine of 5 µm plus a 5 µm waviness of 8 λc, over 6 λc (λc = 50 µm). The parameters must equal those of the
    ideal (infinitely long) Gaussian roughness profile, Σ a_i (1 - T_i) sin(...), over the central 5 λc. v0.10
    evaluated the whole line including the filter's end artefacts: Rp 1.13, Rv 1.43 instead of about 1.0."""
    step, lam_c = 0.5, 50.0
    n = round(6 * lam_c / step) + 1
    x = np.arange(n) * step

    def t(lam):  # ISO 16610-21 amplitude transmission of the Gaussian low-pass
        return math.exp(-math.pi * (metrology.ALPHA * lam_c / lam) ** 2)

    for a_short in (1.0, 0.0):  # with roughness, and pure waviness (whose roughness residual is 5 (1 - T(400)))
        p = a_short * np.sin(2 * np.pi * x / 5.0) + 5.0 * np.sin(2 * np.pi * x / (8 * lam_c) + 0.7)
        ideal = (a_short * (1 - t(5.0)) * np.sin(2 * np.pi * x / 5.0)
                 + 5.0 * (1 - t(8 * lam_c)) * np.sin(2 * np.pi * x / (8 * lam_c) + 0.7))
        secs = _sections(step, n, lam_c, 5)
        ev = np.concatenate(secs)
        r = ideal[ev] - ideal[ev].mean()
        want = {"Rq": float(np.sqrt(np.mean(r * r))), "Ra": float(np.mean(np.abs(r))), "Rp": float(r.max()),
                "Rv": float(-r.min()),
                "Rz": float(np.mean([ideal[s].max() - ideal[s].min() for s in secs]))}
        res = metrology.profile_parameters(p, step, lam_c_um=lam_c)
        for k, v in want.items():
            assert res[k] == pytest.approx(v, rel=0.03), (a_short, k, res[k], v)
        assert res["n_sections"] == 5 and res["evaluation_length_um"] == pytest.approx(250.0)


def test_rz_averages_sections_of_one_cutoff_each():
    """ISO 21920-2 Rz is the mean peak-to-valley height of sections of length λc. Ten sections, each holding one spike,
    alternately 1.0 and 0.2 µm high: Rz is about 0.6, not the 1.0 of five double-length sections (v0.10). The
    ground truth is the spike train minus its convolution with the ISO weighting function (numpy, zero outside)."""
    step, lam_c = 0.5, 50.0
    n = round(11 * lam_c / step) + 1  # 10 sections + λc/2 run-in and run-out
    p = np.zeros(n)
    for k in range(10):  # one spike at the centre of every section: x = 50, 100, ..., 500 µm
        p[round((lam_c + k * lam_c) / step)] = 1.0 if k % 2 == 0 else 0.2
    ideal = p - np.convolve(p, _iso_weighting(step, lam_c), mode="same")
    secs = _sections(step, n, lam_c, 10)
    rz_true = float(np.mean([ideal[s].max() - ideal[s].min() for s in secs]))
    assert rz_true == pytest.approx(0.6, rel=0.02)
    res = metrology.profile_parameters(p, step, lam_c_um=lam_c)
    assert res["Rz"] == pytest.approx(rz_true, rel=0.01), (res["Rz"], rz_true)
    assert res["n_sections"] == 10


# ---------------------------------------------------------------- ISO 25178-2 volume parameters


def _vm(a, m):
    return a * (math.sin(math.pi * m) / math.pi - m * math.cos(math.pi * m))


def _vv(a, m):
    return a * ((1 - m) * math.cos(math.pi * m) + math.sin(math.pi * m) / math.pi)


@pytest.mark.parametrize("period_px, phase_px, tol", [
    (200, 0.25, {"Vmp": 0.01, "Vmc": 0.001, "Vvc": 0.001, "Vvv": 0.003}),
    (200, 0.37, {"Vmp": 0.01, "Vmc": 0.001, "Vvc": 0.001, "Vvv": 0.003}),
    (200, 0.5, {"Vmp": 0.01, "Vmc": 0.001, "Vvc": 0.001, "Vvv": 0.003}),
    (80, 0.5, {"Vmp": 0.04, "Vmc": 0.002, "Vvc": 0.002, "Vvv": 0.01}),  # the case found in review
])
def test_volume_parameters_of_a_sampled_sine_match_the_closed_form_at_any_phase(period_px, phase_px, tol):
    """A = 2 sine, a whole number of periods: Vm(m) = A [sin(pi m)/pi - m cos(pi m)], Vv(m) = A [(1-m) cos(pi m) +
    sin(pi m)/pi], p = 10 %, q = 80 %. v0.10 looked up c(mr) without interpolation on a grid offset by half a sample
    from the one Sk uses: at 80 px per period and 0.5 px phase Vmp was +40 % and Vvv -17.6 %. The remaining error is
    the sampling of the sine itself (its top is not sampled), so it shrinks with the period."""
    a = 2.0
    x = np.arange(4 * period_px) + phase_px
    z = np.tile(a * np.sin(2 * np.pi * x / period_px), (100, 1)).astype(np.float32)
    want = {"Vmp": _vm(a, 0.1), "Vmc": _vm(a, 0.8) - _vm(a, 0.1), "Vvc": _vv(a, 0.1) - _vv(a, 0.8),
            "Vvv": _vv(a, 0.8)}
    fp = metrology.functional_parameters(z)
    for k, v in want.items():
        assert fp[k] == pytest.approx(v, rel=tol[k]), (k, fp[k], v)


# ---------------------------------------------------------------- Sds: numerical-noise floor


def _edf(z):
    fin = np.nan_to_num(z, nan=float(np.nanmin(z)))
    return ((fin - fin.min()) / max(float(np.ptp(fin)), 1e-12) * 255).astype(np.float32)


def test_summit_density_of_a_tilted_plane_is_zero_and_of_an_egg_crate_is_exact():
    """A float32 plane keeps a rounding residual of ~1e-7 µm after plane removal; v0.10 counted its maxima as summits
    (Sds about 6.5e5 /mm²). The closed form is 0. An egg-crate A cos(2 pi x/L) cos(2 pi y/L) on the same plane has two
    summits per L x L cell, at (iL, jL) and ((i+1/2)L, (j+1/2)L): the floor must not lose any of them."""
    h, w, d, lam = 200, 300, 0.5, 20
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    plane = 0.01 * xx + 0.02 * yy + 3.0
    area_mm2 = h * w * (d * 1e-3) ** 2
    r = metrology.analyze(plane.astype(np.float32), _edf(plane), d, "µm", 1.0, form_order=1, lam_s_um=0.0)
    assert r["Sds_per_mm2"] == 0.0
    assert "Ssc_per_um" not in r
    f = metrology.analyze(plane.astype(np.float32), _edf(plane), d, "µm", 1.0, form_order=1, lam_s_um=0.0,
                          features=True)
    assert f["Sds_per_mm2"] == 0.0 and "Spd_per_mm2" not in f and f["topo_grain_count"] == 0

    crate = np.cos(2 * np.pi * xx / lam) * np.cos(2 * np.pi * yy / lam) + plane
    rows = [r_ for r_ in range(0, h, lam // 2) if 1 <= r_ <= h - 2]
    cols = [c_ for c_ in range(0, w, lam // 2) if 1 <= c_ <= w - 2]
    n_summits = sum(1 for r_ in rows for c_ in cols if (r_ // (lam // 2)) % 2 == (c_ // (lam // 2)) % 2)
    assert n_summits == 9 * 14 + 10 * 15
    e = metrology.analyze(crate.astype(np.float32), _edf(crate), d, "µm", 1.0, form_order=1, lam_s_um=0.0)
    assert e["Sds_per_mm2"] == pytest.approx(n_summits / area_mm2, rel=1e-9)


# ---------------------------------------------------------------- interactive tools: mask rectangle, step height, 3-point


@pytest.mark.parametrize("rect, cols, rows", [
    ((-30, 10, -2, 20), (0, 0), (10, 20)),  # entirely left of the image
    ((10, -40, 30, -1), (10, 30), (0, 0)),  # entirely above
    ((90, 10, 120, 20), (80, 80), (10, 20)),  # entirely right
    ((10, 60, 30, 90), (10, 30), (50, 50)),  # entirely below
    ((-5, 10, 3, 20), (0, 3), (10, 20)),  # partly left: only columns 0..2 are on the image
    ((3, 20, -5, 10), (0, 3), (10, 20)),  # the same, corners given the other way round
    ((70, 45, 95, 70), (70, 80), (45, 50)),  # partly over the bottom-right corner
])
def test_mask_rectangle_off_the_image_is_clipped_not_wrapped(rect, cols, rows):
    """A clipped end coordinate of -2 used to become the slice end -2 (Python counts from the end), so a rectangle
    dragged entirely off the left edge masked 780 of 4000 pixels. Only the on-image part may be masked."""
    z = np.zeros((50, 80), np.float32)
    want = np.zeros((50, 80), bool)
    want[rows[0]:rows[1], cols[0]:cols[1]] = True  # built with already clipped, non-negative bounds
    assert np.array_equal(np.isnan(metrology.apply_exclusion_mask(z, rect)), want)
    assert np.array_equal(np.isnan(metrology.apply_exclusion_mask(z, rect, invert=True)), ~want)


def test_step_region_off_the_image_is_refused():
    """A region dragged off the image used to wrap round and select most of the image (3950 px), and a step was
    reported. It has no pixels and must be refused."""
    z = np.zeros((50, 80), np.float32)
    z[:, 40:] = 1.0
    with pytest.raises(ValueError, match="region A has no pixels"):
        metrology.step_height_regions(z, (-30, 0, -1, 50), (50, 0, 70, 50))
    with pytest.raises(ValueError, match="region B has no pixels"):
        metrology.step_height_regions(z, (5, 0, 30, 50), (50, -20, 70, -3))
    # a region partly off the image keeps its on-image part: an exact step of 1 with no tilt
    m = metrology.step_height_regions(z, (-10, 5, 10, 45), (50, 5, 90, 45))
    assert m["step_height"] == pytest.approx(1.0, abs=1e-9)
    assert (m["pixels_a"], m["pixels_b"]) == (10 * 40, 30 * 40)


@pytest.mark.parametrize("ra, rb", [
    ((5, 5, 25, 6), (5, 30, 25, 31)),  # one row each: y is a combination of the two indicators
    ((5, 5, 6, 30), (30, 5, 31, 30)),  # one column each: x is
])
def test_step_height_on_parallel_one_pixel_regions_is_refused(ra, rb):
    """With one-row (or one-column) regions the tilt column is collinear with the region indicators; lstsq returned
    its minimum-norm solution, a step of 0.19 instead of 1.0 with a standard error of 3e-17. Refuse instead."""
    yy, xx = np.mgrid[0:40, 0:40]
    z = np.where(yy >= 20 if ra[3] - ra[1] == 1 else xx >= 20, 1.0, 0.0).astype(np.float32)
    with pytest.raises(ValueError, match="parallel lines"):
        metrology.step_height_regions(z, ra, rb)


def test_step_height_on_a_row_and_a_column_is_exact():
    """A one-row region and a one-column region are not parallel, so tilt and step are separable and the exact step
    of a tilted ideal terrace is recovered: the rank test must not refuse a design that is identifiable."""
    yy, xx = np.mgrid[0:40, 0:40].astype(np.float64)
    z = (np.where(yy >= 20, 1.0, 0.0) + 0.01 * xx - 0.02 * yy).astype(np.float64)
    m = metrology.step_height_regions(z, (5, 5, 25, 6), (30, 25, 31, 36))
    assert m["step_height"] == pytest.approx(1.0, abs=1e-9)


def _ols_contrast_se(z, ma, mb):
    """Independent OLS standard error of c_b - c_a for z = c_a I_a + c_b I_b + gx x + gy y (raw coordinates)."""
    yy, xx = np.mgrid[0:z.shape[0], 0:z.shape[1]].astype(np.float64)
    both = ma | mb
    X = np.column_stack([ma[both], mb[both], xx[both], yy[both]]).astype(np.float64)
    y = z[both].astype(np.float64)
    beta = np.linalg.solve(X.T @ X, X.T @ y)
    r = y - X @ beta
    s2 = r @ r / (y.size - 4)
    C = np.linalg.inv(X.T @ X)
    return beta[1] - beta[0], math.sqrt(s2 * (C[0, 0] + C[1, 1] - 2 * C[0, 1]))


def test_step_height_standard_error_includes_the_fitted_tilt():
    """Two 80 x 10 px terraces 80 rows apart on a tilted surface. The step is strongly correlated with the fitted
    y-tilt, so its scatter over repeated noise is 15 x the sqrt(var_a/n_a + var_b/n_b) that v0.10 reported.
    The reported SE must equal the OLS contrast standard error (independent computation) and match the Monte Carlo
    scatter of the step over 300 noise realisations (sampling error of that SD about 4 %)."""
    h, w, sigma = 100, 100, 0.05
    rng = np.random.default_rng(0)
    yy, xx = np.mgrid[0:h, 0:w]
    truth = np.where(yy >= 50, 1.0, 0.0) + 0.002 * xx + 0.003 * yy
    rect_a, rect_b = (10, 0, 90, 10), (10, 90, 90, 100)
    ma = np.zeros((h, w), bool)
    ma[0:10, 10:90] = True
    mb = np.zeros((h, w), bool)
    mb[90:100, 10:90] = True
    steps, ses = [], []
    for _ in range(300):
        z = (truth + rng.normal(0, sigma, truth.shape)).astype(np.float32)
        m = metrology.step_height_regions(z, rect_a, rect_b)
        step_ref, se_ref = _ols_contrast_se(z, ma, mb)
        assert m["step_height"] == pytest.approx(step_ref, abs=1e-9)
        assert m["standard_error"] == pytest.approx(se_ref, rel=1e-6)
        steps.append(m["step_height"])
        ses.append(m["standard_error"])
    assert np.mean(steps) == pytest.approx(1.0, abs=0.01)
    assert np.mean(ses) == pytest.approx(np.std(steps, ddof=1), rel=0.15)
