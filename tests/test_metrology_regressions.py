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
