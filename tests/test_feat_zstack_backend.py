"""Slice 3 feature acceptance: breathing-scale registration, OpenCL focus backends, the sff DOF ruler,
tiled z-stack reconstruction with RGB EDF, and stage-coupled LiveEngine sweeps.

These tests run fast (no simulator video I/O beyond tiny in-memory renders) and are the acceptance
commands for Slice 3 together with the four regression files.
"""

from __future__ import annotations

import itertools
import math
import sys

import cv2
import numpy as np
import pytest
from conftest import require_opencl

import microtopos.optics as _optics  # monkeypatch target for the wavelength sweep
from microtopos import objectives, zstack
from microtopos.backend import focus_map_backend, guided_filter_backend, parity
from microtopos.calibration import CalibrationStore, bgr_to_luma
from microtopos.camera import SimulatorSource
from microtopos.focus import focus_map
from microtopos.kinematics import Override
from microtopos.live import LiveEngine
from microtopos.registration import (
    PairRegistration,
    estimate_breathing_scales,
    register_sweep,
    warp_scale_shift,
    warp_to_reference,
)
from microtopos.sff import theoretical_fwhm_um
from microtopos.simulator import SpecimenSimulator
from microtopos.stage import SimulatedStageDriver
from microtopos.zstack import ZSettings

OBJ = objectives.resolve("10x / 0.25", objset=objectives.default_set())


def _texture(seed: int, shape: tuple[int, int] = (120, 160)) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.clip(rng.normal(128, 35, shape), 0, 255).astype(np.uint8)


def _sim_sweep(seed: int, n_frames: int = 9, size: tuple[int, int] = (240, 320)):
    """Luma frames plus simulator ground-truth z for one monotonic focus sweep."""
    sim = SpecimenSimulator(seed, OBJ, out_w=size[1], out_h=size[0])
    zs = sim.sweep_trajectory(n_frames=n_frames)
    frames, truths = [], []
    for z_k in zs:  # render() never moves focus: drive the sweep ourselves so labels match the frames
        sim.focus_z_um = float(z_k)
        bgr, truth = sim.render()
        frames.append(bgr_to_luma(bgr))
        truths.append(truth.focus_z_um)
    return frames, np.array(truths, dtype=np.float64)


def _zoom_frames(texture: np.ndarray, n: int, beta: float, ref: int, shape: tuple[int, int]) -> list[np.ndarray]:
    """Frames of `shape` showing the (larger) texture magnified by exp(beta * (k - ref)) about the frame centre:
    an exact focus-breathing model with no empty borders."""
    h, w = shape
    c = np.array([(w - 1) / 2.0, (h - 1) / 2.0])
    out = []
    for k in range(n):
        s = math.exp(beta * (k - ref))
        A = np.eye(2) / s  # frame pixel -> texture pixel
        t = np.array([texture.shape[1] / 2.0, texture.shape[0] / 2.0]) - A @ c
        M = np.hstack([A, t[:, None]]).astype(np.float32)
        out.append(cv2.warpAffine(texture, M, (w, h), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                                  borderMode=cv2.BORDER_REFLECT))
    return out


def _smooth_texture(seed: int, shape: tuple[int, int]) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return cv2.GaussianBlur(rng.normal(128, 40, shape).astype(np.float32), (0, 0), 1.5)

# ------------------------------------------------------------------------------------------------ B1


def test_breathing_scales_recover_exact_linear_drift():
    z = np.arange(17.0)
    beta = 0.001
    ref = 8
    true = np.exp(beta * (z - z[ref]))
    pairs = [PairRegistration(np.eye(2, 3, dtype=np.float32), 0.99, True, "ecc",
                              scale_step=float(true[k + 1] / true[k])) for k in range(len(z) - 1)]
    scales = estimate_breathing_scales(pairs, ref, z.tolist())
    assert scales.shape == (17,)
    assert scales[ref] == pytest.approx(1.0, abs=1e-9)
    assert np.allclose(scales, true, rtol=1e-3, atol=1e-6)


def test_breathing_scales_are_clamped_inside_the_registration_envelope():
    z = np.arange(41.0)
    beta = 0.02
    ref = 20
    true = np.exp(beta * (z - z[ref]))
    pairs = [PairRegistration(np.eye(2, 3, dtype=np.float32), 0.99, True, "ecc",
                              scale_step=float(true[k + 1] / true[k])) for k in range(len(z) - 1)]
    scales = estimate_breathing_scales(pairs, ref, z.tolist())
    assert float(np.max(np.abs(scales - 1.0))) <= 0.015 + 1e-6
    assert float(np.median(np.diff(scales) / np.diff(z))) > 0


def test_breathing_scales_negative_controls_return_unity():
    z = [0.0, 1.0, 2.0, 3.0]
    flat = [PairRegistration(np.eye(2, 3, dtype=np.float32), 0.5, False, "ecc") for _ in range(3)]
    assert np.all(estimate_breathing_scales(flat, 1, z) == 1.0)
    unity = [PairRegistration(np.eye(2, 3, dtype=np.float32), 0.99, True, "ecc", scale_step=1.0)
             for _ in range(3)]
    assert np.all(estimate_breathing_scales(unity, 1, z) == 1.0)
    assert np.all(estimate_breathing_scales([], 0, []) == np.array([1.0]))


def test_register_sweep_observes_breathing_scale_steps():
    texture = _smooth_texture(2, (480, 640))
    frames = _zoom_frames(texture, 11, 0.01, 5, (240, 320))
    z = [0.5 * k for k in range(len(frames))]
    warps, pairs = register_sweep(frames, ref=5, factor=1, workers=1,
                                  correct_breathing=True, z=z)
    assert len(warps) == len(frames)  # identity warp at the reference frame plus one per pair
    assert len(pairs) == len(frames) - 1
    expected = math.exp(0.01)
    assert sum(p.ok for p in pairs) == len(pairs)
    assert all(abs((p.scale_step - 1) - (expected - 1)) <= 0.2 * (expected - 1) for p in pairs)  # within 20 % of the zoom step
    # without breathing correction the per-frame scale model is not folded into the warps,
    # but ECC still sees the magnification change between consecutive frames
    _warps2, pairs2 = register_sweep(frames, ref=5, factor=1, workers=1, correct_breathing=False, z=z)
    assert all(abs((p.scale_step - 1) - (expected - 1)) <= 0.3 * (expected - 1) for p in pairs2 if p.ok)


# ---------------------------------------------------------------- breathing: scale against known zoom, warning, end to end

def test_breathing_warps_reproduce_a_known_zoom_and_realign_the_frames():
    n, ref, beta = 15, 7, 0.001
    frames = _zoom_frames(_smooth_texture(3, (480, 640)), n, beta, ref, (240, 320))
    z = [float(k) for k in range(n)]
    true = np.exp(beta * (np.arange(n) - ref))
    res = {}
    for on in (False, True):
        warps, _pairs = register_sweep(frames, ref=ref, factor=1, workers=1, correct_breathing=on, z=z)
        scale = np.array([warp_scale_shift(W, (240, 320))[0] for W in warps])
        aligned = [warp_to_reference(frames[k], warps[k]) for k in range(n)]
        sl = (slice(20, -20), slice(30, -30))
        res[on] = (scale, float(np.mean([np.abs(aligned[k][sl] - frames[ref][sl]).mean() for k in range(n)])))
    signal = float(np.max(np.abs(true - 1)))  # 0.7 %
    assert float(np.max(np.abs(res[True][0] - true))) <= 0.2 * signal, (res[True][0] - 1, true - 1)
    assert np.all(res[False][0] == 1.0)  # off: the warps carry no magnification at all
    assert res[True][1] < 0.5 * res[False][1], (res[True][1], res[False][1])  # the frames really line up better


def test_breathing_beyond_the_validated_range_is_reported_not_silently_clamped():
    n, ref = 15, 7
    z = [float(k) for k in range(n)]
    big = _zoom_frames(_smooth_texture(3, (480, 640)), n, 0.004, ref, (240, 320))  # 5.6 % over the sweep
    notes: list[str] = []
    warps, _ = register_sweep(big, ref=ref, factor=1, workers=1, correct_breathing=True, z=z, notes=notes)
    assert any("exceeds" in m and "limited to 1.5%" in m for m in notes), notes
    span = max(warp_scale_shift(W, (240, 320))[0] for W in warps) - min(warp_scale_shift(W, (240, 320))[0] for W in warps)
    assert span == pytest.approx(0.015, abs=1e-4)  # what is applied is exactly the stated limit
    small = _zoom_frames(_smooth_texture(3, (480, 640)), n, 0.001, ref, (240, 320))  # 1.4 %: inside the range
    notes = []
    register_sweep(small, ref=ref, factor=1, workers=1, correct_breathing=True, z=z, notes=notes)
    assert notes == []
    # unusable evidence is reported too: flat frames cannot be measured
    flat = [np.full((240, 320), 100.0, np.float32) for _ in range(n)]
    notes = []
    register_sweep(flat, ref=ref, factor=1, workers=1, correct_breathing=True, z=z, notes=notes)
    assert any("no correction applied" in m for m in notes), notes


def _breathing_sweep(seed: int, per_um: float, blur: bool, n: int = 24):
    sim = SpecimenSimulator(seed, OBJ, out_w=320, out_h=240)
    sim.breathing_per_um = per_um / OBJ.dof_um
    zs = sim.sweep_trajectory(n_frames=n)
    frames, truths = [], []
    for z_k in zs:
        sim.focus_z_um = float(z_k)
        bgr, truth = sim.render(blur=blur)
        frames.append(bgr_to_luma(bgr))
        truths.append(truth)
    return sim, frames, truths


@pytest.mark.parametrize("seed", [1, 2])
def test_breathing_correction_in_the_simulator_recovers_the_true_drift_and_improves_alignment(seed):
    _sim, frames, truths = _breathing_sweep(seed, 1.0e-3, blur=False)  # 1.1 % drift, geometry only (no defocus)
    n, ref = len(frames), len(frames) // 2
    z = [t.focus_z_um for t in truths]
    true = np.array([t.breathing_scale / truths[ref].breathing_scale for t in truths])
    res = {}
    for on in (False, True):
        warps, _ = register_sweep(frames, ref=ref, factor=1, workers=1, correct_breathing=on, z=z)
        aligned = [warp_to_reference(frames[k], warps[k]) for k in range(n)]
        sl = (slice(20, -20), slice(20, -20))
        res[on] = (np.array([warp_scale_shift(W, (240, 320))[0] for W in warps]),
                   float(np.mean([np.abs(aligned[k][sl] - frames[ref][sl]).mean() for k in range(n)])))
    assert float(np.max(np.abs(res[True][0] - true))) <= 0.2 * float(np.max(np.abs(true - 1)))
    assert res[True][1] < 0.9 * res[False][1], (res[True][1], res[False][1])


def test_reconstruct_applies_and_reports_breathing_correction():
    _sim, frames, truths = _breathing_sweep(2, 1.0e-3, blur=True, n=24)
    z = np.array([t.focus_z_um for t in truths])
    ref = len(frames) // 2
    true = np.array([t.breathing_scale / truths[ref].breathing_scale for t in truths])
    on = zstack.reconstruct(frames, z, OBJ, 1.0, "um", settings=ZSettings(correct_breathing=True, allow_tiling=False))
    off = zstack.reconstruct(frames, z, OBJ, 1.0, "um", settings=ZSettings(correct_breathing=False, allow_tiling=False))
    assert off["breathing_scale_range"] == 0.0
    assert on["breathing_scale_range"] == pytest.approx(float(np.ptp(true)), rel=0.2)  # kills correct_breathing=False
    # the same sweep with a drift beyond the limit: reported in the result's warnings
    _sim, frames, truths = _breathing_sweep(2, 2.5e-3, blur=True, n=24)
    z = np.array([t.focus_z_um for t in truths])
    r = zstack.reconstruct(frames, z, OBJ, 1.0, "um", settings=ZSettings(correct_breathing=True, allow_tiling=False))
    assert any("Focus breathing" in m and "exceeds" in m for m in r["warnings"]), r["warnings"]
    assert not any("Focus breathing" in m for m in on["warnings"])


def _height_error(sim, truths, r, ref) -> float:
    """RMS height error (um) against the simulator's ground truth in the reference frame's geometry, away from the
    border, after removing the global offset."""
    M = sim._matrix(truths[ref].breathing_scale, truths[ref].content_shift_px, 0.0, 0.0)
    truth = sim._sample(sim.world_height_pf, M)
    h = r["height"]
    ok = np.isfinite(h)
    ok[:12], ok[-12:], ok[:, :12], ok[:, -12:] = False, False, False, False
    d = (h - truth)[ok]
    d = d - np.median(d)
    return float(np.sqrt(np.mean(d * d)))


def test_breathing_correction_height_error_end_to_end_with_and_without():
    """Measured, not assumed: on simulated sweeps with 1.1 % breathing the height error is the same with and
    without the correction, because it is set by the focus-peak fit (about 0.15 DOF), not by the 1-2 px edge
    misalignment. The correction therefore stays experimental and default-off; it must at least not hurt."""
    errs = {False: [], True: []}
    for seed in (1, 2, 3):
        sim, frames, truths = _breathing_sweep(seed, 1.0e-3, blur=True, n=24)
        z = np.array([t.focus_z_um for t in truths])
        for on in (False, True):
            r = zstack.reconstruct(frames, z, OBJ, 1.0, "um", settings=ZSettings(correct_breathing=on, allow_tiling=False))
            errs[on].append(_height_error(sim, truths, r, len(frames) // 2))
    off, on = float(np.mean(errs[False])), float(np.mean(errs[True]))
    assert off < 0.25 * OBJ.dof_um and on < 0.25 * OBJ.dof_um, (off, on)  # both are accurate: far below one DOF
    assert on <= off * 1.1, (off, on)  # the experimental correction must not make heights worse
    print(f"height RMS error vs ground truth (um): off {off:.3f}  on {on:.3f}  (DOF {OBJ.dof_um} um)")


# ------------------------------------------------------------------------------------------------ B2


PARITY_TOL = 1e-5  # relative L2 between the OpenCL result and the CPU reference (float32 arithmetic order only)


@pytest.mark.parametrize("measure", ["DoG band-pass", "Tenengrad", "SML"])
@pytest.mark.parametrize("window", [5, 9, 31])
@pytest.mark.parametrize("shape", [(120, 160), (97, 211)])
def test_focus_map_backend_opencl_equals_the_cpu_reference(measure, window, shape):
    """The OpenCL path must compute the SAME measure as focus.focus_map (same kernels, window, border mode)."""
    require_opencl()
    img = _texture(3, shape).astype(np.float32)
    psf = OBJ.psf_sigma_px(1.0)
    out_cpu, used_cpu = focus_map_backend(img, measure, window, psf, "cpu")
    assert used_cpu == "cpu"
    assert np.array_equal(out_cpu, focus_map(img, measure, window, psf))
    out_cl, used_cl = focus_map_backend(img, measure, window, psf, "opencl")
    assert used_cl == "opencl", f"{measure}: OpenCL was available but the CPU ran"
    assert out_cl.shape == img.shape and np.isfinite(out_cl).all()
    rel, _mx = parity(out_cpu, out_cl)
    assert rel <= PARITY_TOL, f"{measure} window {window} {shape}: OpenCL differs from the CPU reference, rel L2 {rel:.3e}"


@pytest.mark.parametrize("measure", ["DoG band-pass", "Tenengrad", "SML"])
def test_focus_map_backend_opencl_honours_the_window(measure):
    """A window of 9 and 31 must give different maps on OpenCL, as they do on the CPU."""
    require_opencl()
    img = _texture(3).astype(np.float32)
    a, ua = focus_map_backend(img, measure, 9, 1.0, "opencl")
    b, ub = focus_map_backend(img, measure, 31, 1.0, "opencl")
    assert ua == ub == "opencl"
    assert parity(a, b)[0] > 0.05


@pytest.mark.parametrize("measure", ["Vollath F4", "Vollath F5"])
def test_focus_map_backend_reports_cpu_for_measures_without_an_opencl_path(measure):
    img = _texture(3).astype(np.float32)
    out, used = focus_map_backend(img, measure, 9, 1.0, "opencl")
    assert used == "cpu"
    assert np.array_equal(out, focus_map(img, measure, 9, 1.0))

def test_focus_map_backend_unknown_measure_raises():
    img = _texture(3)
    with pytest.raises(ValueError):
        focus_map_backend(img, "not-a-measure", 9, 1.0, "opencl")
    with pytest.raises(ValueError):
        focus_map(img, "not-a-measure", 9, 1.0)


def test_guided_filter_backend_parity_and_nan():
    rng = np.random.default_rng(0)
    p = rng.normal(100.0, 20.0, (64, 80)).astype(np.float32)
    nan_mask = np.zeros((64, 80), bool)
    nan_mask[::7, ::7] = True
    p[nan_mask] = np.nan
    w = rng.uniform(0.2, 1.0, (64, 80)).astype(np.float32)
    guide = cv2.GaussianBlur(np.nan_to_num(p), (0, 0), 3)
    out_cpu, used_cpu = guided_filter_backend(p, w, guide, 5, 1e-2, "cpu")
    assert used_cpu == "cpu" and out_cpu.shape == p.shape
    assert np.isfinite(out_cpu).all()  # NaN in p is weighted 0: finite output, no crash
    require_opencl()
    out_cl, used_cl = guided_filter_backend(p, w, guide, 5, 1e-2, "opencl")
    assert used_cl == "opencl"
    assert out_cl.shape == p.shape and np.isfinite(out_cl).all()
    rel, _mx = parity(out_cpu, out_cl)
    assert rel <= 1e-4, f"guided filter: OpenCL differs from the CPU reference, rel L2 {rel:.3e}"
    assert np.array_equal(np.isnan(out_cpu), np.isnan(out_cl))

def test_benchmark_reports_every_measure_with_parity_and_a_working_negative_control():
    from microtopos import backend

    r = backend.benchmark(96, 64, reps=2)
    assert set(r["measures"]) == {"DoG band-pass", "Tenengrad", "SML", "guided filter"}
    assert all(m["cpu_ms"] > 0 for m in r["measures"].values())
    require_opencl()
    for name, m in r["measures"].items():
        assert m["opencl_with_transfer_ms"] > 0, name
        assert m["parity"]["ok"] and m["parity"]["negative_control_detected"], (name, m["parity"])
    # the control is a real control: a 1 % corrupted output must fail the very comparison the bench uses
    a = focus_map(_texture(3).astype(np.float32), "SML", 9, 1.0)
    assert backend._parity_entry(a, a)["ok"] and backend._parity_entry(a, a)["negative_control_detected"]
    assert not backend._parity_entry(a, a * 1.01)["ok"]

def test_guided_filter_backend_shape_mismatch_and_all_nan():
    p = np.zeros((10, 10), np.float32)
    w = np.ones((10, 10), np.float32)
    g = np.zeros((8, 8), np.float32)
    with pytest.raises(ValueError):
        guided_filter_backend(p, w, g, 5)
    out, _ = guided_filter_backend(np.full((8, 8), np.nan, np.float32),
                                   np.ones((8, 8), np.float32), np.zeros((8, 8), np.float32), 5)
    assert np.isnan(out).all()


# ------------------------------------------------------------------------------------------------ B3


def _make_obj(na: float, mag: float = 10.0) -> objectives.Objective:
    return objectives.Objective(f"ruler {na:g}", float(mag), float(na))


def _measure_fn(obj: objectives.Objective, um_per_px: float):
    psf = obj.psf_sigma_px(um_per_px)
    return lambda img: focus_map(img, "DoG band-pass", 9, psf)


def test_theoretical_fwhm_decreases_monotonically_with_na():
    tex = _texture(5)
    fwhms = []
    for na in (0.25, 0.35, 0.5, 0.7, 0.9):
        obj = _make_obj(na)
        f = theoretical_fwhm_um(obj, 1.0, _measure_fn(obj, 1.0), tex)
        assert f is not None and np.isfinite(f) and f > 0
        fwhms.append(f)
    assert all(later < earlier for earlier, later in itertools.pairwise(fwhms))


def test_theoretical_fwhm_is_of_order_dof(monkeypatch):
    tex = _texture(5)
    for lam in (0.45, 0.55, 0.70):
        monkeypatch.setattr(_optics, "WAVELENGTH_UM", lam)
        obj = _make_obj(0.5)  # dof is recomputed from the wavelength
        f = theoretical_fwhm_um(obj, 1.0, _measure_fn(obj, 1.0), tex)
        assert f is not None and np.isfinite(f) and f > 0
        # the ruler's width in um of defocus scales with the depth of focus (geometric blur model)
        assert 0.5 * obj.dof_um < f < 4.0 * obj.dof_um, (f, obj.dof_um)


@pytest.mark.parametrize(("name", "um_per_px", "cycles_per_px", "hand_fwhm_um"), [
    # hand values: FWHM = 2 sqrt(2 ln 2) * um_per_px / (pi * f * tan(theta)), tan(theta) = tan(asin(NA)) for air
    #   10x / 0.25: tan = 0.25 / sqrt(1 - 0.25^2) = 0.258199  ->  2.354820 * 1.00 / (pi / 16 * 0.258199) = 46.449 um
    #   40x / 0.85: tan = 0.85 / sqrt(1 - 0.85^2) = 1.613570  ->  2.354820 * 0.25 / (pi / 32 * 1.613570) =  3.716 um
    ("10x / 0.25", 1.0, 1 / 16, 46.449),
    ("40x / 0.85", 0.25, 1 / 32, 3.716),
])
def test_theoretical_fwhm_matches_the_hand_computed_gaussian_response(name, um_per_px, cycles_per_px, hand_fwhm_um):
    """A pure sinusoid and a peak-to-peak measure respond as exp(-2 pi^2 f^2 s^2) to a blur s = |dz| tan(theta) / 2:
    a Gaussian in defocus whose FWHM is known in closed form, independent of the code under test."""
    obj = objectives.resolve(name, objset=objectives.default_set())
    x = np.arange(256)
    tex = np.tile((128 + 60 * np.cos(2 * np.pi * cycles_per_px * x)).astype(np.float32), (256, 1))
    got = theoretical_fwhm_um(obj, um_per_px, lambda im: np.full(im.shape, float(im.max() - im.min()), np.float32), tex)
    assert got is not None
    assert got == pytest.approx(hand_fwhm_um, rel=0.01), (got, hand_fwhm_um)

def test_theoretical_fwhm_negative_controls_return_none():
    tex = _texture(5)
    m = _measure_fn(_make_obj(0.5), 1.0)
    assert theoretical_fwhm_um(None, 1.0, m, tex) is None
    assert theoretical_fwhm_um(_make_obj(0.0), 1.0, m, tex) is None
    assert theoretical_fwhm_um(_make_obj(0.1), 1.0, m, tex) is None  # below 0.24 NA: no ruler
    blank = np.zeros((64, 64), np.uint8)
    assert theoretical_fwhm_um(_make_obj(0.5), 1.0,
                               lambda img: focus_map(img, "DoG band-pass", 9, 1.0), blank) is None


# ------------------------------------------------------------------------------------------------ B4


_SKIP_KEYS = ("timings", "timestamp", "n_strips")  # run-dependent bookkeeping, not results


def _strip_context(r_one_piece: dict, width: int):
    n, (ph, pw) = r_one_piece["frames_used"], r_one_piece["edf_proc"].shape
    halo = zstack.strip_halo_rows("DoG band-pass", r_one_piece["focus_window_px"], OBJ.psf_sigma_px(1.0 * width / pw))
    return n, ph, pw, halo


def _force_strips(monkeypatch, r_one_piece: dict, width: int, n_strips: int) -> None:
    """Make the memory query report just enough memory for `n_strips` strips and not for fewer."""
    n, ph, pw, halo = _strip_context(r_one_piece, width)
    avail = int(zstack.estimate_tiled_peak_bytes(n, ph, pw, n_strips, halo) / 0.8) + 1
    assert zstack.plan_strips(n, ph, pw, halo, avail, 0.8, True) == n_strips
    monkeypatch.setattr(zstack, "available_memory_bytes", lambda: avail)


def _colour_sweep(seed: int, size: tuple[int, int], n_frames: int):
    sim = SpecimenSimulator(seed, OBJ, out_w=size[1], out_h=size[0])
    frames, colours, zs = [], [], []
    for z_k in sim.sweep_trajectory(n_frames=n_frames):
        sim.focus_z_um = float(z_k)
        bgr, truth = sim.render()
        frames.append(bgr_to_luma(bgr))
        colours.append(bgr)
        zs.append(truth.focus_z_um)
    return frames, colours, np.array(zs, dtype=np.float64)


@pytest.mark.parametrize(("size", "n_frames", "n_strips"), [((240, 320), 9, 3), ((360, 480), 13, 12)])
def test_strip_wise_reconstruction_equals_the_one_piece_reconstruction(monkeypatch, size, n_frames, n_strips):
    """Strip-wise (tiled) reconstruction to bound peak memory must give the same result as one piece, output by
    output. Exact equality is expected (every strip carries its focus halo, the global steps run on the stitched
    maps); the only intended difference is the capture-resolution fusion, which reads the weights as uint16 (EDF
    within 1 grey level). The numeric tolerance 1e-4 um (~1e-5 of the height spread) only absorbs float32 order of
    operations on another CPU."""
    frames, colours, zs = _colour_sweep(11, size, n_frames)
    one = zstack.reconstruct(frames, zs, OBJ, 1.0, "um", color_frames=colours,
                             settings=ZSettings(allow_tiling=False))
    assert one["n_strips"] == 1
    _force_strips(monkeypatch, one, size[1], n_strips)
    tiled = zstack.reconstruct(frames, zs, OBJ, 1.0, "um", color_frames=colours, settings=ZSettings(allow_tiling=True))
    assert tiled["n_strips"] == n_strips
    assert set(tiled) == set(one)
    for k in one:
        if k in _SKIP_KEYS or k in ("edf", "edf_rgb"):
            continue
        a, b = one[k], tiled[k]
        if isinstance(a, np.ndarray) and a.dtype.kind == "f":
            assert a.shape == b.shape, k
            assert np.array_equal(np.isnan(a), np.isnan(b)), f"{k}: NaN/support masks differ"
            np.testing.assert_allclose(b, a, rtol=1e-6, atol=1e-4, equal_nan=True, err_msg=k)
        elif isinstance(a, np.ndarray):
            assert np.array_equal(a, b), k
        elif isinstance(a, float):
            assert b == pytest.approx(a, rel=1e-6, abs=1e-6, nan_ok=True), k
        else:
            assert a == b, k
    assert tiled["support_fraction"] == one["support_fraction"] and one["support_fraction"] > 0.6
    assert float(np.nanmax(np.abs(tiled["height"] - one["height"]))) <= 1e-4
    assert not np.array_equal(tiled["height_raw"], tiled["height"], equal_nan=True)  # raw fit is not the refined map
    assert float(np.abs(tiled["edf"] - one["edf"]).max()) <= 1.0
    assert int(np.abs(tiled["edf_rgb"].astype(int) - one["edf_rgb"].astype(int)).max()) <= 1


def test_strip_wise_reconstruction_lowers_peak_memory(monkeypatch):
    import gc
    import tracemalloc

    frames, zs = _sim_sweep(11, n_frames=25, size=(480, 640))

    def peak(settings):
        gc.collect()
        tracemalloc.start()
        try:
            r = zstack.reconstruct(frames, zs, OBJ, 1.0, "um", settings=settings)
            return tracemalloc.get_traced_memory()[1], r
        finally:
            tracemalloc.stop()

    p_one, one = peak(ZSettings(allow_tiling=False))
    _force_strips(monkeypatch, one, 640, 16)
    p_tiled, tiled = peak(ZSettings(allow_tiling=True))
    assert tiled["n_strips"] == 16
    assert np.array_equal(tiled["height"], one["height"], equal_nan=True)
    assert p_tiled < 0.6 * p_one, (p_tiled / 2**20, p_one / 2**20)


def test_memory_guard_refuses_a_job_that_cannot_fit_even_in_strips(monkeypatch):
    frames, zs = _sim_sweep(11, n_frames=9, size=(240, 320))
    one = zstack.reconstruct(frames, zs, OBJ, 1.0, "um", settings=ZSettings(allow_tiling=False))
    n, ph, pw, halo = _strip_context(one, 320)
    floor = zstack.estimate_tiled_peak_bytes(n, ph, pw, ph, halo)
    # budget below even the one-row strip: refused although tiling is allowed (the default)
    monkeypatch.setattr(zstack, "available_memory_bytes", lambda: int(0.5 * floor / 0.8))
    with pytest.raises(MemoryError, match="in strips"):
        zstack.reconstruct(frames, zs, OBJ, 1.0, "um", settings=ZSettings(allow_tiling=True))
    # fits in strips but not in one piece: refused when tiling is switched off (the base-branch refusal)
    _force_strips(monkeypatch, one, 320, 4)
    with pytest.raises(MemoryError, match="lower the processing width"):
        zstack.reconstruct(frames, zs, OBJ, 1.0, "um", settings=ZSettings(allow_tiling=False))
    # a job that needs many strips (more than 16, the base branch's cap) is processed, not refused
    need40 = int(zstack.estimate_tiled_peak_bytes(25, 480, 640, 40, 14) / 0.8) + 1
    assert zstack.plan_strips(25, 480, 640, 14, need40, 0.8, True) == 40
    # plenty of memory: one piece, no strips
    monkeypatch.setattr(zstack, "available_memory_bytes", lambda: 64 * 2**30)
    assert zstack.reconstruct(frames, zs, OBJ, 1.0, "um")["n_strips"] == 1


@pytest.mark.skipif(sys.platform != "win32", reason="Windows memory query (ctypes GlobalMemoryStatusEx)")
def test_windows_available_memory_matches_the_os_report():
    import subprocess

    def os_free_bytes() -> int:
        out = subprocess.run(["powershell", "-NoProfile", "-Command",
                              "(Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory"],
                             capture_output=True, text=True, timeout=60, check=False).stdout.strip()
        return int(out) * 1024  # reported in KiB

    # free memory moves while the suite runs (it moved 450 MB between two back-to-back readings under load): the
    # query must lie between two OS readings taken just before and just after it, so a unit error (KiB vs bytes) or
    # total-instead-of-available is still caught
    before = os_free_bytes()
    got = zstack.available_memory_bytes()
    after = os_free_bytes()
    assert got is not None
    assert 0.75 * min(before, after) <= got <= 1.25 * max(before, after), (before, got, after)

def test_zstack_rgb_edf_is_a_sharp_colour_image():
    sim = SimulatorSource(9, OBJ, 160, 120)
    zs = sim.sim.sweep_trajectory(n_frames=7)
    frames, colors, truths = [], [], []
    for _z in zs:  # render() never moves focus: drive the sweep ourselves
        sim.sim.focus_z_um = float(_z)
        bgr, truth = sim.sim.render()
        frames.append(bgr_to_luma(bgr))
        colors.append(bgr)
        truths.append(truth.focus_z_um)
    r = zstack.reconstruct(frames, np.array(truths, dtype=np.float64), OBJ, 1.0, "um",
                           color_frames=colors)
    assert r["edf_rgb"] is not None
    assert r["edf_rgb"].shape == (120, 160, 3)
    assert r["edf_rgb"].dtype == np.uint8
    assert float(r["edf_rgb"].std()) > 5.0
    ch = r["edf_rgb"].reshape(-1, 3).astype(np.float32)
    colour = float(np.abs(ch[:, 0] - ch[:, 1]).mean() + np.abs(ch[:, 1] - ch[:, 2]).mean())
    assert colour > 1.0  # not a gray-scale duplicate


def test_zstack_rejects_mismatched_colour_frames():
    frames, zs = _sim_sweep(1, n_frames=5, size=(80, 80))
    with pytest.raises(ValueError):
        zstack.reconstruct(frames, zs, OBJ, 1.0, "um", color_frames=frames[:2])


def _banded_colour_stack(dtype):
    """9 frames, 3 vertical bands; band r is sharpest in frame SHARP[r] and blurred in proportion to the distance
    from it. Colour frame k = luma * tint[k] (BGR), the tints of the sharpest frames being strongly different, so
    the fused colour of a band identifies which frame it came from and the channel order."""
    sharp = (2, 4, 6)
    h, w, n = 96, 180, 9
    rng = np.random.default_rng(5)
    base = [cv2.GaussianBlur(rng.uniform(40, 215, (h, w)).astype(np.float32), (0, 0), 1.0) for _ in range(3)]
    tint = rng.uniform(0.35, 1.0, (n, 3)).astype(np.float32)  # BGR
    tint[2], tint[4], tint[6] = (0.2, 0.5, 1.0), (1.0, 0.3, 0.3), (0.3, 1.0, 0.4)  # red-ish, blue-ish, green-ish
    band = np.minimum(np.arange(w) * 3 // w, 2)
    lum, bgr = [], []
    for k in range(n):
        y = np.empty((h, w), np.float32)
        for r in range(3):
            s = 2.5 * abs(k - sharp[r])
            img = base[r] if s == 0 else cv2.GaussianBlur(base[r], (0, 0), s)
            y[:, band == r] = img[:, band == r]
        lum.append(y)
        bgr.append(y[..., None] * tint[k][None, None, :])
    expected = np.empty((h, w, 3), np.float32)  # RGB
    for r in range(3):
        expected[:, band == r] = (bgr[sharp[r]][:, band == r])[..., ::-1]
    frames = [np.clip(b, 0, 255) for b in bgr]
    colour = [np.rint(f).astype(np.uint8) if dtype == np.uint8 else np.rint(f * 257.0).astype(np.uint16) for f in frames]
    return lum, colour, np.arange(float(n)), expected, band


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16])
def test_colour_edf_reproduces_the_colours_of_the_sharpest_frames(dtype):
    lum, colour, zs, expected, band = _banded_colour_stack(dtype)
    r = zstack.reconstruct(lum, zs, OBJ, 1.0, "um", color_frames=colour)
    rgb = r["edf_rgb"]
    assert rgb.dtype == np.uint8 and rgb.shape == expected.shape
    inner = np.zeros(band.shape, bool)
    for b in range(3):  # interior of each band: away from the seams between the bands and from the image border
        cols = np.nonzero(band == b)[0]
        inner[cols[25:-25] if b == 1 else (cols[8:-25] if b == 0 else cols[25:-8])] = True
    diff = np.abs(rgb[8:-8][:, inner].astype(np.float32) - expected[8:-8][:, inner])
    assert float(diff.mean()) < 2.0, float(diff.mean())
    assert float(np.percentile(diff, 99)) < 12.0, float(np.percentile(diff, 99))
    # the three bands really are three different colours (channel order: band 0 red-dominant, 1 blue, 2 green)
    means = [rgb[8:-8][:, np.nonzero(inner & (band == b))[0]].reshape(-1, 3).mean(0) for b in range(3)]
    assert means[0].argmax() == 0 and means[1].argmax() == 2 and means[2].argmax() == 1, means
    assert float(rgb.std()) > 20  # a 16-bit input must not saturate (it did: mean 254.9, std 0.35)


def test_colour_edf_grey_colour_frames_are_accepted():
    lum, _colour, zs, expected, _band = _banded_colour_stack(np.uint8)
    grey = [np.rint(l_).astype(np.uint8) for l_ in lum]
    r = zstack.reconstruct(lum, zs, OBJ, 1.0, "um", color_frames=grey)
    assert r["edf_rgb"].shape == expected.shape
    assert int(np.abs(r["edf_rgb"][..., 0].astype(int) - r["edf_rgb"][..., 2]).max()) == 0  # R == B for grey input


def test_rescale_heights_full_invariants():
    h = np.random.default_rng(0).normal(0, 1, (32, 32)).astype(np.float32)
    edf = np.full((32, 32), 128, np.float32)
    cutoff = objectives.resolve("10x / 0.25").lateral_cutoff_um
    from microtopos import metrology
    m_def = metrology.analyze(h, edf, 0.5, "frames", cutoff, lam_s_um=None)
    r = {
        **m_def, "height": h, "height_raw": h.copy(), "edf_proc": edf, "um_per_px": 0.5,
        "objective": "10x / 0.25", "z_values": [0.0, 1.0, 2.0], "z_first": 0.0, "z_last": 2.0,
        "fwhm_z": 4.0, "frame_indices": [10, 11, 12], "sample_positions": [1.0, 2.0, 3.0],
        "sigma_inst": 0.1, "repeat_offset": 0.2, "repeat_overlap": 0.9, "sigma_inst_lt_Sq_over_10": True,
    }
    r2 = zstack.rescale_heights(r, -2.0, "um")
    assert r2["filter_notes"] == m_def["filter_notes"]  # default unresolvable filter_notes preserved
    assert r2["z_values"] == [-4.0, -2.0, -0.0]  # scaled then reversed
    assert r2["z_first"] == -4.0 and r2["z_last"] == -0.0
    assert r2["frame_indices"] == [12, 11, 10] and r2["sample_positions"] == [3.0, 2.0, 1.0]
    assert r2["fwhm_z"] == pytest.approx(8.0)  # scaled by abs(factor)
    assert "sigma_inst" not in r2 and "repeat_offset" not in r2  # repeatability keys cleared
    assert any("Repeatability cleared" in w for w in r2["warnings"])


# ------------------------------------------------------------------------------------------------ B5


def test_stage_coupled_live_engine_z_sweep_populates_z_flags_and_z_labels():
    sim = SimulatorSource(4, OBJ, 160, 120)
    stage = SimulatedStageDriver(sim_source=sim)
    eng = LiveEngine(CalibrationStore(None), OBJ)
    eng.sync_callbacks = True
    caps: list = []
    eng.on_sweep = caps.append
    eng.set_source(sim)
    eng.sm.override = Override.Z
    trajectory = sim.sim.sweep_trajectory(n_frames=14)
    for i, z_um in enumerate(trajectory):
        stage.move_to(z_um=float(z_um))  # the stage drive and the simulator share one Z
        assert stage.pose().z_um == pytest.approx(z_um)
        bgr, truth = sim.sim.render()
        eng.process_frame(bgr, i * 0.04, i + 1, truth=truth)
    eng.flush()
    assert len(caps) == 1
    cap = caps[0]
    assert cap.manual is True  # sweep captured under manual Z override
    assert cap.simulated is True  # frames came from a simulator source
    assert cap.z is not None and len(cap.z) == len(cap.frames) == len(trajectory)
    # the label VALUES are the stage Z that was commanded for each frame, in order (not merely "not None")
    assert np.allclose(cap.z, trajectory, atol=1e-9), (list(cap.z), list(trajectory))
    assert cap.seqs == list(range(1, len(trajectory) + 1))  # one label per source frame, none dropped or reordered
    # and they are usable: reconstruction with them reports a known orientation and the commanded z range
    r = zstack.reconstruct(cap.frames, cap.z, OBJ, sim.sim.um_per_px, "um")
    assert r["z_orientation"] == "known"
    assert (r["z_first"], r["z_last"]) == pytest.approx((trajectory[0], trajectory[-1]))


def test_stage_coupled_negative_control_without_truth_there_are_no_z_labels_and_no_simulated_flag():
    sim = SimulatorSource(4, OBJ, 160, 120)
    eng = LiveEngine(CalibrationStore(None), OBJ)
    eng.sync_callbacks = True
    caps: list = []
    eng.on_sweep = caps.append
    eng.set_source(sim)
    eng.sm.override = Override.Z
    for i, _z in enumerate(sim.sim.sweep_trajectory(n_frames=14)):
        sim.sim.focus_z_um = float(_z)
        bgr, _truth = sim.sim.render()
        eng.process_frame(bgr, i * 0.04, i + 1, truth=None)  # a real camera: no ground truth
    eng.flush()
    assert len(caps) == 1
    assert caps[0].manual is True  # the override still marks it manual
    assert caps[0].simulated is False and caps[0].z is None  # but nothing may invent z labels or a simulator flag


def test_live_sweep_ended_event_carries_the_manual_flag_from_override():
    sim = SimulatorSource(5, OBJ, 160, 120)
    eng = LiveEngine(CalibrationStore(None), OBJ)
    eng.sync_callbacks = True
    caps: list = []
    eng.on_sweep = caps.append
    eng.sm.override = Override.Z
    for i in range(12):
        bgr, truth = sim.sim.render()
        eng.process_frame(bgr, i * 0.04, i + 1, truth=truth)
    eng.sm.override = Override.AUTO  # ending a manual sweep must flag the capture manual=True
    bgr, truth = sim.sim.render()
    eng.process_frame(bgr, 12 * 0.04, 13, truth=truth)
    assert len(caps) == 1
    assert caps[0].manual is True


def test_live_manual_z_sweep_ended_by_switching_to_another_override_is_flagged_manual():
    sim = SimulatorSource(5, OBJ, 160, 120)
    eng = LiveEngine(CalibrationStore(None), OBJ)
    eng.sync_callbacks = True
    caps: list = []
    eng.on_sweep = caps.append
    eng.sm.override = Override.Z
    for i, z_um in enumerate(sim.sim.sweep_trajectory(n_frames=12)):
        sim.sim.focus_z_um = float(z_um)
        bgr, truth = sim.sim.render()
        eng.process_frame(bgr, i * 0.04, i + 1, truth=truth)
    eng.sm.override = Override.PAN  # leaving manual Z for a forced mode ends the sweep through the other branch
    bgr, truth = sim.sim.render()
    eng.process_frame(bgr, 12 * 0.04, 13, truth=truth)
    assert len(caps) == 1
    assert caps[0].manual is True
    assert caps[0].z is not None and len(caps[0].z) == 12

def test_live_request_calibration_refuses_simulator_against_disk_store(tmp_path):
    sim = SimulatorSource(6, OBJ, 120, 90)
    eng = LiveEngine(CalibrationStore(tmp_path / "cal"), OBJ)
    eng.set_source(sim)
    eng.request_calibration("dark", 4)
    bgr, truth = sim.sim.render()
    eng.process_frame(bgr, 0.0, 1, truth=truth)
    assert eng._calib_req is None
    assert eng.calib_error and "disk" in eng.calib_error.lower()
    assert not (tmp_path / "cal" / "calibration.json").exists()


def test_live_dark_frames_are_recorded_for_non_simulator_source(tmp_path):
    eng = LiveEngine(CalibrationStore(tmp_path / "cal2"), OBJ)
    eng.set_source(None)
    eng.request_calibration("dark", 3)
    frame = _texture(8)
    for i in range(3):
        eng.process_frame(frame, i * 0.04, i + 1, truth=None)
    assert eng._calib_req is None
    assert (tmp_path / "cal2" / "calibration.json").exists()


# ------------------------------------------------------------------------------------------------ B6


def _colour_bgr(seed: int, shape: tuple[int, int] = (120, 160)) -> np.ndarray:
    tex = _texture(seed, shape)
    return np.stack([tex, tex, tex], axis=-1)  # grayscale colour frame, uint8


def test_live_engine_rgb_sweep_records_colour_frames_and_rgb_mosaic():
    sim = SimulatorSource(4, OBJ, 160, 120)
    eng = LiveEngine(CalibrationStore(None), OBJ)
    eng.color_mode = True
    eng.sync_callbacks = True
    caps: list = []
    eng.on_sweep = caps.append
    eng.set_source(sim)
    eng.sm.override = Override.Z
    trajectory = sim.sim.sweep_trajectory(n_frames=14)
    for i, z_um in enumerate(trajectory):  # drive the sweep ourselves, like the stage-coupled test
        sim.sim.focus_z_um = float(z_um)
        bgr, truth = sim.sim.render()
        eng.process_frame(bgr, i * 0.04, i + 1, truth=truth)
    eng.flush()
    assert len(caps) == 1
    cap = caps[0]
    assert cap.manual is True and cap.simulated is True
    assert cap.color_frames is not None
    assert len(cap.color_frames) == len(cap.frames) >= 8
    assert all(isinstance(c, np.ndarray) and c.shape == (120, 160, 3) and c.dtype == np.uint8
               for c in cap.color_frames)

    # a colour keyframe lands in the RGB mosaic canvas and in the live preview
    eng.sm.override = Override.PAN
    bgr, _ = sim.sim.render()
    eng.process_frame(bgr, 100 * 0.04, 100, truth=None)
    snap = eng.snapshot()
    assert snap.preview_rgb is not None and snap.preview_rgb.shape == (120, 160, 3)
    eng._render_now()
    view, _partitions, _nkf, _npatches = eng.mosaic_image()
    assert view.ndim == 3 and view.shape[2] == 4  # RGBA preview
    assert bool(view[..., 3].any())  # colour content was blended into the canvas
    assert bool(eng.mosaic.tiles_rgb)


def test_live_engine_accepts_16bit_colour_frames_in_a_sweep():
    sim = SimulatorSource(7, OBJ, 160, 120)
    eng = LiveEngine(CalibrationStore(None), OBJ)
    eng.color_mode = True
    eng.sync_callbacks = True
    caps: list = []
    eng.on_sweep = caps.append
    eng.set_source(sim)
    eng.sm.override = Override.Z
    trajectory = sim.sim.sweep_trajectory(n_frames=14)
    for i, z_um in enumerate(trajectory):
        sim.sim.focus_z_um = float(z_um)
        bgr, truth = sim.sim.render()
        eng.process_frame((bgr.astype(np.uint16) * 257), i * 0.04, i + 1, truth=truth)
    eng.flush()
    assert len(caps) == 1
    cap = caps[0]
    assert cap.color_frames is not None
    assert len(cap.color_frames) == len(cap.frames) >= 8
    assert all(c.dtype == np.uint16 and c.shape == (120, 160, 3) for c in cap.color_frames)


def test_live_engine_load_mosaic_state_reloads_graph_and_insets():
    eng = LiveEngine(CalibrationStore(None), OBJ)
    eng.color_mode = True
    eng.sm.override = Override.PAN
    eng.process_frame(_colour_bgr(3), 0.0, 1, truth=None)  # seeds keyframe 0 with an RGB tile
    assert len(eng.mosaic.keyframes) == 1 and bool(eng.mosaic.tiles_rgb)
    eng.mosaic.add_inset("10x / 0.25", 0.625, 5.0, 7.0,
                         np.full((32, 32), 120, np.uint8))
    meta, arrays = eng.mosaic.to_state_arrays()
    assert meta["insets"] and "inset_img_0" in arrays and "kf_img_0" in arrays

    eng2 = LiveEngine(CalibrationStore(None), OBJ)
    eng2.color_mode = True  # the saved RGB keyframe tiles re-render as a colour view
    eng2.load_mosaic_state(meta, arrays)
    assert len(eng2.mosaic.keyframes) == 1
    assert eng2.mosaic.areas == 1
    assert eng2.mosaic_version == 1
    assert len(eng2.mosaic.insets) == 1
    ins = eng2.mosaic.insets[0]
    assert ins.objective == "10x / 0.25" and ins.um_per_px == 0.625
    assert np.asarray(ins.image).shape == (32, 32)
    eng2._render_now()
    img = eng2.mosaic_image()[0]
    assert img.ndim == 3 and img.shape[2] == 4 and bool(img[..., 3].any())


def test_live_engine_reset_mosaic_keeps_insets_only_when_requested():
    eng = LiveEngine(CalibrationStore(None), OBJ)
    eng.mosaic.add_inset("4x / 0.10", 2.5, 1.0, 2.0, np.zeros((8, 8), np.uint8))
    eng.reset_mosaic()  # default: insets are cleared with the rest of the graph
    assert not eng.mosaic.insets and not eng.mosaic.keyframes and not eng.mosaic.tiles
    eng.mosaic.add_inset("4x / 0.10", 2.5, 1.0, 2.0, np.zeros((8, 8), np.uint8))
    eng.reset_mosaic(keep_insets=True)
    assert len(eng.mosaic.insets) == 1
    assert not eng.mosaic.keyframes and not eng.mosaic.tiles


def test_live_engine_load_mosaic_state_rejects_bad_state():
    eng = LiveEngine(CalibrationStore(None), OBJ)
    bad_meta = {"keyframes": [{"x": 0.0, "y": 0.0, "sharpness": 0.0, "area": 0,
                               "angle_deg": 0.0, "yields": [], "has_rgb": False}]}
    with pytest.raises(KeyError):
        eng.load_mosaic_state(bad_meta, {})


# ------------------------------------------------------------------------------------------ skip policy (conftest)


def _run_policy_pytest(tmp_path, test_src: str, **env):
    import os
    import pathlib
    import subprocess
    import sys

    (tmp_path / "conftest.py").write_text((pathlib.Path(__file__).parent / "conftest.py").read_text(encoding="utf-8"),
                                          encoding="utf-8")
    (tmp_path / "test_x.py").write_text(test_src, encoding="utf-8")
    e = {k: v for k, v in os.environ.items() if not k.startswith(("MICROTOPOS_", "PYTEST_"))}
    e.update(env)
    p = subprocess.run([sys.executable, "-m", "pytest", "-p", "no:cacheprovider", "-q", "-rsf", "test_x.py"],
                       cwd=tmp_path, env=e, capture_output=True, text=True, timeout=120, check=False)
    return p.stdout + p.stderr


def test_strict_skips_keep_allowlisted_skips_and_fail_wrong_result_skips(tmp_path):
    src = ("import pytest\n"
           "def test_absent():\n    pytest.skip('no OpenCL device')\n"
           "def test_wrong():\n    pytest.skip('OpenCL device diverges from the CPU reference for SML (rel L2 0.77)')\n")
    out = _run_policy_pytest(tmp_path, src, MICROTOPOS_STRICT_SKIPS="1")
    assert "1 failed, 1 skipped" in out, out
    assert "diverges from the CPU reference" in out and "not an absent capability" in out
    lenient = _run_policy_pytest(tmp_path, src)  # without the switch nothing changes
    assert "2 skipped" in lenient and "failed" not in lenient, lenient


def test_strict_skips_also_judge_collection_time_skips(tmp_path):
    out = _run_policy_pytest(tmp_path, "import pytest\npytest.importorskip('no_such_module_xyz')\n",
                             MICROTOPOS_STRICT_SKIPS="1")
    assert "not an absent capability" in out and "no_such_module_xyz" in out, out


def test_require_opencl_skips_or_fails_per_environment(tmp_path, monkeypatch):
    src = ("import conftest\nimport pytest\n"
           "def test_a(monkeypatch):\n"
           "    monkeypatch.setattr(conftest, 'opencl_device_available', lambda: False)\n"
           "    conftest.require_opencl()\n")
    skipped = _run_policy_pytest(tmp_path, src)
    assert "1 skipped" in skipped and "no OpenCL device" in skipped, skipped
    required = _run_policy_pytest(tmp_path, src, MICROTOPOS_REQUIRE_OPENCL="1")
    assert "1 failed" in required and "MICROTOPOS_REQUIRE_OPENCL=1" in required, required
    strict = _run_policy_pytest(tmp_path, src, MICROTOPOS_STRICT_SKIPS="1")
    assert "1 skipped" in strict, strict  # the one allowed reason stays a skip even in strict mode


# ---------------------------------------------------------------- colour all-in-focus is reachable from the GUI

@pytest.fixture
def gui_window(tmp_path_factory):
    import os

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    pg = pytest.importorskip("pyqtgraph")
    from PySide6.QtCore import QSettings, QStandardPaths
    from PySide6.QtWidgets import QApplication

    from microtopos.gui.main_window import MainWindow

    QStandardPaths.setTestModeEnabled(True)
    QSettings.setPath(QSettings.NativeFormat, QSettings.UserScope, str(tmp_path_factory.mktemp("qsettings")))
    app = QApplication.instance() or QApplication([])
    pg.setConfigOptions(imageAxisOrder="row-major")
    base = tmp_path_factory.mktemp("gui_data")
    os.environ["XDG_DATA_HOME"] = str(base / "data")
    os.environ["XDG_CONFIG_HOME"] = str(base / "config")
    w = MainWindow(simulator_only=True)
    w.show()
    for _ in range(20):
        app.processEvents()
    w.timer.stop()
    yield app, w
    w.close()
    app.processEvents()


def _wait_until(app, cond, timeout=60.0, what=None):
    import time

    t0 = time.time()
    while time.time() - t0 < timeout and not cond():
        app.processEvents()
        time.sleep(0.02)
    assert cond(), f"timed out after {timeout:.0f} s: {what() if what else ''}"


def _gui_sweep_result(app, w):
    """Run the simulator's focus sweep and return the reconstruction. The demo sweep is random: when it passes through
    focus over less than half of the field the window (rightly) does not reconstruct it by itself, so the same
    "Reconstruct last sweep" button the user has is pressed instead."""
    prev = getattr(w, "last_cap", None)
    w.result = None
    w._demo("sweep")
    _wait_until(app, lambda: getattr(w, "last_cap", None) is not prev, what=lambda: (
        f"state={w.engine.sm.state}, sweep frames={len(w.engine._buf)}, fps={w.engine._snap.fps_in:.1f}"))
    _wait_until(app, lambda: w.job is None or not w.job.isRunning(), timeout=120)  # an automatic reconstruction
    cap = w.last_cap

    def mine():
        return w.result is not None and w.result.get("sweep_seq_range") == [cap.seqs[0], cap.seqs[-1]]

    if not mine():  # nothing yet, or a late result of an earlier capture: reconstruct THIS capture
        w.result = None
        w._reconstruct_last()
        _wait_until(app, mine, timeout=120)
    for _ in range(10):
        app.processEvents()
    return w.result, cap

def test_gui_colour_toggle_yields_a_displayed_colour_all_in_focus(gui_window):
    app, w = gui_window
    assert w.engine.color_mode is False
    w.chk_rgb.setChecked(True)
    assert w.engine.color_mode is True  # the toggle used to set only the window's flag; the engine never kept colour
    _wait_until(app, lambda: len(w.engine._preroll) > 0 and all(item[5] is not None for item in w.engine._preroll))
    r, cap = _gui_sweep_result(app, w)
    assert cap.color_frames is not None and len(cap.color_frames) == len(cap.frames)
    assert r["edf_rgb"].ndim == 3 and r["edf_rgb"].shape[2] == 3 and r["edf_rgb"].dtype == np.uint8
    assert r["edf_rgb"].shape[:2] == r["edf"].shape[:2]
    shown = w.p_edf.data
    assert shown.ndim == 3 and shown.shape[2] == 3  # the colour image, not the grey one, is on the All-in-focus panel
    assert "colour" in w.lbl_edf.text()
    w.chk_rgb.setChecked(False)
    assert w.engine.color_mode is False
    _wait_until(app, lambda: len(w.engine._preroll) > 0 and all(item[5] is None for item in w.engine._preroll))
    r2, cap2 = _gui_sweep_result(app, w)
    assert cap2.color_frames is None
    assert r2.get("edf_rgb") is None and w.p_edf.data.ndim == 2  # colour off: the grey EDF, no colour claim