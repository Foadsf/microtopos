"""Slice 3 feature acceptance: breathing-scale registration, OpenCL focus backends, the sff DOF ruler,
tiled z-stack reconstruction with RGB EDF, and stage-coupled LiveEngine sweeps.

These tests run fast (no simulator video I/O beyond tiny in-memory renders) and are the acceptance
commands for Slice 3 together with the four regression files.
"""

from __future__ import annotations

import itertools
import math

import cv2
import numpy as np
import pytest

import microtopos.optics as _optics  # monkeypatch target for the wavelength sweep
from conftest import require_opencl
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


def _zoom_frames(texture: np.ndarray, n: int, beta: float, ref: int) -> list[np.ndarray]:
    """Fixed-size frames whose content is zoomed by exp(beta * (k - ref))."""
    h, w = texture.shape
    out = np.empty((n, h, w), np.float32)
    for k in range(n):
        s = float(math.exp(beta * (k - ref)))
        nh, nw = max(4, round(h * s)), max(4, round(w * s))
        big = cv2.resize(texture, (nw, nh), interpolation=cv2.INTER_LINEAR)
        canvas = np.zeros((h, w), np.float32)
        oh, ow = min(h, nh), min(w, nw)
        canvas[h // 2 - oh // 2:h // 2 - oh // 2 + oh,
               w // 2 - ow // 2:w // 2 - ow // 2 + ow] = big[nh // 2 - oh // 2:nh // 2 - oh // 2 + oh,
                                                             nw // 2 - ow // 2:nw // 2 - ow // 2 + ow]
        out[k] = canvas
    return [out[k] for k in range(n)]


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
    texture = _texture(2, (96, 96)).astype(np.float32)
    frames = _zoom_frames(texture, 11, 0.01, 5)
    z = [0.5 * k for k in range(len(frames))]
    warps, pairs = register_sweep(frames, ref=5, factor=1, workers=1,
                                  correct_breathing=True, z=z)
    assert len(warps) == len(frames)  # identity warp at the reference frame plus one per pair
    assert len(pairs) == len(frames) - 1
    expected = math.exp(0.01)
    assert all(abs(p.scale_step - expected) < 0.009 for p in pairs if p.ok)  # breathing is observed
    # without breathing correction the per-frame scale model is not folded into the warps,
    # but ECC still sees the magnification change between consecutive frames
    _warps2, pairs2 = register_sweep(frames, ref=5, factor=1, workers=1, correct_breathing=False, z=z)
    assert all(abs(p.scale_step - expected) < 0.02 for p in pairs2 if p.ok)


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


def test_zstack_tiled_and_untiled_reconstructions_agree():
    frames, zs = _sim_sweep(11, n_frames=9, size=(240, 320))
    r_untiled = zstack.reconstruct(frames, zs, OBJ, 1.0, "um",
                                   settings=ZSettings(allow_tiling=False))
    r_tiled = zstack.reconstruct(frames, zs, OBJ, 1.0, "um",
                                 settings=ZSettings(allow_tiling=True, memory_fraction=1e-5))
    assert r_tiled["frames_used"] == r_untiled["frames_used"] == len(frames)
    valid = (r_untiled["support"] & r_tiled["support"]
             & np.isfinite(r_untiled["height"]) & np.isfinite(r_tiled["height"]))
    assert valid.sum() > 0.5 * valid.size
    diff = np.abs(r_untiled["height"][valid] - r_tiled["height"][valid])
    assert float(np.median(diff)) < 0.6, float(np.median(diff))
    assert float(np.percentile(diff, 95)) < 2.5, float(np.percentile(diff, 95))
    for r in (r_untiled, r_tiled):
        assert np.isfinite(r["fwhm_z"]) and r["fwhm_z"] > 0
        assert r["ecc_mean"] is not None and r["ecc_mean"] > 0.5
        assert r["support_fraction"] > 0.3


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
    assert cap.z is not None and len(cap.z) == len(cap.frames)
    assert all(v is not None for v in cap.z)  # z labels populated from stage/simulator truth


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
                       cwd=tmp_path, env=e, capture_output=True, text=True, timeout=120)
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