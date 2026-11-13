"""Scientific-regression tests for MicroTopos.

Validates:
1. Revisit sharpness preservation (preventing blur when revisiting mapped areas).
2. Subpixel phase-correlation registration accuracy (< 0.05 px error).
3. Pose-graph loop closure and drift reduction.
4. Stage backlash model hysteresis compensation.
5. Confidence and uncertainty map consistency.
6. Diagnostic bundle export integrity.
7. Strict input validation for CLI crops and numbers.
"""

from __future__ import annotations

import argparse
import math
import zipfile
from pathlib import Path

import cv2
import numpy as np
import pytest

from microtopos.logging import create_diagnostic_bundle, log_environment
from microtopos.mosaic import BacklashModel, Mosaic, PoseGraph, sharpness_score
from microtopos.registration import register_pair
from microtopos.validation import (
    parse_crop,
    parse_nonnegative_int,
    parse_positive_float,
    validate_warp_matrix,
)


def _synthetic_texture(h: int = 256, w: int = 256, seed: int = 42) -> np.ndarray:
    """Generate a high-frequency textured surface with deterministic features."""
    rng = np.random.default_rng(seed)
    base = rng.uniform(40, 200, (h, w)).astype(np.float32)
    # Add structured scratch and edge features
    cv2.line(base, (30, 40), (220, 200), 240, 2)
    cv2.circle(base, (128, 128), 45, 10, 2)
    cv2.rectangle(base, (60, 160), (100, 210), 220, -1)
    return cv2.GaussianBlur(base, (0, 0), 0.8)


def test_revisit_sharpness_not_blurred():
    """Scientific regression: real pan-out + defocused return genuinely triggers revisit guard without blur."""
    from microtopos.calibration import bgr_to_luma
    from microtopos.optics import OBJECTIVES
    from microtopos.simulator import SpecimenSimulator

    obj = OBJECTIVES["40x / 0.85"]
    sim = SpecimenSimulator(12, obj, 240, 180)
    _, hz = sim.truth_patch()
    z_best = float(np.median(hz))
    sim.focus_z_um = z_best
    x0, y0 = sim.stage_x_um, sim.stage_y_um

    def grab(dx_um: float, dy_um: float, z: float) -> np.ndarray:
        sim.stage_x_um, sim.stage_y_um, sim.focus_z_um = x0 + dx_um, y0 + dy_um, z
        bgr, _ = sim.render(tremor=False, noise=False, breathing=False)
        return bgr_to_luma(bgr)

    f0 = grab(0.0, 0.0, z_best)
    f_pan = grab(sim.um_per_px * 60.0, 0.0, z_best)  # > 0.2 * 180 = 36 px -> adds keyframe 1
    # Return to origin with steady small subpixel offset (1.2 px) and mild defocus
    # so recent-frame sharp_enough gate passes (~0.82 * s0 >= 0.80 * s0) and revisit guard fires.
    f_return_blurry = grab(sim.um_per_px * 1.2, sim.um_per_px * 0.8, z_best + 0.15)
    s0 = sharpness_score(f0)
    assert 0.80 * s0 <= sharpness_score(f_return_blurry) < 0.95 * s0

    m_sharp = Mosaic(fusion="sharpest")
    for fr in (f0, f_pan, f_pan, f_return_blurry, f_return_blurry):
        m_sharp.track(fr)

    assert len(m_sharp.revisit_diagnostics) >= 1
    assert m_sharp.revisit_diagnostics[-1]["decision"] == "retained_sharp"
    assert sharpness_score(m_sharp.keyframes[0].image) >= 0.95 * s0

    # Now revisit with an initially defocused keyframe 0 and a sharper return: must upgrade subpixel-aligned!
    m_up = Mosaic(fusion="sharpest")
    f0_soft = grab(0.0, 0.0, z_best + 0.6)
    f_ret_crisp = grab(sim.um_per_px * 0.6, -sim.um_per_px * 0.4, z_best)
    for fr in (f0_soft, f_pan, f_pan, f_ret_crisp, f_ret_crisp):
        m_up.track(fr)
    assert any(d["decision"] == "upgraded_to_sharper" for d in m_up.revisit_diagnostics)
    assert sharpness_score(m_up.keyframes[0].image) > 1.15 * sharpness_score(f0_soft)


def test_subpixel_registration_accuracy():
    """Scientific regression: phase correlation + ECC must estimate subpixel translations within 0.05 px."""
    template = _synthetic_texture(256, 256, seed=777)
    true_dx, true_dy = 0.35, -0.28
    W_true = np.float32([[1, 0, true_dx], [0, 1, true_dy]])
    moving = cv2.warpAffine(template, W_true, (256, 256), flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REFLECT)

    reg = register_pair(template, moving, factor=1, model="translation")
    assert reg.ok, "Registration should succeed with high correlation"

    est_dx, est_dy = float(reg.warp[0, 2]), float(reg.warp[1, 2])
    error_px = math.hypot(est_dx - true_dx, est_dy - true_dy)
    assert error_px < 0.05, f"Subpixel registration error must be < 0.05 px (got {error_px:.4f} px)"


def test_pose_graph_loop_closure():
    """Scientific regression: PoseGraph global relaxation must close loops and eliminate drift."""
    pg = PoseGraph()
    # 4 corners of a 100x100 square loop
    pg.add_node(0, 0.0, 0.0)
    pg.add_node(1, 100.0, 0.0)
    pg.add_node(2, 100.0, 100.0)
    pg.add_node(3, 0.0, 100.0)

    # Add noisy step edges with 0.5 px accumulated drift
    pg.add_edge(0, 1, 100.5, 0.0, 1.0)
    pg.add_edge(1, 2, 0.0, 100.5, 1.0)
    pg.add_edge(2, 3, -100.5, 0.0, 1.0)
    # Loop closure edge directly connecting 3 back to 0
    pg.add_edge(3, 0, 0.0, -100.0, 5.0)

    opt = pg.optimize(anchor_id=0, anchor_weight=50.0)
    # Anchor node 0 should remain at (0, 0)
    assert abs(opt[0][0]) < 0.05 and abs(opt[0][1]) < 0.05
    # Loop error from node 3 to node 0 should be eliminated
    loop_residual = math.hypot(opt[3][0] - 0.0, opt[3][1] - 100.0)
    assert loop_residual < 0.5, f"Loop residual must be bounded after relaxation (got {loop_residual:.3f} px)"


def test_backlash_model():
    """Scientific regression: BacklashModel must apply correction offset only upon direction reversal."""
    bm = BacklashModel(backlash_x=1.5, backlash_y=2.0)
    # Moving forward along X
    cx1, cy1 = bm.apply(10.0, 0.0)
    assert cx1 == 0.0 and cy1 == 0.0

    # Continuing forward: no backlash
    cx2, cy2 = bm.apply(5.0, 0.0)
    assert cx2 == 0.0 and cy2 == 0.0

    # Reversing direction (moving negative along X): backlash correction applied!
    cx3, cy3 = bm.apply(-4.0, 0.0)
    assert cx3 == -1.5, f"Expected backlash correction of -1.5 px on X reversal, got {cx3}"

    # Continuing in negative direction: no extra backlash
    cx4, cy4 = bm.apply(-3.0, 0.0)
    assert cx4 == 0.0


def test_confidence_and_uncertainty_maps():
    """Scientific regression: confidence map values must be in [0, 1] and uncertainty map must reflect 1 - conf."""
    m = Mosaic()
    img = _synthetic_texture(128, 128, seed=99)
    m.track(img)

    conf, (cx0, cy0) = m.confidence_map()
    uncert, (ux0, uy0) = m.uncertainty_map()

    assert cx0 == ux0 and cy0 == uy0
    valid_conf = conf[np.isfinite(conf)]
    valid_uncert = uncert[np.isfinite(uncert)]

    assert len(valid_conf) > 0
    assert (valid_conf >= 0.0).all() and (valid_conf <= 1.0).all()
    # Uncertainty is 1 - confidence (clamped)
    np.testing.assert_allclose(valid_uncert, np.maximum(0.05, 1.0 - valid_conf), atol=1e-5)


def test_diagnostic_bundle_generation(tmp_path: Path):
    """Scientific regression: diagnostic bundle export creates a valid zip containing all diagnostic manifests."""
    log_environment()
    zip_target = tmp_path / "test_diagnostics.zip"
    created = create_diagnostic_bundle(zip_target)
    assert created.exists()
    assert created.stat().st_size > 0

    with zipfile.ZipFile(created, "r") as zf:
        namelist = zf.namelist()
        for expected in ("versions.json", "commandline.txt", "logs.txt", "config.json", "calibration.json"):
            assert expected in namelist, f"Missing {expected} in diagnostic bundle"


def test_crop_validation():
    """Scientific regression: parse_crop enforces x,y,w,h integer format, w>0, h>0, non-negative coordinates."""
    # Valid crop
    assert parse_crop("10, 20, 300, 400") == (10, 20, 300, 400)
    assert parse_crop("0,0,1920,1080") == (0, 0, 1920, 1080)

    # Invalid crops
    with pytest.raises(argparse.ArgumentTypeError, match="exactly 4"):
        parse_crop("10,20,300")

    with pytest.raises(argparse.ArgumentTypeError, match="integers"):
        parse_crop("10,20,foo,400")

    with pytest.raises(argparse.ArgumentTypeError, match="non-negative"):
        parse_crop("-5,10,100,100")

    with pytest.raises(argparse.ArgumentTypeError, match="strictly positive"):
        parse_crop("0,0,0,100")


def test_positive_float_and_nonnegative_int():
    """Scientific regression: strict numerical validators for seconds, exposure, scales."""
    assert parse_positive_float("180.5") == 180.5
    with pytest.raises(argparse.ArgumentTypeError):
        parse_positive_float("-1.0")
    with pytest.raises(argparse.ArgumentTypeError):
        parse_positive_float("0.0")

    assert parse_nonnegative_int("300") == 300
    assert parse_nonnegative_int("0") == 0
    with pytest.raises(argparse.ArgumentTypeError):
        parse_nonnegative_int("-5")


def test_validate_warp_matrix():
    """Scientific regression: validate_warp_matrix checks shape, finiteness, determinant, and translation."""
    W_good = np.float32([[1.01, 0.02, 12.5], [-0.02, 1.01, -8.3]])
    assert validate_warp_matrix(W_good)

    W_nan = np.float32([[1.0, 0.0, np.nan], [0.0, 1.0, 0.0]])
    assert not validate_warp_matrix(W_nan)

    W_flipped = np.float32([[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])  # inverted orientation
    assert not validate_warp_matrix(W_flipped)

    W_excessive = np.float32([[1.0, 0.0, 50000.0], [0.0, 1.0, 0.0]])
    assert not validate_warp_matrix(W_excessive)


def test_lens_distortion_model():
    """Scientific regression: LensDistortionModel correctly models camera intrinsics, undistorts frames, and serializes."""
    from microtopos.calibration import LensDistortionModel

    K = np.array([[800.0, 0.0, 320.0], [0.0, 800.0, 240.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    dist = np.array([-0.15, 0.05, 0.0, 0.0, 0.0], dtype=np.float64)
    model = LensDistortionModel(camera_matrix=K, dist_coeffs=dist, image_size=(640, 480))
    d = model.to_dict()
    assert "camera_matrix" in d and "dist_coeffs" in d
    model2 = LensDistortionModel.from_dict(d)
    assert np.allclose(model2.camera_matrix, K)
    assert np.allclose(model2.dist_coeffs, dist)

    test_img = np.zeros((480, 640), dtype=np.uint8)
    cv2.rectangle(test_img, (50, 50), (590, 430), 255, 2)
    undist = model.undistort(test_img)
    assert undist.shape == test_img.shape


def test_backlash_learning():
    """Scientific regression: BacklashModel learns directional backlash from measured revisit offsets."""
    model = BacklashModel(backlash_x=0.0, backlash_y=0.0)
    for _ in range(25):
        model.update_from_revisit(dx_obs=0.8, dy_obs=0.5, dir_x=1, dir_y=-1, learning_rate=0.25)
    assert abs(model.backlash_x - 0.8) < 0.05
    assert abs(model.backlash_y - 0.5) < 0.05


def test_joint_intensity_height_registration():
    """Scientific regression: register_joint_intensity_height locks onto topographic gradients when optical contrast is low."""
    from microtopos.registration import register_joint_intensity_height

    img1 = np.full((120, 160), 128.0, dtype=np.float32)
    h1 = np.zeros((120, 160), dtype=np.float32)
    h1[40:80, 40:120] = 50.0

    M = np.float32([[1, 0, 8.0], [0, 1, -4.0]])
    img2 = cv2.warpAffine(img1, M, (160, 120))
    h2 = cv2.warpAffine(h1, M, (160, 120))

    dx, dy, _ = register_joint_intensity_height(img1, h1, img2, h2, weight_height=0.7)
    assert abs(dx - 8.0) < 0.5
    assert abs(dy - (-4.0)) < 0.5


def test_revisit_registration_benchmark_milestone():
    """Scientific milestone verification: 100 revisit trials achieve mean error < 0.1 px and p95 < 0.2 px."""
    from microtopos.mosaic import revisit_registration_benchmark

    res = revisit_registration_benchmark(cycles=100)
    assert res["mean_error_px"] < 0.10, f"Mean error {res['mean_error_px']} exceeded 0.1 px"
    assert res["p95_error_px"] < 0.20, f"95th percentile {res['p95_error_px']} exceeded 0.2 px"
    assert res["sub_0_1_rate"] >= 0.95


def test_stage_repeatability_benchmark():
    """Scientific regression: stage_repeatability_benchmark calculates repeatability statistics across 100 cycles."""
    from microtopos.mosaic import stage_repeatability_benchmark

    res = stage_repeatability_benchmark(cycles=100)
    assert res["cycles"] == 100.0
    assert res["mean_error_euclidean"] > 0.0
    assert res["p95_error_euclidean"] > 0.0


def test_mosaic_residual_map():
    """Scientific regression: Mosaic.residual_map computes spatial alignment residuals."""
    m = Mosaic()
    f1 = _synthetic_texture(128, 128, seed=1)
    f2 = _synthetic_texture(128, 128, seed=2)
    m.track(f1)
    m.track(f2)
    res_map, origin = m.residual_map()
    assert res_map.ndim == 2


def test_p1_regressions_units_registration_calibration_and_replay(tmp_path: Path):
    """Adversarial regression tests for F02-F10 fixes."""
    import json
    import warnings
    from types import SimpleNamespace
    from unittest.mock import patch

    from microtopos import export, focus, metrology, registration, sff, superres, zstack
    from microtopos.calibration import CalibrationStore
    from microtopos.camera import FrameSource, ReplaySource
    from microtopos.gui import main_window as gui
    from microtopos.live import LiveEngine
    from microtopos.optics import OBJECTIVES

    # F04: height_mosaic never blends incompatible units ('um' vs 'frames')
    e = np.full((24, 24), 100.0, np.float32)
    m = Mosaic()
    m.add_height_patch(0, 0, e, e * 0 + 10.0, "um")
    m.add_height_patch(0, 0, e, e * 0 + 30.0, "frames")
    hm_default = m.height_mosaic()
    assert hm_default is not None and float(hm_default[0][12, 12]) == 10.0
    hm_frames = m.height_mosaic(z_unit="frames")
    assert hm_frames is not None and float(hm_frames[0][12, 12]) == 30.0

    # F05: flat/unobservable frames produce zero shift in registration and dwell_superres
    flat = np.full((64, 96), 117.0, np.float32)
    reg_flat = registration.register_pair(flat, flat, factor=1)
    assert not reg_flat.ok and reg_flat.ecc == 0.0
    assert np.allclose(reg_flat.warp[:, 2], 0.0)
    sr_flat = superres.dwell_superres([flat, flat], 0.6, deconvolve=False)
    assert np.allclose(sr_flat["shifts"][1], (0.0, 0.0))

    # F10: Vollath F4/F5 no circular border wrap, form order 0 leaves plane untouched, blank spatial = {}
    step_img = np.repeat(np.r_[np.full(24, 50.0, np.float32), np.full(72, 200.0, np.float32)][None, :], 48, axis=0)
    ctrl_img = step_img.copy()
    ctrl_img[:, :6] = ctrl_img[:, -1:]
    for meas in ("Vollath F4", "Vollath F5"):
        assert abs(float(focus.focus_map(step_img, meas, 9)[24, -1])) < 1e-4
        assert abs(float(focus.focus_map(ctrl_img, meas, 9)[24, -1])) < 1e-4

    plane = np.broadcast_to(np.linspace(-2.0, 2.0, 64), (48, 64)).astype(np.float32)
    assert float(np.std(metrology.remove_form(plane, order=0))) > 1.0
    assert float(np.std(metrology.remove_form(plane, order=1))) < 1e-5
    with warnings.catch_warnings(record=True) as wrec:
        warnings.simplefilter("always")
        assert metrology.spatial_parameters(np.zeros((64, 64), np.float32), 1.0) == {}
        assert math.isnan(sff.theoretical_fwhm_um(OBJECTIVES["40x / 0.85"], 0.25, focus.focus_map, np.zeros((96, 96), np.float32)))
        assert not [w for w in wrec if issubclass(w.category, RuntimeWarning)]

    # F09: strict JSON & session roundtrip preserves edf_proc, sl_surface, warnings
    res = {"edf": e, "height": e * 0.1, "confidence": e / 100.0, "edf_proc": e * 0.5,
           "sl_surface": e * 0.2, "nonfinite": np.float32("nan"), "warnings": ["w1"]}
    files = export.write_report_bundle(tmp_path / "rpt", res, {})
    json.loads(files[0].read_text(), parse_constant=lambda c: (_ for _ in ()).throw(ValueError(c)))
    export.save_session(tmp_path / "sess.npz", res, {})
    loaded, _ = export.load_session(tmp_path / "sess.npz")
    assert {"edf", "edf_proc", "height", "confidence", "sl_surface", "warnings"} <= set(loaded)

    # F03: calibration writes to the result's objective, not a newly clicked UI objective
    store = CalibrationStore(None)
    win_stub = SimpleNamespace(
        result={"objective": "40x / 0.85", "z_status": "NOMINAL", "frames_used": 49, "frame_span": 48.0},
        obj=OBJECTIVES["10x / 0.25"],
        store=store,
        _refresh_checklist=lambda: None,
        notify=lambda *_: None,
    )
    with patch.object(gui, "QInputDialog", SimpleNamespace(getDouble=lambda *_: (6.0, True))):
        gui.MainWindow._calibrate_z_travel(win_stub)
    assert store.get("40x / 0.85").z_um_per_frame == 0.125
    assert store.get("10x / 0.25").z_um_per_frame is None

    # F08 & F11-seq: set_source resets _last_seq and cancels pending _calib_req
    class IdleSource(FrameSource):
        pass

    eng = LiveEngine(CalibrationStore(None), OBJECTIVES["40x / 0.85"])
    eng.request_calibration("dark", n=2)
    eng._process(np.full((24, 24, 3), 60, np.uint8), None, 25, 25, 1)
    eng.set_source(IdleSource())
    assert eng._last_seq == -1
    assert eng._calib_req is None
    eng._process(np.full((24, 30, 3), 100, np.uint8), None, 25, 25, 1)
    assert not eng.store.get("40x / 0.85").darks

    # F06: ReplaySource skips corrupt JPEG cleanly without returning None
    rdir = tmp_path / "rep"
    rdir.mkdir()
    ok, jpg = cv2.imencode(".jpg", np.full((24, 24, 3), 90, np.uint8))
    assert ok
    (rdir / "000000.jpg").write_bytes(jpg.tobytes())
    (rdir / "000001.jpg").write_bytes(b"corrupt")
    (rdir / "frames.json").write_text(json.dumps({"timestamps": [0.0, 0.1]}))
    rep = ReplaySource(str(rdir))
    decoded = list(rep.frames())
    assert len(decoded) == 1 and decoded[0] is not None

    # F02: frame_indices preserves spacing across 2:1 thinned sweeps
    obj = OBJECTIVES["40x / 0.85"]
    rng = np.random.default_rng(7)
    tex = 128 + 30 * cv2.GaussianBlur(rng.normal(0, 1, (80, 128)).astype(np.float32), (0, 0), 0.7)
    _, xx = np.indices((80, 128))
    zs = np.linspace(-2.5, 2.5, 41)
    full_fr = [np.where(xx < 64, cv2.GaussianBlur(tex, (0, 0), obj.defocus_sigma_um(-0.8 - z) / 0.25),
                        cv2.GaussianBlur(tex, (0, 0), obj.defocus_sigma_um(0.8 - z) / 0.25)) for z in zs]
    thinned = full_fr[::2]
    idx_even = np.arange(0, len(full_fr), 2, dtype=np.float64)
    out = zstack.reconstruct(thinned, None, obj, 0.25, "µm", float(zs[1] - zs[0]), frame_indices=idx_even)
    step_delta = float(np.nanmedian(out["height"][16:-16, 76:112]) - np.nanmedian(out["height"][16:-16, 16:52]))
    assert abs(step_delta - 1.6) < 0.20

