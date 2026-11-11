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
    """Scientific regression: revisiting a location must retain sharpness and NOT cause blur.

    In classic 'average' fusion, averaging two frames with a small subpixel shift causes
    noticeable blur (Laplacian variance drops). In 'sharpest' fusion mode, the sharpest
    frame is preserved and the high-frequency sharpness is retained.
    """
    img0 = _synthetic_texture(256, 256, seed=123)
    # Simulate a return pass with a 0.4 pixel subpixel misalignment and slight defocus
    shift_mat = np.float32([[1, 0, 0.4], [0, 1, 0.3]])
    img_shifted = cv2.warpAffine(img0, shift_mat, (256, 256), borderMode=cv2.BORDER_REFLECT)
    img_defocused = cv2.GaussianBlur(img_shifted, (0, 0), 1.2)

    s0 = sharpness_score(img0)
    s_defocused = sharpness_score(img_defocused)
    assert s_defocused < s0, "Defocused frame must have lower sharpness score than original"

    # Mosaic in sharpest mode
    m_sharp = Mosaic(fusion="sharpest")
    m_sharp.track(img0)
    assert len(m_sharp.keyframes) == 1
    # Now simulate moving slightly and returning with the defocused frame
    m_sharp.track(img_defocused)

    # In sharpest mode, keyframe image should remain the crisp original img0
    kf_sharp = m_sharp.keyframes[0].image
    recovered_sharpness = sharpness_score(kf_sharp)
    assert recovered_sharpness >= 0.95 * s0, (
        f"Sharpest mode must preserve keyframe sharpness (expected >= {0.95 * s0:.1f}, got {recovered_sharpness:.1f})"
    )


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

