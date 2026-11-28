"""Estimated radial distortion from a panned specimen (microtopos.distortion), checked against a known distortion.

Ground truth: frames of the simulated specimen are rendered while the stage pans along a 2-D (Lissajous) path and
then distorted by a KNOWN Brown radial model with cv2.remap at render resolution. The inverse mapping that the
remap needs is computed by OpenCV (cv2.undistortPoints with the half-diagonal camera matrix), not by the module
under test. The frames are rendered M px larger on each side than the output so that the barrel case samples real
specimen at the output corners. The motion fed to the estimator is measured by the app's own registration
(register_pair), as the live engine would. Everything runs on the simulator; nothing here exercises a real camera.

Cases: barrel k = (-0.012, -0.003), a corner displacement of -1.5 % of the half-diagonal (-2.99 px at 320 x 240);
pincushion k = (+0.012, +0.004), +1.6 % (+3.19 px); zero distortion.
"""

from __future__ import annotations

import functools
import json
import math

import cv2
import numpy as np
import pytest

from microtopos.calibration import bgr_to_luma
from microtopos.distortion import (
    MIN_PAIRS,
    DistortionEstimator,
    DistortionResult,
    NoMeasurableDistortion,
    RadialModel,
)
from microtopos.objectives import default_set, resolve
from microtopos.registration import phase_shift, register_pair
from microtopos.simulator import SpecimenSimulator

W, H, M = 320, 240, 12
CX, CY = (W - 1) / 2.0, (H - 1) / 2.0
R = math.hypot(CX, CY)  # the stated normalisation: r = 1 at the corner pixel centres
BARREL = (-0.012, -0.003)
PINCUSHION = (0.012, 0.004)
ZERO = (0.0, 0.0)


def camera_matrix() -> np.ndarray:
    return np.array([[R, 0.0, CX], [0.0, R, CY], [0.0, 0.0, 1.0]])


def opencv_inverse_maps(k: tuple[float, float]) -> tuple[np.ndarray, np.ndarray]:
    """For every observed (distorted) output pixel p, where in the M-padded render to sample: U(p) + M, with U from
    OpenCV's iterative undistortPoints (converged to 1e-14 in normalised units)."""
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    pts = np.stack([xx.ravel(), yy.ravel()], 1)[:, None, :]
    crit = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 100, 1e-14)
    u = cv2.undistortPoints(pts, camera_matrix(), np.array([k[0], k[1], 0.0, 0.0, 0.0]), None, None, None, crit)
    u = u[:, 0, :] * R + (CX, CY) + M
    return u[:, 0].reshape(H, W).astype(np.float32), u[:, 1].reshape(H, W).astype(np.float32)


def lissajous(n: int, amp: float) -> np.ndarray:
    """View positions in field units (x in widths, y in heights): motion in 2-D."""
    t = np.linspace(0.0, 1.0, n)
    return np.stack([amp * np.sin(2 * np.pi * t + 0.3), amp * np.sin(3 * np.pi * t + 1.1)], 1)


@functools.lru_cache(maxsize=8)
def pan(seed: int, k: tuple[float, float], still: bool = False) -> tuple[list[np.ndarray], list, float]:
    """(distorted luma frames, register_pair shifts with None first, dark level) for a 40-frame pan of 0.6 fields
    amplitude (or 40 frames of a still view with hand tremor only)."""
    s = SpecimenSimulator(seed, resolve("10x / 0.25", objset=default_set()), W + 2 * M, H + 2 * M)
    mx, my = opencv_inverse_maps(k)
    frames, shifts = [], []
    for fx, fy in (np.zeros((40, 2)) if still else lissajous(40, 0.6)):
        s.stage_x_um, s.stage_y_um = fx * W * s.um_per_px, fy * H * s.um_per_px
        bgr, _ = s.render(blur=False, breathing=False)
        luma = cv2.remap(bgr_to_luma(bgr), mx, my, cv2.INTER_LINEAR)
        if frames:
            reg = register_pair(frames[-1], luma)
            shifts.append((float(reg.warp[0, 2]), float(reg.warp[1, 2])))
        else:
            shifts.append(None)
        frames.append(luma)
    return frames, shifts, s.dark_level


def estimator(seed: int, k: tuple[float, float], n: int | None = None, still: bool = False) -> DistortionEstimator:
    frames, shifts, dark = pan(seed, k, still)
    est = DistortionEstimator((H, W), dark=dark)
    for f, d in list(zip(frames, shifts))[:n]:
        est.add(f, d)
    return est


@functools.lru_cache(maxsize=8)
def result(seed: int, k: tuple[float, float]) -> DistortionResult:
    return estimator(seed, k).result()


def tile_residual(frames: list[np.ndarray], shifts: list, model: RadialModel | None, gap: int = 2,
                  tile: int = 64) -> float:
    """Independent check of geometric consistency: RMS deviation (px) of the shifts of 64 px tiles, measured by
    phase correlation between frames `gap` apart, from the median shift of their pair. Under a pure translation all
    tiles of a pair move alike; distortion makes the tile shifts depend on field position. Tiles keep 8 px from
    the border, where a model that pushes samples outward would read the remap border."""
    imgs = [model.undistort(f) if model is not None else f for f in frames]
    edge, step = 8, 32
    dev = []
    for i in range(gap, len(imgs), gap):
        d = np.round(np.sum([shifts[j] for j in range(i - gap + 1, i + 1)], axis=0)).astype(int)
        loc = []
        for y in range(edge, H - edge - tile + 1, step):
            for x in range(edge, W - edge - tile + 1, step):
                yb, xb = y + d[1], x + d[0]
                if min(yb, xb) < edge or yb + tile > H - edge or xb + tile > W - edge:
                    continue
                dx, dy, _ = phase_shift(imgs[i - gap][y:y + tile, x:x + tile], imgs[i][yb:yb + tile, xb:xb + tile])
                loc.append((dx, dy))
        if len(loc) >= 6:
            loc = np.array(loc)
            dev.append(loc - np.median(loc, axis=0))
    e = np.concatenate(dev)
    return float(np.sqrt(np.mean(np.sum(e * e, 1))))


# ---------------------------------------------------------------------------------------------- the model
@pytest.mark.parametrize("k", [BARREL, PINCUSHION])
def test_undistortion_maps_equal_opencv_with_the_half_diagonal_camera_matrix(k):
    """The stated convention: OpenCV's model with camera matrix [[R, 0, cx], [0, R, cy]], R the half-diagonal."""
    model = RadialModel((H, W), *k)
    assert model.radius == pytest.approx(R) and model.centre == (CX, CY)
    mx, my = model.undistort_maps((H, W))
    ox, oy = cv2.initUndistortRectifyMap(camera_matrix(), np.array([k[0], k[1], 0.0, 0.0, 0.0]), None,
                                         camera_matrix(), (W, H), cv2.CV_32FC1)
    assert np.max(np.abs(mx - ox)) < 2e-3 and np.max(np.abs(my - oy)) < 2e-3
    # the corner displacement is R (k1 + k2), the radial shift of a corner pixel centre
    corner = model.distort_points(np.array([[0.0, 0.0]]))[0]
    assert math.hypot(corner[0] - CX, corner[1] - CY) - R == pytest.approx(model.corner_displacement_px, abs=1e-9)
    assert model.corner_displacement_px == pytest.approx(R * (k[0] + k[1]), abs=1e-12)


@pytest.mark.parametrize("k", [BARREL, PINCUSHION])
def test_point_inverse_is_exact_and_agrees_with_opencv(k):
    model = RadialModel((H, W), *k)
    rng = np.random.default_rng(0)
    u = rng.uniform((0, 0), (W - 1, H - 1), (500, 2))
    np.testing.assert_allclose(model.undistort_points(model.distort_points(u)), u, atol=1e-8)
    p = model.distort_points(u)
    crit = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 100, 1e-14)
    ocv = cv2.undistortPoints(p[:, None, :], camera_matrix(), np.array([k[0], k[1], 0.0, 0.0, 0.0]), None, None,
                              None, crit)[:, 0, :] * R + (CX, CY)
    np.testing.assert_allclose(model.undistort_points(p), ocv, atol=1e-6)


# ---------------------------------------------------------------------------------------------- recovery
@pytest.mark.parametrize("seed", (1, 2))
@pytest.mark.parametrize("k", [BARREL, PINCUSHION], ids=["barrel", "pincushion"])
def test_recovers_known_distortion(seed, k):
    r = result(seed, k)
    truth = R * (k[0] + k[1])
    assert abs(truth) > 2.9  # really about 1.5 % of the half-diagonal
    # measured errors 0.6 - 2.4 % on these four cases (seeds 1-7: up to 2.6 %)
    assert abs(r.corner_displacement_px - truth) < 0.05 * abs(truth)
    # the reported uncertainty covers the error (corner_u_px includes the 2 % Type B term)
    assert abs(r.corner_displacement_px - truth) < 3 * r.corner_u_px
    assert r.corner_u_px < 0.04 * abs(truth)
    # k1 and k2 are correlated over a field that ends at r = 1; k1 measured within 0.0014 of the truth
    assert abs(r.k1 - k[0]) < 0.002
    assert r.pairs >= MIN_PAIRS and r.field_coverage >= 0.75


@pytest.mark.parametrize("k", [BARREL, PINCUSHION], ids=["barrel", "pincushion"])
def test_estimated_map_matches_the_true_map(k):
    """Over the whole field, the estimated undistortion samples within 0.12 px of where the true one does (measured
    <= 0.08 px). A 5 % error in the corner displacement alone would be 0.15 px at the corners."""
    r = result(1, k)
    ex, ey = r.undistort_maps((H, W))
    tx, ty = RadialModel((H, W), *k).undistort_maps((H, W))
    assert float(np.max(np.hypot(ex - tx, ey - ty))) < 0.12


@pytest.mark.parametrize("k", [BARREL, PINCUSHION], ids=["barrel", "pincushion"])
def test_undistorting_frames_removes_the_field_dependent_residual(k):
    """Independent of the estimator's own feature matches: tile shifts by phase correlation on the frames.

    The residual of the truth-undistorted frames is the floor of the tile measurement itself (noise, JPEG,
    phase-correlation precision, about 0.13 px). The part of the residual attributable to distortion,
    sqrt(res^2 - floor^2), must drop at least 5-fold (measured: to zero within the floor's own scatter), and the raw
    residual must exceed the estimated-undistorted one by 1.6x (measured 1.9x; the floor bounds this ratio)."""
    frames, shifts, _ = pan(1, k)
    frames, shifts = frames[:24], shifts[:24]
    r = result(1, k)
    raw = tile_residual(frames, shifts, None)
    est = tile_residual(frames, shifts, r.model)
    floor = tile_residual(frames, shifts, RadialModel((H, W), *k))
    assert raw > 1.6 * est
    assert est**2 - floor**2 < (raw**2 - floor**2) / 25
    # the estimator's own match statistics say the same (measured 0.21 px -> 0.044 px)
    assert r.residual_after_px < r.residual_before_px / 3


@pytest.mark.parametrize("k", [BARREL, PINCUSHION], ids=["barrel", "pincushion"])
def test_the_inverse_model_makes_the_residual_worse(k):
    """Sign check: undistorting with -k doubles the distortion instead of removing it, so the distortion part of
    the tile residual grows (about 4-fold in squared terms; asserted >= 3) and the residual exceeds the raw one."""
    frames, shifts, _ = pan(1, k)
    frames, shifts = frames[:24], shifts[:24]
    r = result(1, k)
    raw = tile_residual(frames, shifts, None)
    floor = tile_residual(frames, shifts, RadialModel((H, W), *k))
    inv = tile_residual(frames, shifts, RadialModel((H, W), -r.k1, -r.k2))
    assert inv > 1.5 * raw
    assert inv**2 - floor**2 > 3 * (raw**2 - floor**2)


# ---------------------------------------------------------------------------------------------- refusals
@pytest.mark.parametrize("seed", (1, 2, 3))
def test_zero_distortion_is_reported_as_no_measurable_distortion(seed):
    est = estimator(seed, ZERO)
    assert est.progress()["ready"]  # the refusal below is a measurement outcome, not missing data
    with pytest.raises(NoMeasurableDistortion) as exc:
        est.result()
    # a tight upper bound, not a gate that refuses everything: measured |estimate| <= 0.02 px, bound <= 0.05 px
    assert exc.value.bound_px < 0.1
    assert abs(exc.value.corner_displacement_px) < 0.05


def test_a_still_view_is_not_ready():
    est = estimator(1, BARREL, still=True)
    p = est.progress()
    assert p["frames"] == 1 and p["pairs"] == 0 and not p["ready"]
    with pytest.raises(ValueError, match="not ready") as exc:
        est.result()
    assert not isinstance(exc.value, NoMeasurableDistortion)


def test_too_few_pairs_is_not_ready():
    est = estimator(1, BARREL, n=8)
    p = est.progress()
    assert 0 < p["pairs"] < MIN_PAIRS and not p["ready"]
    assert any("pairs" in m for m in p["missing"])
    with pytest.raises(ValueError, match="not ready") as exc:
        est.result()
    assert not isinstance(exc.value, NoMeasurableDistortion)


# ---------------------------------------------------------------------------------------------- persistence
def test_json_round_trip_is_exact():
    r = result(1, BARREL)
    back = DistortionResult.from_json(json.loads(json.dumps(r.to_json())))
    assert back == r
    assert back.model.k1 == r.model.k1 and back.model.k2 == r.model.k2  # bit-exact floats
    for a, b in zip(back.undistort_maps(), r.undistort_maps()):
        assert np.array_equal(a, b)
    m = RadialModel((H, W), -0.0123456789012345, 0.00098765432109876)
    assert RadialModel.from_json(json.loads(json.dumps(m.to_json()))) == m
