"""Mosaic capabilities: area joining, insets, two-band blending, rotation tracking, RGB mosaics, session
persistence, and the lateral-scale decision. Every numerical claim is checked against an independently built
ground truth (a known synthetic world seen through known camera poses)."""

from __future__ import annotations

import dataclasses
import io
import json
import math
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import cv2
import numpy as np
import pytest

from microtopos.calibration import CalibrationStore, CalStatus, ObjectiveCalibration
from microtopos.export import ExportError, load_session, save_session
from microtopos.mosaic import Mosaic
from microtopos.objectives import resolve

W, H = 320, 240


def _world(seed: int, size: int = 1024) -> np.ndarray:
    """Deterministic textured world; a frame is a W x H window of it at stage position (cx, cy)."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, (size, size), dtype=np.uint8).astype(np.float32)


def _grab(world: np.ndarray, cx: float, cy: float = 0.0) -> np.ndarray:
    h, w = world.shape
    x0 = round(cx - W / 2)
    y0 = round(cy - H / 2)
    xs = np.clip(np.arange(x0, x0 + W), 0, w - 1)
    ys = np.clip(np.arange(y0, y0 + H), 0, h - 1)
    return world[np.ix_(ys, xs)]


def _rgb_of(frame: np.ndarray) -> np.ndarray:
    return np.stack(
        [frame, np.clip(frame * 0.6 + 20, 0, 255), np.clip(frame * 1.3 - 10, 0, 255)],
        axis=-1,
    ).astype(np.float32)


def _track_sequence(m: Mosaic, world, positions):
    return [m.track(_grab(world, x)) for x in positions]


def _graph_connected(m: Mosaic) -> bool:
    seen = {0}
    stack = [0]
    while stack:
        i = stack.pop()
        for a, b, *_ in m.edges:
            if a == i and b not in seen:
                seen.add(b)
                stack.append(b)
            elif b == i and a not in seen:
                seen.add(a)
                stack.append(a)
    return seen == set(range(len(m.keyframes)))


# ------------------------------------------------------------------------------------------- area joining


def test_areas_join_after_a_gap():
    """Pan out, lose tracking long enough to start a second area, then pan back in small steps: the returning
    chain loop-closes onto the first area, everything merges into one area, and every keyframe ends at the
    position it was captured at (the stage position of its frame): a bridge that joined the areas without moving
    the second one into the first one's coordinates would leave it ~700 px off."""
    world = _world(3)
    m = Mosaic()
    truth: list[float] = []  # stage x of the frame each keyframe was made from

    def track(cx):
        added, q = m.track(_grab(world, cx))
        if added:
            truth.append(cx)
        return q

    assert all(track(x) > 0.3 for x in [0.0, 30.0, 60.0, 90.0, 120.0])
    assert m.areas == 1 and len(m.keyframes) >= 1
    # jump far away: content no longer overlaps -> 40+ lost frames start a separate area
    assert not m.track(_grab(world, 700.0))[0]
    for _ in range(45):
        track(700.0)
    assert m.areas == 2
    # the second area is where the stage was NOT known to be: it sits next to the first, not at 700
    assert abs(m.keyframes[-1].x - 700.0) > 100.0
    # pan back in small steps so every frame is tracked; the end overlaps area 0's content
    assert all(track(float(x)) > 0.3 for x in np.arange(680.0, 90.0, -20.0))
    assert m.areas == 1
    assert all(k.area == 0 for k in m.keyframes)
    assert len(m.keyframes) >= 4 and len(truth) == len(m.keyframes)
    assert _graph_connected(m)
    assert math.isfinite(m.x) and math.isfinite(m.y)
    for i, (k, cx) in enumerate(zip(m.keyframes, truth)):
        assert abs(k.x - cx) < 1.0 and abs(k.y) < 1.0, f"keyframe {i} at ({k.x:.1f}, {k.y:.1f}), stage was at {cx}"


# ------------------------------------------------------------------------------------------- insets


def test_insets_stored_and_serialized():
    m = Mosaic()
    m.track(_grab(_world(1), 0.0))
    inset = np.full((40, 40), 150.0, np.float32)
    m.add_inset("40x / 0.85", 0.25, 10.0, 20.0, inset)
    assert len(m.insets) == 1
    it = m.insets[0]
    assert it.objective == "40x / 0.85"
    assert it.um_per_px == 0.25 and it.x_um == 10.0 and it.y_um == 20.0
    assert np.array_equal(it.image, inset)

    meta, arrays = m.to_state_arrays()
    assert len(meta["insets"]) == 1 and "inset_img_0" in arrays
    m2 = Mosaic()
    m2.load_state_arrays(meta, arrays)
    assert len(m2.insets) == 1
    it2 = m2.insets[0]
    assert (it2.objective, it2.um_per_px, it2.x_um, it2.y_um) == ("40x / 0.85", 0.25, 10.0, 20.0)
    assert np.array_equal(it2.image, inset)


# ------------------------------------------------------------------------------------------- two-band blending


def _noisy_pair_mosaic() -> Mosaic:
    """Two keyframes 60 px apart over the same world, each frame with its own sensor noise (std 6)."""
    world = _world(5)
    rng = np.random.default_rng(1)
    m = Mosaic()
    for x in (0.0, 60.0, 60.0):  # the dwell at 60 makes the stage steady, so a keyframe is added there
        m.track(_grab(world, x) + rng.normal(0, 6, (H, W)).astype(np.float32))
    assert len(m.keyframes) == 2
    return m


def test_two_band_blend_has_the_extent_origin_and_nan_mask_of_the_feather_render():
    """The "multiband" mode is a two-band blend (low band feather-weighted, high band weighted by feather**4).
    It must cover exactly what the feather render covers: same shape, same origin, same NaN mask (empty canvas
    is NaN, not 0), and differ from it only where keyframes overlap."""
    m = _noisy_pair_mosaic()
    feather, o_f = m.image()
    m.set_blending("multiband")
    band, o_b = m.image()

    x1 = round(m.keyframes[1].x - m.keyframes[0].x)
    assert 55 <= x1 <= 65
    assert feather.shape == band.shape == (H, W + x1) and o_f == o_b
    assert not np.isnan(feather).any()  # the union of two overlapping views here is a full rectangle
    assert np.array_equal(np.isnan(feather), np.isnan(band))
    # the defect rendered the empty canvas as exact zeros; noise clipped at 0 is legitimate, so compare to feather
    assert (band == 0.0).mean() <= (feather == 0.0).mean() + 0.001

    # one contributor: both modes give the image itself (the outermost 12 px keep only the low band, where the
    # feather weight**4 is below the blend's floor)
    only_first, overlap, only_second = slice(12, x1), slice(x1, W), slice(W, W + x1 - 12)
    for sl in (only_first, only_second):
        assert np.abs(band[12:-12, sl] - feather[12:-12, sl]).max() < 0.05
    diff = np.abs(band[:, overlap] - feather[:, overlap])
    assert diff.mean() > 0.3  # the overlap is blended differently

    # the NaN mask also agrees when the canvas has holes: two separate areas
    world = _world(5)
    far = Mosaic()
    far.track(_grab(world, 0.0))
    for _ in range(45):
        far.track(_grab(world, 700.0))
    assert far.areas == 2
    holes_f, of = far.image()
    far.set_blending("multiband")
    holes_b, ob = far.image()
    assert np.isnan(holes_f).any() and of == ob and holes_f.shape == holes_b.shape
    assert np.array_equal(np.isnan(holes_f), np.isnan(holes_b))


def test_empty_two_band_mosaic_renders_like_an_empty_feather_one():
    m = Mosaic()
    m.set_blending("multiband")
    img, origin = m.image()
    assert img.shape == (8, 8) and not np.isfinite(img).any() and origin == (0, 0)


# ------------------------------------------------------------------------------------------- rotation tracking


def _smooth_world(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    w = cv2.GaussianBlur(rng.integers(0, 256, (1200, 1600), dtype=np.uint8).astype(np.float32), (0, 0), 1.2)
    return (w - w.mean()) * 3 + 128


def _rview(world: np.ndarray, cx: float, cy: float, theta: float) -> np.ndarray:
    """W x H view of the world centred at (cx, cy) whose content is turned by theta degrees in the sense of
    cv2.getRotationMatrix2D (positive = counter-clockwise on screen)."""
    M = cv2.getRotationMatrix2D((cx, cy), theta, 1.0)
    M[0, 2] += W / 2 - cx
    M[1, 2] += H / 2 - cy
    return cv2.warpAffine(world, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def _ncc(a: np.ndarray, b: np.ndarray, margin: int = 40) -> float:
    a = a[margin:-margin, margin:-margin].astype(np.float64)
    b = b[margin:-margin, margin:-margin].astype(np.float64)
    a -= a.mean()
    b -= b.mean()
    return float((a * b).sum() / math.sqrt((a * a).sum() * (b * b).sum()))


def _rotated_pan(world, theta: float, n: int = 7, track_rotation: bool = True):
    """Frame 0 unrotated; the view then turns to theta in two steps (the per-frame rotation must stay inside the
    capture range) and stays there while the stage moves 60 px per frame. Returns (mosaic, stage x of each
    keyframe's frame)."""
    m = Mosaic(track_rotation=track_rotation)
    ramp = [0.0, theta / 2, theta] + [theta] * (n - 3)
    cxs = []
    for i, th in enumerate(ramp):
        cx = 400 + 60.0 * i
        added, _q = m.track(_rview(world, cx, 500, th))
        if added:
            cxs.append(cx)
        assert not m.lost, f"frame {i} lost"
    return m, cxs


@pytest.mark.parametrize("theta", [5.0, -5.0])
def test_rotated_pan_has_signed_angle_and_derotated_keyframes(theta):
    """A view turned by +theta must read +theta (cv2 sense, documented in the module), and every stored keyframe
    must be the UNROTATED view at its position: NCC with the true unrotated view >= 0.98 (the defect derotated
    with the wrong sign: NCC 0.01)."""
    world = _smooth_world(7)
    m, cxs = _rotated_pan(world, theta)
    assert m.angle_deg == pytest.approx(theta, abs=0.3)
    assert len(m.keyframes) >= 3 and len(cxs) == len(m.keyframes)
    for i, (k, cx) in enumerate(zip(m.keyframes, cxs)):
        assert abs(k.x - (cx - 400)) < 0.5 and abs(k.y) < 0.5, f"keyframe {i} pose ({k.x:.2f}, {k.y:.2f})"
        truth = _rview(world, cx, 500, 0.0)
        assert _ncc(k.image.astype(np.float32), truth) >= 0.98, f"keyframe {i} is not the unrotated view"
        # the matching images are derived from the derotated keyframe
        assert _ncc(k.bp.astype(np.float32), Mosaic._bp(truth), margin=20) >= 0.95
        if i >= 1:
            assert k.angle_deg == pytest.approx(theta, abs=0.3)  # recorded signed angle at capture


@pytest.mark.parametrize("theta", [5.0, -5.0])
def test_rendered_mosaic_of_a_rotated_pan_equals_the_unrotated_mosaic(theta):
    world = _smooth_world(7)
    rot, _ = _rotated_pan(world, theta)
    ref, _ = _rotated_pan(world, 0.0, track_rotation=False)
    (a, oa), (b, ob) = rot.image(), ref.image()
    x0, y0 = max(oa[0], ob[0]) + 40, max(oa[1], ob[1]) + 40
    x1 = min(oa[0] + a.shape[1], ob[0] + b.shape[1]) - 40
    y1 = min(oa[1] + a.shape[0], ob[1] + b.shape[0]) - 40
    ca = a[y0 - oa[1]:y1 - oa[1], x0 - oa[0]:x1 - oa[0]]
    cb = b[y0 - ob[1]:y1 - ob[1], x0 - ob[0]:x1 - ob[0]]
    assert ca.shape == cb.shape and ca.size > 100 * 100
    assert _ncc(ca, cb, margin=1) >= 0.97


@pytest.mark.parametrize("sign", [1.0, -1.0])
def test_rotation_tracking_reads_the_signed_angle_over_many_keyframes(sign):
    """The view turns 2 deg per frame while the stage moves 25 px per frame (26 deg in total, ~3 keyframes): the
    angle is tracked frame by frame with its sign, and keyframes captured at 10 and 20 deg are stored upright."""
    world = _smooth_world(7)
    m = Mosaic(track_rotation=True)
    cxs = []
    for i in range(14):
        added, _q = m.track(_rview(world, 400 + 25.0 * i, 500, sign * 2.0 * i))
        assert not m.lost
        assert m.angle_deg == pytest.approx(sign * 2.0 * i, abs=0.4)
        assert abs(m.x - 25.0 * i) < 0.6 and abs(m.y) < 0.6
        if added:
            cxs.append(400 + 25.0 * i)
    assert len(m.keyframes) >= 3
    for k, cx in zip(m.keyframes, cxs):
        true_angle = sign * 2.0 * round((cx - 400) / 25.0)
        assert k.angle_deg == pytest.approx(true_angle, abs=0.4)
        # margin 80: at up to 26 deg the derotated corners hold reflected border the truth view lacks; 0.97: the
        # view was interpolated once to rotate it and once more to derotate it (measured 0.98; a sign error: 0.0)
        assert _ncc(k.image.astype(np.float32), _rview(world, cx, 500, 0.0), margin=80) >= 0.97
    assert m.keyframes[-1].angle_deg * sign > 15.0


def test_without_rotation_tracking_the_angle_stays_flat():
    world = _smooth_world(7)
    m = Mosaic()
    for i in range(4):
        m.track(_rview(world, 400, 500, 1.0 * i))
    assert m.angle_deg == 0.0 and m.keyframes[0].angle_deg == 0.0


def _track_jump(world, step: float, theta: float) -> tuple[Mosaic, float]:
    m = Mosaic(track_rotation=True)
    m.track(_rview(world, 400, 500, 0.0))
    m.track(_rview(world, 400 + step, 500, theta))
    return m, math.hypot(m.x - step, m.y)


@pytest.mark.parametrize("seed", [1, 2])
@pytest.mark.parametrize("step", [0.0, 30.0])
@pytest.mark.parametrize("theta", [-3.0, -2.0, -1.0, 1.0, 2.0, 3.0])
def test_rotation_capture_range_is_matched_accurately(seed, step, theta):
    """Per-frame rotations up to 3 deg (Mosaic.ROT_CAPTURE_DEG = 3.5) are matched with position error <= 0.3 px
    and angle error <= 0.1 deg (measured: <= 0.10 px, <= 0.02 deg over 3 worlds x 3 shifts)."""
    m, err = _track_jump(_smooth_world(seed), step, theta)
    assert not m.lost
    assert err < 0.3 and abs(m.angle_deg - theta) < 0.1


@pytest.mark.parametrize("step", [0.0, 30.0])
@pytest.mark.parametrize("theta", [-12.0, -10.0, -8.0, 8.0, 10.0, 12.0])
def test_rotation_beyond_the_capture_range_is_not_matched(step, theta):
    """A single-frame rotation of 8 deg or more must fail safe: reported as lost, pose and angle untouched (the
    defect returned a pose 36 px off)."""
    m, _ = _track_jump(_smooth_world(1), step, theta)
    assert m.lost and (m.x, m.y) == (0.0, 0.0) and m.angle_deg == 0.0


@pytest.mark.parametrize("sign", [1.0, -1.0])
@pytest.mark.parametrize("theta", [4.0, 5.0, 6.0, 7.0])
def test_rotation_between_the_capture_range_and_8_degrees_is_never_a_wrong_pose(theta, sign):
    """Measured: 4-7 deg is matched only partly by the translation estimate; whatever is accepted must be
    right (position error < 1 px), the rest must be reported lost. Never a silent wrong pose."""
    for step in (0.0, 30.0):
        m, err = _track_jump(_smooth_world(1), step, sign * theta)
        assert m.lost or err < 1.0


# ------------------------------------------------------------------------------------------- RGB mosaic


def test_rgb_mosaic_has_the_source_colours_in_channel_order():
    """Where a single keyframe contributes, the colour mosaic is that frame's colour image: channel 0 is the
    source's R (identity), channel 1 is 0.6 R + 20, channel 2 is 1.3 R - 10, alpha marks valid canvas."""
    world = _world(9)
    m = Mosaic()
    frames = {}
    for x in (0.0, 150.0, 300.0):
        fr = _grab(world, x)
        frames[x] = fr
        assert m.track(fr, rgb=_rgb_of(fr))[0]
    rgba = m.image_rgb()
    assert rgba is not None and rgba.dtype == np.uint8 and rgba.shape[2] == 4
    valid = rgba[..., 3] > 0
    rows, cols = np.where(valid)
    crop = rgba[rows.min():rows.max() + 1, cols.min():cols.max() + 1]
    assert crop.shape[:2] == (H, 300 + W) and valid.sum() == crop.shape[0] * crop.shape[1]
    # the first 75 columns see only keyframe 0 (the next one starts at x = 150)
    expect = np.clip(_rgb_of(frames[0.0]) + 0.5, 0, 255).astype(np.uint8)
    assert np.abs(crop[:, :75, :3].astype(int) - expect[:, :75].astype(int)).max() <= 1
    # likewise the last 75 columns see only the last keyframe
    expect_last = np.clip(_rgb_of(frames[300.0]) + 0.5, 0, 255).astype(np.uint8)
    assert np.abs(crop[:, -75:, :3].astype(int) - expect_last[:, -75:].astype(int)).max() <= 1
    # the three channels are really three different quantities (a swap or an inversion cannot hide)
    assert not np.array_equal(crop[..., 0], crop[..., 1]) and not np.array_equal(crop[..., 1], crop[..., 2])


# ------------------------------------------------------------------------------------------- persistence


def _build_mosaic(blending: str) -> tuple[Mosaic, np.ndarray]:
    """A mosaic built only through the public API: a blurred first pass and a sharp revisit (so keyframes yield
    to each other), loop edges, colour, two overlapping height patches, an inset."""
    world = _world(21)
    m = Mosaic(blending=blending) if blending == "feather" else Mosaic()
    if blending != "feather":
        m.set_blending(blending)
    plan = [(0.0, True), (60.0, True), (120.0, False), (180.0, False), (240.0, False), (180.0, False),
            (120.0, False)]
    for x, blurred in plan:
        fr = _grab(world, x)
        if blurred:
            fr = cv2.GaussianBlur(fr, (0, 0), 1.6)
        m.track(fr, rgb=_rgb_of(fr))
    rng = np.random.default_rng(3)
    ph, pw = 120, 160
    yy, xx = np.mgrid[0:ph, 0:pw].astype(np.float32)
    h1 = 0.05 * xx + 0.02 * yy
    m.add_height_patch(20.0, 30.0, rng.uniform(50, 200, (ph, pw)).astype(np.float32), h1, "um", rid="a")
    m.add_height_patch(100.0, 40.0, rng.uniform(50, 200, (ph, pw)).astype(np.float32),
                       (0.05 * (xx + 80) + 0.02 * (yy + 10) + 3.0).astype(np.float32), "um", rid="b")
    m.add_inset("10x / 0.25", 0.5, 1.0, 2.0, _grab(world, 0.0)[:40, :40])
    return m, world


def _same_array(a, b) -> bool:
    if a is None or b is None:
        return a is None and b is None
    return a.dtype == b.dtype and np.array_equal(a, b, equal_nan=True)


def _assert_same_mosaic(a: Mosaic, b: Mosaic) -> None:
    """Every persisted field, compared exactly."""
    assert (a.tile, a.blending, a.track_rotation, a.areas, a.current_area, a.ref) == \
           (b.tile, b.blending, b.track_rotation, b.areas, b.current_area, b.ref)
    assert (a.x, a.y, a.angle_deg, a.prev_xy, a.velocity, a.lost, a.lost_frames) == \
           (b.x, b.y, b.angle_deg, b.prev_xy, b.velocity, b.lost, b.lost_frames)
    assert list(a.recent_b) == list(b.recent_b)
    assert _same_array(a.prev_frame, b.prev_frame) and _same_array(a.prev_bp, b.prev_bp)
    for fa, fb in (a.last_good, b.last_good), (a.lost_view, b.lost_view):
        assert (fa is None) == (fb is None)
        if fa is not None:
            assert all(_same_array(x, y) for x, y in zip(fa[:2], fb[:2])) and tuple(fa[2:]) == tuple(fb[2:])
    assert len(a.keyframes) == len(b.keyframes)
    for ka, kb in zip(a.keyframes, b.keyframes):
        assert (ka.x, ka.y, ka.sharpness, ka.area, ka.angle_deg) == (kb.x, kb.y, kb.sharpness, kb.area, kb.angle_deg)
        assert _same_array(ka.image, kb.image) and _same_array(ka.rgb, kb.rgb)
        assert _same_array(ka.bp.astype(np.float32), kb.bp.astype(np.float32))
        assert [i for i, _ in ka.yields] == [i for i, _ in kb.yields]
        assert all(np.array_equal(ga, gb) for (_, ga), (_, gb) in zip(ka.yields, kb.yields))
    assert a.edges == b.edges
    assert len(a.patches) == len(b.patches)
    for pa, pb in zip(a.patches, b.patches):
        assert (pa.x, pa.y, pa.z_unit, pa.offset, pa.kf, tuple(pa.rel), pa.z_key, pa.rid) == \
               (pb.x, pb.y, pb.z_unit, pb.offset, pb.kf, tuple(pb.rel), pb.z_key, pb.rid)
        assert _same_array(pa.edf, pb.edf) and _same_array(pa.height, pb.height)
    assert len(a.insets) == len(b.insets)
    for ia, ib in zip(a.insets, b.insets):
        assert (ia.objective, ia.um_per_px, ia.x_um, ia.y_um) == (ib.objective, ib.um_per_px, ib.x_um, ib.y_um)
        assert _same_array(ia.image, ib.image)


def _assert_same_render(a: Mosaic, b: Mosaic) -> None:
    """Intensity mosaic, colour mosaic and height mosaic. The canvas is rebuilt from keyframes and patches in
    index order while the original was built incrementally (and relaxed): float32 summation order differs, so
    intensities may differ by rounding noise only (<= 1e-3 grey levels, i.e. a tolerance of 1/1000 of one level
    where the defect had 94)."""
    (ia, oa), (ib, ob) = a.image(), b.image()
    assert oa == ob and ia.shape == ib.shape
    assert np.array_equal(np.isnan(ia), np.isnan(ib))
    assert np.nanmax(np.abs(ia - ib)) <= 1e-3
    ra, rb = a.image_rgb(), b.image_rgb()
    assert ra is not None and np.array_equal(ra[..., 3], rb[..., 3])
    assert np.abs(ra.astype(int) - rb.astype(int)).max() <= 1
    ha, hb = a.height_mosaic(), b.height_mosaic()
    assert ha is not None and ha[1] == hb[1]
    assert _same_array(ha[0], hb[0])


@pytest.mark.parametrize("blending", ["feather", "multiband"])
def test_session_round_trip_reproduces_the_mosaic_and_tracking_continues(tmp_path, blending):
    m, world = _build_mosaic(blending)
    assert len(m.edges) >= 4 and any(k.yields for k in m.keyframes)  # the test is not vacuous
    assert len(m.patches) == 2 and m.patches[0].offset != m.patches[1].offset
    p = tmp_path / "session.npz"
    save_session(p, {"mosaic": m, "um_per_px": 0.4, "z_unit": "um"}, {"objective": "10x / 0.25"})
    result, settings = load_session(p)
    assert settings == {"objective": "10x / 0.25"} and result["session_schema"] == 2
    loaded = result["mosaic"]
    _assert_same_mosaic(m, loaded)
    _assert_same_render(m, loaded)

    # the live path: the engine-style snapshot (taken under a lock) loads the same state
    meta, arrays = m.to_state_arrays()
    twin = Mosaic()
    twin.load_state_arrays(meta, arrays)
    _assert_same_mosaic(m, twin)

    # the next frames of the same pan go to the original and to the reloaded mosaic: same poses (<= 0.05 px)
    for x in (60.0, 0.0):
        fr = cv2.GaussianBlur(_grab(world, x), (0, 0), 1.6) if x == 0.0 else _grab(world, x)
        m.track(fr, rgb=_rgb_of(fr))
        loaded.track(fr, rgb=_rgb_of(fr))
        assert math.hypot(m.x - loaded.x, m.y - loaded.y) <= 0.05
        assert len(m.keyframes) == len(loaded.keyframes)


def test_session_round_trip_keeps_areas_and_a_lost_tracker(tmp_path):
    """Two areas (an unconnected view placed beside the first), and a mosaic saved in the middle of a loss."""
    world = _world(3)
    m = Mosaic()
    for x in (0.0, 30.0, 60.0, 90.0, 120.0):
        m.track(_grab(world, x))
    for _ in range(3):  # lost, but not yet an area of its own
        m.track(_grab(world, 700.0))
    assert m.lost and m.lost_frames == 3 and m.lost_view is not None and m.last_good is not None
    meta, arrays = m.to_state_arrays()
    lost_twin = Mosaic()
    lost_twin.load_state_arrays(meta, arrays)
    _assert_same_mosaic(m, lost_twin)
    for _ in range(45):  # the loss goes on: both start the second area at the same frame
        m.track(_grab(world, 700.0))
        lost_twin.track(_grab(world, 700.0))
        assert (m.areas, m.lost, m.lost_frames) == (lost_twin.areas, lost_twin.lost, lost_twin.lost_frames)
    assert m.areas == 2 and m.current_area == 1 and m.keyframes[-1].area == 1

    p = tmp_path / "areas.npz"
    save_session(p, {"mosaic": m}, {})
    loaded = load_session(p)[0]["mosaic"]
    assert loaded.areas == 2 and loaded.current_area == 1
    assert [k.area for k in loaded.keyframes] == [k.area for k in m.keyframes]
    _assert_same_mosaic(m, loaded)
    (a, oa), (b, ob) = m.image(), loaded.image()
    assert oa == ob and np.array_equal(a, b, equal_nan=True)


def test_state_without_tracker_fields_still_loads_and_continues():
    """A mosaic state written before the tracker state was saved (no "tracking" entry, no prev arrays)."""
    m, world = _build_mosaic("feather")
    meta, arrays = m.to_state_arrays()
    meta.pop("tracking")
    old = {k: v for k, v in arrays.items() if not k.startswith(("prev_", "last_good", "lost_view"))}
    loaded = Mosaic()
    loaded.load_state_arrays(meta, old)
    assert len(loaded.keyframes) == len(m.keyframes) and loaded.edges == m.edges
    loaded.track(_grab(world, 120.0))
    assert not loaded.lost and abs(loaded.x - 120.0) < 1.0


def test_failed_state_load_leaves_the_mosaic_untouched():
    m, _ = _build_mosaic("feather")
    before, ob = m.image()
    n = len(m.keyframes)
    meta, arrays = m.to_state_arrays()
    arrays.pop("kf_img_2")  # a damaged state
    with pytest.raises(KeyError):
        m.load_state_arrays(meta, arrays)
    after, oa = m.image()
    assert len(m.keyframes) == n and oa == ob and np.array_equal(before, after, equal_nan=True)


def test_rebuilding_the_canvas_keeps_height_patch_splats():
    """Switching the blend rebuilds the canvas; a height patch's all-in-focus image must stay on it."""
    m, _ = _build_mosaic("feather")
    before, ob = m.image()
    m.set_blending("multiband")
    m.set_blending("feather")
    after, oa = m.image()
    assert oa == ob and np.array_equal(np.isnan(before), np.isnan(after))
    assert np.nanmax(np.abs(before - after)) <= 1e-3


def test_session_without_a_mosaic_graph_loads(tmp_path):
    """A session in the format of the base branch (no mosaic graph at all), written by hand."""
    blob = {"schema": 2, "result": {"warnings": ["keep me"], "um_per_px": 0.25, "z_unit": "um"}, "settings": {"s": 1},
            "has_result": False}
    buf = io.BytesIO()
    np.savez_compressed(buf, __meta__=np.array(json.dumps(blob)), edf=np.ones((16, 16), np.float32))
    (tmp_path / "base.npz").write_bytes(buf.getvalue())
    r, st = load_session(tmp_path / "base.npz")
    assert "mosaic" not in r and r["session_schema"] == 2 and st == {"s": 1}
    assert r["warnings"] == ["keep me"] and np.array_equal(r["edf"], np.ones((16, 16), np.float32))


def test_session_round_trip_without_mosaic_unchanged(tmp_path):
    result = {"edf": np.zeros((16, 16), np.float32), "warnings": ["keep me"], "um_per_px": 0.25, "z_unit": "um"}
    p = tmp_path / "plain.npz"
    save_session(p, result, {"s": 1})
    r2, st = load_session(p)
    assert np.array_equal(r2["edf"], result["edf"])
    assert r2["warnings"] == ["keep me"] and r2["session_schema"] == 2 and st == {"s": 1}
    assert "mosaic" not in r2


def test_saved_file_layout_keeps_the_mosaic_out_of_band(tmp_path):
    m, _ = _build_mosaic("feather")
    p = tmp_path / "session.npz"
    save_session(p, {"mosaic": m}, {})
    with np.load(p) as zf:
        names = set(zf.files)
        blob = json.loads(str(zf["__meta__"]))
    assert "mosaic_meta" in blob and "mosaic" not in names
    assert {"mosaic_kf_img_0", "mosaic_patch_h_0", "mosaic_inset_img_0", "mosaic_prev_frame"} <= names


# ------------------------------------------------------------------------------------------- negative controls


def test_empty_mosaic_has_no_rgb_and_invalid_blending_is_rejected():
    m = Mosaic()
    assert m.image_rgb() is None
    img, _ = m.image()
    assert img.shape == (8, 8) and not np.isfinite(img).any()
    with pytest.raises(ValueError):
        m.set_blending("not-a-mode")


def test_load_session_rejects_bad_meta(tmp_path):
    buf = io.BytesIO()
    np.savez_compressed(buf, edf=np.zeros((4, 4), np.float32))
    (tmp_path / "real.npz").write_bytes(buf.getvalue())
    with pytest.raises(ValueError, match="not a MicroTopos session"):
        load_session(tmp_path / "real.npz")


def test_load_session_rejects_newer_schema(tmp_path):
    blob = json.dumps({"schema": 99, "result": {}, "settings": {}})
    buf = io.BytesIO()
    np.savez_compressed(buf, __meta__=np.array(blob), edf=np.zeros((4, 4), np.float32))
    (tmp_path / "new.npz").write_bytes(buf.getvalue())
    with pytest.raises(ValueError, match="session schema 99 is newer"):
        load_session(tmp_path / "new.npz")


def test_save_session_rejects_non_mosaic(tmp_path):
    with pytest.raises(ExportError):
        save_session(tmp_path / "x.npz", {"mosaic": "not a mosaic"}, {})


# ------------------------------------------------------------------------------------------- lateral scale


def _cal(frame=(640, 360), value=0.33) -> ObjectiveCalibration:
    return ObjectiveCalibration(objective="10x / 0.25", lateral_um_per_px_full=value, lateral_frame=frame)


OBJ = resolve("10x / 0.25")
MODES = {"1920x1080": (1920, 1080), "1280x720": (1280, 720), "1440x1080": (1440, 1080), "1080x1080": (1080, 1080),
         "1600x1200": (1600, 1200), "preview 640x360": (640, 360), "preview 640x480": (640, 480)}


def test_lateral_scale_covers_modes_of_the_calibrated_aspect_ratio_by_width():
    c = _cal(frame=(640, 360))
    for name in ("1920x1080", "1280x720", "preview 640x360"):
        w, h = MODES[name]
        assert c.lateral_status_for(w, h) is CalStatus.CALIBRATED, name
        assert c.um_per_px(w, OBJ, h) == pytest.approx(0.33 * 1920 / w), name


@pytest.mark.parametrize("mode", ["1440x1080", "1080x1080", "1600x1200", "preview 640x480"])
def test_lateral_scale_of_another_aspect_ratio_is_not_calibrated_and_never_guessed(mode):
    """A camera cannot tell a crop from a rescale: a 4:3 or 1:1 mode is NOT covered by a 16:9 calibration, so it
    is flagged NOMINAL and gets the nameplate scale (by width), not a number derived from the calibration."""
    c = _cal(frame=(640, 360))
    w, h = MODES[mode]
    assert c.lateral_status_for(w, h) is CalStatus.NOMINAL
    assert not c.covers(w, h)
    assert c.um_per_px(w, OBJ, h) == pytest.approx(OBJ.nominal_um_per_px(w))
    assert c.um_per_px(w, OBJ, h) != pytest.approx(0.33 * 1920 / w)


def test_a_calibration_made_in_a_4_3_mode_covers_4_3_and_not_16_9():
    c = _cal(frame=(640, 480), value=0.44)
    assert c.lateral_status_for(1440, 1080) is CalStatus.CALIBRATED
    assert c.um_per_px(1440, OBJ, 1080) == pytest.approx(0.44 * 1920 / 1440)
    assert c.lateral_status_for(1920, 1080) is CalStatus.NOMINAL


def test_legacy_calibration_without_a_recorded_frame_scales_by_width_for_every_mode():
    """Files written before the frame size was recorded keep the base-branch behaviour."""
    c = ObjectiveCalibration(objective="10x / 0.25", lateral_um_per_px_full=0.33)
    assert c.lateral_frame is None
    for name, (w, h) in MODES.items():
        assert c.um_per_px(w, OBJ, h) == pytest.approx(0.33 * 1920 / w), name
        assert c.um_per_px(w, OBJ) == c.um_per_px(w, OBJ, h)
        assert c.lateral_status_for(w, h) is CalStatus.CALIBRATED


def test_uncalibrated_objective_is_nominal_in_every_mode():
    c = ObjectiveCalibration(objective="10x / 0.25")
    for name, (w, h) in MODES.items():
        assert c.um_per_px(w, OBJ, h) == pytest.approx(OBJ.nominal_um_per_px(w)), name
        assert c.lateral_status_for(w, h) is CalStatus.NOMINAL


def test_lateral_frame_survives_the_calibration_file_and_legacy_files_still_load(tmp_path):
    store = CalibrationStore(tmp_path)
    c = store.get("10x / 0.25")
    c.lateral_um_per_px_full, c.lateral_frame = 0.33, (640, 360)
    store.touch("10x / 0.25")
    store.save()
    back = CalibrationStore(tmp_path).get("10x / 0.25")
    assert tuple(back.lateral_frame) == (640, 360) and not back.covers(1440, 1080) and back.covers(1280, 720)
    # a file from before: a removed field is ignored, a missing frame means width scaling
    meta = json.loads((tmp_path / "calibration.json").read_text())
    meta["10x / 0.25"].pop("lateral_frame")
    meta["10x / 0.25"]["calibrated_magnification"] = 10.0
    (tmp_path / "calibration.json").write_text(json.dumps(meta))
    old = CalibrationStore(tmp_path).get("10x / 0.25")
    assert old.lateral_frame is None and old.covers(1440, 1080) and not hasattr(old, "calibrated_magnification")


def test_every_call_site_gives_the_same_scale_for_the_same_camera_mode(monkeypatch):
    """The live engine, the batch runner and a direct call must agree, per camera mode, on the micrometres per
    FULL-FRAME pixel and on the status. The engine's call is made on the preview (640 px wide): its scale times
    preview_w / frame_w is the per-frame-pixel scale (the defect: 0.99 vs 0.33 for a 1440x1080 mode)."""
    from microtopos.live import PREVIEW_W, LiveEngine

    seen = []
    orig = ObjectiveCalibration.lateral_for

    def spy(self, frame_width, frame_height=None, objective=None):
        out = orig(self, frame_width, frame_height, objective)
        seen.append((frame_width, frame_height, out))
        return out

    monkeypatch.setattr(ObjectiveCalibration, "lateral_for", spy)
    for name in ("1920x1080", "1280x720", "1440x1080", "1600x1200"):
        w, h = MODES[name]
        store = CalibrationStore(None)
        store.get(OBJ).lateral_um_per_px_full = 0.33
        store.get(OBJ).lateral_frame = (640, 360)
        eng = LiveEngine(store, OBJ)
        eng.sync_callbacks = True
        seen.clear()
        eng.process_frame(np.random.default_rng(0).integers(0, 255, (h, w), dtype=np.uint8), 0.0, 1)
        pw = min(PREVIEW_W, w)
        calls = [s for s in seen if s[0] == pw]
        assert calls, f"{name}: the engine did not decide a scale"
        live_um, live_status = calls[0][2]
        direct, status = store.get(OBJ).lateral_for(w, h, OBJ)
        assert live_um * pw / w == pytest.approx(direct, rel=1e-3), name
        assert live_status is status
        assert (status is CalStatus.CALIBRATED) == (abs(w / h - 16 / 9) < 0.01), name


# ------------------------------------------------------------------------------------------- GUI session load

pg = pytest.importorskip("pyqtgraph")
from PySide6.QtCore import QSettings, QStandardPaths
from PySide6.QtWidgets import QApplication, QFileDialog


@pytest.fixture(scope="module")
def qapp(tmp_path_factory):
    QStandardPaths.setTestModeEnabled(True)
    QSettings.setPath(QSettings.NativeFormat, QSettings.UserScope, str(tmp_path_factory.mktemp("qsettings")))
    app = QApplication.instance() or QApplication([])
    pg.setConfigOptions(imageAxisOrder="row-major")
    yield app


@pytest.fixture
def window(qapp, tmp_path):
    from microtopos.gui.main_window import MainWindow

    os.environ["XDG_DATA_HOME"] = str(tmp_path / "data")
    os.environ["XDG_CONFIG_HOME"] = str(tmp_path / "config")
    w = MainWindow(simulator_only=True)
    w.show()
    for _ in range(20):
        qapp.processEvents()
    w.timer.stop()
    yield w
    w.close()
    qapp.processEvents()


def _result_arrays() -> dict:
    h = np.ones((16, 16), np.float32)
    return {"height": h, "height_raw": h, "edf": np.zeros((16, 16), np.uint8), "edf_proc": h, "sl_surface": h,
            "confidence": h, "um_per_px": 0.25, "z_unit": "um", "valid_fraction": 1.0, "frames_used": 10,
            "frames_in": 10, "measure": "DoG band-pass"}


def test_failed_session_load_leaves_the_live_mosaic_untouched(window, qapp, tmp_path, monkeypatch):
    """The session's result names an objective that does not exist: resolve_result fails. The saved mosaic must
    not have replaced the live one by then (the defect applied it first)."""
    w = window
    live, world = _build_mosaic("feather")
    meta, arrays = live.to_state_arrays()
    w.engine.load_mosaic_state(meta, arrays)
    before, ob = w.engine.mosaic.image()
    n = len(w.engine.mosaic.keyframes)

    other = Mosaic()
    for x in (0.0, 30.0, 60.0, 90.0, 120.0, 150.0, 180.0, 210.0, 240.0, 270.0, 300.0):
        other.track(_grab(world, x))
    assert len(other.keyframes) != n
    bad = tmp_path / "bad.npz"
    save_session(bad, dict(_result_arrays(), objective="No such objective 99x"), {}, {"mosaic": other})
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *a, **k: (str(bad), ""))
    w._load_session()
    after, oa = w.engine.mosaic.image()
    assert len(w.engine.mosaic.keyframes) == n and oa == ob and np.array_equal(before, after, equal_nan=True)
    assert w.loaded_mosaic is None and w.result is None

    good = tmp_path / "good.npz"
    save_session(good, dict(_result_arrays(), objective="10x / 0.25"), {}, {"mosaic": other})
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *a, **k: (str(good), ""))
    w._load_session()
    assert len(w.engine.mosaic.keyframes) == len(other.keyframes)  # a valid session does replace it


def test_saving_a_session_serializes_the_live_mosaic_under_the_engine_lock(window, qapp, tmp_path, monkeypatch):
    w = window
    live, _ = _build_mosaic("feather")
    w.engine.load_mosaic_state(*live.to_state_arrays())
    held = []
    orig = Mosaic.to_state_arrays

    def spy(self):
        held.append(not w.engine._lock.acquire(blocking=False))  # True: some thread already holds the lock
        if not held[-1]:
            w.engine._lock.release()
        return orig(self)

    monkeypatch.setattr(Mosaic, "to_state_arrays", spy)
    monkeypatch.setattr("microtopos.gui.main_window.MainWindow._save_path", lambda *a, **k: str(tmp_path / "s.npz"))
    w._save_session()
    assert held and all(held)  # every serialization of the live mosaic happened inside the engine lock
    assert len(load_session(tmp_path / "s.npz")[0]["mosaic"].keyframes) == len(live.keyframes)


def test_gui_flags_a_camera_mode_the_calibration_does_not_cover(window, qapp):
    w = window
    c = w.store.get(w.obj)
    c.lateral_um_per_px_full, c.lateral_frame = 0.33, (640, 360)
    real = w.engine.snapshot
    for shape, cal in (((1080, 1920), True), ((1080, 1440), False)):
        monkeypatch_snap = dataclasses.replace(real(), frame_shape=shape)
        w.engine.snapshot = lambda s=monkeypatch_snap: s
        w._refresh_checklist()
        text, tip = w.chip_xy.text(), w.chip_xy.toolTip()
        assert ("CAL " in text) == cal and ("NOMINAL" in text) == (not cal)
        assert ("Not calibrated for this camera mode" in tip) == (not cal)
    del w.engine.snapshot
