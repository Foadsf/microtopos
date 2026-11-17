"""Slice 2 mosaic capabilities: area joining, insets, multiband blending, rotation tracking, RGB
mosaics and session persistence. Includes negative controls for each area."""

from __future__ import annotations

import io
import json
import math

import cv2
import numpy as np
import pytest

from microtopos.export import ExportError, load_session, save_session
from microtopos.mosaic import HeightPatch, Mosaic

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
    """Pan out, lose tracking long enough to start a second area, then pan back in small steps:
    the returning chain loop-closes onto the first area and everything merges into one area."""
    world = _world(3)
    m = Mosaic()
    assert all(q > 0.3 for _, q in _track_sequence(m, world, [0.0, 30.0, 60.0, 90.0, 120.0]))
    assert m.areas == 1 and len(m.keyframes) >= 1
    # jump far away: content no longer overlaps -> 40+ lost frames start a separate area
    assert not m.track(_grab(world, 700.0))[0]
    for _ in range(45):
        m.track(_grab(world, 700.0))
    assert m.areas == 2
    # pan back in small steps so every frame is tracked; the end overlaps area 0's content
    assert all(q > 0.3 for _, q in _track_sequence(m, world, [float(x) for x in np.arange(680.0, 90.0, -20.0)]))
    assert m.areas == 1
    assert all(k.area == 0 for k in m.keyframes)
    assert len(m.keyframes) >= 4
    assert _graph_connected(m)
    assert math.isfinite(m.x) and math.isfinite(m.y)


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


# ------------------------------------------------------------------------------------------- multiband blending


def test_multiband_blending_renders_and_round_trips():
    world = _world(5)
    m = Mosaic()
    m.set_blending("multiband")
    assert m.track(_grab(world, 0.0)) and m.track(_grab(world, 60.0))
    img, origin = m.image()
    assert img.ndim == 2 and np.isfinite(img).all() and img.shape[0] >= H and img.shape[1] >= W
    # feather mode gives a different blend where tiles overlap
    m.set_blending("feather")
    img2, origin2 = m.image()
    oy, ox = origin2[1] - origin[1], origin2[0] - origin[0]
    sub = img[oy : oy + img2.shape[0], ox : ox + img2.shape[1]]
    assert sub.shape == img2.shape
    finite = np.isfinite(img2)
    assert finite.any() and not np.allclose(sub[finite], img2[finite], atol=1e-3)
    with pytest.raises(ValueError):
        m.set_blending("bogus")

    meta, arrays = m.to_state_arrays()
    m3 = Mosaic()
    m3.load_state_arrays(meta, arrays)
    assert m3.blending == "feather"
    img3, _ = m3.image()
    assert np.array_equal(img3, img2)


# ------------------------------------------------------------------------------------------- rotation tracking


def test_rotation_tracking_accumulates_angle():
    rng = np.random.default_rng(11)
    base = (rng.standard_normal((H, W)) * 40 + 100).astype(np.float32)
    M = cv2.getRotationMatrix2D((W / 2, H / 2), 2.0, 1.0)

    m = Mosaic(track_rotation=True)
    for i in range(8):
        base = cv2.warpAffine(base, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        _, q = m.track(base)
        assert q > 0.3
    assert abs(abs(m.angle_deg) - 16.0) < 2.5
    # pure rotation moves the camera negligibly: a single keyframe whose angle stays at origin
    assert len(m.keyframes) == 1 and m.keyframes[0].angle_deg == 0.0

    # without rotation tracking the angle estimate stays flat
    m2 = Mosaic()
    base0 = (rng.standard_normal((H, W)) * 40 + 100).astype(np.float32)
    for i in range(6):
        base0 = cv2.warpAffine(base0, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        m2.track(base0)
    assert m2.angle_deg == 0.0


# ------------------------------------------------------------------------------------------- RGB mosaic


def test_rgb_mosaic_and_state_round_trip():
    world = _world(9)
    m = Mosaic()
    positions = [0.0, 150.0, 300.0]
    for x in positions:
        fr = _grab(world, x)
        assert m.track(fr, rgb=_rgb_of(fr))[0]
    rgba = m.image_rgb()
    assert rgba is not None and rgba.ndim == 3 and rgba.shape[2] == 4
    assert rgba[..., 3].max() > 0
    assert 0 <= rgba[..., :3].min() and rgba[..., :3].max() <= 255

    meta, arrays = m.to_state_arrays()
    assert all(km["has_rgb"] for km in meta["keyframes"])
    assert all(f"kf_rgb_{i}" in arrays for i in range(len(m.keyframes)))
    m2 = Mosaic()
    m2.load_state_arrays(meta, arrays)
    assert len(m2.keyframes) == 3
    for a, b in zip(m.keyframes, m2.keyframes):
        assert (a.x, a.y, a.area, a.angle_deg) == (b.x, b.y, b.area, b.angle_deg)
        assert np.array_equal(a.image, b.image)
        assert np.array_equal(a.rgb, b.rgb)
    assert np.array_equal(m2.image_rgb(), rgba)

    # tracking continues from the loaded state
    fr = _grab(world, 150.0)
    m.track(fr, rgb=_rgb_of(fr))
    m2.track(fr, rgb=_rgb_of(fr))
    assert m2.x == m.x and m2.y == m.y


# ------------------------------------------------------------------------------------------- session persistence


def test_session_round_trip_keeps_mosaic(tmp_path):
    world = _world(4)
    m = Mosaic(track_rotation=True)
    m.set_blending("multiband")
    m.add_inset("10x / 0.25", 0.5, 1.0, 2.0, _grab(world, 0.0)[:40, :40])
    patch = HeightPatch(
        10.0, 20.0, np.full((8, 8), 1.0, np.float32), np.full((8, 8), 5.0, np.float32), "um", 0.5, kf=0, rel=(1.0, 2.0)
    )
    m.patches.append(patch)
    for x in (0.0, 40.0, 80.0):
        m.track(_grab(world, x), rgb=_rgb_of(_grab(world, x)))

    p = tmp_path / "session.npz"
    save_session(p, {"mosaic": m, "um_per_px": 0.4, "z_unit": "um"}, {"objective": "10x / 0.25"})
    result, settings = load_session(p)
    assert settings == {"objective": "10x / 0.25"}
    assert result["session_schema"] == 2
    assert result["um_per_px"] == 0.4 and result["z_unit"] == "um"
    loaded = result["mosaic"]
    assert isinstance(loaded, Mosaic)
    assert loaded.blending == "multiband" and loaded.track_rotation
    assert len(loaded.keyframes) == len(m.keyframes)
    assert len(loaded.insets) == 1 and len(loaded.patches) == 1
    assert np.array_equal(loaded.insets[0].image, m.insets[0].image)
    pi_ = loaded.patches[0]
    assert np.array_equal(pi_.height, patch.height) and pi_.kf == 0 and tuple(pi_.rel) == (1.0, 2.0)
    assert np.array_equal(loaded.image()[0], m.image()[0])
    assert np.array_equal(loaded.image_rgb(), m.image_rgb())
    # the mosaic rides along out of band: meta + mosaic_* arrays, never a bare "mosaic" array in the file
    with np.load(p) as zf:
        names = set(zf.files)
        blob = json.loads(str(zf["__meta__"]))
        assert "mosaic_meta" in blob  # meta rides inside __meta__
        assert "mosaic" not in names  # never a bare mosaic array
        assert any(n.startswith("mosaic_") for n in names)  # arrays out of band under mosaic_ prefix
    assert isinstance(result["mosaic"], Mosaic)


def test_session_round_trip_without_mosaic_unchanged(tmp_path):
    result = {"edf": np.zeros((16, 16), np.float32), "warnings": ["keep me"], "um_per_px": 0.25, "z_unit": "um"}
    p = tmp_path / "plain.npz"
    save_session(p, result, {"s": 1})
    r2, st = load_session(p)
    assert np.array_equal(r2["edf"], result["edf"])
    assert r2["warnings"] == ["keep me"] and r2["session_schema"] == 2 and st == {"s": 1}
    assert "mosaic" not in r2


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
