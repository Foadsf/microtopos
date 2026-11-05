"""Capture, replay, calibration ownership, persistence and export: each fix with a control that must fail without it."""

import json
import time

import cv2
import numpy as np
import pytest

from microtopos import export, zstack
from microtopos.calibration import CalibrationStore
from microtopos.camera import FrameSource, ReplaySource
from microtopos.live import LiveEngine
from microtopos.mosaic import Mosaic
from microtopos.optics import OBJECTIVES
from microtopos.recording import RecordingError, RecordingWriter, load_manifest

OBJ = OBJECTIVES["10x / 0.25"]


def _jpeg(img):
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    return buf.reshape(-1)


def _frame(v=128, shape=(90, 160)):
    rng = np.random.default_rng(int(v))
    return np.clip(rng.normal(v, 20, shape + (3,)), 0, 255).astype(np.uint8)


# ------------------------------------------------------------------------------------------------ recording
def test_writer_raw_and_manifest(tmp_path):
    w = RecordingWriter(tmp_path / "r", "cam")
    for i in range(3):
        w.add(_jpeg(_frame(100 + i)), i + 1, 0.08 * i)
    assert w.close() == 3
    m = load_manifest(tmp_path / "r")
    assert m.mode == "mjpeg-raw" and len(m.files) == 3 and m.seqs == [1, 2, 3] and not m.warnings
    meta = json.loads((tmp_path / "r" / "frames.json").read_text())
    assert meta["complete"] is True and all(e["bytes"] > 0 and "crc32" in e for e in meta["frames"])


def test_writer_decoded_frames_are_png_not_fake_jpeg(tmp_path):
    w = RecordingWriter(tmp_path / "r")
    w.add(_frame(), 1, 0.0)  # a decoded image (backend refused raw mode)
    w.close()
    m = load_manifest(tmp_path / "r")
    assert m.mode == "decoded-png" and m.files[0].suffix == ".png"
    assert cv2.imread(str(m.files[0])) is not None


def test_writer_rejects_non_jpeg_bytes_and_stale_folder(tmp_path):
    w = RecordingWriter(tmp_path / "r")
    with pytest.raises(RecordingError):
        w.add(np.arange(100, dtype=np.uint8), 1, 0.0)
    w.close()
    (tmp_path / "old").mkdir()
    (tmp_path / "old" / "000000.jpg").write_bytes(b"x")
    with pytest.raises(FileExistsError):
        RecordingWriter(tmp_path / "old")


def test_writer_failure_is_visible_and_manifest_marked_incomplete(tmp_path, monkeypatch):
    w = RecordingWriter(tmp_path / "r")
    w.add(_jpeg(_frame()), 1, 0.0)
    import builtins
    real_open = builtins.open

    def failing_open(path, mode="r", *a, **k):
        if str(path).endswith(".jpg") and "w" in mode:
            raise OSError(28, "No space left on device")
        return real_open(path, mode, *a, **k)

    monkeypatch.setattr(builtins, "open", failing_open)
    with pytest.raises(RecordingError):
        w.add(_jpeg(_frame()), 2, 0.1)
    monkeypatch.undo()
    w.close()
    meta = json.loads((tmp_path / "r" / "frames.json").read_text())
    assert meta["complete"] is False and "No space" in meta["error"] and len(meta["frames"]) == 1


def test_legacy_manifest_count_mismatch_is_reported(tmp_path):
    d = tmp_path / "v1"
    d.mkdir()
    for i in range(3):
        (d / f"{i:06d}.jpg").write_bytes(_jpeg(_frame(90 + i)).tobytes())
    (d / "frames.json").write_text(json.dumps({"timestamps": [0.0, 0.08]}))
    m = load_manifest(d)
    assert len(m.files) == 2 and any("3 JPEG files but 2 timestamps" in w for w in m.warnings)
    rs = ReplaySource(str(d))
    assert len(list(rs.frames())) == 2  # direct iteration and timed playback now use the same list


def test_corrupt_frame_is_skipped_and_engine_keeps_running(tmp_path):
    w = RecordingWriter(tmp_path / "r")
    for i in range(6):
        w.add(_jpeg(_frame(80 + 10 * i)), i + 1, 0.08 * i)
    w.close()
    (tmp_path / "r" / "000002.jpg").write_bytes(b"\xff\xd8garbage\xff\xd9")
    rs = ReplaySource(str(tmp_path / "r"), exhaustive=True)
    assert len(list(rs.frames())) == 5 and rs.corrupt == 1
    eng = LiveEngine(CalibrationStore(None), OBJ)
    eng.start()
    rs2 = ReplaySource(str(tmp_path / "r"), exhaustive=True)
    eng.set_source(rs2)
    rs2.start()
    t0 = time.time()
    while not rs2.done and time.time() - t0 < 20:
        time.sleep(0.05)
    time.sleep(0.3)
    eng.stop()
    rs2.stop()
    assert rs2.done and rs2.corrupt == 1 and eng.error is None
    assert eng.snapshot().seq == 6  # the last frame was processed; gaps in seq mean real gaps


def test_engine_error_boundary(monkeypatch):
    class Src(FrameSource):
        def _run(self):
            for i in range(30):
                self._publish(_frame(100))
                time.sleep(0.01)

    eng = LiveEngine(CalibrationStore(None), OBJ)
    calls = {"n": 0}
    orig = eng._process

    def flaky(*a, **k):
        calls["n"] += 1
        if calls["n"] == 3:
            raise RuntimeError("deliberate")
        return orig(*a, **k)

    monkeypatch.setattr(eng, "_process", flaky)
    src = Src()
    eng.set_source(src)
    eng.start()
    src.start()
    time.sleep(0.8)
    eng.stop()
    src.stop()
    assert eng.errors == 1 and "deliberate" in eng.error and calls["n"] > 5  # it went on after the failure


# ------------------------------------------------------------------------------------------------ ownership
def test_dark_frame_cancelled_when_source_changes():
    store = CalibrationStore(None)
    eng = LiveEngine(store, OBJ)
    eng.request_calibration("dark", 2)
    eng.process_frame(np.full((90, 160, 3), 60, np.uint8), 0.0, 1)
    eng.set_source(None)  # a different source before the average is complete
    eng.process_frame(np.full((90, 160, 3), 100, np.uint8), 0.1, 1)
    assert not store.get(OBJ).darks and eng.calib_error and "cancelled" in eng.calib_error


def test_dark_frame_cancelled_on_mode_change_and_saved_otherwise():
    store = CalibrationStore(None)
    eng = LiveEngine(store, OBJ)
    eng.request_calibration("dark", 2)
    eng.process_frame(np.full((90, 160, 3), 60, np.uint8), 0.0, 1)
    eng.process_frame(np.full((80, 160, 3), 60, np.uint8), 0.1, 2)  # capture size changed
    assert not store.get(OBJ).darks and eng.calib_error
    eng.request_calibration("dark", 2)
    eng.process_frame(np.full((90, 160, 3), 60, np.uint8), 0.2, 3)
    eng.process_frame(np.full((90, 160, 3), 62, np.uint8), 0.3, 4)
    (arr,) = store.get(OBJ).darks.values()
    assert abs(float(arr.mean()) - 61.0) < 0.5 and store.get(OBJ).revision == 1


def test_late_height_map_not_pinned_into_a_restarted_mosaic():
    eng = LiveEngine(CalibrationStore(None), OBJ)
    gen = eng.mosaic_generation
    eng.reset_mosaic()
    eng.add_height_patch((0, 0), np.zeros((36, 64), np.float32), np.zeros((36, 64), np.float32), "um",
                         mosaic_generation=gen)
    eng.process_frame(_frame(), 0.0, 1)
    assert not eng.mosaic.patches
    eng.add_height_patch((0, 0), np.zeros((36, 64), np.float32), np.zeros((36, 64), np.float32), "um")
    eng.process_frame(_frame(), 0.1, 2)
    assert len(eng.mosaic.patches) == 1


def test_source_switch_does_not_skip_first_frame():
    class One(FrameSource):
        def __init__(self, v):
            super().__init__()
            self._publish(_frame(v))

        def _run(self):
            pass

    eng = LiveEngine(CalibrationStore(None), OBJ)
    a = One(50)
    eng.set_source(a)
    eng.start()
    time.sleep(0.3)
    b = One(200)  # also sequence number 1
    eng.set_source(b)
    time.sleep(0.3)
    eng.stop()
    assert eng.snapshot().preview is not None and float(eng.snapshot().preview.mean()) > 150


def test_keyframe_correction_removes_exactly_what_was_added():
    m = Mosaic()
    img = np.full((64, 96), 100.25, np.float32)
    m._add_keyframe(img, m._bp(img), 1.0, 0.0, 0.0, None, (0, 0))
    k = m.keyframes[0]
    m._splat(k.image, k.x, k.y, weight=-1.0)  # what _relax does before re-adding at the new position
    num = sum(float(np.abs(n).max()) for n, _ in m.tiles.values())
    assert num < 1e-3


# ------------------------------------------------------------------------------------------------ export
def _result():
    rng = np.random.default_rng(0)
    h = rng.normal(0, 1, (40, 60)).astype(np.float32)
    h[:5, :5] = np.nan
    return {"edf": rng.uniform(0, 255, (80, 120)).astype(np.float32), "height": h, "height_raw": h.copy(),
            "edf_proc": rng.uniform(0, 255, (40, 60)).astype(np.float32), "sl_surface": h * 0.5,
            "confidence": np.ones_like(h), "Sq": np.float32(1.5), "bad": np.float64("nan"), "z_unit": "um",
            "um_per_px": 0.6, "objective": "10x / 0.25", "warnings": ["acquisition warning kept"]}


def test_report_bundle_strict_json_and_encoder_failure(tmp_path, monkeypatch):
    files = export.write_report_bundle(tmp_path / "ok", _result(), {"microtopos": "t"})
    meta = json.loads((tmp_path / "ok" / "report.json").read_text())  # strict parse
    assert meta["bad"] is None and abs(meta["Sq"] - 1.5) < 1e-6 and all(f.exists() for f in files)
    monkeypatch.setattr(cv2, "imencode", lambda *a, **k: (False, None))
    with pytest.raises(export.ExportError):
        export.write_report_bundle(tmp_path / "bad", _result(), {})
    assert not any((tmp_path / "bad").iterdir())  # nothing half-written is left behind


def test_session_round_trip_keeps_original_arrays_and_warnings(tmp_path):
    r = _result()
    export.save_session(tmp_path / "s.npz", r, {"objective": "10x / 0.25"})
    r2, st = export.load_session(tmp_path / "s.npz")
    for k in ("edf_proc", "sl_surface", "height_raw"):
        assert np.array_equal(r2[k], r[k], equal_nan=True)
    assert r2["warnings"] == ["acquisition warning kept"] and r2["session_schema"] == 2 and not r2["migrated"]
    export.save_session(tmp_path / "m.npz", None, {}, {"mosaic_image": np.zeros((8, 8), np.float32)})
    m2, _ = export.load_session(tmp_path / "m.npz")
    assert "mosaic_image" in m2 and "height" not in m2


def test_session_validation_rejects_inconsistent_files(tmp_path):
    r = _result()
    r["confidence"] = np.ones((3, 3), np.float32)
    export.save_session(tmp_path / "s.npz", r, {})
    with pytest.raises(ValueError):
        export.load_session(tmp_path / "s.npz")


def test_height_export_writes_mask_and_provenance(tmp_path):
    h = _result()["height"]
    for fmt in ("gsf", "tif", "stl"):
        paths = export.export_height(tmp_path / f"h.{fmt}", fmt, h, 0.6, "frames", {"objective": "10x"})
        assert all(p.exists() for p in paths)
        info = json.loads((tmp_path / "h.json").read_text())
        assert info["format"] == fmt and info["z_unit"] == "frames" and "invalid_pixels" in info
    mask = cv2.imread(str(tmp_path / "h.mask.png"), cv2.IMREAD_GRAYSCALE)
    assert mask[0, 0] == 0 and mask[-1, -1] == 255


# ------------------------------------------------------------------------------------------------ batch
def _sim_recording(folder, n_sweep=70):
    from microtopos.simulator import SpecimenSimulator

    sim = SpecimenSimulator(3, OBJ)
    zs = sim.sweep_trajectory(n_frames=n_sweep)
    z0 = float(zs[0])
    w = RecordingWriter(folder, "sim")
    path = [z0] * 40 + list(zs) + [float(zs[-1])] * 40
    for i, z in enumerate(path):
        sim.focus_z_um = z
        bgr, _ = sim.render()
        w.add(_jpeg(bgr), i + 1, 0.04 * i)
    w.close()


def test_batch_is_deterministic(tmp_path):
    from microtopos.batch import run_batch

    _sim_recording(tmp_path / "rec")
    a = run_batch(tmp_path / "rec", tmp_path / "a", log=lambda *_: None)
    b = run_batch(tmp_path / "rec", tmp_path / "b", log=lambda *_: None)
    assert a["sweeps"] and "error" not in a["sweeps"][0], a["sweeps"]
    assert [s.get("Sq") for s in a["sweeps"]] == [s.get("Sq") for s in b["sweeps"]]
    import tifffile
    ha = tifffile.imread(tmp_path / "a" / "sweep-01" / "height.tif")
    hb = tifffile.imread(tmp_path / "b" / "sweep-01" / "height.tif")
    assert np.array_equal(ha, hb, equal_nan=True)


def test_rescale_heights_flips_sign_and_scales():
    rng = np.random.default_rng(1)
    h = (np.arange(64)[None, :] * 0.1 + rng.normal(0, 0.05, (48, 64))).astype(np.float32)
    r = {"height": h, "height_raw": h.copy(), "edf_proc": rng.uniform(0, 255, h.shape).astype(np.float32),
         "um_per_px": 1.0, "objective": "10x / 0.25", "z_unit": "frames", "warnings": ["Sweep direction unknown: x"],
         "z_values": [0.0, 1.0], "form_order": 0}
    from microtopos import metrology
    r.update({k: v for k, v in metrology.analyze(h, r["edf_proc"], 1.0, "frames", 1.1, 0).items() if k != "warnings"})
    r2 = zstack.rescale_heights(r, -0.5, "um")
    assert np.allclose(r2["height"], -0.5 * h) and r2["z_unit"] == "um"
    assert abs(r2["Sq"] - 0.5 * r["Sq"]) < 1e-4 and "Sdr_pct" in r2 and not any("direction" in w for w in r2["warnings"])
