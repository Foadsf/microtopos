"""Regression tests verifying and guarding every reported bug in the fix/reported-bugs branch."""

from __future__ import annotations

import json
import os
import subprocess
import threading
import time

import cv2
import numpy as np
import pytest

from microtopos import batch, calibration, export, kinematics, live, metrology, objectives, sff, zstack
from microtopos.camera import CameraInfo, CameraSource, SimulatorSource, V4L2Controls
from microtopos.mosaic import HeightPatch, Mosaic
from microtopos.recording import RecordingError, RecordingWriter


def _jpeg(img: np.ndarray) -> np.ndarray:
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    return buf.reshape(-1)


def _tex(seed: int = 0, shape: tuple[int, int] = (90, 160)) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.clip(rng.normal(128, 25, shape + (3,)), 0, 255).astype(np.uint8)


# ------------------------------------------------------------------------------------------------ A1
def test_recording_writer_marks_incomplete_on_non_io_errors(tmp_path):
    # 1) Non-JPEG bytes
    w1 = RecordingWriter(tmp_path / "r1", "cam")
    with pytest.raises(RecordingError):
        w1.add(np.arange(64, dtype=np.uint8), 1, 0.0)
    w1.close()
    m1 = json.loads((tmp_path / "r1" / "frames.json").read_text())
    assert m1["complete"] is False and m1["error"]

    # 2) Format change mid-recording
    w2 = RecordingWriter(tmp_path / "r2", "cam")
    w2.add(_jpeg(_tex()), 1, 0.0)
    with pytest.raises(RecordingError):
        w2.add(_tex(), 2, 0.1)
    w2.close()
    m2 = json.loads((tmp_path / "r2" / "frames.json").read_text())
    assert m2["complete"] is False and "changed" in m2["error"]

    # 3) Periodic decode check failure (frame 0 has valid SOI/EOI markers but corrupt body)
    w3 = RecordingWriter(tmp_path / "r3", "cam")
    with pytest.raises(RecordingError):
        w3.add(np.frombuffer(b"\xff\xd8garbage\xff\xd9", dtype=np.uint8), 1, 0.0)
    w3.close()
    m3 = json.loads((tmp_path / "r3" / "frames.json").read_text())
    assert m3["complete"] is False and "decode" in m3["error"]


# ------------------------------------------------------------------------------------------------ A3
def test_rescale_heights_preserves_filters_ordering_fwhm_and_clears_repeatability():
    h = np.random.default_rng(0).normal(0, 1, (32, 32)).astype(np.float32)
    edf = np.full((32, 32), 128, np.float32)
    cutoff = objectives.resolve("10x / 0.25").lateral_cutoff_um

    # Case 1: default optical cutoff below 4 px (default_note)
    m_def = metrology.analyze(h, edf, 0.5, "frames", cutoff, lam_s_um=None)
    r = {
        **m_def, "height": h, "height_raw": h.copy(), "edf_proc": edf, "um_per_px": 0.5,
        "objective": "10x / 0.25", "z_values": [0.0, 1.0, 2.0], "z_first": 0.0, "z_last": 2.0,
        "fwhm_z": 4.0, "frame_indices": [10, 11, 12], "sample_positions": [1.0, 2.0, 3.0],
        "sigma_inst": 0.1, "repeat_offset": 0.2, "repeat_overlap": 0.9, "sigma_inst_lt_Sq_over_10": True,
    }
    r2 = zstack.rescale_heights(r, -2.0, "um")
    assert r2["filter_notes"] == m_def["filter_notes"]
    assert r2["z_values"] == [-4.0, -2.0, -0.0]
    assert r2["z_first"] == -4.0 and r2["z_last"] == -0.0
    assert r2["frame_indices"] == [12, 11, 10] and r2["sample_positions"] == [3.0, 2.0, 1.0]
    assert r2["fwhm_z"] == pytest.approx(8.0)
    assert "sigma_inst" not in r2 and "repeat_offset" not in r2
    assert any("Repeatability cleared" in w for w in r2["warnings"])

    # Case 2: explicitly requested unresolvable S-filter and L-filter
    m_req = metrology.analyze(h, edf, 0.5, "frames", cutoff, lam_s_um=1.5, lam_c_um=1.5)
    r_req = {**m_req, "height": h, "height_raw": h.copy(), "edf_proc": edf, "um_per_px": 0.5,
             "objective": "10x / 0.25", "z_values": [0.0, 1.0], "z_first": 0.0, "z_last": 1.0}
    r_req2 = zstack.rescale_heights(r_req, 1.5, "um")
    assert r_req2["lambda_s_requested_um"] == 1.5 and r_req2["lambda_c_requested_um"] == 1.5
    assert any("S-filter not applied" in w for w in r_req2["warnings"])
    assert any("L-filter not applied" in w for w in r_req2["warnings"])


# ------------------------------------------------------------------------------------------------ A5 & B4
def test_mosaic_sharp_over_blurry_and_relax_yield_offsets():
    rng = np.random.default_rng(1)
    sharp = rng.uniform(0, 255, (64, 96)).astype(np.float32)
    blurry = cv2.GaussianBlur(sharp, (0, 0), 3.0)

    mos = Mosaic()
    mos._add_keyframe(blurry, mos._bp(blurry), 1.0, 0.0, 0.0, None, (0, 0))
    mos._add_keyframe(sharp, mos._bp(sharp), 10.0, 4.0, 0.0, 0, (4.0, 0.0))
    assert len(mos.keyframes[0].yields) == 1

    # Add an edge pulling keyframe 1 to (10, 0) and relax; verify exact un-splat/re-splat consistency
    mos.edges.append((0, 1, 10.0, 0.0, 5.0))
    mos._relax()
    for k in mos.keyframes:
        mos._splat(k.image, k.x, k.y, weight=-1.0, mask=mos._yield_mask(k))
    residual = sum(float(np.abs(n).max()) for n, _ in mos.tiles.values())
    assert residual < 1e-3


def test_mosaic_solve_offsets_resets_non_overlapping_and_anchors_components():
    p_disjoint = [
        HeightPatch(0, 0, np.zeros((32, 32)), np.ones((32, 32)), "um", offset=5.0),
        HeightPatch(100, 0, np.zeros((32, 32)), np.ones((32, 32)), "um", offset=-3.0),
    ]
    Mosaic._solve_offsets_for(p_disjoint)
    assert [p.offset for p in p_disjoint] == [0.0, 0.0]

    p_two_comp = [
        HeightPatch(0, 0, np.zeros((32, 32)), np.zeros((32, 32)), "um"),
        HeightPatch(0, 0, np.zeros((32, 32)), np.zeros((32, 32)), "um"),
        HeightPatch(200, 0, np.zeros((32, 32)), np.full((32, 32), 10.0), "um"),
        HeightPatch(200, 0, np.zeros((32, 32)), np.full((32, 32), 20.0), "um"),
    ]
    Mosaic._solve_offsets_for(p_two_comp)
    assert p_two_comp[0].offset == pytest.approx(0.0)
    assert p_two_comp[1].offset == pytest.approx(0.0)
    assert p_two_comp[2].offset == pytest.approx(0.0)
    assert p_two_comp[3].offset == pytest.approx(-10.0)


# ------------------------------------------------------------------------------------------------ A6
def test_objectives_commit_edit_clears_readded_and_handles_reused_name(tmp_path):
    store = calibration.CalibrationStore(tmp_path)
    old = objectives.ObjectiveSet([objectives.Objective("A", 10, 0.25), objectives.Objective("B", 20, 0.40)])
    store.get("A").lateral_um_per_px_full = 0.3

    # Remove A, keep B, then re-add A at 50x: stale 10x lateral calibration must be cleared
    new1, _ = objectives.commit_edit(tmp_path, store, old, [objectives.Objective("B", 20, 0.40)])
    assert calibration.CalibrationStore(tmp_path).get("A").lateral_magnification == 10
    _, notes2 = objectives.commit_edit(tmp_path, store, new1,
                                       [objectives.Objective("A", 50, 0.85), objectives.Objective("B", 20, 0.40)])
    assert store.get("A").lateral_um_per_px_full is None
    assert any("cleared" in n for n in notes2)

    # Rename A(10x) -> C(10x) and add new row named A(20x) in the same edit: C keeps cal, new A gets no false mismatch
    store.get("A").lateral_um_per_px_full = 0.3
    store.get("A").lateral_magnification = 50
    old3 = objectives.ObjectiveSet([objectives.Objective("A", 50, 0.85)])
    _, notes3 = objectives.commit_edit(tmp_path, store, old3,
                                       [objectives.Objective("C", 50, 0.85), objectives.Objective("A", 20, 0.40)],
                                       renames={"A": "C"})
    assert store.get("C").lateral_um_per_px_full == 0.3
    assert not any("A:" in n and "cleared" in n for n in notes3)


def test_objectives_commit_edit_flags_legacy_unstamped_calibration_on_readd(tmp_path):
    """A calibration file written before magnifications were recorded holds a lateral scale for 'A' with no
    magnification. When 'A' comes back at some magnification, nothing can confirm the number belongs to it:
    it is kept (as removal promised) but the user is told, once, that it is unverified."""
    store = calibration.CalibrationStore(tmp_path)
    c = store.get("A")
    c.lateral_um_per_px_full, c.lateral_source = 0.3, "stage micrometer: 100 µm over 333.3 px"
    assert c.lateral_magnification is None
    old = objectives.ObjectiveSet([objectives.Objective("B", 20, 0.40)])
    new_list = [objectives.Objective("A", 50, 0.85), objectives.Objective("B", 20, 0.40)]

    new1, notes = objectives.commit_edit(tmp_path, store, old, new_list)
    assert store.get("A").lateral_um_per_px_full == 0.3
    assert store.get("A").lateral_magnification == 50
    flagged = [n for n in notes if n.startswith("A:") and "predates magnification tracking" in n]
    assert len(flagged) == 1 and "50x" in flagged[0] and "0.3 µm/px" in flagged[0]
    assert calibration.CalibrationStore(tmp_path).get("A").lateral_magnification == 50  # persisted

    # said once: the next edit of the same list is quiet, and a later magnification change clears as usual
    new2, notes2 = objectives.commit_edit(tmp_path, store, new1, new_list)
    assert not any("predates" in n for n in notes2)
    new3, notes3 = objectives.commit_edit(tmp_path, store, new2,
                                          [objectives.Objective("A", 10, 0.25), objectives.Objective("B", 20, 0.40)])
    assert store.get("A").lateral_um_per_px_full is None and any("cleared" in n for n in notes3)

    # an objective that never left the list is stamped quietly: its calibration was measured on this row
    store.get("B").lateral_um_per_px_full = 0.15
    _, notes4 = objectives.commit_edit(tmp_path, store, new3, new3.objectives)
    assert store.get("B").lateral_magnification == 20 and not any(n.startswith("B:") for n in notes4)


# ------------------------------------------------------------------------------------------------ A7 & B1 & B2
def test_live_flush_preserves_simulated_and_manual_z_ends_immediately():
    obj = objectives.default_set().objectives[1]
    eng = live.LiveEngine(calibration.CalibrationStore(None), obj)
    eng.sync_callbacks = True
    caps = []
    eng.on_sweep = caps.append
    sim = SimulatorSource(1, obj, 160, 120)
    zs = sim.sim.sweep_trajectory(n_frames=30)

    # Start manual Z override, feed frames, then finish manual override -> immediate manual SweepCapture
    eng.sm.override = kinematics.Override.Z
    for i, z in enumerate(zs[:15]):  # incomplete (< 50% completeness)
        sim.sim.focus_z_um = float(z)
        bgr, truth = sim.sim.render()
        eng.process_frame(bgr, i * 0.04, i + 1, truth=truth)
    assert len(caps) == 0
    eng.sm.override = kinematics.Override.AUTO
    sim.sim.focus_z_um = float(zs[15])
    bgr, truth = sim.sim.render()
    eng.process_frame(bgr, 15 * 0.04, 16, truth=truth)
    assert len(caps) == 1
    assert caps[0].manual is True and caps[0].simulated is True

    # Also test flush() on an unended simulator sweep
    caps.clear()
    eng2 = live.LiveEngine(calibration.CalibrationStore(None), obj)
    eng2.sync_callbacks = True
    eng2.on_sweep = caps.append
    eng2.sm.override = kinematics.Override.Z
    for i, z in enumerate(zs):
        sim.sim.focus_z_um = float(z)
        bgr, truth = sim.sim.render()
        eng2.process_frame(bgr, i * 0.04, i + 1, truth=truth)
    eng2.flush()
    assert len(caps) == 1 and caps[0].simulated is True


def _half_sweep_frames():
    """A focus movement that stops at best focus and stays there: blurry -> sharp, never blurry again.

    At 640x480 the live engine detects it as a sweep, holds it as incomplete and emits it when the hold
    times out (at 160x120 the engine never emits anything for this input, so nothing would be tested)."""
    obj = objectives.default_set().objectives[1]
    sim = SimulatorSource(2, obj, 640, 480)
    zs = sim.sim.sweep_trajectory(n_frames=40)
    for z in [float(zs[0])] * 15 + [float(v) for v in zs[:20]] + [float(zs[19])] * 70:
        sim.sim.focus_z_um = z
        yield sim.sim.render()[0]


def test_engine_emits_the_half_sweep_as_unusable():
    """Precondition of the batch test below: the incomplete capture really reaches the consumer."""
    eng = live.LiveEngine(calibration.CalibrationStore(None), objectives.default_set().objectives[1])
    eng.sync_callbacks = True
    caps = []
    eng.on_sweep = caps.append
    for i, bgr in enumerate(_half_sweep_frames()):
        eng.process_frame(bgr, 0.04 * i, i + 1)
    eng.flush()
    assert len(caps) == 1
    assert caps[0].completeness < live.MIN_COMPLETENESS and not caps[0].manual
    assert caps[0].usable is False


def test_sweep_capture_usable_is_the_single_completeness_decision():
    def cap(completeness, manual=False):
        return live.SweepCapture([], None, (0.0, 0.0), "10x / 0.25", (4, 4), False, completeness, manual=manual)

    assert live.MIN_COMPLETENESS == 0.5
    assert cap(0.5).usable and cap(1.0).usable
    assert not cap(0.49).usable and not cap(0.0).usable
    assert cap(0.0, manual=True).usable  # the user marked the sweep with Space: their call


def test_batch_skips_incomplete_focus_wiggles(tmp_path):
    w = RecordingWriter(tmp_path / "rec", "sim")
    for i, bgr in enumerate(_half_sweep_frames()):
        w.add(_jpeg(bgr), i + 1, 0.04 * i)
    w.close()
    logs = []
    res = batch.run_batch(tmp_path / "rec", tmp_path / "out", log=logs.append)
    assert res["sweeps"] == []
    assert not (tmp_path / "out" / "sweep-01").exists()
    ignored = [m for m in logs if "ignored incomplete focus movement" in m]
    assert len(ignored) == 1, logs  # the capture reached the filter and the filter said why it dropped it


# ------------------------------------------------------------------------------------------------ B7
def test_v4l2_handles_timeout_and_streaming_defers_apply(monkeypatch):
    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/v4l2-ctl" if name == "v4l2-ctl" else None)

    def boom(*a, **k):
        raise subprocess.TimeoutExpired(a[0], 3)

    monkeypatch.setattr(subprocess, "run", boom)
    ctrl = V4L2Controls("/dev/video0")
    assert ctrl.list() == {} and ctrl.last_error
    assert ctrl.set(exposure_time_absolute=200) == {} and ctrl.last_error

    called = []
    src = CameraSource(CameraInfo(0, "cam", "/dev/video0"))
    src._running = True
    monkeypatch.setattr(src, "_apply_controls", lambda *a: called.append(1) or {})
    src.set_manual_exposure(250)
    assert called == [] and src._apply_pending is True


class _FakeCap:
    """The few cv2.VideoCapture methods the capture thread uses; delivers a small decoded frame every 5 ms."""

    def read(self):
        time.sleep(0.005)
        return True, np.zeros((8, 8, 3), np.uint8)

    def set(self, *a):
        return True

    def get(self, *a):
        return 0.0

    def getBackendName(self):
        return "fake"

    def release(self):
        pass


def _wait_for(cond, timeout=5.0):
    t0 = time.perf_counter()
    while not cond():
        assert time.perf_counter() - t0 < timeout, "timed out"
        time.sleep(0.005)


def test_camera_control_request_never_waits_for_the_device(monkeypatch):
    """The GUI thread asks for a new exposure while the capture thread is in the middle of a slow v4l2-ctl
    call (each control can take up to its 3 s timeout). The request must return at once, and the capture
    thread must go on to apply the newest value and publish the device's answer."""
    src = CameraSource(CameraInfo(0, "cam", "/dev/video0"))
    monkeypatch.setattr(src, "open", lambda: _FakeCap())
    slow, entered, gate, calls = threading.Event(), threading.Event(), threading.Event(), []

    def device(*_a):  # stands in for the v4l2-ctl round trips; reports the exposure it was asked to set
        want = src._want
        calls.append(want[0])
        if slow.is_set():
            entered.set()
            gate.wait(5.0)
        return {"exposure_time_absolute": want[0]}

    monkeypatch.setattr(src, "_apply_controls", device)
    src.set_manual_exposure(200)  # not streaming yet: applied directly
    assert src.applied == {"exposure_time_absolute": 200}
    src.start()
    try:
        _wait_for(lambda: len(calls) >= 2)  # re-applied by the capture thread with its first frame
        slow.set()
        src.set_manual_exposure(300)
        _wait_for(entered.is_set)  # the capture thread is now inside the slow device call
        t0 = time.perf_counter()
        got = src.set_manual_exposure(400)
        waited = time.perf_counter() - t0
        assert waited < 0.25, f"set_manual_exposure blocked for {waited:.2f} s behind the device call"
        assert got == {"exposure_time_absolute": 200}  # documented: the previous report while streaming
        seq_before = src.applied_seq
        slow.clear()
        gate.set()
        _wait_for(lambda: src.applied == {"exposure_time_absolute": 400})  # the newest request wins
        assert src.applied_seq >= seq_before + 2  # one report for 300 (superseded), one for 400
    finally:
        gate.set()
        src.stop()


# ------------------------------------------------------------------------------------------------ B8
def test_find_reversal_ignores_boundary_zero_padding_artifact():
    """A single monotonic pass that starts sharp on another layer (dust, a cover slip) and then passes through
    the surface once. Smoothing with zero padding pulled the first sample down by a third in log energy, which
    turned the decaying start into a second 'focus peak' at frame 1 in every tile; the midpoint between it and
    the real peak was then returned as a turnaround, and half of the sweep was thrown away."""
    n = 30
    k = np.arange(n, dtype=float)[:, None]
    x = np.linspace(-2.5, 2.5, n)[:, None]
    offsets = np.linspace(-0.6, 0.6, 36)[None, :]
    one_pass = 100.0 * np.exp(-(x - offsets) ** 2) + 20.0
    tiles = one_pass + 300.0 * np.exp(-k / 1.5)  # in focus on the other layer at frame 0, gone by frame ~5
    assert sff.find_reversal(tiles) is None

    # positive control: a real forward-and-back sweep (the same pass, then mirrored) still has its turnaround found
    there_and_back = np.vstack([one_pass, one_pass[::-1]])
    t = sff.find_reversal(there_and_back)
    assert t is not None and abs(t - (n - 0.5)) <= 1.5


# ------------------------------------------------------------------------------------------------ GUI tests (A2, A4, B3, B5, B6, C6)
@pytest.mark.characterization("edited only to stop redirecting QSettings by setPath (the settings-isolation fixture does it now)")
def test_gui_fixes_offscreen(tmp_path, monkeypatch):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    pg = pytest.importorskip("pyqtgraph")
    from PySide6.QtCore import QStandardPaths
    from PySide6.QtWidgets import QApplication

    QStandardPaths.setTestModeEnabled(True)
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "cfg"))
    pg.setConfigOptions(imageAxisOrder="row-major")
    app = QApplication.instance() or QApplication([])

    from microtopos.gui.main_window import MainWindow

    w = MainWindow(simulator_only=True)
    try:
        # B5: coverage bottom axis stays hidden at startup
        assert not w.coverage.plot.getAxis("bottom").isVisible()

        # A2: S-filter 0 -> 0.0 (off), and untouched default adapts to result objective in _recompute
        w.sp_ls.setValue(0.0)
        assert w._lam_s(w.obj) == 0.0
        w._on_objective("10x / 0.25")
        assert w._lam_s(objectives.resolve("40x / 0.85")) is None

        # B3: simulator refuses to write or clear disk calibrations
        w.store.get(w.obj).lateral_um_per_px_full = 0.42
        w.store.save()
        w._clear_lateral()
        w._record("dark")
        assert w.store.get(w.obj).lateral_um_per_px_full == 0.42
        assert w.engine.calibration_progress() is None

        # A4: loaded session mosaic keeps its own um_preview and hides live cam_box
        sess = tmp_path / "sess.npz"
        h = np.ones((16, 16), np.float32)
        export.save_session(
            sess,
            {"height": h, "height_raw": h, "edf": np.zeros((16, 16), np.uint8), "edf_proc": h, "sl_surface": h,
             "confidence": h, "um_per_px": 0.25, "z_unit": "um", "objective": "40x / 0.85", "valid_fraction": 1.0,
             "frames_used": 10, "frames_in": 10, "measure": "DoG band-pass"},
            {},
            {"mosaic_image": np.full((32, 32), 128.0, np.float32), "mosaic_origin": [0, 0], "um_preview": 0.123},
        )
        monkeypatch.setattr("PySide6.QtWidgets.QFileDialog.getOpenFileName", lambda *a, **k: (str(sess), ""))
        w._on_objective("4x / 0.10")
        w.tabs.setCurrentIndex(1)
        w._tick()
        # cam_box is a QGraphicsRectItem: isVisible() is its own flag, independent of the window being shown
        assert w.cam_box.isVisible()  # live mosaic: the camera box is part of the view
        w._load_session()
        w.tabs.setCurrentIndex(1)
        w._tick()
        assert w.loaded_mosaic[2] == pytest.approx(0.123)
        assert w.mosaic_um == pytest.approx(0.123)
        assert w.p_mosaic.um_per_px == pytest.approx(0.123)
        assert not w.cam_box.isVisible()

        # B6: stalled camera times out one-shot auto-exposure and re-enables button
        class DummyCam(CameraSource):
            def __init__(self):
                CameraSource.__init__(self, CameraInfo(0, "dummy"))

            def set_manual_exposure(self, exposure, sharpness_off=True):
                return {}

            def exposure_range(self):
                return (50, 1000)

        w.source = DummyCam()
        w.sim_source = None

        # B7: "Device reports" follows what the capture thread applied, not the stale return value of the request
        w.source.readback = {"size": "1920x1080"}
        w._apply_exposure()
        assert "exposure_time_absolute" not in w.lbl_readback.text()
        w.source.applied = {"exposure_time_absolute": 123}  # the capture thread publishes the device's answer
        w.source.applied_seq += 1
        w._tick()
        assert w.lbl_readback.text() == "Device reports: size=1920x1080, exposure_time_absolute=123"

        w._auto_exposure_once()
        assert not w.btn_autoexp_once.isEnabled()
        for _ in range(16):
            w._auto_exposure_step()
        assert w.btn_autoexp_once.isEnabled() and not w._ae_active

        # C6: flat explicit levels don't produce zero-span range
        w.p_height.set_image(np.ones((10, 10), np.float32), 1.0, levels=(3.0, 3.0))
        lo, hi = w.p_height.img.getLevels()
        assert hi > lo

        # The coach's "(enough)" is the engine's decision (live.MIN_COMPLETENESS), not a second copy of the number
        from types import SimpleNamespace

        import microtopos.gui.main_window as mw

        sweeping = SimpleNamespace(state=live.State.Z_SWEEP, sweep_completeness=0.6, sweep_frames=12)
        w._update_guide(sweeping)
        assert "60% of the field has gone sharp and blurry again (enough)" in w.guide.text()
        monkeypatch.setattr(mw, "MIN_COMPLETENESS", 0.8)
        w._update_guide(sweeping)
        assert "(needs 80%)" in w.guide.text()
    finally:
        w.close()
        app.processEvents()
