"""Image-quality controls, each through the control a user operates (offscreen, simulator, isolated settings)."""

import os
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pyqtgraph as pg
import pytest
from PySide6.QtCore import QPoint, QStandardPaths
from PySide6.QtWidgets import QApplication

from microtopos.calibration import white_balance_gains
from microtopos.camera import CameraInfo, CameraSource, FrameSource
from microtopos.gui.main_window import MainWindow


@pytest.fixture(scope="module")
def qapp():
    QStandardPaths.setTestModeEnabled(True)
    app = QApplication.instance() or QApplication([])
    pg.setConfigOptions(imageAxisOrder="row-major")
    yield app


def _wait_for(app, cond, timeout=30.0) -> bool:
    t0 = time.time()
    while time.time() - t0 < timeout:
        app.processEvents()
        time.sleep(0.02)
        if cond():
            return True
    return False


def _open(qapp, tmp_path):
    os.environ["XDG_DATA_HOME"] = str(tmp_path / "data")
    os.environ["XDG_CONFIG_HOME"] = str(tmp_path / "config")
    w = MainWindow(simulator_only=True)
    w.show()
    for _ in range(20):
        qapp.processEvents()
    notes = []
    w.notify = lambda msg, kind="info", *a, **k: notes.append((kind, msg))
    return w, notes


def _close(qapp, w):
    w.close()
    qapp.processEvents()


def test_white_balance_button_sets_persists_and_resets(qapp, tmp_path):
    w, notes = _open(qapp, tmp_path)
    try:
        assert w.engine.wb_gains is None
        assert _wait_for(qapp, lambda: w.source is not None and w.source.latest() is not None)
        w.btn_wb.click()
        frame = w.source.latest()[0]
        g = w.engine.wb_gains
        assert g is not None and notes[-1][0] == "success"
        # the simulator's specimen is warm (B < G < R): the gains lift blue and lower red
        assert g[0] > 1.02 and g[1] == 1.0 and g[2] < 0.98
        assert np.array(g) == pytest.approx(white_balance_gains(frame[::2, ::2]), rel=0.03)
    finally:
        _close(qapp, w)
    w2, _ = _open(qapp, tmp_path)  # same settings directory: the balance is remembered
    try:
        assert w2.engine.wb_gains == pytest.approx(g)
        w2.btn_wb.customContextMenuRequested.emit(QPoint(1, 1))  # right-click: reset
        assert w2.engine.wb_gains is None and w2.settings.value("white_balance", None) is None
    finally:
        _close(qapp, w2)


def test_super_resolution_is_refused_where_the_camera_oversamples_the_optics(qapp, tmp_path):
    """The simulator's 640 px frames through 100x / 1.25 oil: 0.09 um/px against lambda/4NA = 0.11 um, 1.2x
    oversampled, so 2x super-resolution is refused before any job starts; at 10x / 0.25 (0.6x) it runs."""
    w, notes = _open(qapp, tmp_path)
    try:
        w.cmb_obj.setCurrentText("100x / 1.25 oil")
        assert w.obj.name == "100x / 1.25 oil"
        assert _wait_for(qapp, lambda: len(w.engine.dwell_frames()) >= 8, 60)
        w.sr_scale = 2
        before = w.sr_job
        w.btn_sr.click()
        assert w.sr_job is before
        assert notes and notes[-1][0] == "warning" and "adds no detail" in notes[-1][1], notes[-1:]
        w.cmb_obj.setCurrentText("10x / 0.25")
        assert _wait_for(qapp, lambda: len(w.engine.dwell_frames()) >= 8, 60)
        w.btn_sr.click()
        assert w.sr_job is not before  # undersampled: the job starts
        assert _wait_for(qapp, lambda: not w.sr_job.isRunning(), 60)
    finally:
        _close(qapp, w)


def test_stack_button_merges_the_still_frames_and_lowers_the_noise(qapp, tmp_path):
    """The simulator holds still after its demo: 'Stack still frames' merges the dwell buffer; the per-pixel noise of
    the stack falls by about sqrt(frames) (at least 2x for 16 frames) and the result is shown as a stack."""
    w, notes = _open(qapp, tmp_path)
    try:
        assert _wait_for(qapp, lambda: w.sim_source is not None and not w.sim_source.busy(), 60)
        # no hand tremor: the frames then differ by sensor noise only, so the stack's scatter is that noise (with
        # tremor on these undersampled frames, resampling error adds to it; tests/test_stacking.py covers that case)
        w.sim_source.sim.tremor_px = 0.0
        seq0 = w.engine.snapshot().seq
        assert _wait_for(qapp, lambda: w.engine.snapshot().seq > seq0 + 20 and len(w.engine.dwell_frames()) >= 16, 60)
        w.sr_deconvolve = False
        before = w.sr_job
        w.btn_stack.click()
        assert w.sr_job is not before, notes[-1:]
        assert _wait_for(qapp, lambda: not w.sr_job.isRunning() and w.sr_result is not None, 60)
        r = w.sr_result
        assert r["kind"] == "stack" and r["scale"] == 1 and r["frames_used"] >= 8
        assert r["noise_sigma_stack"] < r["noise_sigma_single"] / 2
        assert r["image"].shape == w.engine.dwell_frames()[0].shape  # the camera's own grid, full frame
        assert w.cmb_edf.currentIndex() == 1 and w.lbl_edf.text().startswith("Stack of")
    finally:
        _close(qapp, w)


def test_export_still_writes_the_shown_stack_with_its_scale_and_provenance(qapp, tmp_path, monkeypatch):
    import json

    import tifffile

    w, notes = _open(qapp, tmp_path)
    try:
        img = np.random.default_rng(1).uniform(0, 255, (60, 80)).astype(np.float32)
        col = np.dstack([img * 0.9, img, np.minimum(img * 1.1, 255)]).astype(np.float32)
        w._on_sr({"kind": "stack", "image": img, "colour": col, "um_per_px": 0.734, "scale": 1, "frames_used": 16,
                  "frames_rejected": 0, "noise_sigma_single": 3.0, "noise_sigma_stack": 0.75, "rl_iterations": 0,
                  "pixels_clipped_fraction": 0.01, "notes": ["test"]})
        out = tmp_path / "still.tif"
        monkeypatch.setattr(w, "_save_path", lambda *a: str(out))
        w.btn_export_still.click()
        assert notes[-1][0] == "success", notes[-1]
        np.testing.assert_array_equal(tifffile.imread(out), img)
        with tifffile.TiffFile(out) as t:
            xres = t.pages[0].tags["XResolution"].value
        assert 1e4 / (xres[0] / xres[1]) == pytest.approx(0.734, rel=1e-4)  # px per cm -> um per px
        rgb = tifffile.imread(tmp_path / "still.rgb.tif")
        assert rgb.dtype == np.uint16 and np.allclose(rgb[..., 1] / 257.0, img, atol=0.51)
        assert np.allclose(rgb[..., 0] / 257.0, col[..., 2], atol=0.51)  # BGR -> RGB
        meta = json.loads((tmp_path / "still.json").read_text())
        assert meta["kind"] == "stack" and meta["um_per_px"] == 0.734 and meta["values"] == "camera DN"
    finally:
        _close(qapp, w)


def test_flat_estimate_button_runs_a_pan_and_reports_the_simulators_vignette(qapp, tmp_path):
    """'Estimate flat while panning' + the demo pan (a 2-D loop over ~3 fields): the estimate completes, is not
    written (the simulator never writes calibrations), and its gain is the simulator's own vignette within 2 %."""
    w, notes = _open(qapp, tmp_path)
    try:
        assert _wait_for(qapp, lambda: w.sim_source is not None and not w.sim_source.busy(), 60)
        w.btn_flat_est.click()
        assert w.btn_flat_est.text() == "Cancel flat estimate"
        w._demo("pan")
        assert _wait_for(qapp, lambda: w.engine.flat_status is not None, 120), w.engine.flat_estimate_progress()
        for _ in range(5):
            w._tick()
        st = w.engine.flat_status
        assert st.startswith("Flat estimated, not stored: the simulator never writes to disk"), st
        assert any("not stored" in m for _, m in notes) and w.btn_flat_est.text() == "Estimate flat while panning"
        gain = w.engine.last_flat_estimate["gain"]
        v = w.sim_source.sim.vignette
        r = gain / (v / v.mean())
        assert float(np.sqrt(np.mean((r / r.mean() - 1) ** 2))) < 0.02
    finally:
        _close(qapp, w)


# ---------------------------------------------------------------- camera response and HDR through the GUI
GAMMA = 1.8  # the test camera's tone curve: DN = 255 * (relative exposure) ** (1 / GAMMA)


def _scene(h=120, w=160):
    """Relative radiance with a 100:1 range (a horizontal ramp) and texture; channels differ a little."""
    y, x = np.mgrid[0:h, 0:w].astype(np.float64)
    base = 0.01 * 100.0 ** (x / (w - 1)) * (1.0 + 0.15 * np.sin(y / 3.0) * np.cos(x / 5.0))
    return np.dstack([0.8 * base, base, 1.1 * base])


class _TestCamera(CameraSource):
    """A camera whose exposure setting gives the exposure it states (truthful=True), or one that, like the
    MikrOkular (measured 2026-11-28), quantises it into coarse steps (156, 190 and 220 give the same brightness)."""

    def __init__(self, truthful=True):
        # a new name each time: GUI tests share Qt's test-mode data folder, where responses are stored per camera
        super().__init__(CameraInfo(99, f"Test camera {time.time_ns()}", ""))
        self.truthful = truthful
        self._e = 300
        self.scene = _scene()
        self.rng = np.random.default_rng(0)

    def exposure_range(self):
        return (78, 738)

    def set_manual_exposure(self, exposure, sharpness_off=True):
        self._e = 300 if exposure is None else int(exposure)
        self.applied = {"exposure_time_absolute": self._e}
        return self.applied

    def _effective(self):
        return self._e if self.truthful else 78 * 2 ** round(np.log2(self._e / 78.0))

    def _run(self):
        while self._running:
            x = self.scene * self._effective() / 400.0
            dn = 255.0 * np.clip(x, 0, None) ** (1 / GAMMA) + self.rng.normal(0, 0.5, x.shape)
            self._publish(np.clip(dn + 0.5, 0, 255).astype(np.uint8))
            time.sleep(0.02)


def _use(w, cam):
    w.engine.set_source(None)
    w.source.stop()
    w.sim_source = None
    w.source = cam
    cam.start()
    w.engine.set_source(cam)


def test_measure_response_then_hdr_recovers_the_scene_radiance(qapp, tmp_path):
    w, notes = _open(qapp, tmp_path)
    cam = _TestCamera(truthful=True)
    try:
        _use(w, cam)
        assert _wait_for(qapp, lambda: cam.latest() is not None)
        w.btn_measure_resp.click()
        # as long as the bracket's own timeout (main_window.BRACKET_TIMEOUT_S) plus the solve: a slow runner (Windows
        # CI) did not finish within 120 s; every message is shown when it fails
        assert _wait_for(qapp, lambda: any("Camera response measured" in m for _, m in notes), 240), \
            " | ".join(f"{k}: {m}" for k, m in notes) + f" | job line: {w.lbl_job.text()}"
        curve = w.store.get_response(cam.camera_name, (120, 160))
        assert curve is not None and curve.gamma_equivalent() == pytest.approx(GAMMA, abs=0.1)
        w.engine.linearize = True
        assert w.engine.linearization((120, 160)).startswith("measured camera response")
        # linear luminance through the measured curve is proportional to the scene's (gamma 2.2 is not)
        frame = cam.latest()[0]
        from microtopos.calibration import bgr_to_luma
        lum = bgr_to_luma(frame, True, curve)
        truth = _scene() @ np.array([0.0722, 0.7152, 0.2126])
        ok = (frame.min(axis=2) > 20) & (frame.max(axis=2) < 240)
        r = lum[ok] / truth[ok]
        assert float(np.std(r) / np.mean(r)) < 0.04
        r22 = bgr_to_luma(frame, True)[ok] / truth[ok]
        assert float(np.std(r22) / np.mean(r22)) > 2 * float(np.std(r) / np.mean(r))
        before = w.sr_count if hasattr(w, "sr_count") else 0
        w.btn_hdr.click()
        assert _wait_for(qapp, lambda: getattr(w, "sr_count", 0) > before, 120), notes[-3:]
        res = w.sr_result
        assert res["kind"] == "hdr"
        good = np.isfinite(res["radiance"])
        q = res["radiance"][good] / truth[good]
        assert good.mean() > 0.95 and float(np.std(q) / np.mean(q)) < 0.04  # one ratio over the 100:1 range
        assert cam._e == w.sl_exp.value() or w.chk_autoexp.isChecked()  # the user's exposure is back
    finally:
        cam.stop()
        _close(qapp, w)


def test_a_camera_whose_exposure_steps_are_coarse_is_refused_a_response(qapp, tmp_path):
    w, notes = _open(qapp, tmp_path)
    cam = _TestCamera(truthful=False)
    try:
        _use(w, cam)
        assert _wait_for(qapp, lambda: cam.latest() is not None)
        w.btn_measure_resp.click()
        assert _wait_for(qapp, lambda: any(m.startswith("Camera response:") for _, m in notes), 120), notes[-3:]
        msg = next(m for _, m in notes if m.startswith("Camera response:"))
        assert "gave the same brightness" in msg and "does not apply the exposure time it reports" in msg, msg
        assert w.store.get_response(cam.camera_name, (120, 160)) is None
        w.btn_hdr.click()  # and HDR says why it cannot run
        assert "measured response" in notes[-1][1]
    finally:
        cam.stop()
        _close(qapp, w)


def test_measure_rolling_shutter_button_stores_the_readout(qapp, tmp_path):
    import test_readout_live as trl
    import test_rollingshutter as trs

    class _RSCamera(FrameSource):
        def __init__(self):
            super().__init__()
            self.camera_name = f"RS camera {time.time_ns()}"
            self.script = trl._script(trs.RollingShutterCamera(5, trl.RHO), n_cycles=12)

        def _run(self):
            i = 0
            while self._running:
                f = self.script[i % len(self.script)][0]
                self._publish(np.dstack([f] * 3))
                i += 1
                time.sleep(0.03)

    w, notes = _open(qapp, tmp_path)
    cam = _RSCamera()
    try:
        _use(w, cam)
        w.btn_measure_ro.click()
        assert _wait_for(qapp, lambda: any(m.startswith("Rolling shutter") for _, m in notes), 180), notes[-3:]
        v = w.store.get_camera_value("readout", cam.camera_name, (trs.H, trs.W))
        assert v is not None and v["rho"] == pytest.approx(trl.RHO, rel=0.05), (notes[-1], v)
        assert w.engine.readout_for((trs.H, trs.W)) == pytest.approx(v["rho"])
    finally:
        cam.stop()
        _close(qapp, w)
