"""Lens distortion through the live engine and the GUI: estimated from a pan (simulated frames distorted by a known
barrel model), stored per objective, and removed from the frames the mosaic stitches."""
import time

import numpy as np
import pytest
import test_distortion as td

from microtopos.calibration import CalibrationStore, distortion_for
from microtopos.camera import FrameSource
from microtopos.distortion import RadialModel
from microtopos.live import LiveEngine, Override
from microtopos.objectives import resolve

OBJ = resolve("10x / 0.25")
TRUE_CORNER = td.R * (td.BARREL[0] + td.BARREL[1])  # -3 px at the corner (barrel)


def _feed(eng, frames, seq0=1):
    for i, f in enumerate(frames):
        eng.process_frame(np.dstack([np.clip(f + 0.5, 0, 255).astype(np.uint8)] * 3), (seq0 + i) * 0.04, seq0 + i)


def test_a_pan_through_the_engine_gives_the_objectives_distortion(tmp_path):
    frames, _, _ = td.pan(1, td.BARREL)
    eng = LiveEngine(CalibrationStore(tmp_path / "cal"), OBJ)
    eng.sm.override = Override.PAN
    eng.start_distortion_estimate()
    _feed(eng, frames)
    got = eng.take_distortion_estimator()
    assert got is not None, eng.distortion_progress()
    est, shape, name = got
    assert shape == (td.H, td.W) and name == OBJ.name
    r = est.result()
    assert r.corner_displacement_px == pytest.approx(TRUE_CORNER, rel=0.05)


def test_with_a_stored_distortion_the_mosaic_gets_undistorted_frames(tmp_path):
    store = CalibrationStore(tmp_path / "cal")
    model = RadialModel((td.H, td.W), *td.BARREL)
    store.get(OBJ).distortion[f"{td.W}x{td.H}"] = model.to_json()
    assert distortion_for(store.get(OBJ), (td.H, td.W)).k1 == td.BARREL[0]
    eng = LiveEngine(store, OBJ)
    seen = []
    real = eng.mosaic.track
    eng.mosaic.track = lambda img, rgb=None: (seen.append(img.copy()), real(img, rgb=rgb))[1]
    eng.sm.override = Override.PAN
    frames, _, _ = td.pan(2, td.BARREL)
    _feed(eng, frames[:3])
    prev = eng.snapshot().preview  # the corrected preview of the last frame (no dark/flat here: the frame itself)
    np.testing.assert_allclose(seen[-1], model.undistort(prev), atol=1e-3)
    assert float(np.abs(seen[-1] - prev).max()) > 5  # the correction moved the corners by ~3 px


def test_measure_lens_distortion_button_stores_it(tmp_path):
    import os

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    import pyqtgraph as pg
    from PySide6.QtCore import QStandardPaths
    from PySide6.QtWidgets import QApplication

    from microtopos.gui.main_window import MainWindow

    frames, _, _ = td.pan(3, td.BARREL)

    class _Cam(FrameSource):
        def _run(self):
            i = 0
            while self._running:
                f = np.clip(frames[i % len(frames)] + 0.5, 0, 255).astype(np.uint8)
                self._publish(np.dstack([f] * 3))
                i += 1
                time.sleep(0.04)

    QStandardPaths.setTestModeEnabled(True)
    app = QApplication.instance() or QApplication([])
    pg.setConfigOptions(imageAxisOrder="row-major")
    w = MainWindow(simulator_only=True)
    notes = []
    w.notify = lambda m, k="info", *a, **kw: notes.append((k, m))
    cam = _Cam()
    try:
        w.engine.set_source(None)
        w.source.stop()
        w.sim_source, w.source = None, cam
        cam.start()
        w.engine.set_source(cam)
        cal = w.store.get(w.obj)
        cal.distortion.pop(f"{td.W}x{td.H}", None)  # Qt's test-mode data folder is shared between tests
        w.engine.sm.override = Override.PAN
        w.btn_measure_dist.click()
        t0 = time.time()
        while not any("distortion" in m and "px" in m for _, m in notes) and time.time() - t0 < 180:
            app.processEvents()
            time.sleep(0.02)
        assert any("barrel distortion" in m for _, m in notes), notes[-3:]
        m = distortion_for(cal, (td.H, td.W))
        assert m is not None and m.corner_displacement_px == pytest.approx(TRUE_CORNER, rel=0.05)
    finally:
        cam.stop()
        w.close()
        app.processEvents()
