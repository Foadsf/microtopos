"""Measured lateral chromatic aberration, from the Objective page to every colour output (never the luma)."""
import os
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pyqtgraph as pg
import pytest
import test_chromatic as tc
from PySide6.QtCore import QStandardPaths
from PySide6.QtWidgets import QApplication

from microtopos import stills
from microtopos.calibration import lateral_ca_for
from microtopos.camera import FrameSource
from microtopos.chromatic import LateralCA
from microtopos.gui.main_window import MainWindow
from microtopos.live import DwellStill
from microtopos.objectives import resolve


@pytest.fixture(scope="module")
def planes():
    return tc.specimen(3)


def _ca():
    return LateralCA((tc.W, tc.H), tc.TRUE_R[0], tc.TRUE_R[1], tc.TRUE_B[0], tc.TRUE_B[1], significant=True)


def test_still_colour_is_corrected_and_luma_is_not(planes):
    frames = [tc.render(planes, tc.TRUE_R, tc.TRUE_B, seed=k) for k in range(8)]
    truth = tc.render(planes, noise=0, quantise=False)
    luma = [np.full((tc.H, tc.W), 100, np.uint8)] * 8
    base = DwellStill(luma, None, frames, None, False, None, None, resolve("10x / 0.25"), color_mode=True)
    plain = stills.prepare(base)
    fixed = stills.prepare(DwellStill(luma, None, frames, None, False, None, None, resolve("10x / 0.25"),
                                      color_mode=True, lateral_ca=_ca()))
    before, after = tc.fringe_rms(plain["colour"][0], truth), tc.fringe_rms(fixed["colour"][0], truth)
    assert max(after) < max(before) / 4, (before, after)
    assert any("colour fringes corrected" in n for n in fixed["notes"])
    assert all(np.array_equal(a, b) for a, b in zip(fixed["frames"], plain["frames"]))


class _ColourCamera(FrameSource):
    def __init__(self, planes):
        super().__init__()
        self.planes = planes
        self.k = 0

    def _run(self):
        while self._running:
            self.k += 1
            self._publish(tc.render(self.planes, tc.TRUE_R, tc.TRUE_B, seed=self.k % 5))
            time.sleep(0.05)


def test_measure_colour_fringes_button_stores_them_and_the_colour_preview_loses_them(planes, tmp_path):
    QStandardPaths.setTestModeEnabled(True)
    app = QApplication.instance() or QApplication([])
    pg.setConfigOptions(imageAxisOrder="row-major")
    os.environ["XDG_DATA_HOME"] = str(tmp_path / "data")
    w = MainWindow(simulator_only=True)
    notes = []
    w.notify = lambda m, k="info", *a, **kw: notes.append((k, m))
    cam = _ColourCamera(planes)
    try:
        w.engine.set_source(None)
        w.source.stop()
        w.sim_source, w.source = None, cam
        cam.start()
        w.engine.set_source(cam)
        w.chk_rgb.setChecked(True)
        cal = w.store.get(w.obj)
        cal.lateral_ca.pop(f"{tc.W}x{tc.H}", None)  # Qt's test-mode data folder is shared between tests
        t0 = time.time()
        while cam.latest() is None and time.time() - t0 < 10:
            app.processEvents()
            time.sleep(0.02)
        w.btn_measure_ca.click()
        t0 = time.time()
        while not any("Colour fringes" in m for _, m in notes) and time.time() - t0 < 120:
            app.processEvents()
            time.sleep(0.02)
        assert notes and "measured" in notes[-1][1], notes[-2:]
        ca = lateral_ca_for(cal, (tc.H, tc.W))
        assert ca is not None and ca.scale_r == pytest.approx(tc.TRUE_R[0], abs=3e-4)
        assert ca.scale_b == pytest.approx(tc.TRUE_B[0], abs=3e-4)
        seq = w.engine.snapshot().seq
        t0 = time.time()
        while w.engine.snapshot().seq < seq + 3 and time.time() - t0 < 10:
            app.processEvents()
            time.sleep(0.02)
        shown = w.engine.snapshot().preview_rgb[..., ::-1].astype(np.float32)  # RGB -> BGR
        truth = tc.render(planes, noise=0, quantise=False)
        raw = tc.render(planes, tc.TRUE_R, tc.TRUE_B, noise=0, quantise=False)
        assert max(tc.fringe_rms(shown, truth)) < max(tc.fringe_rms(raw, truth)) / 3
    finally:
        cam.stop()
        w.close()
        app.processEvents()


def test_sweep_colour_frames_are_corrected_before_the_reconstruction(planes, monkeypatch):
    from microtopos.gui import main_window

    seen = {}
    monkeypatch.setattr(main_window.zstack, "reconstruct",
                        lambda *a, color_frames=None, **kw: seen.setdefault("c", color_frames))
    frames = [tc.render(planes, tc.TRUE_R, tc.TRUE_B, seed=k) for k in range(2)]
    main_window._reconstruct_job([], None, color_frames=frames, lateral_ca=_ca())
    truth = tc.render(planes, noise=0, quantise=False)
    assert max(tc.fringe_rms(seen["c"][0], truth)) < max(tc.fringe_rms(frames[0], truth)) / 4
    seen.clear()
    main_window._reconstruct_job([], None, color_frames=frames, lateral_ca=None)
    assert seen["c"] is frames  # nothing measured: the colour frames go through untouched
