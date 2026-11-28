"""Edge sharpness measured with the slanted-edge method, stored per objective, and used to sharpen still images."""
import os
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pyqtgraph as pg
import pytest
import test_edgemtf as te
from PySide6.QtCore import QStandardPaths
from PySide6.QtWidgets import QApplication

from microtopos.calibration import ObjectiveCalibration, psf_for
from microtopos.camera import FrameSource
from microtopos.gui.main_window import MainWindow
from microtopos.objectives import resolve
from microtopos.superres import fit_edge_psf_sigma

SIGMA, TILT = 1.2, 6.0


@pytest.mark.parametrize("sigma", [0.8, 1.5])
def test_the_deconvolution_psf_from_an_edge_has_the_pixel_aperture_removed(sigma):
    """The earlier erf fit read 0.945 px for sigma 0.8 (+18 %): it kept the pixel aperture and interpolation blur."""
    img = te.render_edge(160, 160, sigma, TILT)
    got, src = fit_edge_psf_sigma(img, 9.9)
    assert got == pytest.approx(sigma, rel=0.02) and src.startswith("slanted edge")
    flat = np.full((160, 160), 100.0)
    assert fit_edge_psf_sigma(flat, 9.9)[0] == 9.9  # no edge: the nominal value, said so
    assert fit_edge_psf_sigma(flat, 9.9)[1].startswith("nominal")


def test_a_stored_edge_measurement_replaces_the_theoretical_psf():
    obj = resolve("10x / 0.25")
    cal = ObjectiveCalibration(obj.name)
    assert psf_for(cal, obj, (240, 320), 0.9) == (obj.psf_sigma_px(0.9), "nominal")
    cal.measured_psf["320x240"] = {"psf_sigma_px": 1.37, "mtf50_cyc_per_px": 0.21, "date": "2026-11-28"}
    sigma, src = psf_for(cal, obj, (240, 320), 0.9)
    assert sigma == 1.37 and src.startswith("measured on an edge")
    assert psf_for(cal, obj, (480, 640), 0.45)[1] == "nominal"  # another capture size is not covered


class _EdgeCamera(FrameSource):
    def __init__(self):
        super().__init__()
        self.clean = te.render_edge(240, 320, SIGMA, TILT, dark=40, bright=200)
        self.rng = np.random.default_rng(0)

    def _run(self):
        while self._running:
            f = np.clip(self.clean + self.rng.normal(0, 1.0, self.clean.shape) + 0.5, 0, 255).astype(np.uint8)
            self._publish(np.dstack([f] * 3))
            time.sleep(0.03)


def test_measure_sharpness_button_stores_the_blur_and_the_stack_sharpens_with_it(tmp_path):
    QStandardPaths.setTestModeEnabled(True)
    app = QApplication.instance() or QApplication([])
    pg.setConfigOptions(imageAxisOrder="row-major")
    w = MainWindow(simulator_only=True)
    notes = []
    w.notify = lambda m, k="info", *a, **kw: notes.append((k, m))
    cam = _EdgeCamera()

    def wait(cond, t=60):
        t0 = time.time()
        while time.time() - t0 < t and not cond():
            app.processEvents()
            time.sleep(0.02)
        return cond()
    try:
        w.engine.set_source(None)
        w.source.stop()
        w.sim_source, w.source = None, cam
        cam.start()
        w.engine.set_source(cam)
        cal = w.store.get(w.obj)
        cal.measured_psf.pop("320x240", None)  # Qt's test-mode data folder is shared between tests
        assert wait(lambda: len(w.engine.dwell_frames()) >= 8)
        w.btn_edge.click()
        assert wait(lambda: any("MTF50" in m for _, m in notes)), notes[-2:]
        m = cal.measured_psf["320x240"]
        assert m["psf_sigma_px"] == pytest.approx(SIGMA, rel=0.03)
        assert m["mtf50_cyc_per_px"] == pytest.approx(te.mtf50_true(SIGMA, TILT), rel=0.03)
        w.sr_deconvolve = True
        before = w.sr_job
        w.btn_stack.click()
        assert w.sr_job is not before and wait(lambda: not w.sr_job.isRunning() and w.sr_result is not None)
        assert w.sr_result["psf_sigma_px"] == m["psf_sigma_px"]
        assert w.sr_result["psf_source"].startswith("measured on an edge")
    finally:
        cam.stop()
        w.close()
        app.processEvents()


def test_super_resolution_uses_a_measured_psf_as_given():
    from test_superres_sampling import _frames

    from microtopos.superres import dwell_superres

    frames, _ = _frames(0.6)
    r = dwell_superres(frames, 1.37, scale=2, psf_source="measured on an edge (test)")
    assert r["psf_sigma_lr"] == 1.37 and r["psf_source"] == "measured on an edge (test)"
