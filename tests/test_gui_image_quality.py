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
