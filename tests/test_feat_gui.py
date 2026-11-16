"""Slice 5 GUI integration: ImagePanel clicking/colour, MainWindow capability wiring, interactive height
tools, session save/load round-trip with the mosaic graph, and negative controls.

Runs offscreen with isolated Qt settings, like tests/test_gui_layout.py.
"""

import os
import time
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

pg = pytest.importorskip("pyqtgraph")
from PySide6.QtCore import QPointF, QSettings, QStandardPaths, Qt
from PySide6.QtGui import QMouseEvent
from PySide6.QtWidgets import QApplication, QFileDialog

from microtopos import export
from microtopos.gui.main_window import MainWindow
from microtopos.gui.theme import DARK
from microtopos.gui.widgets import ImagePanel
from microtopos.mosaic import Mosaic


@pytest.fixture(scope="module")
def qapp(tmp_path_factory):
    QStandardPaths.setTestModeEnabled(True)
    QSettings.setPath(QSettings.NativeFormat, QSettings.UserScope, str(tmp_path_factory.mktemp("qsettings")))
    app = QApplication.instance() or QApplication([])
    pg.setConfigOptions(imageAxisOrder="row-major")
    yield app


@pytest.fixture
def window(qapp, tmp_path):
    """A fresh MainWindow per test, aimed at a temporary data folder (never the user's real calibration)."""
    os.environ["XDG_DATA_HOME"] = str(tmp_path / "data")
    os.environ["XDG_CONFIG_HOME"] = str(tmp_path / "config")
    w = MainWindow(simulator_only=True)
    w.show()
    for _ in range(20):
        qapp.processEvents()
    w.timer.stop()  # the 15 Hz refresh drives the chips; tests drive the window themselves
    for _ in range(5):
        qapp.processEvents()
    yield w
    w.close()
    qapp.processEvents()


def _pump(app, n=25):
    for _ in range(n):
        app.processEvents()


def _wait_result(app, w, timeout=30.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        app.processEvents()
        time.sleep(0.02)
        if w.result is not None:
            return True
    return False


def _mouse_press(scene_x, scene_y, button=Qt.LeftButton):
    # PySide6/Qt6 keeps the six-argument form; fall back to the five-argument one (no globalPos).
    try:
        return QMouseEvent(QMouseEvent.Type.MouseButtonPress, QPointF(scene_x, scene_y), QPointF(scene_x, scene_y),
                           button, button, Qt.KeyboardModifier.NoModifier)
    except TypeError:
        return QMouseEvent(QMouseEvent.Type.MouseButtonPress, QPointF(scene_x, scene_y), button,
                           Qt.MouseButtons(button), Qt.KeyboardModifier.NoModifier)


# --------------------------------------------------------------------------- widgets.py

def test_image_panel_clicked_px_emits_unrotated_pixel_coords(qapp):
    panel = ImagePanel(DARK, "empty")
    img = np.arange(100, dtype=np.float32).reshape(10, 10)
    panel.set_image(img, 1.0)
    hits: list[tuple[float, float]] = []
    panel.clicked_px.connect(lambda x, y: hits.append((float(x), float(y))))

    panel._clicked(_mouse_press(150.0, 150.0))  # inside the plot's scene rect, maps into the image
    assert len(hits) == 1
    x, y = hits[0]
    assert 0.0 <= x < 10.0 and 0.0 <= y < 10.0  # unrotated image pixels

    # right-click: nothing emitted
    panel._clicked(_mouse_press(150.0, 150.0, button=Qt.RightButton))
    assert len(hits) == 1

    # click outside the image: nothing emitted
    panel._clicked(_mouse_press(5000.0, 5000.0))
    assert len(hits) == 1

    # no data: nothing emitted
    panel.set_image(None, 1.0)
    panel._clicked(_mouse_press(150.0, 150.0))
    assert len(hits) == 1


def test_image_panel_set_image_rgb_and_zero_span_levels(qapp):
    panel = ImagePanel(DARK, "empty")
    rgb = np.zeros((8, 6, 3), np.uint8)
    rgb[2:6, 2:4] = 255
    panel.set_image(rgb, 1.5, rgb=True)  # colour frame: must render, not crash or blank
    assert panel.data.shape[:2] == (8, 6)

    gray = np.full((6, 8), 5.0, np.float32)
    panel.set_image(gray, 1.0, levels=(5.0, 5.0))  # zero span: clamps, renders, never crashes
    lo, hi = panel.img.getLevels()
    assert hi > lo


# ---------------------------------------------------------------- main window capabilities

def test_main_window_exposes_all_slice5_controls(window):
    w = window
    assert callable(w._replay_bar)
    assert not w.replay_widget.isVisible()  # only shown for a ReplaySource
    for name in ("mask_roi", "step_roi_a", "step_roi_b", "prof_plot", "prof_curve", "prof_label"):
        assert getattr(w, name) is not None, name
    for name in ("cmb_mblend", "chk_rot_track", "chk_mosaic_rgb", "btn_mosaic_inset", "chk_keep_insets"):
        assert getattr(w, name) is not None, name
    for name in ("cmb_stg", "cmb_stg_port", "btn_stg_connect", "btn_stg_zero", "btn_stg_zsweep", "lbl_stg_pose"):
        assert getattr(w, name) is not None, name
    for name in ("sl_sr_pixfrac", "chk_sr_deconv", "cmb_sr_scale", "sr_scale", "sr_pixfrac", "sr_deconvolve"):
        assert getattr(w, name) is not None, name
    for name in ("chk_3d_breathing", "chk_3d_tiling", "cmb_3d_backend", "zst"):
        assert getattr(w, name) is not None, name
    # tab order is frozen (Live, Mosaic, All-in-focus, Height, 3D, Report)
    titles = [w.tabs.tabText(i) for i in range(w.tabs.count())]
    assert titles == ["Live", "Mosaic", "All-in-focus", "Height", "3D", "Report"]
    assert w.tabs.count() == 6


# ---------------------------------------------------------------- interactive height tools

def test_height_tools_levelling_mask_and_step(window, qapp):
    w = window
    assert w.sim_source is not None
    w._demo("sweep")
    assert _wait_result(qapp, w), "the sweep never reconstructed"
    assert w.result is not None and "height" in w.result

    w.tabs.setCurrentIndex(3)  # Height
    _pump(qapp)
    base = w.result["sl_surface"]
    assert np.isfinite(base).any(), "the sweep's surface map is empty"
    h, wdt = base.shape
    um = w.result.get("um_per_px", 1.0)

    # 3-point levelling replaces the height map with a levelled one
    w.btn_level3.setChecked(True)
    assert w.level_picking
    for x, y in ((0.25 * wdt, 0.25 * h), (0.75 * wdt, 0.25 * h), (0.5 * wdt, 0.75 * h)):
        w.p_height.clicked_px.emit(float(x), float(y))
        _pump(qapp, 3)
    assert not w.level_picking
    h_after = np.array(w.result["height"], copy=True)
    assert h_after.shape == base.shape
    assert np.isfinite(h_after).any()
    assert not np.array_equal(h_after, base), "3-point levelling did not change the height map"

    # step-height ROIs: a step-height line is written to the height label (on the raw surface,
    # before the mask ROI NaNs out the centre)
    sa = (max(1, int(0.09 * wdt * um)), max(1, int(0.09 * h * um)))
    sb = (max(1, int(0.09 * wdt * um)), max(1, int(0.09 * h * um)))
    w.step_roi_a.setPos((0.375 * wdt * um, 0.39 * h * um)); w.step_roi_a.setSize(sa)
    w.step_roi_b.setPos((0.565 * wdt * um, 0.39 * h * um)); w.step_roi_b.setSize(sb)
    w.btn_step.click()
    _pump(qapp, 3)
    assert "Step" in w.lbl_height.text(), w.lbl_height.text()

    # exclusion mask ROI: pixels inside the mask become NaN (applied to the "height" view)
    q0 = (int(0.25 * wdt * um), int(0.25 * h * um))
    q1 = (int(0.75 * wdt * um), int(0.75 * h * um))
    w.mask_roi.setPos(q0)
    w.mask_roi.setSize((q1[0] - q0[0], q1[1] - q0[1]))
    assert np.isfinite(w.result["height"]).any()  # levelling may NaN-edge, but data remains
    w.cmb_height.setCurrentIndex(1)
    _pump(qapp)
    w.btn_mask.click()
    _pump(qapp, 3)
    assert np.isnan(w.result["height"]).any()
    mpos = w.mask_roi.pos(); msize = w.mask_roi.size()
    x0 = round(mpos.x() / um); y0 = round(mpos.y() / um)
    x1 = round((mpos.x() + msize.x()) / um); y1 = round((mpos.y() + msize.y()) / um)
    inner = w.result["height"][y0:y1, x0:x1]
    assert np.isnan(inner).all(), "mask ROI did not exclude the pixels inside its rect"

    # ISO 21920 profile label follows the confidence view
    w.cmb_height.setCurrentIndex(2)
    _pump(qapp)
    assert "Confidence" in w.prof_label.text(), w.prof_label.text()


# ---------------------------------------------------------------- session save/load

def test_mosaic_session_save_and_load_roundtrip(window, qapp, tmp_path):
    w = window
    assert w.sim_source is not None
    w._demo("sweep")
    assert _wait_result(qapp, w)
    w._demo("pan")  # stage motion makes tracking add mosaic keyframes
    t0 = time.time()
    while len(w.engine.mosaic.keyframes) < 1 and time.time() - t0 < 8.0:
        qapp.processEvents()
        time.sleep(0.02)
    assert len(w.engine.mosaic.keyframes) >= 1, "the simulator pan produced no mosaic keyframes"
    kf_before = len(w.engine.mosaic.keyframes)

    path = tmp_path / "session.npz"
    export.save_session(path, w.result, {}, {"mosaic": w.engine.mosaic})
    r2, _s2 = export.load_session(path)
    assert isinstance(r2["mosaic"], Mosaic)
    assert len(r2["mosaic"].keyframes) == kf_before
    assert r2["session_schema"] == 2

    # the same file opened through the MainWindow (dialog patched to avoid a real file dialog)
    with mock.patch.object(QFileDialog, "getOpenFileName",
                           return_value=(str(path), "Session (*.npz)")):
        w._load_session()
    _pump(qapp, 25)
    assert len(w.engine.mosaic.keyframes) == kf_before


# ---------------------------------------------------------------- negative controls

def test_bad_session_files_fail_gracefully(tmp_path):
    missing = tmp_path / "does-not-exist.npz"
    with pytest.raises((FileNotFoundError, OSError, ValueError)):
        export.load_session(missing)

    corrupt = tmp_path / "corrupt.npz"
    corrupt.write_bytes(b"not a session at all")
    with pytest.raises(ValueError):
        export.load_session(corrupt)


def test_simulator_calibration_is_rejected(window):
    """With the simulator active, disk calibration is refused and nothing is written to the store."""
    w = window
    assert w.sim_source is not None
    assert w._reject_simulator_cal() is True
    before = w.store.get(w.obj)
    w._record("dark")
    w._record("flat")
    after = w.store.get(w.obj)
    # the simulator refuses disk calibration: nothing is recorded or stored
    assert before.z_um_per_frame is None and after.z_um_per_frame is None
    assert before.darks == {} and before.flats == {} and after.darks == {} and after.flats == {}
    assert before.lateral_um_per_px_full == after.lateral_um_per_px_full  # nominal objective value, untouched
