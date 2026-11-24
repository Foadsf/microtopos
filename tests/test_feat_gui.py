"""Slice 5 GUI integration: ImagePanel clicking/colour, MainWindow capability wiring, interactive height
tools, session save/load round-trip with the mosaic graph, and negative controls.

Runs offscreen with isolated Qt settings, like tests/test_gui_layout.py.
"""

import os
import re
import time
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import cv2
import numpy as np
import pytest

pg = pytest.importorskip("pyqtgraph")
from PySide6.QtCore import QPointF, QSettings, QStandardPaths, Qt
from PySide6.QtGui import QMouseEvent
from PySide6.QtWidgets import QApplication, QFileDialog

from microtopos import export, metrology, zstack
from microtopos import mosaic as mosaic_mod
from microtopos.camera import ReplaySource
from microtopos.gui.main_window import MainWindow
from microtopos.gui.theme import DARK
from microtopos.gui.widgets import ImagePanel
from microtopos.mosaic import Mosaic
from microtopos.recording import RecordingWriter
from microtopos.stage import SimulatedStageDriver


@pytest.fixture(scope="module")
def qapp(tmp_path_factory):
    QStandardPaths.setTestModeEnabled(True)
    # MainWindow reads QSettings("MicroTopos", "MicroTopos"). On Windows the native format is the REGISTRY and
    # setPath does not apply to it, so the user's real saved objective would leak into the pixel size of every
    # result below. Use the ini format, redirected to a temporary folder, for the module.
    previous = QSettings.defaultFormat()
    QSettings.setDefaultFormat(QSettings.IniFormat)
    QSettings.setPath(QSettings.IniFormat, QSettings.UserScope, str(tmp_path_factory.mktemp("qsettings")))
    QSettings.setPath(QSettings.NativeFormat, QSettings.UserScope, str(tmp_path_factory.mktemp("qsettings_native")))
    app = QApplication.instance() or QApplication([])
    pg.setConfigOptions(imageAxisOrder="row-major")
    yield app
    QSettings.setDefaultFormat(previous)


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

def test_main_window_tab_order_and_replay_bar_hidden_without_a_replay(window):
    w = window
    assert not w.replay_widget.isVisible()  # only shown for a ReplaySource
    # tab order is frozen (Live, Mosaic, All-in-focus, Height, 3D, Report)
    titles = [w.tabs.tabText(i) for i in range(w.tabs.count())]
    assert titles == ["Live", "Mosaic", "All-in-focus", "Height", "3D", "Report"]


# ---------------------------------------------------------------- synthetic height maps with known truth
#
# The simulator sweep is reconstructed ONCE per module (9 s); every metrology test then installs a synthetic height
# map with an analytically known answer into a fresh window and lets the window's own recompute job measure it.

SHAPE = (360, 640)
NODATA_COLS = 10  # the left strip carries no height, like a real map's unobserved border


def _xy_um(um):
    yy, xx = np.mgrid[0:SHAPE[0], 0:SHAPE[1]].astype(np.float64)
    return xx * um, yy * um


def _relief(um):
    """Zero-mean test relief of two short wavelengths (3 um in x, 4 um in y): Sa is computed from it, not assumed.
    Short against the 10 um levelling patches, so the plane through three patch medians is off by well under 1 %
    of the injected tilt."""
    x, y = _xy_um(um)
    return 0.4 * np.sin(2 * np.pi * x / 3.0) + 0.3 * np.sin(2 * np.pi * y / 4.0 + 1.0)


def _sa(z):
    v = z[np.isfinite(z)]
    return float(np.mean(np.abs(v - v.mean())))


def _nodata(z):
    z = z.astype(np.float32)
    z[:, :NODATA_COLS] = np.nan
    return z


def _plane_slope(z, um):
    """(dz/dx, dz/dy) in um/um of the least-squares plane through the finite pixels."""
    x, y = _xy_um(um)
    ok = np.isfinite(z)
    A = np.column_stack([np.ones(int(ok.sum())), x[ok], y[ok]])
    return np.linalg.lstsq(A, z[ok].astype(np.float64), rcond=None)[0][1:]


def _wait_jobs(app, w, timeout=90.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        app.processEvents()
        time.sleep(0.01)
        if not w.jobs:
            _pump(app, 5)
            return
    raise AssertionError(f"jobs still running after {timeout} s: {[j.label for j in w.jobs]}")


@pytest.fixture(scope="module")
def swept(qapp, tmp_path_factory):
    """One reconstructed simulator sweep: the result dict every synthetic map below is derived from."""
    base = tmp_path_factory.mktemp("swept")
    os.environ["XDG_DATA_HOME"] = str(base / "data")
    os.environ["XDG_CONFIG_HOME"] = str(base / "config")
    w = MainWindow(simulator_only=True)
    w.show()
    _pump(qapp, 20)
    w.timer.stop()
    w.cmb_obj.setCurrentText("40x / 0.85")  # 0.225 um/px at the simulator's 640 px: wavelengths below are in pixels
    _pump(qapp, 10)
    w._demo("sweep")
    assert _wait_result(qapp, w, 60.0), "the sweep never reconstructed"
    assert w.result["um_per_px"] == pytest.approx(0.225)
    _wait_jobs(qapp, w)
    r = w.result
    w.close()
    qapp.processEvents()
    return r


def _install(app, w, swept, height, form=0, mode=0, base=None):
    """Make `height` the measured height of the window's result and recompute all metrology from it (the same job a
    filter change runs). Form order 0 keeps a tilt visible in the filtered surface."""
    r = dict(base if base is not None else swept)
    r["height"] = height
    r["height_raw"] = height.copy()
    w.sp_form.setValue(form)
    w.cmb_height.blockSignals(True)
    w.cmb_height.setCurrentIndex(mode)
    w.cmb_height.blockSignals(False)
    w.history = [r]
    w.result = r
    w.tabs.setCurrentIndex(3)
    w._recompute()
    _wait_jobs(app, w)
    assert w.result is not r and "Sa" in w.result
    return w.result


def _three_point_expected(z, pts, radius):
    """Plane through the medians of the (2 radius + 1)-px square patches at the 3 points, subtracted from z."""
    rows, vals = [], []
    for px, py in pts:
        ix, iy = round(px), round(py)
        patch = z[max(0, iy - radius):iy + radius + 1, max(0, ix - radius):ix + radius + 1]
        rows.append([1.0, px, py])
        vals.append(np.nanmedian(patch))
    c0, cx, cy = np.linalg.solve(np.array(rows), np.array(vals, np.float64))
    yy, xx = np.mgrid[0:z.shape[0], 0:z.shape[1]].astype(np.float64)
    return z - (c0 + cx * xx + cy * yy)


def _click_points(app, w, pts):
    w.btn_level3.setChecked(True)
    assert w.level_picking
    for x, y in pts:
        w.p_height.clicked_px.emit(float(x), float(y))
        _pump(app, 3)
    _wait_jobs(app, w)


# ---------------------------------------------------------------- interactive height tools

TILT = np.array([0.25, 0.10])  # um/um in x and y: 36 um across the 144 um field, ~90x the 0.4 um relief


@pytest.mark.parametrize("mode", [0, 1, 2], ids=["filtered", "raw-fit", "confidence"])
def test_three_point_levelling_removes_the_tilt_whatever_layer_is_shown(window, qapp, swept, mode):
    w = window
    um = swept["um_per_px"]
    x, y = _xy_um(um)
    relief = _relief(um)
    tilted = _nodata(relief + TILT[0] * x + TILT[1] * y)
    before = _install(qapp, w, swept, tilted, form=0, mode=mode)
    conf_before = before["confidence"]
    # the tilt is really there, in the measured height and (form order 0) in the filtered surface
    assert np.allclose(_plane_slope(before["height"], um), TILT, rtol=0.01)
    assert np.allclose(_plane_slope(before["sl_surface"], um), TILT, rtol=0.01)
    assert before["Sa"] > 5 * _sa(relief[:, NODATA_COLS:])

    h, wd = SHAPE
    pts = [(0.25 * wd, 0.25 * h), (0.75 * wd, 0.25 * h), (0.5 * wd, 0.75 * h)]
    _click_points(qapp, w, pts)
    after = w.result
    assert not w.level_picking

    # the measured height AND the filtered surface are flat: less than 1 % of the injected slope remains
    for layer in ("height", "sl_surface"):
        residual = np.hypot(*_plane_slope(after[layer], um))
        assert residual < 0.01 * np.hypot(*TILT), (layer, residual)
    # what is on screen follows the displayed layer (the confidence layer is not height and is untouched)
    shown = w.p_height.data
    if mode == 2:
        assert np.array_equal(shown, conf_before) and np.array_equal(after["confidence"], conf_before)
    else:
        layer = after["sl_surface"] if mode == 0 else after["height"]
        assert np.array_equal(shown, layer, equal_nan=True)
        assert np.hypot(*_plane_slope(shown, um)) < 0.01 * np.hypot(*TILT)
    # levelling subtracted a plane and nothing else
    diff = after["height"] - before["height"]
    c = _plane_slope(diff, um)
    x_, y_ = _xy_um(um)
    ok = np.isfinite(diff)
    A = np.column_stack([np.ones(int(ok.sum())), x_[ok], y_[ok]])
    coef = np.linalg.lstsq(A, diff[ok].astype(np.float64), rcond=None)[0]
    assert np.abs(A @ coef - diff[ok]).max() < 1e-3
    assert np.allclose(c, -TILT, rtol=0.03)
    # the plane is the one through the medians of three 10 um patches around the picked points, computed here
    # independently with numpy; every parameter was recomputed from that height (Sa is the plain mean |z - mean|)
    expected = _three_point_expected(tilted, pts, max(3, round(10 / um)))
    assert np.allclose(after["height"], expected, atol=2e-4, equal_nan=True)
    assert after["Sa"] == pytest.approx(_sa(expected), rel=1e-3)
    assert after["Sa"] < 0.1 * before["Sa"]
    # the edit is recorded where the user and the report can see it
    assert after["height_tools"][0]["tool"] == "level3"
    assert any(wm.startswith("Height edited by the user") and "levelled" in wm for wm in after["warnings"])
    assert "levelled" in w.report.toPlainText()
    assert w.btn_tools_reset.isEnabled()


@pytest.mark.parametrize("mode", [0, 1], ids=["filtered", "raw-fit"])
def test_exclusion_roi_masks_exactly_the_rectangle_and_recomputes_parameters(window, qapp, swept, mode):
    w = window
    um = swept["um_per_px"]
    relief = _relief(um)
    z = _nodata(relief)
    before = _install(qapp, w, swept, z, form=0, mode=mode)
    assert not w.mask_roi.isVisible()
    w.btn_mask.click()  # first press: arm the tool, show its rectangle
    _pump(qapp, 3)
    assert w.mask_roi.isVisible() and w.btn_tool_cancel.isVisible()
    x0, y0, x1, y1 = 200, 100, 350, 180
    w.mask_roi.setPos((x0 * um, y0 * um))
    w.mask_roi.setSize(((x1 - x0) * um, (y1 - y0) * um))
    w.btn_mask.click()  # second press: apply
    _wait_jobs(qapp, w)
    after = w.result
    expected_nan = np.isnan(z)
    expected_nan[y0:y1, x0:x1] = True
    assert np.array_equal(np.isnan(after["height"]), expected_nan)
    shown = w.p_height.data
    assert np.isnan(shown[y0:y1, x0:x1]).all() and np.isfinite(shown[:, 400:]).all()
    assert not w.mask_roi.isVisible()
    # parameters come from the remaining pixels only
    kept = np.where(expected_nan, np.nan, relief)
    assert after["Sa"] == pytest.approx(_sa(kept), rel=0.01)
    assert after["valid_fraction"] == pytest.approx(1 - expected_nan.mean(), abs=2e-3)
    assert before["valid_fraction"] > after["valid_fraction"]
    assert after["height_tools"][0] == {"tool": "exclude", "rect_px": [x0, y0, x1, y1]}
    assert any("excluded" in wm and wm.startswith("Height edited by the user") for wm in after["warnings"])
    assert "excluded" in w.report.toPlainText()


def _step_value(text):
    m = re.search(r"Step A→B:\s*([+-][0-9.]+)\s*(\S+)", text)
    assert m, text
    return float(m.group(1)), m.group(2)


def test_step_height_reads_the_known_step_in_the_result_unit(window, qapp, swept):
    w = window
    um = swept["um_per_px"]
    x, _y = _xy_um(um)
    z = _nodata(0.05 * _relief(um) + 3.5 * (x >= 0.5 * SHAPE[1] * um) + 0.002 * x)  # terraces 3.5 um apart
    r = _install(qapp, w, swept, z, form=0)
    w.btn_step.click()
    _pump(qapp, 3)
    assert w.step_roi_a.isVisible() and w.step_roi_b.isVisible()
    w.btn_step.click()
    _pump(qapp, 3)
    value, unit = _step_value(w.lbl_height.text())
    assert unit == "µm" and value == pytest.approx(3.5, rel=0.02)
    assert not w.step_roi_a.isVisible() and not w.step_roi_b.isVisible()

    # the same step in an uncalibrated sweep is reported in frames, not micrometres
    frames = zstack.rescale_heights(r, 1.0, "frames")
    w.history = [frames]
    w.result = frames
    w._show_result()
    w.btn_step.click()
    _pump(qapp, 3)
    w.btn_step.click()
    _pump(qapp, 3)
    value, unit = _step_value(w.lbl_height.text())
    assert unit == "frames" and value == pytest.approx(3.5, rel=0.02)
    assert "µm" not in w.lbl_height.text()


def test_tool_rois_are_shown_only_while_their_tool_is_armed(window, qapp, swept):
    w = window
    um = swept["um_per_px"]
    _install(qapp, w, swept, _nodata(_relief(um)))
    rois = (w.mask_roi, w.step_roi_a, w.step_roi_b)
    assert not any(r.isVisible() for r in rois)
    w.btn_step.click()
    _pump(qapp, 3)
    assert w.step_roi_a.isVisible() and w.step_roi_b.isVisible() and not w.mask_roi.isVisible()
    w.btn_mask.click()  # arming the other tool swaps the rectangles
    _pump(qapp, 3)
    assert w.mask_roi.isVisible() and not w.step_roi_a.isVisible() and not w.step_roi_b.isVisible()
    w.btn_tool_cancel.click()
    _pump(qapp, 3)
    assert not any(r.isVisible() for r in rois) and not w.btn_tool_cancel.isVisible()
    w.btn_mask.click()
    w.btn_level3.setChecked(True)  # picking points hides them too
    _pump(qapp, 3)
    assert w.level_picking and not any(r.isVisible() for r in rois)
    w.btn_level3.setChecked(False)
    assert not w.level_picking


def test_reset_restores_the_reconstructed_height_and_every_displayed_layer(window, qapp, swept):
    w = window
    um = swept["um_per_px"]
    x, y = _xy_um(um)
    z = _nodata(_relief(um) + TILT[0] * x + TILT[1] * y)
    original = _install(qapp, w, swept, z, form=0)
    h0, s0, sa0 = original["height"].copy(), original["sl_surface"].copy(), original["Sa"]
    assert not w.btn_tools_reset.isEnabled()
    h, wd = SHAPE
    _click_points(qapp, w, [(0.25 * wd, 0.25 * h), (0.75 * wd, 0.25 * h), (0.5 * wd, 0.75 * h)])
    w.btn_mask.click()
    w.mask_roi.setPos((200 * um, 100 * um))
    w.mask_roi.setSize((100 * um, 60 * um))
    w.btn_mask.click()
    _wait_jobs(qapp, w)
    assert len(w.result["height_tools"]) == 2
    assert not np.array_equal(w.result["height"], h0, equal_nan=True)
    w.btn_tools_reset.click()
    _wait_jobs(qapp, w)
    r = w.result
    assert np.array_equal(r["height"], h0, equal_nan=True)
    assert np.array_equal(r["sl_surface"], s0, equal_nan=True)
    assert r["Sa"] == pytest.approx(sa0, rel=1e-6)
    assert "height_tools" not in r and not any(wm.startswith("Height edited") for wm in r["warnings"])
    assert not w.btn_tools_reset.isEnabled()


# ---------------------------------------------------------------- profile roughness label (ISO 21920)

LAM_SHORT, LAM_LONG, A_SHORT, A_LONG = 8.0, 40.0, 0.5, 2.0  # roughness and waviness sines (um, um, height um)
ALPHA = float(np.sqrt(np.log(2) / np.pi))


def _two_sine_surface(um):
    x, _ = _xy_um(um)
    row = A_SHORT * np.sin(2 * np.pi * x / LAM_SHORT) + A_LONG * np.sin(2 * np.pi * x / LAM_LONG + 0.7)
    return _nodata(row)  # constant in y, so any horizontal line sees the same profile


def _expected_roughness(lam_c, um, x_px0, x_px1):
    """Ra, Rq of the roughness profile of the two-sine line: each sine is attenuated by the ISO 16610-21 Gaussian
    L-filter transmission exp(-pi (alpha lam_c / lam)^2) before being subtracted from the profile."""
    def t(lam):
        return np.exp(-np.pi * (ALPHA * lam_c / lam) ** 2)

    xs = np.arange(x_px0, x_px1, 0.01 / um) * um
    rough = (A_SHORT * (1 - t(LAM_SHORT)) * np.sin(2 * np.pi * xs / LAM_SHORT)
             + A_LONG * (1 - t(LAM_LONG)) * np.sin(2 * np.pi * xs / LAM_LONG + 0.7))
    return float(np.mean(np.abs(rough))), float(np.sqrt(np.mean(rough ** 2)))


def _label_value(text, name):
    m = re.search(rf"\b{name} ([0-9.eE+-]+)", text)
    assert m, text
    return float(m.group(1))


def test_profile_label_reports_iso_roughness_at_the_chosen_cutoff(window, qapp, swept):
    w = window
    um = swept["um_per_px"]
    _install(qapp, w, swept, _two_sine_surface(um), form=0, mode=0)
    p0, p1 = w.prof_roi.getHandles()
    length = float(np.hypot(p1.pos().x() - p0.pos().x(), p1.pos().y() - p0.pos().y()))
    x_a, x_b = p0.pos().x() / um, p1.pos().x() / um
    assert p0.pos().y() == pytest.approx(p1.pos().y())  # a horizontal line

    w.sp_lc.setValue(15.0)
    _pump(qapp, 5)
    text = w.prof_label.text()
    ra_exp, rq_exp = _expected_roughness(15.0, um, x_a, x_b)
    assert _label_value(text, "Ra") == pytest.approx(ra_exp, rel=0.02), text
    assert _label_value(text, "Rq") == pytest.approx(rq_exp, rel=0.02)
    assert "λc 15" in text and "L-filter setting" in text

    # the waviness (40 um, 2 um high) dominates the unfiltered line, so Ra/Rq depend on lambda_c as the Gaussian
    # transmission says; a different lambda_c gives a different, equally predicted, value
    w.sp_lc.setValue(0.0)  # "off": the label falls back to a fifth of the line, and says so
    _pump(qapp, 5)
    text0 = w.prof_label.text()
    lam_default = length / metrology.PROFILE_EVAL_CUTOFFS
    ra0, rq0 = _expected_roughness(lam_default, um, x_a, x_b)
    assert "default" in text0
    assert _label_value(text0, "Ra") == pytest.approx(ra0, rel=0.02)
    assert _label_value(text0, "Rq") == pytest.approx(rq0, rel=0.02)
    assert abs(ra0 - ra_exp) / ra_exp > 0.05  # the two cut-offs really are distinguishable

    # the raw-fit layer shows the same measured height, so the same roughness
    w.sp_lc.setValue(15.0)
    w.cmb_height.setCurrentIndex(1)
    _pump(qapp, 5)
    assert _label_value(w.prof_label.text(), "Ra") == pytest.approx(ra_exp, rel=0.02)
    w.cmb_height.setCurrentIndex(2)
    _pump(qapp, 5)
    assert "Confidence" in w.prof_label.text()


def test_profile_label_refuses_a_cutoff_the_line_cannot_support(window, qapp, swept):
    w = window
    um = swept["um_per_px"]
    _install(qapp, w, swept, _two_sine_surface(um), form=0, mode=0)
    w.sp_lc.setValue(100.0)  # five cut-offs = 500 um on a 115 um line
    _pump(qapp, 5)
    text = w.prof_label.text()
    assert text.startswith("Profile too short for λc = 100")
    assert "Ra " not in text
    # the reason is the shared metrology rule, not a GUI invention
    n = len(w.prof_curve.getData()[0])
    assert metrology.profile_cutoff_problem(n, w.prof_curve.getData()[0][1], 100.0) is not None


# ---------------------------------------------------------------- on-demand feature parameters and hills

def _hill_map(um):
    """6 x 5 lattice of Gaussian hills (1 um high, sigma 12 px) on a 100 x 70 px pitch: 30 peaks, of which
    (6-2) x (5-2) = 12 catchments do not touch the border, each 100 x 70 px."""
    h, w = SHAPE
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    z = np.zeros(SHAPE)
    for i in range(6):
        for j in range(5):
            z += np.exp(-((xx - (70 + 100 * i)) ** 2 + (yy - (40 + 70 * j)) ** 2) / (2 * 12.0 ** 2))
    return z.astype(np.float32)


def test_feature_checkbox_computes_spd_and_the_known_hill_count_and_area(window, qapp, swept):
    w = window
    um = swept["um_per_px"]
    r = _install(qapp, w, swept, _hill_map(um), form=0)
    assert "Spd_per_mm2" not in r and "topo_grain_count" not in r
    assert "computed on demand" in w.report.toPlainText()
    assert "Hill count" not in w.report.toPlainText()

    w.chk_topo_grains.setChecked(True)
    _wait_jobs(qapp, w)
    r = w.result
    area_mm2 = SHAPE[0] * SHAPE[1] * um * um * 1e-6
    assert r["Spd_per_mm2"] == pytest.approx(30 / area_mm2, rel=1e-6)
    assert r["topo_grain_count"] == 12
    assert r["topo_mean_grain_area_um2"] == pytest.approx(100 * 70 * um * um, rel=0.05)
    text = w.report.toPlainText()
    assert re.search(r"Hill count\s+12\b", text), text
    assert re.search(r"Spd\s+[0-9.e+]+\s+peaks/mm²", text), text

    # a result computed later (new filter) drops the stale features and does not pretend to have them
    w.chk_topo_grains.setChecked(False)
    w._recompute()
    _wait_jobs(qapp, w)
    assert "Spd_per_mm2" not in w.result and "Hill count" not in w.report.toPlainText()


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
    # Stop the pan and let the engine catch up on a still view: the session is saved from, and loaded into, an
    # engine that is no longer adding keyframes, so the counts below compare the same mosaic.
    with w.sim_source.lock:
        w.sim_source.script = []
    still_from = w.sim_source.latest()[2] + 2  # the first frames rendered after the stage stopped
    assert _wait_for(qapp, lambda: w.sim_source.consumed_seq >= still_from + 5, timeout=20.0)
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


# ---------------------------------------------------------------- replay transport, stage driver, mosaic toggles

def _make_recording(tmp_path, n_frames: int = 8):
    d = tmp_path / "rec"
    d.mkdir()
    w = RecordingWriter(d, "sim")
    for i in range(n_frames):
        frame = np.full((16, 16, 3), fill_value=(10 * i) % 256, dtype=np.uint8)
        w.add(frame, seq=i, t=0.1 * i)
    w.close()
    return d


def _wait_for(qapp, cond, timeout: float = 5.0) -> bool:
    t0 = time.time()
    while time.time() - t0 < timeout:
        qapp.processEvents()
        time.sleep(0.02)
        if cond():
            return True
    return False


def test_replay_ui_transport_bar(tmp_path, qapp):
    """Opening MainWindow(replay=...) shows the replay bar and the pause / seek / step / speed /
    loop controls drive the ReplaySource (offscreen, no real camera)."""
    rec_dir = _make_recording(tmp_path)
    w = MainWindow(simulator_only=False, replay=str(rec_dir))
    w.show()
    _pump(qapp)
    try:
        assert isinstance(w.source, ReplaySource)
        assert w.replay_widget.isVisible()
        src: ReplaySource = w.source
        assert src.total_frames == 8

        # pause: the playback thread stops advancing
        w.btn_replay_pause.click()
        _pump(qapp, 10)
        assert src.paused is True
        assert w.btn_replay_pause.text() == "▶ Play"

        # resume
        w.btn_replay_pause.click()
        _pump(qapp, 10)
        assert src.paused is False
        assert w.btn_replay_pause.text() == "‖ Pause"

        # step forward: pauses and advances one frame
        w.btn_replay_pause.click()  # pause first
        _pump(qapp, 10)
        start = src.position
        w.btn_replay_step_f.click()
        assert src.paused is True
        assert src._seek_target == min(start + 1, src.total_frames - 1)
        assert _wait_for(qapp, lambda: src.position == min(start + 1, src.total_frames - 1))

        # seek via the slider
        w.replay_seek.setValue(3)
        w.replay_seek.sliderReleased.emit()
        assert src._seek_target == 3
        assert _wait_for(qapp, lambda: src.position == 3)

        # speed
        w.cmb_replay_speed.setCurrentText("2×")
        _pump(qapp, 10)
        assert src.speed == 2.0

        # loop
        w.chk_replay_loop.setChecked(True)
        _pump(qapp, 10)
        assert src.loop is True
        w.chk_replay_loop.setChecked(False)
        _pump(qapp, 10)
        assert src.loop is False
    finally:
        w.close()
        qapp.processEvents()


def _stage_idle(w) -> bool:
    return w.stage_job is None or not w.stage_job.isRunning()


def test_simulated_stage_connect_jog_zero_and_disconnect(window, qapp):
    """The stage panel with the simulated drive. Stage operations run on a worker, so each step waits for it.
    (The Z sweep itself is tested end to end in tests/test_feat_hardware.py.)"""
    w = window
    assert w.sim_source is not None
    assert w.cmb_stg.currentText() == "Simulated stage"
    assert not w.btn_stg_zero.isEnabled() and not w.btn_stg_zsweep.isEnabled()  # nothing connected yet

    w.btn_stg_connect.click()
    assert _wait_for(qapp, lambda: isinstance(w.stage_driver, SimulatedStageDriver) and _stage_idle(w), timeout=20.0)
    assert w.btn_stg_connect.text() == "Disconnect"
    assert w.btn_stg_zero.isEnabled() and w.btn_stg_zsweep.isEnabled()
    assert w.source.stage_driver is w.stage_driver  # attached to the active source
    sim = w.sim_source.sim
    x0 = sim.stage_x_um

    w.stage_jog_buttons[1].click()  # "X +10 µm"
    assert _wait_for(qapp, lambda: _stage_idle(w) and f"X {x0 + 10.0:+.2f} µm" in w.lbl_stg_pose.text(), timeout=20.0)
    assert sim.stage_x_um == pytest.approx(x0 + 10.0)  # the simulated specimen moved, not only the label

    w.stage_driver.move_to(x_um=3.0, y_um=-1.5, z_um=2.0)
    w.btn_stg_zero.click()  # the simulated drive returns to the simulator's origin
    assert _wait_for(qapp, lambda: _stage_idle(w) and "X +0.00 µm  Y +0.00 µm  Z +0.00 µm" in w.lbl_stg_pose.text(), timeout=20.0)
    assert (sim.stage_x_um, sim.stage_y_um, sim.focus_z_um) == (0.0, 0.0, 0.0)

    w.btn_stg_connect.click()
    assert _wait_for(qapp, lambda: w.stage_driver is None and _stage_idle(w), timeout=20.0)
    assert w.btn_stg_connect.text() == "Connect"
    assert "disconnected" in w.lbl_stg_pose.text()
    assert w.source.stage_driver is None
    assert not w.btn_stg_zero.isEnabled() and not w.btn_stg_zsweep.isEnabled()


def test_mosaic_blend_toggle_changes_what_the_renderer_reads(window, qapp, monkeypatch):
    """The combo box must change the value Mosaic.image() consumes: a low-frequency tile layer only for multiband."""
    w = window
    seen = []
    real = mosaic_mod.render_tiles

    def spy(tiles, tile, d, low):
        seen.append(low is not None)
        return real(tiles, tile, d, low)

    monkeypatch.setattr(mosaic_mod, "render_tiles", spy)
    w.cmb_mblend.setCurrentText("multiband")
    _pump(qapp)
    w.engine.mosaic.image()
    w.cmb_mblend.setCurrentText("feather")
    _pump(qapp)
    w.engine.mosaic.image()
    assert seen == [True, False]


def test_rotation_toggle_changes_the_rotation_the_mosaic_measures(window, qapp):
    """A frame rotated by 3 degrees: with the box ticked the mosaic measures it (2.9-3.1 deg), unticked it does not."""
    w = window
    rng = np.random.default_rng(3)
    a = cv2.GaussianBlur(rng.normal(size=(200, 200)).astype(np.float32), (0, 0), 2.5)
    a = (a - a.min()) / (a.max() - a.min())
    M = cv2.getRotationMatrix2D((99.5, 99.5), 3.0, 1.0)
    f = cv2.warpAffine(a, M, (200, 200), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    w.chk_rot_track.setChecked(True)
    _pump(qapp)
    assert w.track_rotation is True and w.engine.mosaic.track_rotation is True
    w.engine.mosaic._verified(a, f, 0.0, 0.0)
    assert abs(abs(w.engine.mosaic._last_drot) - 3.0) < 0.1
    w.chk_rot_track.setChecked(False)
    _pump(qapp)
    assert w.track_rotation is False and w.engine.mosaic.track_rotation is False
    w.engine.mosaic._verified(a, f, 0.0, 0.0)
    assert w.engine.mosaic._last_drot == 0.0


def test_mosaic_rgb_toggle_sets_the_flag_the_view_reads(window, qapp):
    w = window
    w.chk_mosaic_rgb.setChecked(True)
    _pump(qapp)
    assert w.mosaic_rgb is True
    w.chk_mosaic_rgb.setChecked(False)
    _pump(qapp)
    assert w.mosaic_rgb is False


def test_backend_and_breathing_toggles_reach_the_reconstruction(window, qapp, monkeypatch):
    """Whatever the 3D page shows must be what zstack.reconstruct is given in its ZSettings."""
    w = window
    got = []

    def recorder(frames, z, obj, um, unit, zcal, st, **kw):
        got.append(st)
        raise RuntimeError("recorded; no reconstruction wanted")

    monkeypatch.setattr(zstack, "reconstruct", recorder)
    w.cmb_obj.setCurrentText("40x / 0.85")  # the objective whose simulated sweep passes through focus
    _pump(qapp, 10)
    w._demo("sweep")
    t0 = time.time()
    while not got and time.time() - t0 < 60.0:
        qapp.processEvents()
        time.sleep(0.02)
    assert got, "the sweep never reached the reconstruction"
    _wait_jobs(qapp, w)
    assert (got[0].backend, got[0].correct_breathing) == ("cpu", False)

    w.cmb_3d_backend.setCurrentText("OpenCL")
    w.chk_3d_breathing.setChecked(True)
    w._reconstruct_last()
    t0 = time.time()
    while len(got) < 2 and time.time() - t0 < 30.0:
        qapp.processEvents()
        time.sleep(0.02)
    assert len(got) == 2
    assert (got[1].backend, got[1].correct_breathing) == ("opencl", True)
    _wait_jobs(qapp, w)

    w.cmb_3d_backend.setCurrentText("CPU")
    w.chk_3d_breathing.setChecked(False)
    w._reconstruct_last()
    t0 = time.time()
    while len(got) < 3 and time.time() - t0 < 30.0:
        qapp.processEvents()
        time.sleep(0.02)
    assert len(got) == 3 and (got[2].backend, got[2].correct_breathing) == ("cpu", False)
    _wait_jobs(qapp, w)


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
