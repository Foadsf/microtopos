"""Main-window layout: a stable header and a left panel that is never cut off (offscreen, isolated settings)."""

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pg = pytest.importorskip("pyqtgraph")
from PySide6.QtCore import QSettings, QStandardPaths  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402


@pytest.fixture(scope="module")
def window(tmp_path_factory):
    QStandardPaths.setTestModeEnabled(True)  # calibration / objective files go to Qt's test location
    QSettings.setPath(QSettings.NativeFormat, QSettings.UserScope, str(tmp_path_factory.mktemp("qsettings")))
    pg.setConfigOptions(imageAxisOrder="row-major")
    app = QApplication.instance() or QApplication([])
    from microtopos.gui.main_window import MainWindow

    w = MainWindow(simulator_only=True)
    w.resize(1920, 1050)
    w.show()
    for _ in range(20):
        app.processEvents()
    w.timer.stop()  # the 15 Hz refresh rewrites the chips from live values; the tests drive them themselves
    for _ in range(5):
        app.processEvents()
    yield app, w
    w.close()
    app.processEvents()


def _settle(app, n=6):
    for _ in range(n):
        app.processEvents()


def test_header_buttons_and_height_do_not_move_while_chips_change(window):
    """The fps chip changes text every frame and CLIPPING toggles with the exposure: neither may move the header
    buttons between rows or change the header height (the "Light theme" button flickered between two rows)."""
    app, w = window
    # a middle panel as narrow as on a 1920 px screen with both side panels open, where the old header wrapped
    texts = ["9.9 fps in · 9.9 processed", "12.5 fps in · 12.1 processed", "25.0 fps in · 24.9 processed",
             "0.0 fps in · 0.0 processed"]
    w.resizeDocks([w.dock_left, w.dock_right], [380, 420], pg.QtCore.Qt.Horizontal)
    w.chip_fps.set(texts[0], "neutral")
    _settle(app, 20)  # the header re-flows in deferred (singleShot) callbacks: let them all run before measuring
    seen = set()
    for i in range(16):
        w.chip_fps.set(texts[i % len(texts)], "neutral")
        w.chip_sat.setVisible(i % 3 == 0)
        w.chip_state.set(["DWELL", "PAN", "Z_SWEEP  37 frames", "OBJECTIVE_CHANGE"][i % 4], "info")
        _settle(app, 3)
        g = w.btn_theme.geometry()
        seen.add((w.btn_theme.mapTo(w, g.topLeft() * 0).toTuple(), w.centralWidget().layout().itemAt(0).geometry().height()))
    assert len(seen) == 1, f"header moved: {sorted(seen)}"
    w.resizeDocks([w.dock_left, w.dock_right], [360, 390], pg.QtCore.Qt.Horizontal)
    _settle(app)


def test_left_panel_pages_fit_their_panel(window):
    """The pages scroll vertically only, so each page must fit the panel's width; a long camera name must not
    widen the page past it (the Camera page's right side was cut off under the middle panel)."""
    app, w = window
    w.cmb_source.addItem("MikrOkularFullHD: MikrOkularFullHD camera with a long name  (/dev/video4)")
    w.resizeDocks([w.dock_left], [w.dock_left.minimumWidth()], pg.QtCore.Qt.Horizontal)  # as narrow as allowed
    _settle(app)
    for i in range(w.stack.count()):
        w._open_step(i)
        _settle(app)
        sc = w.stack.widget(i)
        page = sc.widget()
        assert page.minimumSizeHint().width() <= sc.viewport().width(), (
            f"step {i + 1}: page needs {page.minimumSizeHint().width()} px, panel shows {sc.viewport().width()} px")
    w._open_step(0)
