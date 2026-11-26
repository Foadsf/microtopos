"""Main-window layout: a stable header and a left panel that is never cut off (offscreen, isolated settings)."""

import os
import subprocess
import sys

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pyqtgraph as pg
from PySide6.QtCore import QStandardPaths
from PySide6.QtWidgets import QApplication


@pytest.fixture(scope="module")
def window(tmp_path_factory):
    QStandardPaths.setTestModeEnabled(True)  # calibration / objective files go to Qt's test location
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


# Limits on what the window may demand of the screen. Fonts differ between platforms (Qt's offscreen platform has the
# widest, which is what CI uses), so the limits leave room for them; the real measurements, in Qt's device-independent
# pixels with Segoe UI on Windows: the window's minimum width was 2361 before the tool bars wrapped, 1092 after.
# (Offscreen: 3477 before, 1723 after.)
MAX_WINDOW_MIN_WIDTH = 1800
MAX_PAGE_MIN_WIDTH = 600
MAX_LEFT_PANEL_MIN_WIDTH = 650


def test_no_page_or_panel_sets_a_wide_minimum_width(window):
    """Every control in a row of a QHBoxLayout adds its width to the window's MINIMUM width, so the Live tab's row of
    checkboxes, combo boxes and buttons made 0.10.0 unable to shrink below 2361 px: on a 1920 px screen the window
    could not be narrowed, and maximised it lost its right panel off the screen. Tool bars wrap instead."""
    app, w = window
    w.replay_widget.setVisible(True)  # the replay transport bar is part of the Live tab while a recording plays
    _settle(app)
    try:
        wide = {w.tabs.tabText(i): w.tabs.widget(i).minimumSizeHint().width() for i in range(w.tabs.count())
                if w.tabs.widget(i).minimumSizeHint().width() > MAX_PAGE_MIN_WIDTH}
    finally:
        w.replay_widget.setVisible(False)
    assert not wide, f"tabs that demand more than {MAX_PAGE_MIN_WIDTH} px of width: {wide}"
    assert w.dock_left.minimumWidth() <= MAX_LEFT_PANEL_MIN_WIDTH, w.dock_left.minimumWidth()
    assert w.minimumSizeHint().width() <= MAX_WINDOW_MIN_WIDTH, (
        f"the window cannot be narrower than {w.minimumSizeHint().width()} px")


@pytest.mark.parametrize("size", [(1920, 1080), (1800, 1000)])
def test_the_window_takes_the_size_it_is_given_and_keeps_both_panels_inside(window, size):
    """Resizing (which is what the window manager does on maximise) must give exactly that size, with the session
    panel, the middle panel and the telemetry panel all inside it."""
    app, w = window
    w.resize(*size)
    _settle(app, 20)
    assert (w.width(), w.height()) == size, f"asked for {size}, the window is {(w.width(), w.height())}"
    left, mid, right = w.dock_left.geometry(), w.centralWidget().geometry(), w.dock_right.geometry()
    assert w.dock_left.isVisible() and w.dock_right.isVisible()
    assert left.left() >= 0 and right.right() <= w.width() - 1, (left.getRect(), right.getRect())
    assert mid.width() > 0 and left.right() < mid.left() + 2 and mid.right() < right.left() + 2, (
        left.getRect(), mid.getRect(), right.getRect())


def test_the_image_panel_gets_the_width_the_side_panels_do_not_need(window):
    """A fresh window (no saved layout) gives the session and telemetry panels fixed, modest widths. Left to their
    size hints the telemetry panel took 600+ px and the image only 480 px of a 1536 px screen."""
    from microtopos.gui.main_window import DEFAULT_RIGHT_DOCK_WIDTH, MainWindow

    app, _ = window  # the fixture sets the application up; this test needs a window of its own
    w = MainWindow(simulator_only=True)
    try:
        w.timer.stop()
        w.resize(1920, 1080)
        w.show()
        _settle(app, 30)
        mid, total = w.centralWidget().width(), w.width()
        assert total == 1920, total
        assert w.dock_right.width() == DEFAULT_RIGHT_DOCK_WIDTH, w.dock_right.width()  # offscreen: 468 without it
        assert mid >= 0.48 * total, (
            f"image panel {mid} px of {total} px (session {w.dock_left.width()}, telemetry {w.dock_right.width()})")
    finally:
        w.close()
        app.processEvents()


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



# window geometry and dock layout saved by MicroTopos on a two-monitor desktop (1920x1080 each)
_SAVED_GEOMETRY = (
    '01d9d0cb00030000000007800000000000000eff00000437000007a00000004000000db7000003eb00000001020000000780000007800000002000000eff00000437')
_SAVED_STATE = (
    '000000ff00000000fd00000002000000000000014b00000418fc0200000001fb0000000e00730065007300730069006f006e0100000000000004180000016700ffffff000000010000018600000418fc0200000001fb0000001200740065006c0065006d00650074007200790100000000000004180000005e00ffffff000004ad0000041800000004000000040000000800000008fc00000000')

_REOPEN_MAXIMISED = r"""
import sys
import os
from PySide6.QtCore import QByteArray, QStandardPaths
QStandardPaths.setTestModeEnabled(True)
os.environ["MICROTOPOS_SETTINGS_DIR"] = sys.argv[1]
import pyqtgraph as pg
from PySide6.QtWidgets import QApplication, QWidget
pg.setConfigOptions(imageAxisOrder="row-major")
app = QApplication([])
from microtopos.gui.main_window import MainWindow, app_settings
def pump(n):
    import time
    for _ in range(n):
        app.processEvents(); time.sleep(0.02)
s = app_settings()  # as a real session left it: maximised on a second monitor, normal size 1560x940
s.setValue("geometry", QByteArray.fromHex(sys.argv[2].encode())); s.setValue("state", QByteArray.fromHex(sys.argv[3].encode())); s.sync()
w = MainWindow(simulator_only=True); w.show(); pump(100)
edges = [w.dock_left.geometry().right() + 1, w.dock_right.geometry().left() - 1]
stray = [c.geometry().getRect() for c in w.findChildren(QWidget, "qt_qmainwindow_extended_splitter")
         if c.isVisible() and not any(abs(c.geometry().center().x() - e) <= 3 for e in edges)]
print("maximised", w.isMaximized(), "edges", edges, "stray", stray)
w.close(); pump(5)
sys.exit(3 if not w.isMaximized() else 1 if stray else 0)
"""


def test_reopening_a_maximised_window_leaves_no_stray_dock_separators(tmp_path, xvfb_and_window_manager):
    """A window saved maximised (here: on a second monitor) and reopened must have dock separators only at the dock
    edges. Restoring the dock
    layout into a window already marked maximised (but not yet shown) left separators from the unmaximised size,
    each a 1-px line across the middle panel. Only a real window manager, which maps the window first and
    maximises it a moment later, shows this."""
    script = tmp_path / "reopen.py"
    script.write_text(_REOPEN_MAXIMISED)
    env = {k: v for k, v in os.environ.items() if k != "QT_QPA_PLATFORM"}
    r = subprocess.run(["xvfb-run", "-a", "-s", "-screen 0 1920x1080x24", "sh", "-c",
                        (f"metacity --replace >/dev/null 2>&1 & sleep 2; '{sys.executable}' '{script}' '{tmp_path}' "
                         f"{_SAVED_GEOMETRY} {_SAVED_STATE}")],
                       capture_output=True, text=True, timeout=180, env=env, check=False)
    report = [line for line in r.stdout.splitlines() if line.startswith("maximised")]
    assert r.returncode != 3, f"the window manager did not maximise the window: {report}"
    assert r.returncode == 0, f"stray dock separators: {report or r.stderr[-1500:]}"
