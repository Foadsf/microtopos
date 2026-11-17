"""Main-window layout: a stable header and a left panel that is never cut off (offscreen, isolated settings)."""

import os
import shutil
import subprocess
import sys

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pg = pytest.importorskip("pyqtgraph")
from PySide6.QtCore import QSettings, QStandardPaths
from PySide6.QtWidgets import QApplication


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



# window geometry and dock layout saved by MicroTopos on a two-monitor desktop (1920x1080 each)
_SAVED_GEOMETRY = (
    '01d9d0cb00030000000007800000000000000eff00000437000007a00000004000000db7000003eb00000001020000000780000007800000002000000eff00000437')
_SAVED_STATE = (
    '000000ff00000000fd00000002000000000000014b00000418fc0200000001fb0000000e00730065007300730069006f006e0100000000000004180000016700ffffff000000010000018600000418fc0200000001fb0000001200740065006c0065006d00650074007200790100000000000004180000005e00ffffff000004ad0000041800000004000000040000000800000008fc00000000')

_REOPEN_MAXIMISED = r"""
import sys
from PySide6.QtCore import QByteArray, QSettings, QStandardPaths
QStandardPaths.setTestModeEnabled(True)
QSettings.setPath(QSettings.NativeFormat, QSettings.UserScope, sys.argv[1])
import pyqtgraph as pg
from PySide6.QtWidgets import QApplication, QWidget
pg.setConfigOptions(imageAxisOrder="row-major")
app = QApplication([])
from microtopos.gui.main_window import MainWindow
def pump(n):
    import time
    for _ in range(n):
        app.processEvents(); time.sleep(0.02)
s = QSettings("MicroTopos", "MicroTopos")  # as a real session left it: maximised on a second monitor, normal size 1560x940
s.setValue("geometry", QByteArray.fromHex(sys.argv[2].encode())); s.setValue("state", QByteArray.fromHex(sys.argv[3].encode())); s.sync()
w = MainWindow(simulator_only=True); w.show(); pump(100)
edges = [w.dock_left.geometry().right() + 1, w.dock_right.geometry().left() - 1]
stray = [c.geometry().getRect() for c in w.findChildren(QWidget, "qt_qmainwindow_extended_splitter")
         if c.isVisible() and not any(abs(c.geometry().center().x() - e) <= 3 for e in edges)]
print("maximised", w.isMaximized(), "edges", edges, "stray", stray)
w.close(); pump(5)
sys.exit(3 if not w.isMaximized() else 1 if stray else 0)
"""


@pytest.mark.skipif(not (shutil.which("xvfb-run") and shutil.which("metacity")),
                    reason="needs Xvfb and a window manager (metacity): offscreen Qt maximises synchronously")
def test_reopening_a_maximised_window_leaves_no_stray_dock_separators(tmp_path):
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
