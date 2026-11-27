"""Mosaic defects of the November 2026 audit seen through the window: an opened session drawn at the selected
objective's scale, insets dropped by an objective change, insets never drawn.

Runs offscreen with isolated settings (conftest.py) on the simulator; no camera, no stage."""

import os
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import cv2
import numpy as np
import pyqtgraph as pg
import pytest
from PySide6.QtCore import QStandardPaths
from PySide6.QtWidgets import QApplication, QFileDialog

from microtopos.export import save_session
from microtopos.gui.main_window import MainWindow
from microtopos.mosaic import Mosaic


@pytest.fixture(scope="module")
def qapp():
    QStandardPaths.setTestModeEnabled(True)
    app = QApplication.instance() or QApplication([])
    pg.setConfigOptions(imageAxisOrder="row-major")
    yield app


def _wait_for(app, cond, timeout=20.0) -> bool:
    t0 = time.time()
    while time.time() - t0 < timeout:
        app.processEvents()
        time.sleep(0.02)
        if cond():
            return True
    return False


@pytest.fixture
def window(qapp, tmp_path):
    os.environ["XDG_DATA_HOME"] = str(tmp_path / "data")
    os.environ["XDG_CONFIG_HOME"] = str(tmp_path / "config")
    w = MainWindow(simulator_only=True)
    w.show()
    for _ in range(20):
        qapp.processEvents()
    w.timer.stop()  # tests drive _tick themselves
    yield w
    w.close()
    qapp.processEvents()


def _mosaic_10x() -> Mosaic:
    """Six 320 x 240 views of a random texture, panned along x: a mosaic of 3 keyframes."""
    rng = np.random.default_rng(1)
    world = cv2.GaussianBlur(rng.integers(0, 256, (600, 1400)).astype(np.float32), (0, 0), 1.2)
    m = Mosaic()
    for cx in range(100, 500, 20):
        m.track(world[200:440, cx:cx + 320].copy())
    assert len(m.keyframes) >= 3
    return m


def _draw_mosaic_tab(w):
    w.tabs.setCurrentIndex(1)
    w.last_mosaic_version = -1
    w._tick()


@pytest.mark.parametrize("recorded", [True, False], ids=["scale-in-state", "older-file"])
def test_an_opened_mosaic_is_drawn_at_its_own_scale_whatever_objective_is_selected(window, qapp, tmp_path,
                                                                                    monkeypatch, recorded):
    """A session mapped at 10x (0.9 µm per preview px) is opened while 40x is selected. The restored pose graph has
    more than one keyframe, so the window drew it at the 40x preview scale: µm axes, span label and exported height
    mosaic 4x too small, while 40x frames were tracked into it. It must be drawn at its own scale (recorded in the
    mosaic state; for an older file, the session's preview scale), and the label must say it is not tracked."""
    w = window
    w.cmb_obj.setCurrentText("40x / 0.85")
    m = _mosaic_10x()
    if recorded:  # saved by this version: the state says what it is; the file's preview scale is another (40x) one
        m.objective, m.um_per_px = "10x / 0.25", 0.9
        settings, um_file = {"objective": "40x / 0.85"}, 0.225
    else:  # 0.10.x: the mosaic was made with the objective selected when saving, at the file's preview scale
        settings, um_file = {"objective": "10x / 0.25"}, 0.9
    p = tmp_path / "s.npz"
    save_session(p, None, settings, {"mosaic": m, "um_preview": um_file})
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *a, **k: (str(p), ""))
    w._load_session()
    assert len(w.engine.mosaic.keyframes) == len(m.keyframes)
    assert _wait_for(qapp, lambda: w.engine.snapshot().mosaic_hold is not None)
    _draw_mosaic_tab(w)
    (img, _origin, d), *_ = w.engine.mosaic_image()
    assert w.p_mosaic.um_per_px / d == pytest.approx(0.9, rel=1e-9)
    assert f"{img.shape[1] * d * 0.9:.0f} × {img.shape[0] * d * 0.9:.0f} µm" in w.lbl_mosaic.text()
    assert "not tracking" in w.lbl_mosaic.text() and "10x / 0.25" in w.lbl_mosaic.text()
    assert len(w.engine.mosaic.keyframes) == len(m.keyframes)  # the live 40x frames were not tracked into it


def _wait_mapping(qapp, w):
    """The simulator is at rest: wait until the engine has a keyframe and is tracking into its mosaic."""
    with w.sim_source.lock:
        w.sim_source.script = []
    assert _wait_for(qapp, lambda: w.engine.snapshot().preview is not None and len(w.engine.mosaic.keyframes) >= 1
                     and w.engine.snapshot().mosaic_hold is None and not w.engine.snapshot().mosaic_lost)


def test_an_inset_is_drawn_in_the_mosaic_tab(window, qapp):
    """'Add inset (current view)' stored an inset nothing drew: the Mosaic tab was pixel-identical. A bright inset
    placed 1000 preview px to the left of the mapped area must appear in the displayed image, there."""
    w = window
    _wait_mapping(qapp, w)
    _draw_mosaic_tab(w)
    before = w.p_mosaic.data.copy()
    s = w.engine.snapshot()
    um = w.engine.mosaic.um_per_px
    (_img, origin, d0), *_ = w.engine.mosaic_image()
    w.engine.add_mosaic_inset("other", um, (origin[0] - 1000) * um, origin[1] * um, np.full_like(s.preview, 255))
    w._mosaic_add_inset()  # and the button's own inset, at the live view
    w.engine._render_now()
    _draw_mosaic_tab(w)
    shown = w.p_mosaic.data
    (_img2, origin2, d), *_ = w.engine.mosaic_image()
    assert shown.shape[1] * d >= before.shape[1] * d0 + 1000 and origin2[0] <= origin[0] - 1000
    x = (origin[0] - 1000 + s.preview.shape[1] // 2 - origin2[0]) // d  # inside the bright inset
    y = (origin[1] + s.preview.shape[0] // 2 - origin2[1]) // d
    assert tuple(shown[y, x]) == (255, 255, 255, 255)


def test_an_objective_change_keeps_the_insets_and_the_overview(window, qapp):
    """The objective handler called engine.set_objective (which added the old objective's overview as an inset)
    and then engine.reset_mosaic() without keep_insets, which deleted it and every other inset whatever 'Keep insets
    on reset' said. With the box ticked the user's inset and the overview of the old mosaic must survive."""
    w = window
    _wait_mapping(qapp, w)
    w._mosaic_add_inset()
    old, gen = w.obj, w.engine.mosaic_generation
    assert w.chk_keep_insets.isChecked() and w.engine.keep_overview_insets
    other = next(n for n in (w.cmb_obj.itemText(i) for i in range(w.cmb_obj.count()))
                 if w.objset.get(n).magnification != old.magnification)
    w.cmb_obj.setCurrentText(other)
    assert w.engine.mosaic_generation > gen and w.engine.mosaic.objective == other  # restarted for it
    assert sorted(it.objective for it in w.engine.mosaic.insets) == [old.name, old.name]  # the inset + the overview
