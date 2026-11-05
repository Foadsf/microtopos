"""Regenerate the documentation screenshots by driving the real GUI on the built-in simulator.

    uv run python docs/tools/screenshots.py [--out docs/modules/ROOT/images] [--platform offscreen|xcb]

Every screenshot is of the actual application; the specimen is the synthetic one, so the images are labelled
"simulator" in the manual. Settings and calibration go to a temporary folder, never to the user's own.
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
import time
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "images"))
    ap.add_argument("--platform", default="offscreen")
    ap.add_argument("--size", default="1600x980")
    ap.add_argument("--only", nargs="*", help="save only these shots (others are not even grabbed)")
    a = ap.parse_args()
    os.environ["QT_QPA_PLATFORM"] = a.platform
    tmp = tempfile.mkdtemp(prefix="microtopos-shots-")
    os.environ["XDG_DATA_HOME"] = tmp
    os.environ["XDG_CONFIG_HOME"] = tmp
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    import pyqtgraph as pg
    from PySide6.QtWidgets import QApplication

    pg.setConfigOptions(imageAxisOrder="row-major", antialias=True)
    app = QApplication.instance() or QApplication(sys.argv)
    from microtopos.gui.main_window import MainWindow
    from microtopos.gui.smoke import pump, wait_frames

    w = MainWindow(simulator_only=True)
    W, H = (int(v) for v in a.size.split("x"))
    w.resize(W, H)
    w.show()
    wait_frames(app, w)
    pump(app, 2)

    def trimmed(img):
        """Cut the empty panel background below the last row with content (keeps a 14 px margin)."""
        bg = img.pixel(img.width() - 2, img.height() - 2)
        y = img.height() - 1
        while y > 40 and all(img.pixel(x, y) == bg for x in range(0, img.width(), 3)):
            y -= 1
        return img.copy(0, 0, img.width(), min(img.height(), y + 15))

    def shot(name, widget=None, trim=False):
        if a.only and name not in a.only:
            return
        pump(app, 0.6)
        img = (widget or w).grab().toImage()
        (trimmed(img) if trim else img).save(str(out / f"{name}.png"))
        print("saved", name)

    def dock_of(widget):
        p = widget
        while p is not None and p.metaObject().className() != "QDockWidget":
            p = p.parentWidget()
        return p

    def wait_state(name, timeout=30):
        t0 = time.time()
        while time.time() - t0 < timeout:
            pump(app, 0.1)
            if w.engine.snapshot().state.value == name:
                return True
        return False

    def wait_result(prev, timeout=120):
        t0 = time.time()
        while time.time() - t0 < timeout:
            pump(app, 0.1)
            if w.result is not None and w.result is not prev:
                return True
        return False

    session = dock_of(w.stack)
    telemetry = dock_of(w.coverage)
    # first start: the checklist and the coach line
    w.tabs.setCurrentIndex(0)
    shot("overview-start")
    for k, name in enumerate(["camera", "objective", "darkflat", "lateral", "zscale", "inspect"]):
        w._open_step(k)
        shot(f"step-{name}", session, trim=True)
        if name == "lateral":
            w.tabs.setCurrentIndex(0)
            shot("live-lateral")  # the blue calibration line on the Live view
    w._open_step(5)

    # a focus sweep, caught half-way, then its result
    w.chk_focus.setChecked(True)
    w._demo("sweep")
    wait_state("Z_SWEEP")
    t0 = time.time()
    while time.time() - t0 < 30 and w.engine.snapshot().sweep_completeness < 0.3:
        pump(app, 0.2)
    shot("live-sweep")
    shot("telemetry-sweep", telemetry, trim=True)
    prev = w.result
    for _attempt in range(3):  # grabbing the window mid-sweep can starve a busy computer: sweep again, no grabs
        if wait_result(prev, timeout=40):
            break
        w._demo("sweep")
    if w.result is None:
        raise SystemExit("the demo sweep was never reconstructed; close other programs and run again")
    w.chk_focus.setChecked(False)
    pump(app, 2)
    for i, name in ((2, "tab-edf"), (3, "tab-height"), (5, "tab-report")):
        w.tabs.setCurrentIndex(i)
        shot(name)
        shot(name.replace("tab-", "view-"), w.tabs)  # the view alone, for cards
    w.tabs.setCurrentIndex(3)
    w.cmb_height.setCurrentIndex(2)  # confidence
    shot("tab-height-confidence")
    w.cmb_height.setCurrentIndex(0)
    w.chk_grains.setChecked(True)
    w.tabs.setCurrentIndex(2)
    shot("tab-edf-grains")
    w.chk_grains.setChecked(False)
    w.tabs.setCurrentIndex(4)
    pump(app, 2)
    shot("tab-3d")

    # mapping: pan, then the mosaic with the height map pinned in place
    w._demo("pan")
    pump(app, 3)
    w.tabs.setCurrentIndex(0)
    shot("live-pan")
    w.tabs.setCurrentIndex(1)
    pump(app, 14)
    w.cmb_mview.setCurrentIndex(1)
    w.sl_hop.setValue(65)
    shot("tab-mosaic-overlay")
    shot("view-mosaic-overlay", w.tabs)
    w.cmb_mview.setCurrentIndex(0)
    shot("tab-mosaic")
    shot("view-mosaic", w.tabs)

    # rotated views (camera turned in the eyepiece)
    w.sp_rot.setValue(15.0)
    w.tabs.setCurrentIndex(0)
    w._open_step(0)
    shot("live-rotated")
    w.sp_rot.setValue(0.0)

    # light theme, for the landing page
    w._toggle_theme()
    w.tabs.setCurrentIndex(3)
    w._open_step(5)
    shot("overview-light")
    w._toggle_theme()
    w.tabs.setCurrentIndex(3)
    shot("overview-dark")
    w.close()
    pump(app, 0.5)


if __name__ == "__main__":
    main()
