"""Offscreen GUI smoke test: drives the simulator through pan, sweep, reversal, super-resolution, theme switch,
metrology recompute and exports, and fails on any uncaught exception (including ones raised in Qt slots)."""

from __future__ import annotations

import os
import sys
import tempfile
import time
import traceback
from pathlib import Path


def run_gui_smoke(seconds: float = 60.0, out=print) -> bool:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    tmp = tempfile.mkdtemp(prefix="microtopos-smoke-")
    os.environ["XDG_DATA_HOME"] = tmp  # never touch the user's real calibration
    os.environ["XDG_CONFIG_HOME"] = tmp
    import pyqtgraph as pg
    from PySide6.QtWidgets import QApplication

    from .. import export
    from .main_window import MainWindow

    errors: list[str] = []
    old_hook = sys.excepthook
    sys.excepthook = lambda t, v, tb: errors.append("".join(traceback.format_exception(t, v, tb)))
    pg.setConfigOptions(imageAxisOrder="row-major")
    app = QApplication.instance() or QApplication([])
    w = MainWindow(simulator_only=True)
    w.show()
    log: list[str] = []
    results: list[dict] = []
    w.job_done_hook = results

    def wait_result(n, timeout=90):
        t0 = time.time()
        while time.time() - t0 < timeout:
            app.processEvents()
            time.sleep(0.02)
            if w.result is not None and id(w.result) not in [id(x) for x in results]:
                results.append(w.result)
            if len(results) >= n:
                return True
        return False

    def step(name, fn):
        try:
            fn()
            log.append(f"ok    {name}")
        except Exception:
            errors.append(f"{name}: {traceback.format_exc()}")
            log.append(f"ERROR {name}")

    t_start = time.time()
    step("wait for frames", lambda: wait_frames(app, w))
    step("demo sweep", lambda: w._demo("sweep"))
    got1 = wait_result(1)
    log.append(f"{'ok' if got1 else 'FAIL'}    sweep reconstructed automatically")
    step("tabs", lambda: [w.tabs.setCurrentIndex(i) or app.processEvents() for i in range(6)])
    step("height modes", lambda: [w.cmb_height.setCurrentIndex(i) for i in (2, 1, 0)])
    step("grain overlay", lambda: w.chk_grains.setChecked(True))
    step("recompute metrology", w._recompute)
    step("demo pan", lambda: (w.tabs.setCurrentIndex(1), w._demo("pan")))
    pump(app, 6)
    step("2.5D mosaic toggle", lambda: w.chk_hmosaic.setChecked(True))
    pump(app, 1)
    step("sweep with reversal", lambda: w._demo("rev"))
    got2 = wait_result(2)
    log.append(f"{'ok' if got2 else 'FAIL'}    reversal sweep reconstructed")
    pump(app, 2)
    step("super-resolution", w._run_sr)
    pump(app, 8)
    step("theme toggle twice", lambda: (w._toggle_theme(), w._toggle_theme()))
    step("objective change", lambda: w.cmb_obj.setCurrentText("40x / 0.85"))
    pump(app, 2)
    if w.result is not None:
        d = Path(tmp)
        r = w.result
        step("export gsf", lambda: export.write_gwyddion_gsf(d / "h.gsf", r["sl_surface"], r["um_per_px"], True))
        step("export stl", lambda: export.write_binary_stl(d / "h.stl", r["sl_surface"], r["um_per_px"]))
        step("export tiff", lambda: export.write_float32_tiff(d / "h.tif", r["sl_surface"], r["um_per_px"]))
        step("export report", lambda: export.write_report_bundle(d / "rep", r, {}))
        step("save+load session", lambda: export.load_session(_save(export, d, r)))
    st = w.store.get(w.obj)
    contaminated = st.z_um_per_frame is not None or st.lateral_um_per_px_full is not None
    log.append(f"{'FAIL' if contaminated else 'ok'}    simulator did not write calibration")
    w.close()
    pump(app, 0.5)
    sys.excepthook = old_hook
    ok = got1 and got2 and not errors and not contaminated
    for line in log:
        out("  " + line)
    for e in errors[:5]:
        out("  ---\n" + e[-1500:])
    out(f"GUI smoke test: {'PASS' if ok else 'FAIL'} ({time.time() - t_start:.0f} s, {len(errors)} exception(s))")
    return ok


def _save(export, d, r):
    export.save_session(d / "s.npz", r, {})
    return d / "s.npz"


def pump(app, seconds):
    t0 = time.time()
    while time.time() - t0 < seconds:
        app.processEvents()
        time.sleep(0.02)


def wait_frames(app, w, timeout=10):
    t0 = time.time()
    while time.time() - t0 < timeout:
        app.processEvents()
        if w.engine.snapshot().preview is not None:
            return
        time.sleep(0.02)
    raise RuntimeError("no frames from the simulator")


if __name__ == "__main__":
    sys.exit(0 if run_gui_smoke() else 1)
