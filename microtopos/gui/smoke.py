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
    step("mosaic height overlay", lambda: (w.cmb_mview.setCurrentIndex(1), w.sl_hop.setValue(70)))
    pump(app, 1)
    step("mosaic height only", lambda: w.cmb_mview.setCurrentIndex(2))
    pump(app, 1)
    step("height history", lambda: w.cmb_hist.setCurrentIndex(w.cmb_hist.count() - 1))
    pump(app, 1)
    step("sweep with reversal", lambda: w._demo("rev"))
    got2 = wait_result(2)
    log.append(f"{'ok' if got2 else 'FAIL'}    reversal sweep reconstructed")
    pump(app, 2)

    def rotate():
        w.sp_rot.setValue(30.0)
        app.processEvents()
        assert abs(w.p_mosaic.group.rotation() - 30.0) < 1e-6 and abs(w.p_live.group.rotation() - 30.0) < 1e-6
        # the cursor readout must still report unrotated image coordinates
        pt = w.p_live.group.mapToScene(w.p_live.img.mapToParent(w.p_live.img.boundingRect().center()))
        back = w.p_live.group.mapFromScene(pt)
        c = w.p_live.img.mapToParent(w.p_live.img.boundingRect().center())
        assert abs(back.x() - c.x()) < 1e-6 and abs(back.y() - c.y()) < 1e-6
        w.sp_rot.setValue(0.0)

    step("rotate views", rotate)
    n_before = len(results)
    step("sweep kept, not reconstructed", lambda: (w.chk_autorec.setChecked(False), w._demo("sweep")))
    cap_before, t0 = getattr(w, "last_cap", None), time.time()
    while getattr(w, "last_cap", None) is cap_before and time.time() - t0 < 60:
        pump(app, 0.5)
    pump(app, 2)
    log.append(f"{'ok' if len(results) == n_before else 'FAIL'}    auto-reconstruct off: no automatic map")
    step("reconstruct last sweep", w._reconstruct_last)
    got3 = wait_result(n_before + 1)
    log.append(f"{'ok' if got3 else 'FAIL'}    manual reconstruction of the last sweep")
    w.chk_autorec.setChecked(True)

    def profile_confidence():
        w.tabs.setCurrentIndex(3)
        w.cmb_height.setCurrentIndex(2)
        app.processEvents()
        assert w.prof_label.text().startswith("Confidence") or "No confidence" in w.prof_label.text(), \
            w.prof_label.text()
        w.cmb_height.setCurrentIndex(0)

    step("profile follows the confidence view", profile_confidence)

    def panels():
        w.btn_left.click()
        app.processEvents()
        assert not w.dock_left.isVisible()
        w._reset_layout()
        app.processEvents()
        assert w.dock_left.isVisible() and w.dock_right.isVisible()
        assert w.card.isVisible() and "RESULT" not in w.lbl_card.text() and w.lbl_card.text()

    step("panels hide and reset; result card shown", panels)
    sr_before = getattr(w, "sr_count", 0)
    w.tabs.setCurrentIndex(0)
    if w.sim_source is not None:  # hand tremor gives the sub-pixel motion super-resolution needs
        with w.sim_source.lock:
            w.sim_source.sim.tremor_px = 0.6
    t0 = time.time()
    while len(w.engine.dwell_frames()) < 8 and time.time() - t0 < 30:  # still frames for the dwell buffer
        pump(app, 0.2)
    log.append(f"{'ok' if len(w.engine.dwell_frames()) >= 8 else 'FAIL'}    view still for super-resolution")
    step("super-resolution", w._run_sr)
    t0 = time.time()
    while getattr(w, "sr_count", 0) == sr_before and time.time() - t0 < 30:
        pump(app, 0.2)
    log.append(f"{'ok' if getattr(w, 'sr_count', 0) > sr_before else 'FAIL'}    super-resolution produced a new image"
               f" ({w.lbl_job.text()[:120]})")
    step("theme toggle twice", lambda: (w._toggle_theme(), w._toggle_theme()))
    step("objective change", lambda: w.cmb_obj.setCurrentText("40x / 0.85"))
    pump(app, 1)

    def objective_editor():
        from PySide6.QtWidgets import QDialogButtonBox

        from .. import objectives as objlib

        dlg = w._edit_objectives()
        app.processEvents()
        assert dlg.isVisible()
        dlg.table.item(0, 2).setText("1.4")  # a dry objective with NA 1.4: refused, OK disabled
        assert not dlg.buttons.button(QDialogButtonBox.Ok).isEnabled(), dlg.lbl_problems.text()
        dlg.apply_preset(objlib.LEITZ_NPL_PRESET)
        assert dlg.buttons.button(QDialogButtonBox.Ok).isEnabled(), dlg.lbl_problems.text()
        dlg.accept()
        app.processEvents()
        names = [o.name for o in objlib.PRESETS[objlib.LEITZ_NPL_PRESET]]
        assert [w.cmb_obj.itemText(i) for i in range(w.cmb_obj.count())] == names
        assert "keys 1–4" in w.lbl_obj_title.text() and objlib.load_set(w.data_dir).names == names
        w._select_objective_index(3)  # key 4
        app.processEvents()
        assert w.obj.name == "NPL Fluotar 50x / 0.85" and w.engine.objective.na == 0.85, w.obj
        w._select_objective_index(0)  # key 1
        app.processEvents()
        assert w.obj.name == "NPL 5x / 0.09" and "5×" in w.step_chips[1].text()
        dlg = w._edit_objectives()  # back to the built-in list for the remaining steps
        dlg.apply_preset(objlib.DEFAULT_PRESET)
        dlg.accept()
        app.processEvents()
        assert w.cmb_obj.findText("40x / 0.85") >= 0

    step("objective editor: validation, Leitz preset, keys switch objectives", objective_editor)
    step("objective change back", lambda: w.cmb_obj.setCurrentText("40x / 0.85"))
    pump(app, 2)
    if w.result is not None:
        d = Path(tmp)
        r = w.result
        step("export gsf", lambda: export.write_gwyddion_gsf(d / "h.gsf", r["sl_surface"], r["um_per_px"], True))
        step("export stl", lambda: export.write_binary_stl(d / "h.stl", r["sl_surface"], r["um_per_px"]))
        step("export tiff", lambda: export.write_float32_tiff(d / "h.tif", r["sl_surface"], r["um_per_px"]))
        step("export report", lambda: export.write_report_bundle(d / "rep", r, {}))
        step("save+load session", lambda: export.load_session(_save(export, d, r)))

        def mosaic_only():
            (img, origin), parts = w.engine.mosaic_full()
            export.save_session(d / "m.npz", None, {}, {"mosaic_image": img, "mosaic_origin": list(origin)})
            m, _ = export.load_session(d / "m.npz")
            assert "mosaic_image" in m and "height" not in m

        step("mosaic-only session", mosaic_only)
    st = w.store.get(w.obj)
    contaminated = st.z_um_per_frame is not None or st.lateral_um_per_px_full is not None
    log.append(f"{'FAIL' if contaminated else 'ok'}    simulator did not write calibration")
    # close while a reconstruction is running: the window must cancel and wait for it, not leave it orphaned
    while w.job is not None and w.job.isRunning():
        pump(app, 0.2)
    before = w.job
    w._reconstruct_last()
    running = w.job if w.job is not before else None
    started = running is not None and running.isRunning()
    pump(app, 0.3)
    w.close()
    pump(app, 0.5)
    log.append(f"{'ok' if started and not running.isRunning() else 'FAIL'}    close during a running "
               f"reconstruction cancelled and joined it (started: {started})")
    sys.excepthook = old_hook
    ok = got1 and got2 and not errors and not contaminated and not any(x.startswith("FAIL") for x in log)
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
