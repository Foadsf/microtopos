"""Defects found in the v0.9.0 -> v0.10.1 review (PR #14 code), each through the control a user operates.

Runs offscreen with isolated settings (conftest.py), on the simulator or a short recording; no camera, no stage.
"""

import os
import threading
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pyqtgraph as pg
import pytest
from PySide6.QtCore import QStandardPaths
from PySide6.QtWidgets import QApplication, QCheckBox, QSlider

from microtopos.camera import FrameSource, ReplaySource
from microtopos.gui import main_window
from microtopos.gui.main_window import MainWindow
from microtopos.recording import RecordingWriter
from microtopos.stage import StagePose


@pytest.fixture(scope="module")
def qapp():
    QStandardPaths.setTestModeEnabled(True)
    app = QApplication.instance() or QApplication([])
    pg.setConfigOptions(imageAxisOrder="row-major")
    yield app


def _pump(app, n=25):
    for _ in range(n):
        app.processEvents()


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
    _pump(qapp, 20)
    w.timer.stop()  # tests drive _tick themselves
    yield w
    w.close()
    qapp.processEvents()


def _wait_preview(app, w, cond=lambda s: True, timeout=20.0):
    assert _wait_for(app, lambda: w.engine.snapshot().preview is not None and cond(w.engine.snapshot()), timeout)
    return w.engine.snapshot()


# ---------------------------------------------------------------- stage labels
class _Stage:
    """A connected stage that never moves: Z 1234.5 µm."""

    def pose(self):
        return StagePose(0.0, 0.0, 1234.5, False, True)


class _Pub(FrameSource):
    def _run(self):
        pass


def test_attached_stage_labels_frames_only_during_a_motorised_sweep():
    """With a stage attached, hand-focused (or replayed) frames keep no label, so their Z is the frame index; a
    constant stage Z on every frame left one usable frame per sweep and reconstruction failed."""
    src = _Pub()
    src.stage_driver = _Stage()
    src._publish(np.zeros((4, 4), np.uint8))
    assert src.latest()[3] is None
    src.label_with_stage = True  # what stage.capture_z_sweep sets while it drives the stage
    src._publish(np.zeros((4, 4), np.uint8))
    assert src.latest()[3].focus_z_um == 1234.5


# ---------------------------------------------------------------- mosaic: inset, lock, session controls
def test_add_inset_places_the_live_view_at_its_mosaic_position(window, qapp):
    w = window
    s = _wait_preview(qapp, w)
    w._mosaic_add_inset()
    ins = w.engine.mosaic.insets[-1]
    um = w.obj.nominal_um_per_px(s.preview.shape[1])  # simulator: nominal preview scale
    assert ins.um_per_px == pytest.approx(um, rel=1e-12)
    assert np.asarray(ins.image).shape == s.preview.shape
    mx, my = w.engine.snapshot().mosaic_xy  # the simulator stage is at rest: the live position does not change
    assert (ins.x_um, ins.y_um) == pytest.approx((mx * um, my * um), abs=2 * um)


def test_blend_mode_change_waits_for_the_engine_lock(window, qapp):
    """The GUI must not rebuild the mosaic canvas while the engine thread splats into it."""
    w = window
    done = threading.Event()
    with w.engine._lock:
        t = threading.Thread(target=lambda: (w._mosaic_blend("multiband"), done.set()))
        t.start()
        time.sleep(0.3)
        assert not done.is_set() and w.engine.mosaic.blending == "feather"
    t.join(5)
    assert done.is_set() and w.engine.mosaic.blending == "multiband"


def test_opened_session_shows_its_blend_mode_and_rotation_tracking(window, qapp):
    w = window
    w.engine.mosaic.set_blending("multiband")
    w.engine.mosaic.track_rotation = True
    meta, arrays = w.engine.mosaic.to_state_arrays()
    w.engine.mosaic.set_blending("feather")
    w.engine.mosaic.track_rotation = False
    w.cmb_mblend.setCurrentText("feather")
    w.chk_rot_track.setChecked(False)
    w.engine.load_mosaic_state(meta, arrays)
    w._sync_mosaic_controls()  # what Open session calls after loading the mosaic
    assert w.cmb_mblend.currentText() == "multiband" and w.chk_rot_track.isChecked()


# ---------------------------------------------------------------- super-resolution scale
@pytest.mark.parametrize("k", [2, 3, 4])
def test_super_resolution_pixel_size_follows_the_chosen_scale(window, qapp, k):
    w = window
    w.cmb_sr_scale.setCurrentText(f"{k}×")
    frames = [np.full((360, 640), 100, np.float32)] * 8
    w.engine.dwell_frames = lambda: frames
    captured = {}

    def fake_start(job, on_done, main=False):
        captured["cb"] = on_done
        return True

    w._start_job = fake_start
    w._run_sr()
    fake = {"image": np.zeros((360 * k, 360 * k), np.float32), "frames_used": 8, "frames_rejected": 0,
            "subpixel_bins": 4, "rl_iterations": 10, "psf_sigma_lr": 1.0, "psf_source": "test"}
    captured["cb"](fake)
    um_full = w.obj.nominal_um_per_px(640)  # simulator: nominal scale of the captured frame
    assert w.sr_result["um_per_px"] == pytest.approx(um_full / k, rel=1e-12)
    assert f"{k}× linear, {k * k}× pixels" in w.lbl_job.text()


# ---------------------------------------------------------------- live colour, dead controls
def test_colour_preview_shows_a_colour_live_image(window, qapp):
    w = window
    w.chk_rgb.setChecked(True)
    s = _wait_preview(qapp, w, lambda s: s.preview_rgb is not None)
    w._tick()
    assert w.p_live.data.ndim == 3 and w.p_live.data.shape[:2] == s.preview.shape
    w.chk_rgb.setChecked(False)
    _wait_for(qapp, lambda: w.engine.snapshot().preview_rgb is None)
    w._tick()
    assert w.p_live.data.ndim == 2


def test_no_control_promises_a_camera_bit_depth_that_is_never_requested(window):
    labels = [c.text() for c in window.findChildren(QCheckBox)]
    assert not any("bit depth" in t.lower() for t in labels)


def test_serial_ports_can_be_rescanned_after_start(window, qapp, monkeypatch):
    w = window
    monkeypatch.setattr(main_window, "list_serial_ports", lambda: ["/dev/ttyACM7"])
    w.cmb_stg.setCurrentText("GRBL serial port")
    assert "/dev/ttyACM7" in [w.cmb_stg_port.itemText(i) for i in range(w.cmb_stg_port.count())]
    monkeypatch.setattr(main_window, "list_serial_ports", lambda: ["/dev/ttyACM7", "/dev/ttyUSB3"])
    w.btn_stg_rescan.click()
    assert [w.cmb_stg_port.itemText(i) for i in range(w.cmb_stg_port.count())] == ["/dev/ttyACM7", "/dev/ttyUSB3"]


# ---------------------------------------------------------------- height tools across results
def test_a_new_result_clears_level_points_picked_on_the_previous_map(window, qapp):
    w = window
    w.level_picking = True
    w.level_pts = [(10.0, 10.0)]
    w.btn_level3.setChecked(True)
    w.mask_roi.show()
    r = {"height": np.zeros((20, 20), np.float32), "valid_fraction": 1.0, "warnings": [], "objective": w.obj.name,
         "label": "t", "um_per_px": 1.0, "um_per_px_capture": 1.0, "timings": {"total": 1.0}}
    w._show_result = lambda: None  # only the tool state is under test here
    w._pin = lambda r: None
    w._remember = lambda r: None
    w._on_result(r)
    assert w.level_pts == [] and not w.btn_level3.isChecked() and not w.mask_roi.isVisible()


# ---------------------------------------------------------------- replay transport
def _recording(tmp_path, n):
    d = tmp_path / "rec"
    d.mkdir()
    wr = RecordingWriter(d, "sim")
    for i in range(n):
        wr.add(np.full((16, 16, 3), (7 * i) % 256, np.uint8), seq=i, t=0.1 * i)
    wr.close()
    return d


@pytest.fixture
def replay_window(qapp, tmp_path):
    os.environ["XDG_DATA_HOME"] = str(tmp_path / "data")
    os.environ["XDG_CONFIG_HOME"] = str(tmp_path / "config")
    made = []

    def make(n):
        w = MainWindow(simulator_only=False, replay=str(_recording(tmp_path, n)))
        w.show()
        _pump(qapp)
        made.append(w)
        return w

    yield make
    for w in made:
        w.close()
    qapp.processEvents()


def test_seek_keeps_playing_and_the_button_says_so(replay_window, qapp):
    w = replay_window(400)  # 40 s at 1x: playback alone cannot reach frame 350 within these waits
    src: ReplaySource = w.source
    w.replay_seek.setValue(150)
    w.replay_seek.sliderReleased.emit()  # end of a drag
    assert _wait_for(qapp, lambda: src.position >= 150, timeout=5)
    w._tick()
    assert not src.paused and w.btn_replay_pause.text() == "‖ Pause"  # still playing, and the button says so
    w.replay_seek.setValue(350)  # a click on the track or an arrow key: no drag, no sliderReleased
    assert _wait_for(qapp, lambda: src.position >= 350, timeout=5)
    w.btn_replay_pause.click()  # one click pauses
    assert src.paused and w.btn_replay_pause.text() == "▶ Play"


def test_end_of_recording_reads_play_and_one_click_restarts(replay_window, qapp):
    w = replay_window(6)
    src: ReplaySource = w.source
    assert _wait_for(qapp, lambda: src.done)
    w._tick()
    assert w.btn_replay_pause.text() == "▶ Play"
    w.btn_replay_pause.click()  # one click plays again from the first frame
    assert not src.paused and not src.done
    assert _wait_for(qapp, lambda: src.position < 5 or src.done)


def test_a_recreated_player_keeps_speed_and_loop_and_plays(replay_window, qapp):
    w = replay_window(400)
    w.cmb_replay_speed.setCurrentText("2×")
    w.chk_replay_loop.setChecked(True)
    w.btn_replay_pause.click()  # paused
    w._on_source(w.cmb_source.currentIndex())  # the player is rebuilt (source re-selected)
    src: ReplaySource = w.source
    assert src.speed == 2.0 and src.loop is True and not src.paused
    w._tick()
    assert w.btn_replay_pause.text() == "‖ Pause"
    assert not w.cmb_res.isVisible()  # a capture size means nothing to a replay


def test_seek_slider_follows_steps_while_paused_and_seeks_from_there(replay_window, qapp):
    """Step +1 pauses; the slider must still show each stepped frame, so a later arrow key (single step on the
    slider) seeks from the frame on screen, not from where the replay was paused."""
    w = replay_window(400)
    src: ReplaySource = w.source
    assert _wait_for(qapp, lambda: src.position >= 5, timeout=10)
    w.btn_replay_pause.click()
    _pump(qapp, 5)
    paused_at = src.position
    w._tick()
    for k in range(1, 7):
        w.btn_replay_step_f.click()
        assert _wait_for(qapp, lambda k=k: src.position == paused_at + k, timeout=5)
        w._tick()
        assert w.replay_seek.value() == paused_at + k
    w.replay_seek.triggerAction(QSlider.SliderAction.SliderSingleStepAdd)  # what the Right arrow key does
    assert _wait_for(qapp, lambda: src.position == paused_at + 7, timeout=5), (paused_at, src.position)
