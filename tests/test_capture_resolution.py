"""The camera opens at full HD by default, the size list is what the device offers, and the user's choice is kept.

Regression guarded here (v0.10.0, PR #14): the size list started with "640x480", index 0 was the default and the
choice was not saved, so every launch asked the camera for 640x480. The Bresser MikrOkular does not offer that size
and its driver silently delivered 1280x720, 2.25x fewer pixels than v0.9, in every Z-stack.

The device is a fake: `v4l2-ctl --list-formats-ext` is answered with the output recorded from a Bresser MikrOkular
Full HD (2026-11-26), and the capture source never opens hardware.
"""

import os
import subprocess
from typing import ClassVar

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pyqtgraph as pg
import pytest
from PySide6.QtCore import QStandardPaths
from PySide6.QtWidgets import QApplication

from microtopos import camera
from microtopos.camera import CameraInfo, CameraSource
from microtopos.gui import main_window

# `v4l2-ctl -d /dev/video4 --list-formats-ext`, Bresser MikrOkular Full HD, 2026-11-26 (intervals trimmed)
MIKROKULAR_FORMATS = """ioctl: VIDIOC_ENUM_FMT
\tType: Video Capture

\t[0]: 'MJPG' (Motion-JPEG, compressed)
\t\tSize: Discrete 1920x1080
\t\t\tInterval: Discrete 0.040s (25.000 fps)
\t\tSize: Discrete 1280x1024
\t\t\tInterval: Discrete 0.033s (30.000 fps)
\t\tSize: Discrete 1280x720
\t\t\tInterval: Discrete 0.033s (30.000 fps)
"""

# A webcam that offers YUYV at more sizes than MJPG: only the MJPG sizes count, because MJPG is what is requested.
WEBCAM_FORMATS = """ioctl: VIDIOC_ENUM_FMT
\tType: Video Capture

\t[0]: 'YUYV' (YUYV 4:2:2)
\t\tSize: Discrete 640x480
\t\tSize: Discrete 320x240
\t[1]: 'MJPG' (Motion-JPEG, compressed)
\t\tSize: Discrete 640x480
\t\tSize: Discrete 1280x720
"""


def _fake_v4l2(monkeypatch, formats: str):
    def run(cmd, *a, **k):
        out = formats if "--list-formats-ext" in cmd else ""
        return subprocess.CompletedProcess(cmd, 0, stdout=out, stderr="")

    monkeypatch.setattr(camera.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(camera.subprocess, "run", run)


def test_modes_are_the_mjpeg_sizes_the_device_lists_largest_first(monkeypatch):
    _fake_v4l2(monkeypatch, MIKROKULAR_FORMATS)
    assert camera.camera_modes(CameraInfo(4, "MikrOkularFullHD", "/dev/video4")) == [
        "1920x1080", "1280x1024", "1280x720"]
    _fake_v4l2(monkeypatch, WEBCAM_FORMATS)
    assert camera.camera_modes(CameraInfo(0, "webcam", "/dev/video0")) == ["1280x720", "640x480"]


def test_a_fourcc_with_a_space_starts_its_own_format(monkeypatch):
    """v4l2-ctl prints the four FourCC characters between the quotes, spaces included ('Y16 '): those sizes are
    Y16's, not MJPG's (crediting them to MJPG offered 1920x1080 on a camera whose MJPG stops at 1280x720)."""
    _fake_v4l2(monkeypatch, """ioctl: VIDIOC_ENUM_FMT
\tType: Video Capture

\t[0]: 'MJPG' (Motion-JPEG, compressed)
\t\tSize: Discrete 1280x720
\t\tSize: Discrete 640x480
\t[1]: 'Y16 ' (16-bit Greyscale)
\t\tSize: Discrete 1920x1080
""")
    modes = camera.camera_modes(CameraInfo(0, "cam", "/dev/video0"))
    assert modes == ["1280x720", "640x480"]
    assert camera.pick_resolution(modes, None) == "1280x720"


def test_modes_are_read_from_the_older_v4l_utils_layout(monkeypatch):
    """Older v4l-utils print `Pixel Format: 'MJPG'` blocks; _v4l2_is_capture accepts that layout, so camera_modes
    reads it too instead of offering the generic list of sizes the device does not have."""
    _fake_v4l2(monkeypatch, """ioctl: VIDIOC_ENUM_FMT
\tIndex       : 0
\tType        : Video Capture
\tPixel Format: 'YUYV'
\tName        : YUYV 4:2:2
\t\tSize: Discrete 640x480
\tIndex       : 1
\tType        : Video Capture
\tPixel Format: 'MJPG' (compressed)
\tName        : Motion-JPEG
\t\tSize: Discrete 1280x720
\t\tSize: Discrete 640x480
""")
    assert camera.camera_modes(CameraInfo(0, "cam", "/dev/video0")) == ["1280x720", "640x480"]


def test_modes_fall_back_to_the_generic_list_when_the_device_cannot_be_asked(monkeypatch):
    monkeypatch.setattr(camera.shutil, "which", lambda name: None)
    modes = camera.camera_modes(CameraInfo(0, "cam", "/dev/video0"))
    assert modes[0] == "1920x1080" and modes == list(camera.SUPPORTED_RESOLUTIONS)


def test_pick_resolution_prefers_the_saved_size_then_full_hd_then_the_largest():
    modes = ["1920x1080", "1280x1024", "1280x720"]
    assert camera.pick_resolution(modes, "1280x720") == "1280x720"
    assert camera.pick_resolution(modes, None) == "1920x1080"
    assert camera.pick_resolution(modes, "640x480") == "1920x1080"  # saved size this camera does not offer
    assert camera.pick_resolution(["2592x1944", "1280x720"], None) == "2592x1944"


class _FakeCamera(CameraSource):
    """CameraSource that records the size it was asked for and never touches a device."""

    requested: ClassVar[list[tuple[int, int]]] = []

    def __init__(self, info, width=1920, height=1080, fps=25):
        super().__init__(info, width, height, fps)
        _FakeCamera.requested.append((width, height))

    def start(self):
        self._running = True

    def stop(self):
        self._running = False

    def exposure_range(self):
        return None

    def set_manual_exposure(self, exposure, sharpness_off=True):
        return {}


@pytest.fixture
def camera_window_factory(monkeypatch, tmp_path):
    QStandardPaths.setTestModeEnabled(True)
    app = QApplication.instance() or QApplication([])
    pg.setConfigOptions(imageAxisOrder="row-major")
    os.environ["XDG_DATA_HOME"] = str(tmp_path / "data")
    os.environ["XDG_CONFIG_HOME"] = str(tmp_path / "config")
    _fake_v4l2(monkeypatch, MIKROKULAR_FORMATS)
    cam = CameraInfo(4, "MikrOkularFullHD: MikrOkularFul", "/dev/video4")
    monkeypatch.setattr(main_window, "list_cameras", lambda: [cam])
    monkeypatch.setattr(main_window, "CameraSource", _FakeCamera)
    _FakeCamera.requested = []
    made = []

    def make():
        w = main_window.MainWindow()
        w.timer.stop()
        for _ in range(5):
            app.processEvents()
        made.append(w)
        return w

    yield make
    for w in made:
        w.close()
    app.processEvents()


def test_camera_opens_at_full_hd_and_lists_only_device_sizes(camera_window_factory):
    w = camera_window_factory()
    assert [w.cmb_res.itemText(i) for i in range(w.cmb_res.count())] == ["1920x1080", "1280x1024", "1280x720"]
    assert w.cmb_res.currentText() == "1920x1080"
    assert _FakeCamera.requested[-1] == (1920, 1080)
    assert (w.source.req[0], w.source.req[1]) == (1920, 1080)


def test_camera_page_fits_without_scrolling_when_no_stage_is_used(camera_window_factory):
    """v0.9.0's Camera page fitted a 1536x864 screen (content 436 px in a 573 px viewport); v0.10 added an always
    open Stage section (750 px of content) that made a camera user scroll past stage controls they do not have."""
    w = camera_window_factory()
    w.resize(1536, 864)
    w.show()
    app = QApplication.instance()
    for _ in range(30):
        app.processEvents()
    assert w.cmb_stg.currentText() == "None" and not w.stage_details.isVisible()
    sc = w.stack.widget(0)  # the Camera page's scroll area
    assert sc.verticalScrollBar().maximum() == 0, (sc.widget().sizeHint().height(), sc.viewport().height())
    w.cmb_stg.setCurrentText("GRBL serial port")  # choosing a driver shows the stage controls
    assert w.stage_details.isVisibleTo(w)


def test_chosen_size_is_used_at_once_and_kept_for_the_next_launch(camera_window_factory):
    w = camera_window_factory()
    w.cmb_res.setCurrentText("1280x720")
    assert (w.source.req[0], w.source.req[1]) == (1280, 720)
    assert w.cmb_res.currentText() == "1280x720"
    w.close()
    w2 = camera_window_factory()
    assert w2.cmb_res.currentText() == "1280x720"
    assert (w2.source.req[0], w2.source.req[1]) == (1280, 720)
