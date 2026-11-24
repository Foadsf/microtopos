"""Feature tests for hardware integration: stage driver, calibration aspect ratios, 16-bit recording, replay transport."""

from __future__ import annotations

import json
import os
import re
import sys
import time

import cv2
import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QSettings, QStandardPaths
from PySide6.QtWidgets import QApplication

from microtopos import objectives, zstack
from microtopos.calibration import CalibrationStore, ObjectiveCalibration, bgr_to_luma, bgr_to_rgb_u8
from microtopos.camera import (
    FrameSource,
    ReplaySource,
    SimulatorSource,
)
from microtopos.gui.main_window import MainWindow
from microtopos.kinematics import Override
from microtopos.live import LiveEngine
from microtopos.recording import RecordingError, RecordingWriter, decode
from microtopos.simulator import SpecimenSimulator
from microtopos.stage import (
    GrblStageDriver,
    SimulatedStageDriver,
    StageError,
    capture_z_sweep,
    list_serial_ports,
    parse_grbl_status,
)

OBJ = objectives.resolve("10x / 0.25", objset=objectives.default_set())


# ------------------------------------------------------------------------------------------------ stage
class FakeGrbl:
    """A stateful model of a Grbl 1.1 controller behind a pyserial-like port (write / readline / flush / close).

    It follows the Grbl v1.1 interface documentation (gnea/grbl wiki, written from the published interface, not
    run against a real board):
      * "Real-time status reports": `?` (no newline) is answered immediately by
        `<State|MPos:x,y,z|FS:f,s>` or `<State|WPos:x,y,z|FS:f,s>`; machine positions are mm with 3 decimals.
        `WCO:x,y,z` (WPos = MPos - WCO) is added to the first report, to the first report after any change of
        the offsets and then every 10th report.
      * "$10 - Status report options": bit 0 selects MPos (1) instead of WPos (0).
      * "Response messages": every G-code line gets `ok` or `error:N` (error:9 = G-code locked out during
        alarm, error:20 = unsupported command, error:22 = feed rate not set, error:2 = bad number format).
      * "Push messages": `ALARM:N` pushed when an alarm is raised (soft limit = ALARM:2); the state is then
        `Alarm` and G-code is rejected with error:9 until `$X`.
      * G92 / G10 L20 P1 set the work offset so the *current* position reads as the given work value (the
        machine position does not change); G53 makes one G0/G1 line use machine coordinates; G4 and G92 wait for
        the motion queue to drain; `G1` returns `ok` as soon as the move is queued, and the state is `Run`
        until the simulated move time (distance / feed) has elapsed.
    Test hooks: `silent` (never answers), `garble` (next N status reports are corrupted), `travel_mm` (soft
    limits), `trigger_alarm()`.
    """

    def __init__(self, report: str = "MPos", wco=(0.0, 0.0, 0.0), start=(0.0, 0.0, 0.0), feed_mm_min=None):
        self.is_open = True
        self.written: list[str] = []
        self.mpos = list(start)
        self.wco = list(wco)
        self.report = report
        self.feed = feed_mm_min  # modal F; None = not yet set (error:22 on a G1)
        self.silent = False
        self.garble = 0
        self.travel_mm: tuple[float, float] | None = None  # (min, max) soft limit for every axis
        self.alarm = False
        self._out: list[bytes] = []
        self._move: tuple[list[float], list[float], float, float] | None = None  # start, end, t0, duration
        self._reports = 0
        self._wco_dirty = True
        self.model = "relative"

    # -- simulated physics
    def _position(self) -> list[float]:
        if self._move is None:
            return self.mpos
        a, b, t0, dur = self._move
        f = 1.0 if dur <= 0 else min(1.0, (time.monotonic() - t0) / dur)
        if f >= 1.0:
            self.mpos, self._move = list(b), None
            return self.mpos
        return [a[i] + (b[i] - a[i]) * f for i in range(3)]

    def _moving(self) -> bool:
        self._position()
        return self._move is not None

    def _settle(self) -> None:
        if self._move is not None:
            self.mpos, self._move = list(self._move[1]), None

    def trigger_alarm(self, code: int = 1) -> None:
        self._settle()
        self.alarm = True
        self._out.append(f"ALARM:{code}\r\n".encode())

    # -- serial API
    def write(self, data: bytes) -> int:
        self.written.append(data.decode("ascii", errors="replace"))
        if self.silent:
            return len(data)
        for ch in data.decode("ascii"):
            if ch == "?":
                self._out.append(self._status_report())
        for line in data.decode("ascii").replace("?", "").splitlines():
            if line.strip():
                self._out.append((self._execute(line.strip()) + "\r\n").encode())
        return len(data)

    def flush(self) -> None:
        pass

    def reset_input_buffer(self) -> None:
        self._out.clear()

    def readline(self) -> bytes:
        if self._out:
            return self._out.pop(0)
        time.sleep(0.002)  # pyserial returns b"" once its read timeout expires
        return b""

    def close(self) -> None:
        self.is_open = False

    # -- protocol
    def _status_report(self) -> bytes:
        self._reports += 1
        pos = self._position()
        state = "Alarm" if self.alarm else ("Run" if self._moving() else "Idle")
        wpos = [pos[i] - self.wco[i] for i in range(3)]
        key, val = ("MPos", pos) if self.report == "MPos" else ("WPos", wpos)
        fields = [state, f"{key}:" + ",".join(f"{v:.3f}" for v in val), "FS:0,0"]
        if self._wco_dirty or self._reports % 10 == 0:
            fields.append("WCO:" + ",".join(f"{v:.3f}" for v in self.wco))
            self._wco_dirty = False
        if self.garble > 0:
            self.garble -= 1
            fields = [state, "MPos:1.0,,3.0", "FS:0,0"]
        return f"<{'|'.join(fields)}>\r\n".encode()

    def _execute(self, line: str) -> str:
        if line == "$X":
            self.alarm = False
            return "ok"
        if line.startswith("$10="):
            self.report = "MPos" if int(line[4:]) & 1 else "WPos"
            self._wco_dirty = True
            return "ok"
        words = re.findall(r"([A-Z])\s*(-?\d+\.?\d*)", line.upper())
        if not words or line.startswith("$"):
            return "error:20"
        g = [float(v) for k, v in words if k == "G"]
        axes = {k: float(v) for k, v in words if k in "XYZ"}
        for k, v in words:
            if k == "F":
                if float(v) <= 0:
                    return "error:22"
                self.feed = float(v)
        if any(k not in "GXYZFPLN" for k, _ in words):
            return "error:20"
        if self.alarm:
            return "error:9"
        if 21.0 in g or (90.0 in g and not any(x in g for x in (0.0, 1.0, 53.0, 92.0))):
            return "ok"
        if 4.0 in g:
            self._settle()
            return "ok"
        if 92.0 in g:
            self._settle()
            for k, v in axes.items():
                i = "XYZ".index(k)
                self.wco[i] = self.mpos[i] - v
            self._wco_dirty = True
            return "ok"
        if 10.0 in g:  # G10 L20 P1 (set the G54 offset so the current position reads as the given value)
            self._settle()
            for k, v in axes.items():
                i = "XYZ".index(k)
                self.wco[i] = self.mpos[i] - v
            self._wco_dirty = True
            return "ok"
        if 1.0 in g or 0.0 in g:
            if 1.0 in g and not self.feed:
                return "error:22"
            here = list(self._position())
            target = list(here)
            for k, v in axes.items():
                i = "XYZ".index(k)
                target[i] = v if 53.0 in g else v + self.wco[i]
            if self.travel_mm and any(not (self.travel_mm[0] <= t <= self.travel_mm[1]) for t in target):
                self.trigger_alarm(2)
                return "ok"  # the alarm is pushed; the line itself was accepted
            dist = max(abs(target[i] - here[i]) for i in range(3))
            dur = 0.0 if not self.feed else dist / self.feed * 60.0
            self._move = (here, target, time.monotonic(), dur)
            self.mpos = here
            return "ok"
        return "error:20"


def _grbl(fake: FakeGrbl, **kw) -> GrblStageDriver:
    kw.setdefault("timeout", 0.3)
    kw.setdefault("feedrate_mm_min", 6000.0)
    return GrblStageDriver("COM-FAKE", serial_factory=lambda *a, **k: fake, **kw)


def test_list_serial_ports_degrades_without_pyserial(monkeypatch):
    import microtopos.stage as st

    monkeypatch.setattr(st, "_HAS_PYSERIAL", False)
    ports = list_serial_ports()
    assert isinstance(ports, list)


def test_simulated_stage_move_zero_home_poll():
    drv = SimulatedStageDriver()
    drv.connect()
    assert drv.pose().connected is True

    p = drv.move_to(x_um=10.0, y_um=20.0, z_um=5.0)
    assert (p.x_um, p.y_um, p.z_um) == (10.0, 20.0, 5.0)
    assert p.connected is True

    p = drv.move_rel(dx_um=5.0)
    assert p.x_um == pytest.approx(15.0)

    p = drv.zero()
    assert (p.x_um, p.y_um, p.z_um) == (0.0, 0.0, 0.0)

    p = drv.pose()
    assert (p.x_um, p.y_um, p.z_um) == (0.0, 0.0, 0.0)

    drv.close()
    assert drv.pose().connected is False


def test_simulated_stage_refuses_move_when_closed():
    drv = SimulatedStageDriver()
    drv.connect()
    drv.move_to(z_um=3.0)
    drv.close()
    with pytest.raises(StageError, match="not connected"):
        drv.move_to(z_um=7.0)
    drv.connect()  # reconnecting restores a working stage, and the refused move changed nothing
    assert drv.pose().z_um == 3.0


def test_simulated_stage_z_sweep_moves_through_the_positions():
    drv = SimulatedStageDriver()
    drv.connect()
    seen = [drv.move_to(z_um=z).z_um for z in [0.0, 10.0, -5.0, 20.0]]
    assert seen == [0.0, 10.0, -5.0, 20.0]
    assert drv.pose().z_um == pytest.approx(20.0)


def test_simulated_stage_with_sim_source():
    sim = SimulatorSource(1, OBJ, 160, 120)
    sim.start()
    try:
        drv = SimulatedStageDriver(sim_source=sim)
        p = drv.pose()
        assert p.connected is True
        assert p.x_um == pytest.approx(sim.sim.stage_x_um)
        assert p.z_um == pytest.approx(sim.sim.focus_z_um)
    finally:
        sim.stop()


# -- Grbl driver against the simulated controller (no pyserial needed: the port is injected)
def _xyz(p) -> tuple[float, float, float]:
    return (p.x_um, p.y_um, p.z_um)


def test_grbl_connect_sets_units_and_reads_the_position():
    fake = FakeGrbl(start=(1.234, 2.345, 3.456))
    drv = _grbl(fake)
    drv.connect()
    sent = [w.strip() for w in fake.written]
    assert "G21" in sent and "G90" in sent
    p = drv.pose()
    assert _xyz(p) == pytest.approx((1234.0, 2345.0, 3456.0))
    assert p.connected is True and p.moving is False
    drv.close()
    assert not fake.is_open
    assert drv.pose().connected is False


@pytest.mark.parametrize("report", ["MPos", "WPos"])
def test_grbl_zero_pose_and_move_share_one_frame(report):
    # the controller still carries an offset from an earlier G92 (WCO != 0) and may report MPos or WPos ($10)
    fake = FakeGrbl(report=report, wco=(1.0, 2.0, 3.0), start=(10.0, 20.0, 30.0))
    drv = _grbl(fake)
    drv.connect()
    assert _xyz(drv.pose()) == pytest.approx((10000.0, 20000.0, 30000.0))  # machine frame, whatever is reported

    drv.move_to(x_um=15000.0, z_um=25000.0)
    assert fake.mpos == pytest.approx([15.0, 20.0, 25.0])
    assert _xyz(drv.pose()) == pytest.approx((15000.0, 20000.0, 25000.0))

    assert _xyz(drv.zero()) == (0.0, 0.0, 0.0)
    assert _xyz(drv.pose()) == pytest.approx((0.0, 0.0, 0.0))  # zero and pose agree (the audited bug)
    assert fake.wco == [1.0, 2.0, 3.0]  # the controller's own offsets are not touched

    drv.move_to(x_um=100.0)  # 100 um from the new origin
    assert fake.mpos == pytest.approx([15.1, 20.0, 25.0])
    assert _xyz(drv.pose()) == pytest.approx((100.0, 0.0, 0.0))
    assert _xyz(drv.move_rel(dy_um=50.0)) == pytest.approx((100.0, 50.0, 0.0))

    fake._execute("G92 X0 Y0 Z0")  # another sender redefines the work offset behind the driver's back
    assert _xyz(drv.pose()) == pytest.approx((100.0, 50.0, 0.0))
    assert _xyz(drv.move_to(z_um=40.0)) == pytest.approx((100.0, 50.0, 40.0))
    assert fake.mpos == pytest.approx([15.1, 20.05, 25.04])


@pytest.mark.parametrize("report", ["MPos", "WPos"])
def test_grbl_pose_after_zero_is_zero_not_the_old_machine_position(report):
    fake = FakeGrbl(report=report, start=(5.0, 0.0, 2.5))  # a plain controller, no prior offsets
    drv = _grbl(fake)
    drv.connect()
    assert _xyz(drv.pose()) == pytest.approx((5000.0, 0.0, 2500.0))
    assert _xyz(drv.zero()) == (0.0, 0.0, 0.0)
    assert _xyz(drv.pose()) == pytest.approx((0.0, 0.0, 0.0))
    assert _xyz(drv.move_to(x_um=250.0)) == pytest.approx((250.0, 0.0, 0.0))
    assert fake.mpos == pytest.approx([5.25, 0.0, 2.5])


def test_grbl_units_are_converted_in_one_place():
    fake = FakeGrbl()
    drv = _grbl(fake, um_per_unit=500.0)  # one controller unit = 500 um
    drv.connect()
    p = drv.move_to(x_um=1000.0)
    assert any("X2.00000" in w for w in fake.written)
    assert fake.mpos[0] == pytest.approx(2.0)
    assert p.x_um == pytest.approx(1000.0)


def test_grbl_move_waits_for_idle_and_nowait_reports_run():
    fake = FakeGrbl()
    drv = _grbl(fake, feedrate_mm_min=300.0)  # 5 mm/s: a 1 mm move takes 0.2 s
    drv.connect()
    p = drv.move_to(x_um=1000.0, wait=False)
    assert p.moving is True and p.x_um < 1000.0  # the controller reports Run, not the target
    t0 = time.perf_counter()
    done = drv.move_to(x_um=2000.0, wait=True)
    assert time.perf_counter() - t0 >= 0.1  # it really waited for the motion to finish
    assert done.moving is False and done.x_um == pytest.approx(2000.0)
    assert fake.mpos[0] == pytest.approx(2.0)


def test_grbl_error_reply_raises_and_pose_stays_truthful():
    fake = FakeGrbl(start=(1.0, 0.0, 0.0))
    drv = _grbl(fake, feedrate_mm_min=0.0)  # F0 -> the controller answers error:22
    drv.connect()
    with pytest.raises(StageError, match="error:22"):
        drv.move_to(x_um=5000.0, z_um=2500.0)
    assert fake.mpos == [1.0, 0.0, 0.0]
    assert _xyz(drv.pose()) == pytest.approx((1000.0, 0.0, 0.0))  # not the optimistic (5000, 0, 2500)


def test_grbl_alarm_raises_and_blocks_until_cleared():
    fake = FakeGrbl()
    fake.travel_mm = (0.0, 10.0)
    drv = _grbl(fake)
    drv.connect()
    with pytest.raises(StageError, match="ALARM:2"):
        drv.move_to(x_um=50000.0)  # beyond the soft limit
    assert fake.mpos == [0.0, 0.0, 0.0]
    with pytest.raises(StageError, match="Alarm"):
        drv.pose()  # no position is reported while the controller is locked
    with pytest.raises(StageError):
        drv.move_to(x_um=1000.0)
    fake.alarm = False  # operator unlocks ($X)
    assert _xyz(drv.move_to(x_um=1000.0)) == pytest.approx((1000.0, 0.0, 0.0))


def test_grbl_silent_controller_raises_instead_of_a_stale_pose():
    fake = FakeGrbl(start=(1.0, 2.0, 3.0))
    drv = _grbl(fake, timeout=0.1)
    drv.connect()
    assert drv.pose().x_um == pytest.approx(1000.0)
    fake.silent = True
    with pytest.raises(StageError, match="no status report"):
        drv.pose()
    with pytest.raises(StageError):
        drv.move_to(x_um=9000.0)
    assert fake.mpos == [1.0, 2.0, 3.0]


def test_grbl_connect_fails_cleanly_when_the_controller_is_silent():
    fake = FakeGrbl()
    fake.silent = True
    drv = _grbl(fake, timeout=0.1)
    with pytest.raises(StageError):
        drv.connect()
    assert not fake.is_open
    assert drv.pose().connected is False


def test_grbl_garbled_status_is_retried_then_reported_as_stage_error():
    fake = FakeGrbl(start=(1.0, 2.0, 3.0))
    drv = _grbl(fake)
    drv.connect()
    fake.garble = 1  # one corrupted report: the next one is good
    assert _xyz(drv.pose()) == pytest.approx((1000.0, 2000.0, 3000.0))
    fake.garble = 99
    with pytest.raises(StageError, match="garbled") as ei:
        drv.pose()
    assert not isinstance(ei.value, ValueError)


@pytest.mark.parametrize("line, expected", [
    ("<Idle|MPos:1.000,2.000,3.000|FS:0,0>", ("Idle", (1.0, 2.0, 3.0), None, None)),
    ("<Hold:0|WPos:-1.5,0.000,2.250|WCO:1.000,1.000,1.000>", ("Hold:0", None, (-1.5, 0.0, 2.25), (1.0, 1.0, 1.0))),
    ("<Run|MPos:0.000,0.000,0.000|WPos:0.000,0.000,0.000|Ov:100,100,100>",
     ("Run", (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), None)),
])
def test_parse_grbl_status(line, expected):
    assert parse_grbl_status(line) == expected


@pytest.mark.parametrize("line", ["", "ok", "<Idle|MPos:1.0,,3.0>", "<Idle|MPos:1.0,2.0>", "<|MPos:1,2,3>",
                                  "<Idle|MPos:a,b,c>", "Idle|MPos:1,2,3"])
def test_parse_grbl_status_rejects_garbage(line):
    with pytest.raises(ValueError):
        parse_grbl_status(line)


def test_grbl_stage_driver_raises_when_not_connected():
    drv = GrblStageDriver("/dev/nonexistent")
    with pytest.raises(StageError, match="stage not connected"):
        drv.move_to(x_um=100.0)
    with pytest.raises(RuntimeError):  # StageError is a RuntimeError for callers that only know that
        drv.zero()


def test_grbl_connect_without_pyserial_says_how_to_install_it(monkeypatch):
    monkeypatch.setitem(sys.modules, "serial", None)  # makes `import serial` fail like a missing package
    with pytest.raises(ImportError, match=r"microtopos\[serial\]"):
        GrblStageDriver("COM9").connect()


# ------------------------------------------------------------------------------------------------ calibration
def test_um_per_px_aspect_ratios():
    c = ObjectiveCalibration(objective="10x / 0.25", lateral_um_per_px_full=0.30)

    # Full-width scaling (16:9 native)
    v_16_9 = c.um_per_px(1920)
    assert v_16_9 == pytest.approx(0.30 * 1920 / 1920)

    # 1:1 crop from 1080p height -> vertical reference scaling
    v_1_1 = c.um_per_px(1080, frame_height=1080)
    assert v_1_1 == pytest.approx(0.30 * (1920 * 9.0 / 16.0) / 1080)

    # 4:3 crop -> vertical reference scaling
    v_4_3 = c.um_per_px(1440, frame_height=1080)
    assert v_4_3 == pytest.approx(0.30 * (1920 * 9.0 / 16.0) / 1080)

    # Backwards-compatible: frame_height omitted -> full-width scaling
    v_no_h = c.um_per_px(1080)
    assert v_no_h == pytest.approx(0.30 * 1920 / 1080)


def test_um_per_px_nominal_fallback():
    c = ObjectiveCalibration(objective="10x / 0.25")  # no calibrated lateral scale
    v = c.um_per_px(1920)
    expected = OBJ.nominal_um_per_px(1920)
    assert v == pytest.approx(expected)


def test_uint16_bgr_to_luma():
    img16 = np.full((8, 8, 3), 32768, dtype=np.uint16)
    y = bgr_to_luma(img16)
    assert y.dtype == np.float32
    # ~mid-gray: 32768/257 ≈ 127.5
    assert y.mean() == pytest.approx(127.5, abs=1.0)

    img8 = np.full((8, 8, 3), 128, dtype=np.uint8)
    y8 = bgr_to_luma(img8)
    assert y8.dtype == np.float32
    assert y8.mean() == pytest.approx(128.0, abs=1.0)

    # uint16 linearize path
    ylin = bgr_to_luma(img16, linearize=True)
    assert ylin.dtype == np.float32


def test_bgr_to_rgb_u8():
    img_bgr = np.zeros((4, 4, 3), dtype=np.uint8)
    img_bgr[..., 0] = 10  # B
    img_bgr[..., 1] = 20  # G
    img_bgr[..., 2] = 30  # R
    rgb = bgr_to_rgb_u8(img_bgr)
    assert rgb.dtype == np.uint8
    assert rgb[0, 0, 0] == 30  # R
    assert rgb[0, 0, 1] == 20  # G
    assert rgb[0, 0, 2] == 10  # B

    # uint16 input
    img16 = np.full((4, 4, 3), 32768, dtype=np.uint16)
    rgb16 = bgr_to_rgb_u8(img16)
    assert rgb16.dtype == np.uint8
    assert rgb16[0, 0, 0] == pytest.approx(128, abs=1)

    # grayscale input
    gray = np.full((4, 4), 50, dtype=np.uint8)
    rgb_gray = bgr_to_rgb_u8(gray)
    assert rgb_gray.shape == (4, 4, 3)
    assert np.all(rgb_gray == 50)


def test_lateral_magnification_round_trip(tmp_path):
    store = CalibrationStore(tmp_path)
    c = store.get("10x / 0.25")
    c.lateral_um_per_px_full = 0.3
    c.lateral_magnification = 10.0
    c.lateral_source = "stage micrometer"
    store.touch("10x / 0.25")
    store.save()

    back = CalibrationStore(tmp_path)
    c2 = back.get("10x / 0.25")
    assert c2.lateral_um_per_px_full == pytest.approx(0.3)
    assert c2.lateral_magnification == pytest.approx(10.0)
    assert c2.lateral_source == "stage micrometer"
    assert c2.revision == 1


def test_calibrated_magnification_set_on_get():
    store = CalibrationStore(None)
    c = store.get(OBJ)
    c.lateral_um_per_px_full = 0.3
    # Re-get to trigger the magnification recording logic
    c2 = store.get(OBJ)
    assert c2.calibrated_magnification == OBJ.magnification


# ------------------------------------------------------------------------------------------------ recording
def test_16bit_png_write_and_decode_round_trip(tmp_path):
    img16 = np.random.default_rng(0).integers(0, 65535, (20, 30), dtype=np.uint16)
    w = RecordingWriter(tmp_path / "r16")
    w.add(img16, 1, 0.0)
    w.close()
    m = json.loads((tmp_path / "r16" / "frames.json").read_text())
    assert m["complete"] is True
    assert m["frames"][0]["file"].endswith(".png")
    got = decode(tmp_path / "r16" / "000000.png")
    assert got is not None
    assert got.dtype == np.uint16
    assert np.array_equal(got, img16)


def test_16bit_png_keep_8bit_behavior(tmp_path):
    img8 = np.random.default_rng(1).integers(0, 256, (20, 30, 3), dtype=np.uint8)
    w = RecordingWriter(tmp_path / "r8")
    w.add(img8, 1, 0.0)
    w.close()
    got = decode(tmp_path / "r8" / "000000.png")
    assert got is not None
    assert got.dtype == np.uint8
    assert got.shape == img8.shape


def test_non_jpeg_1d_bytes_marks_incomplete(tmp_path):
    w = RecordingWriter(tmp_path / "bad")
    with pytest.raises(RecordingError):
        w.add(np.arange(64, dtype=np.uint8), 1, 0.0)
    w.close()
    m = json.loads((tmp_path / "bad" / "frames.json").read_text())
    assert m["complete"] is False
    assert m["error"]


# ------------------------------------------------------------------------------------------------ replay transport
def _make_replay_recording(tmp_path, n=10):
    d = tmp_path / "rec"
    d.mkdir()
    w = RecordingWriter(d, "sim")
    for i in range(n):
        img = np.full((24, 32, 3), fill_value=10 * i, dtype=np.uint8)
        ok, buf = cv2.imencode(".png", img)
        assert ok
        (d / f"{i:06d}.png").write_bytes(buf.tobytes())
        w.entries.append({"file": f"{i:06d}.png", "seq": i + 1, "t": 0.1 * i, "bytes": len(buf.tobytes()),
                          "crc32": 0})
        w.bytes += len(buf.tobytes())
    w.meta["mode"] = "decoded-png"
    w.meta["complete"] = True
    w._write_manifest()
    return d


def test_replay_transport_pause_resume_toggle(tmp_path):
    d = _make_replay_recording(tmp_path)
    rs = ReplaySource(str(d), speed=10.0)
    rs.start()
    try:
        rs.pause()
        assert rs.paused is True
        rs.resume()
        assert rs.paused is False
        assert rs.toggle_pause() is True
        assert rs.toggle_pause() is False
    finally:
        rs.stop()


def test_replay_seek_clamps_out_of_range(tmp_path):
    d = _make_replay_recording(tmp_path)
    rs = ReplaySource(str(d))
    # out-of-range high
    rs.seek(999)
    assert rs._seek_target == rs.total_frames - 1
    # out-of-range low
    rs.seek(-5)
    assert rs._seek_target == 0


def test_replay_step_frames(tmp_path):
    d = _make_replay_recording(tmp_path)
    rs = ReplaySource(str(d))
    rs.step_frames(3)
    assert rs.paused is True
    assert rs._seek_target == 3


def test_replay_set_speed_and_loop(tmp_path):
    d = _make_replay_recording(tmp_path)
    rs = ReplaySource(str(d))
    rs.set_speed(2.5)
    assert rs.speed == pytest.approx(2.5)
    rs.set_loop(True)
    assert rs.loop is True
    rs.set_loop(False)
    assert rs.loop is False


def test_replay_total_frames(tmp_path):
    d = _make_replay_recording(tmp_path, n=7)
    rs = ReplaySource(str(d))
    assert rs.total_frames == 7


def test_replay_seek_publishes_immediately(tmp_path):
    d = _make_replay_recording(tmp_path)
    rs = ReplaySource(str(d))
    rs.seek(5)
    frames = list(rs.frames())
    assert len(frames) == 10  # frames() iterates all, independent of seek


# ------------------------------------------------------------------------------------------------ stage -> frames -> height map
class _RenderedStageSource(FrameSource):
    """A camera-like source (no simulator truth of its own): it renders the specimen at whatever Z its stage is
    at and publishes the bare frame, so the only Z label a frame can get is the attached stage's readout."""

    def __init__(self, sim, stage):
        super().__init__()
        self.sim = sim
        self.stage = stage
        self.stage_driver = stage

    def _run(self):
        while self._running:
            self.sim.focus_z_um = self.stage.pose().z_um
            bgr, _ = self.sim.render()
            self._publish(bgr)
            time.sleep(0.04)


def _reference_height(sim, h):
    """Simulator relief resampled onto the reconstruction grid (um), and the valid central crop."""
    _, gt = sim.truth_patch()
    g = cv2.resize(gt, (h.shape[1], h.shape[0]), interpolation=cv2.INTER_AREA)
    c = (slice(16, -16), slice(16, -16))
    return g[c], h[c]


def _check_height_matches_relief(h, sim):
    g, hh = _reference_height(sim, h)
    ok = np.isfinite(hh)
    # Valid-pixel coverage depends on how many frames the live thread happened to capture per Z (observed 0.35-0.70
    # over repeated runs); r and slope below are the acceptance criteria, this floor only keeps them meaningful.
    assert ok.mean() > 0.2
    r = float(np.corrcoef(g[ok], hh[ok])[0, 1])
    slope = float(np.polyfit(g[ok], hh[ok], 1)[0])
    assert r >= 0.95, f"Pearson r {r:.3f}"
    assert abs(slope - 1.0) <= 0.10, f"height slope {slope:.3f}"


def test_engine_stage_z_labels_a_non_simulated_source_and_reconstructs_in_um():
    """Real-camera path: frames carry the attached stage's Z (um), the capture is NOT flagged simulated, and the
    sweep reconstructed with those labels reproduces the specimen relief."""
    obj = objectives.default_set().objectives[1]
    sim = SpecimenSimulator(3, obj, 320, 180)
    stage = SimulatedStageDriver()
    lo, hi = SimulatorSource(3, obj, 320, 180).relief_range_um()
    zs = [float(z) for z in np.linspace(lo, hi, 36)]
    stage.move_to(z_um=hi)
    src = _RenderedStageSource(sim, stage)
    caps: list = []
    eng = LiveEngine(CalibrationStore(None), obj)
    eng.on_sweep = caps.append
    eng.start()
    eng.set_source(src)
    src.start()
    try:
        poses = capture_z_sweep(stage, src, eng, zs, settle_s=5.0)
        t0 = time.time()
        while not caps and time.time() - t0 < 10:
            time.sleep(0.02)
    finally:
        src.stop()
        eng.stop()
    assert [p.z_um for p in poses] == zs
    assert len(caps) == 1
    cap = caps[0]
    assert cap.manual is True and cap.simulated is False  # a stage measurement is not a simulation
    assert cap.z is not None and len(cap.z) == len(cap.frames)
    assert set(np.unique(cap.z)) >= set(zs)  # every commanded position was captured
    assert set(np.unique(cap.z)) <= set(zs) | {hi}  # and no other Z was ever labelled
    assert np.all(np.diff(cap.z) >= 0) and cap.z[-1] == zs[-1]  # monotonic and in the commanded direction
    res = zstack.reconstruct(cap.frames, cap.z, obj, obj.nominal_um_per_px(cap.frames[0].shape[1]), "um")
    _check_height_matches_relief(res["height"], sim)


def test_capture_z_sweep_needs_a_stage_attached_to_the_source():
    """Negative control: without the stage readout on the frames no Z label exists, so the sweep refuses."""
    obj = objectives.default_set().objectives[1]
    stage = SimulatedStageDriver()
    src = _RenderedStageSource(SpecimenSimulator(3, obj, 160, 90), stage)
    src.stage_driver = None  # the stage moves the focus, but nothing labels the frames
    eng = LiveEngine(CalibrationStore(None), obj)
    eng.start()
    eng.set_source(src)
    src.start()
    try:
        with pytest.raises(StageError, match="no frame was captured"):
            capture_z_sweep(stage, src, eng, [0.0, 1.0, 2.0], settle_s=0.5)
    finally:
        src.stop()
        eng.stop()
    assert eng.sm.override == Override.AUTO  # the capture was ended again after the failure


def test_frames_are_unlabelled_not_invented_when_the_stage_fails(monkeypatch):
    class _Broken(SimulatedStageDriver):
        def pose(self):
            raise StageError("no status report within 2 s")

    src = FrameSource()
    src.stage_driver = _Broken()
    src._publish(np.zeros((4, 4, 3), np.uint8))
    assert src.latest()[3] is None
    ok = SimulatedStageDriver()
    ok.move_to(z_um=1.5)
    src.stage_driver = ok
    src._publish(np.zeros((4, 4, 3), np.uint8))
    assert src.latest()[3].focus_z_um == 1.5
    ok.close()
    src._publish(np.zeros((4, 4, 3), np.uint8))
    assert src.latest()[3] is None  # a closed stage labels nothing


# -- GUI (offscreen)
@pytest.fixture(scope="module")
def qapp(tmp_path_factory):
    QStandardPaths.setTestModeEnabled(True)
    QSettings.setPath(QSettings.NativeFormat, QSettings.UserScope, str(tmp_path_factory.mktemp("qsettings")))
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture
def window(qapp, tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    w = MainWindow(simulator_only=True)
    w.show()
    _pump(qapp, 20)
    w.timer.stop()
    yield w
    w.close()
    _pump(qapp, 5)


def _pump(app, n=25):
    for _ in range(n):
        app.processEvents()


def _wait(app, cond, timeout=30.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        app.processEvents()
        if cond():
            return True
        time.sleep(0.02)
    return False


def _stage_idle(w):
    return not any(j.isRunning() for j in w.jobs)


def test_gui_grbl_connect_without_pyserial_shows_a_clear_message(window, qapp, monkeypatch):
    w = window
    monkeypatch.setitem(sys.modules, "serial", None)  # pyserial not installed
    seen: list[str] = []
    real_notify = w.notify
    monkeypatch.setattr(w, "notify", lambda text, kind="info": (seen.append(text), real_notify(text, kind))[1])
    crashed: list = []
    monkeypatch.setattr(sys, "excepthook", lambda *a: crashed.append(a))
    w.cmb_stg.setCurrentText("GRBL serial port")
    w.btn_stg_connect.click()
    assert _wait(qapp, lambda: _stage_idle(w) and any("microtopos[serial]" in s for s in seen))
    assert not crashed
    assert w.stage_driver is None
    assert w.btn_stg_connect.text() == "Connect" and not w.btn_stg_zero.isEnabled()


def test_gui_grbl_stage_error_is_reported_not_raised(window, qapp, monkeypatch):
    w = window
    fake = FakeGrbl()
    fake.silent = True  # a controller that does not answer
    import microtopos.gui.main_window as mw

    monkeypatch.setattr(mw, "GrblStageDriver", lambda port: GrblStageDriver(port, serial_factory=lambda *a, **k: fake,
                                                                          timeout=0.1))
    seen: list[str] = []
    real_notify = w.notify
    monkeypatch.setattr(w, "notify", lambda text, kind="info": (seen.append(text), real_notify(text, kind))[1])
    w.cmb_stg.setCurrentText("GRBL serial port")
    w.btn_stg_connect.click()
    assert _wait(qapp, lambda: _stage_idle(w) and any("no status report" in s or "no reply" in s for s in seen))
    assert w.stage_driver is None


def test_gui_grbl_stage_connect_jog_zero_with_the_fake_controller(window, qapp, monkeypatch):
    w = window
    fake = FakeGrbl(start=(5.0, 0.0, 2.5))
    import microtopos.gui.main_window as mw

    monkeypatch.setattr(mw, "GrblStageDriver", lambda port: GrblStageDriver(port, serial_factory=lambda *a, **k: fake,
                                                                          timeout=0.3, feedrate_mm_min=6000.0))
    w.cmb_stg.setCurrentText("GRBL serial port")
    w.btn_stg_connect.click()
    assert _wait(qapp, lambda: w.stage_driver is not None and _stage_idle(w))
    assert "+5000.00" in w.lbl_stg_pose.text() and "+2500.00" in w.lbl_stg_pose.text()
    w.btn_stg_zero.click()
    assert _wait(qapp, lambda: _stage_idle(w) and "X +0.00" in w.lbl_stg_pose.text())  # the label follows zero()
    assert "Z +0.00" in w.lbl_stg_pose.text()
    assert w.source.stage_driver is w.stage_driver  # attached to the active source
    w.btn_stg_connect.click()  # disconnect
    assert _wait(qapp, lambda: w.stage_driver is None)
    assert w.source.stage_driver is None and not fake.is_open


def test_gui_stage_z_sweep_captures_stage_z_and_reconstructs_in_um(window, qapp):
    """Connect the simulated stage, press 'Z sweep': the stage is driven through the range while the live engine
    captures it; the capture's z labels are the commanded stage Z (um) and the height map reproduces the relief."""
    w = window
    w.btn_stg_connect.click()
    assert _wait(qapp, lambda: w.stage_driver is not None and _stage_idle(w))
    assert w.source.stage_driver is w.stage_driver  # attached to the active source
    z0, z1, n = w.sp_stg_z0.value(), w.sp_stg_z1.value(), w.sp_stg_n.value()
    assert z1 > z0 and n >= 16
    commanded = np.linspace(z0, z1, n)
    assert commanded[-1] - commanded[0] > 2.0  # the default range is the specimen relief, not a token +/-5 um
    w.btn_stg_zsweep.click()
    assert _wait(qapp, lambda: w.result is not None, timeout=60.0), "no height map was reconstructed"
    cap = w.last_cap
    assert cap.manual is True and cap.simulated is True
    assert cap.z is not None and len(cap.z) == len(cap.frames) >= n
    uz = np.unique(cap.z)
    assert set(np.round(commanded, 9)) <= set(np.round(uz, 9))  # every commanded position captured
    assert set(np.round(uz, 9)) <= set(np.round(commanded, 9))  # and nothing else (pre-roll sits at the first Z)
    assert np.all(np.diff(cap.z) >= 0)  # monotonic
    assert cap.z[0] == pytest.approx(z0) and cap.z[-1] == pytest.approx(z1)  # sign: ascending, as commanded
    assert w.result["z_unit"] == "um" and w.result["z_status"] == "SIMULATED"
    _check_height_matches_relief(w.result["height"], w.sim_source.sim)
    assert w.stage_driver.pose().z_um == pytest.approx(z1)  # the stage ends where the sweep ended
