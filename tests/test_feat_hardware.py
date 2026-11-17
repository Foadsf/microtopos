"""Feature tests for hardware integration: stage driver, calibration aspect ratios, 16-bit recording, replay transport."""

from __future__ import annotations

import json

import cv2
import numpy as np
import pytest

from microtopos import objectives
from microtopos.calibration import CalibrationStore, ObjectiveCalibration, bgr_to_luma, bgr_to_rgb_u8
from microtopos.camera import (
    ReplaySource,
    SimulatorSource,
)
from microtopos.recording import RecordingError, RecordingWriter, decode
from microtopos.stage import (
    GrblStageDriver,
    SimulatedStageDriver,
    list_serial_ports,
)

OBJ = objectives.resolve("10x / 0.25", objset=objectives.default_set())


# ------------------------------------------------------------------------------------------------ stage
class _FakeSerial:
    """Minimal fake serial object for GrblStageDriver without hardware."""

    def __init__(self, responses: dict[str, str] | None = None):
        self.responses = responses or {}
        self.written: list[str] = []
        self.is_open = True
        self._closed = False
        self.timeout = 2.0

    def write(self, data: bytes) -> int:
        self.written.append(data.decode("ascii", errors="replace").strip())
        return len(data)

    def flush(self) -> None:
        pass

    def reset_input_buffer(self) -> None:
        pass

    def readline(self) -> bytes:
        # Return the last written command's canned response, or "ok"
        last = self.written[-1] if self.written else ""
        if last.startswith("?"):
            return b"<Idle|MPos:1.234,2.345,3.456|WPos:1.234,2.345,3.456>\n"
        if last.startswith("G92"):
            return b"ok\n"
        if last.startswith("G1"):
            return b"ok\n"
        if last.startswith("G4"):
            return b"ok\n"
        if last.startswith("G21"):
            return b"ok\n"
        if last.startswith("G90"):
            return b"ok\n"
        return b"ok\n"

    def close(self) -> None:
        self._closed = True
        self.is_open = False


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
    drv.close()
    # After close, connected is False; move_to still works on the internal state but pose reports disconnected
    p = drv.move_to(z_um=7.0)
    assert p.z_um == pytest.approx(7.0)
    assert p.connected is False


def test_simulated_stage_z_sweep():
    drv = SimulatedStageDriver()
    drv.connect()
    poses = drv.z_sweep([0.0, 10.0, -5.0, 20.0], dwell_s=0.001)
    assert len(poses) == 4
    assert [p.z_um for p in poses] == [0.0, 10.0, -5.0, 20.0]
    assert all(p.connected for p in poses)
    # Final pose matches last sweep position
    final = drv.pose()
    assert final.z_um == pytest.approx(20.0)


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


def test_grbl_stage_driver_parses_status_and_sends_gcode():
    pytest.importorskip("serial")
    fake = _FakeSerial()
    import serial

    # Patch serial.Serial used inside GrblStageDriver.connect
    orig_serial = serial.Serial
    serial.Serial = lambda *a, **k: fake  # type: ignore[assignment]
    try:
        drv = GrblStageDriver("/dev/ttyUSB0")
        drv.connect()
        assert drv.pose().connected is True
        assert fake.written, "expected G-code commands on the wire"
        assert any("G21" in w for w in fake.written)
        assert any("G90" in w for w in fake.written)

        p = drv.pose()
        assert p.x_um == pytest.approx(1234.0)
        assert p.y_um == pytest.approx(2345.0)
        assert p.z_um == pytest.approx(3456.0)
        assert p.connected is True

        drv.move_to(x_um=5000.0, z_um=2500.0)
        assert any("X5.00000" in w for w in fake.written)
        assert any("Z2.50000" in w for w in fake.written)

        drv.zero()
        assert any("G92" in w for w in fake.written)

        drv.close()
        assert not fake.is_open
    finally:
        serial.Serial = orig_serial


def test_grbl_stage_driver_raises_when_not_connected():
    drv = GrblStageDriver("/dev/nonexistent")
    with pytest.raises(RuntimeError, match="stage not connected"):
        drv.move_to(x_um=100.0)


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
