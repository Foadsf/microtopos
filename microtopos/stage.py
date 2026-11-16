"""Motorised Z-focus drives and X-Y stage encoders (serial G-code / Grbl and simulated drives).

Provides a unified, thread-safe interface (`StageDriver`) for:
  * reading live physical (x_um, y_um, z_um) positions from stage encoders or stepper controllers,
  * jogging X, Y and Z in micrometres,
  * executing deterministic, stepped motorised Z-sweeps that feed exact `z_labels` into `zstack.reconstruct`.
"""

from __future__ import annotations

import glob
import sys
import threading
import time
from dataclasses import dataclass

_HAS_PYSERIAL: bool = False
try:
    import serial.tools.list_ports  # noqa: F401  # availability check only
    _HAS_PYSERIAL = True
except ImportError:  # pyserial is optional
    pass


@dataclass(frozen=True)
class StagePose:
    x_um: float = 0.0
    y_um: float = 0.0
    z_um: float = 0.0
    moving: bool = False
    connected: bool = False


def list_serial_ports() -> list[str]:
    """Available serial ports for motorised stage / focus controllers."""
    if _HAS_PYSERIAL:
        try:
            from serial.tools.list_ports import comports

            return [p.device for p in comports()]
        except ImportError:  # pragma: no cover - guarded by _HAS_PYSERIAL above
            pass
    if sys.platform.startswith("linux"):
        return sorted(glob.glob("/dev/ttyACM*") + glob.glob("/dev/ttyUSB*"))
    if sys.platform == "darwin":
        return sorted(glob.glob("/dev/cu.usb*"))
    return []


class StageDriver:
    """Abstract base class for X-Y-Z stage and focus drives."""

    name: str = "none"

    def connect(self) -> None:
        pass

    def close(self) -> None:
        pass

    def pose(self) -> StagePose:
        return StagePose()

    def move_to(self, x_um: float | None = None, y_um: float | None = None, z_um: float | None = None,
                wait: bool = True) -> StagePose:
        raise NotImplementedError

    def move_rel(self, dx_um: float = 0.0, dy_um: float = 0.0, dz_um: float = 0.0, wait: bool = True) -> StagePose:
        p = self.pose()
        return self.move_to(
            p.x_um + dx_um if dx_um else None,
            p.y_um + dy_um if dy_um else None,
            p.z_um + dz_um if dz_um else None,
            wait=wait,
        )

    def zero(self) -> StagePose:
        raise NotImplementedError


class SimulatedStageDriver(StageDriver):
    """Virtual motorised X-Y-Z drive coupled to a `SimulatorSource` (or standalone for testing)."""

    name = "simulated"

    def __init__(self, sim_source=None):
        self._lock = threading.Lock()
        self.sim_source = sim_source
        self._x = 0.0
        self._y = 0.0
        self._z = 0.0
        self._connected = True

    def connect(self) -> None:
        self._connected = True

    def close(self) -> None:
        self._connected = False

    def pose(self) -> StagePose:
        with self._lock:
            if self.sim_source is not None:
                with self.sim_source.lock:
                    s = self.sim_source.sim
                    return StagePose(s.stage_x_um, s.stage_y_um, s.focus_z_um, self.sim_source.busy(), self._connected)
            return StagePose(self._x, self._y, self._z, False, self._connected)

    def move_to(self, x_um: float | None = None, y_um: float | None = None, z_um: float | None = None,
                wait: bool = True) -> StagePose:
        with self._lock:
            if self.sim_source is not None:
                with self.sim_source.lock:
                    s = self.sim_source.sim
                    if x_um is not None:
                        s.stage_x_um = float(x_um)
                    if y_um is not None:
                        s.stage_y_um = float(y_um)
                    if z_um is not None:
                        s.focus_z_um = float(z_um)
                    return StagePose(s.stage_x_um, s.stage_y_um, s.focus_z_um, False, self._connected)
            if x_um is not None:
                self._x = float(x_um)
            if y_um is not None:
                self._y = float(y_um)
            if z_um is not None:
                self._z = float(z_um)
            return StagePose(self._x, self._y, self._z, False, self._connected)

    def zero(self) -> StagePose:
        return self.move_to(0.0, 0.0, 0.0)

    def z_sweep(self, z_um_list: list[float], dwell_s: float = 0.02) -> list[StagePose]:
        """Move Z through a list of positions with dwell between each.

        Returns the sequence of poses after each move. With a `sim_source` attached, each
        pose includes the live simulator stage position as ground-truth z_labels suitable
        for feeding into `zstack.reconstruct`.
        """
        poses: list[StagePose] = []
        for z_um in z_um_list:
            p = self.move_to(z_um=z_um)
            if dwell_s > 0:
                time.sleep(dwell_s)
            poses.append(p)
        return poses


class GrblStageDriver(StageDriver):
    """Serial G-code (Grbl / Marlin / OpenFlexure-style) X-Y-Z controller.

    Coordinates on the wire are in millimetres (G21); `um_per_unit` converts controller units to µm
    (default 1000.0 µm/mm).
    """

    def __init__(self, port: str, baudrate: int = 115200, um_per_unit: float = 1000.0,
                 feedrate_mm_min: float = 120.0, timeout: float = 2.0):
        self.port = port
        self.baudrate = baudrate
        self.um_per_unit = float(um_per_unit)
        self.feedrate = float(feedrate_mm_min)
        self.timeout = timeout
        self.name = f"Grbl ({port})"
        self._ser = None
        self._lock = threading.Lock()
        self._pose = StagePose()

    def connect(self) -> None:
        import serial

        with self._lock:
            self._ser = serial.Serial(self.port, self.baudrate, timeout=self.timeout)
            time.sleep(0.3)
            self._ser.reset_input_buffer()
            self._cmd("G21")  # metric (mm)
            self._cmd("G90")  # absolute positioning
        self.pose()

    def close(self) -> None:
        with self._lock:
            if self._ser is not None:
                try:
                    self._ser.close()
                finally:
                    self._ser = None
                    self._pose = StagePose()

    def _cmd(self, line: str) -> str:
        if self._ser is None:
            raise RuntimeError("stage not connected")
        self._ser.write((line.strip() + "\n").encode("ascii"))
        self._ser.flush()
        resp = []
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < self.timeout:
            raw = self._ser.readline().decode("ascii", errors="replace").strip()
            if not raw:
                continue
            resp.append(raw)
            if raw.startswith(("ok", "error")):
                break
        return "\n".join(resp)

    def pose(self) -> StagePose:
        with self._lock:
            if self._ser is None:
                return StagePose()
            self._ser.write(b"?")
            self._ser.flush()
            raw = self._ser.readline().decode("ascii", errors="replace").strip()
            # Grbl status format: <Idle|MPos:0.000,0.000,0.000|...> or WPos:
            moving = "Run" in raw or "Jog" in raw
            for tag in ("WPos:", "MPos:"):
                if tag in raw:
                    part = raw.split(tag, 1)[1].split("|")[0].split(">")[0]
                    vals = [float(v) * self.um_per_unit for v in part.split(",")[:3]]
                    if len(vals) == 3:
                        self._pose = StagePose(vals[0], vals[1], vals[2], moving, True)
                        break
            else:
                self._pose = StagePose(self._pose.x_um, self._pose.y_um, self._pose.z_um, moving, True)
            return self._pose

    def move_to(self, x_um: float | None = None, y_um: float | None = None, z_um: float | None = None,
                wait: bool = True) -> StagePose:
        with self._lock:
            parts = []
            if x_um is not None:
                parts.append(f"X{x_um / self.um_per_unit:.5f}")
            if y_um is not None:
                parts.append(f"Y{y_um / self.um_per_unit:.5f}")
            if z_um is not None:
                parts.append(f"Z{z_um / self.um_per_unit:.5f}")
            if parts:
                self._cmd(f"G1 {' '.join(parts)} F{self.feedrate:.1f}")
                if wait:
                    self._cmd("G4 P0.02")  # dwell barrier: completes once planner buffer drains
                self._pose = StagePose(
                    self._pose.x_um if x_um is None else float(x_um),
                    self._pose.y_um if y_um is None else float(y_um),
                    self._pose.z_um if z_um is None else float(z_um),
                    False,
                    True,
                )
        return self.pose()

    def zero(self) -> StagePose:
        with self._lock:
            self._cmd("G92 X0 Y0 Z0")
            self._pose = StagePose(0.0, 0.0, 0.0, False, True)
            return self._pose
