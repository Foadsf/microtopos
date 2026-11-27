"""Motorised Z-focus drives and X-Y stage encoders (serial G-code / Grbl and simulated drives).

Provides a unified, thread-safe interface (`StageDriver`) for:
  * reading live physical (x_um, y_um, z_um) positions from stage encoders or stepper controllers,
  * jogging X, Y and Z in micrometres,
  * driving a motorised Z-sweep while the live engine captures it (`capture_z_sweep`): every frame then
    carries the stage's Z in micrometres, which `zstack.reconstruct` uses as its z labels.

Every position is in micrometres in the driver's own frame; `zero()` redefines that frame's origin.
Failures of a real controller are reported as `StageError`, never as a stale or optimistic position.
"""

from __future__ import annotations

import glob
import re
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

_HAS_PYSERIAL: bool = False
try:
    import serial.tools.list_ports  # noqa: F401  # availability check only
    _HAS_PYSERIAL = True
except ImportError:  # pyserial is optional
    pass


class StageError(RuntimeError):
    """The stage refused, did not answer, or reported something that cannot be trusted."""


@dataclass(frozen=True)
class StagePose:
    x_um: float = 0.0
    y_um: float = 0.0
    z_um: float = 0.0
    moving: bool = False
    connected: bool = False


@dataclass(frozen=True)
class StageReadout:
    """What a stage reported while a camera frame was published. Unlike the simulator's `FrameTruth` this is a
    measurement of a real (or simulated) drive, not synthetic ground truth: it carries no 'simulated' claim."""

    stage_x_um: float
    stage_y_um: float
    focus_z_um: float


def pyserial_version() -> str | None:
    """Version of the optional `pyserial` library, or None when it is not installed (the Grbl driver is then off).
    The release build records this in the executable's self-test report so a build without it cannot ship unnoticed."""
    if not _HAS_PYSERIAL:
        return None
    import serial

    return str(serial.__version__)


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
            if not self._connected:
                raise StageError("stage not connected")
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
        """Move back to the simulator's origin. This drive has no frame of its own: its coordinates are the
        simulator's (the frames carry them as simulation truth), so unlike `GrblStageDriver.zero` it cannot
        re-define where zero is."""
        return self.move_to(0.0, 0.0, 0.0)


# ------------------------------------------------------------------------------------------------ Grbl
class _GarbledStatus(ValueError):
    pass


_ERROR_RE = re.compile(r"^error:\s*(\d+)")
_REPORT_INCHES_RE = re.compile(r"^\$13\s*=\s*(\d+)")
MM_PER_INCH = 25.4

# Grbl 1.1 machine states ("Real-time Status Reports"): a commanded move has finished only in Idle. Run/Jog/Home move;
# Hold (feed hold, Hold:1 still decelerating) and Door (safety door) suspend the motion, which resumes on cycle start.
# Check ($C, G-code is parsed but not executed) and Sleep never carry out a move.
_MOVING_STATES = ("Run", "Jog", "Home")
_SUSPENDED_STATES = ("Hold", "Door")
_ALARM_RE = re.compile(r"^ALARM:\s*(\d+)")


def _triple(text: str) -> tuple[float, float, float]:
    parts = text.split(",")
    if len(parts) < 3:
        raise _GarbledStatus(f"expected x,y,z in {text!r}")
    try:
        return float(parts[0]), float(parts[1]), float(parts[2])
    except ValueError as exc:
        raise _GarbledStatus(f"not numeric: {text!r}") from exc


def parse_grbl_status(raw: str) -> tuple[str, tuple | None, tuple | None, tuple | None]:
    """Parse a Grbl 1.1 status report `<State|MPos:x,y,z|WPos:x,y,z|WCO:x,y,z|...>` (mm).

    Returns (state, mpos, wpos, wco); a field the report does not carry is None. Raises `_GarbledStatus` for
    anything that is not a well-formed report (so a corrupted line is never half-parsed into a position).
    """
    line = raw.strip()
    if not (line.startswith("<") and line.endswith(">")):
        raise _GarbledStatus(f"not a status report: {raw!r}")
    fields = line[1:-1].split("|")
    state = fields[0]
    if not state:
        raise _GarbledStatus(f"no machine state in {raw!r}")
    found: dict[str, tuple[float, float, float]] = {}
    for f in fields[1:]:
        key, sep, val = f.partition(":")
        if sep and key in ("MPos", "WPos", "WCO"):
            found[key] = _triple(val)
    return state, found.get("MPos"), found.get("WPos"), found.get("WCO")


class GrblStageDriver(StageDriver):
    """Serial G-code (Grbl 1.1) X-Y-Z controller.

    VERIFIED AGAINST A SIMULATED CONTROLLER ONLY (tests/test_feat_hardware.py models the Grbl 1.1 serial
    interface), NOT ON HARDWARE. Grbl-compatible firmwares other than 1.1 are untested.

    Commands are millimetres (G21); `um_per_unit` converts them to micrometres (default 1000). G20/G21 do not set
    the units of status reports: `$13` does ("report in inches"), so it is read from `$$` at connect and inch
    reports are converted to mm; a controller that does not list `$13` is refused (its report unit is unknown).
    The driver owns its coordinate frame: it keeps a software origin in *machine* coordinates, moves with
    `G53` (machine coordinates) and reads positions as machine positions, so the result does not depend on the
    controller's work offsets (G92, G10, G54..) nor on whether `$10` makes it report MPos or WPos. When only
    WPos is reported the machine position is WPos + WCO (WCO is sent on the first report, after any change and
    periodically). `zero()` moves the software origin; it does not touch the controller's work offsets.

    Every controller failure (`error:N`, `ALARM:N`, Alarm state, no answer, an unparsable report) raises
    `StageError`; a position is only ever returned after the controller confirmed it. A move has finished only
    when the controller is Idle: through a feed hold or an open door `move_to` keeps waiting (until its deadline),
    and in Check mode or Sleep, where the move is never carried out, it raises `StageError`.
    """

    _STATUS_RETRIES = 3
    _WCO_POLLS = 40

    def __init__(self, port: str, baudrate: int = 115200, um_per_unit: float = 1000.0,
                 feedrate_mm_min: float = 120.0, timeout: float = 2.0, serial_factory: Callable | None = None):
        self.port = port
        self.baudrate = baudrate
        self.um_per_unit = float(um_per_unit)
        self.feedrate = float(feedrate_mm_min)
        self.timeout = timeout
        self.name = f"Grbl ({port})"
        self._serial_factory = serial_factory  # (port, baudrate, timeout=...) -> pyserial-like port; default pyserial
        self._ser = None
        self._lock = threading.Lock()  # one conversation at a time on the single serial port
        self._origin_mm = (0.0, 0.0, 0.0)  # machine position of this driver's zero
        self._wco_mm: tuple | None = None  # last WCO the controller reported
        self._report_mm = 1.0  # mm per status-report unit: 1, or 25.4 when the controller reports inches ($13=1)

    # -- units: the one place micrometres and controller units meet
    def _to_um(self, mm: float) -> float:
        return float(mm) * self.um_per_unit

    def _to_mm(self, um: float) -> float:
        return float(um) / self.um_per_unit

    # -- connection
    def connect(self) -> None:
        factory = self._serial_factory
        if factory is None:
            try:
                import serial
            except ImportError as exc:
                raise ImportError("pyserial is required for a GRBL stage: pip install microtopos[serial]") from exc
            factory = serial.Serial
        with self._lock:
            self._ser = factory(self.port, self.baudrate, timeout=self.timeout)
            try:
                time.sleep(0.3)  # Grbl resets on port open and prints its banner
                self._ser.reset_input_buffer()
                self._cmd("G21")  # metric (mm)
                self._cmd("G90")  # absolute positioning
                self._report_mm = self._read_report_unit()
                self._origin_mm = (0.0, 0.0, 0.0)
                self._wco_mm = None
                self._machine_status()  # fail now, not on the first move, if the controller is not answering
            except Exception:
                self._drop_port()
                raise

    def _read_report_unit(self) -> float:
        """mm per status-report unit, from `$13` in the `$$` settings dump (callers hold self._lock)."""
        for line in self._cmd("$$"):
            m = _REPORT_INCHES_RE.match(line)
            if m:
                return MM_PER_INCH if int(m.group(1)) else 1.0
        raise StageError("the controller did not list $13 (report units) in $$; its positions cannot be read in a "
                         "known unit")

    def _drop_port(self) -> None:
        ser, self._ser = self._ser, None
        if ser is not None:
            try:
                ser.close()
            except OSError:
                pass

    def close(self) -> None:
        with self._lock:
            self._drop_port()

    # -- wire protocol (callers hold self._lock)
    def _readline(self) -> str:
        return self._ser.readline().decode("ascii", errors="replace").strip()

    def _require_port(self) -> None:
        if self._ser is None:
            raise StageError("stage not connected")

    def _cmd(self, line: str) -> list[str]:
        """Send one G-code line, return the informational lines before `ok`. Raises StageError on error:/ALARM:
        or when the controller does not answer within `timeout`."""
        self._require_port()
        self._ser.write((line.strip() + "\n").encode("ascii"))
        self._ser.flush()
        info: list[str] = []
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < self.timeout:
            raw = self._readline()
            if not raw:
                continue
            if raw == "ok":
                return info
            m = _ERROR_RE.match(raw)
            if m:
                raise StageError(f"controller rejected {line.strip()!r}: {raw}")
            m = _ALARM_RE.match(raw)
            if m:
                raise StageError(f"controller in {raw} while running {line.strip()!r} (clear it, e.g. $X or homing)")
            info.append(raw)
        raise StageError(f"no reply to {line.strip()!r} within {self.timeout:g} s")

    def _status(self) -> tuple[str, tuple | None, tuple | None, tuple | None]:
        """One confirmed status report; garbled reports are retried, then reported."""
        self._require_port()
        last = ""
        for _ in range(self._STATUS_RETRIES):
            self._ser.write(b"?")
            self._ser.flush()
            t0 = time.perf_counter()
            while time.perf_counter() - t0 < self.timeout:
                raw = self._readline()
                if not raw:
                    continue
                m = _ALARM_RE.match(raw)
                if m:
                    raise StageError(f"controller reported {raw}")
                if not raw.startswith("<"):
                    continue  # a stray 'ok' or a message from an earlier command
                try:
                    return parse_grbl_status(raw)
                except _GarbledStatus as exc:
                    last = str(exc)
                    break  # ask again
            else:
                raise StageError(f"no status report within {self.timeout:g} s")
        raise StageError(f"garbled status report ({last})")

    def _machine_status(self) -> tuple[str, tuple]:
        """(machine state without its sub-state, machine position in mm), from MPos, or from WPos + WCO."""
        for _ in range(self._WCO_POLLS):
            state, mpos, wpos, wco = self._status()
            k = self._report_mm
            mpos, wpos, wco = (None if v is None else tuple(c * k for c in v) for v in (mpos, wpos, wco))
            if wco is not None:
                self._wco_mm = wco
            base = state.split(":")[0]
            if base == "Alarm":
                raise StageError("controller is in the Alarm state (clear it with $X or homing)")
            if mpos is not None:
                return base, mpos
            if wpos is not None and self._wco_mm is not None:
                return base, tuple(w + o for w, o in zip(wpos, self._wco_mm, strict=True))
            if wpos is None:
                raise StageError("status report carries neither MPos nor WPos (check $10)")
            time.sleep(0.01)  # WPos without WCO yet: the next report includes it
        raise StageError("controller reports WPos but never sent WCO; set $10=1 to report MPos")

    def _pose_locked(self) -> StagePose:
        return self._pose_and_state_locked()[0]

    def _pose_and_state_locked(self) -> tuple[StagePose, str]:
        state, m = self._machine_status()
        o = self._origin_mm
        # anything but Idle is busy: a suspended move (Hold, Door) has not arrived either
        return StagePose(*(self._to_um(m[i] - o[i]) for i in range(3)), state != "Idle", True), state

    def pose(self) -> StagePose:
        with self._lock:
            if self._ser is None:
                return StagePose()
            return self._pose_locked()

    def move_to(self, x_um: float | None = None, y_um: float | None = None, z_um: float | None = None,
                wait: bool = True) -> StagePose:
        with self._lock:
            self._require_port()
            axes = [(a, v) for a, v in zip("XYZ", (x_um, y_um, z_um), strict=True) if v is not None]
            if not axes:
                return self._pose_locked()
            target = {a: self._to_mm(v) + self._origin_mm["XYZ".index(a)] for a, v in axes}  # machine mm
            _, here = self._machine_status()
            dist = max(abs(t - here["XYZ".index(a)]) for a, t in target.items())
            self._cmd("G90 G53 G1 " + " ".join(f"{a}{t:.5f}" for a, t in target.items()) + f" F{self.feedrate:.1f}")
            if not wait:
                return self._pose_locked()
        # The port is taken per status poll, not for the whole move: the camera thread reads the pose for every
        # frame it publishes and must not stall until the stage arrives.
        # generous bound: twice the travel time at the commanded feed rate, plus one reply timeout
        deadline = time.perf_counter() + self.timeout
        if self.feedrate > 0:
            deadline += 2.0 * dist / self.feedrate * 60.0
        while True:
            with self._lock:
                self._require_port()  # closed from another thread while waiting
                pose, state = self._pose_and_state_locked()
            if state == "Idle":
                return pose
            if state not in _MOVING_STATES + _SUSPENDED_STATES:
                raise StageError(f"the controller is in the {state} state: the move is not carried out")
            if time.perf_counter() > deadline:
                held = f" (the controller is in {state}: resume it with cycle start)" if state in _SUSPENDED_STATES \
                    else ""
                raise StageError(f"the stage did not finish the move in time{held}")
            time.sleep(0.01)

    def zero(self) -> StagePose:
        """Make the current position this driver's (0, 0, 0). The controller's own offsets are left alone."""
        with self._lock:
            self._require_port()
            _, m = self._machine_status()
            self._origin_mm = tuple(m)
            return self._pose_locked()


def capture_z_sweep(driver: StageDriver, source, engine, z_um_list: list[float], *, settle_s: float = 5.0,
                    cancelled: Callable[[], bool] | None = None,
                    progress: Callable[[float, str], None] | None = None) -> list[StagePose]:
    """Drive `driver` through `z_um_list` (µm) while `engine` captures the sweep under manual-Z override.

    One frame per commanded position is awaited: the engine must have *processed* a frame published after
    the stage settled at that Z, and every frame carries the stage's own Z reading (see
    `FrameSource._publish`), so the emitted `SweepCapture.z` is the stage's Z in µm. The capture is ended
    (override back to automatic) even on failure or cancel; the engine then emits it as usual.
    Raises `StageError` when the stage fails or no frame is processed within `settle_s` per position.
    """
    from .kinematics import Override
    from .live import PREROLL

    if not z_um_list:
        raise ValueError("empty Z list")
    poses: list[StagePose] = []
    # Approach the start with motion analysis paused (a jump in Z would otherwise be taken for a sweep that began
    # early), and let the engine's pre-roll fill with frames at the start Z only.
    engine.set_override(Override.PAUSE)
    source.label_with_stage = True  # frames carry the stage's Z only while the stage drives the focus
    try:
        first = driver.move_to(z_um=z_um_list[0])
        _wait_processed_at(source, first.z_um, settle_s, frames=PREROLL)
        engine.set_override(Override.Z)
        for i, z in enumerate(z_um_list):
            if progress:
                progress(i / len(z_um_list), f"Z {z:+.2f} um ({i + 1}/{len(z_um_list)})")
            if cancelled and cancelled():
                break
            p = driver.move_to(z_um=z)
            poses.append(p)
            _wait_processed_at(source, p.z_um, settle_s)
    finally:
        engine.set_override(Override.AUTO)
        source.label_with_stage = False
    return poses


def _wait_processed_at(source, z_um: float, timeout: float, frames: int = 1) -> None:
    """Block until a frame labelled with Z == z_um has been published and the engine has then processed `frames`
    further frames (each of them was published after the stage settled, so it carries the same Z)."""
    t0 = time.perf_counter()
    seen_seq = None
    last_consumed = None
    got = 0
    while time.perf_counter() - t0 < timeout:
        item = source.latest()
        if item is not None:
            _, _, seq, truth = item
            if seen_seq is None:
                if truth is not None and truth.focus_z_um == z_um:
                    seen_seq = seq
            else:
                c = source.consumed_seq
                if c > seen_seq and c != last_consumed:
                    got += 1
                    last_consumed = c
                    if got >= frames:
                        return
        time.sleep(0.002)
    raise StageError(f"no frame was captured with the stage at Z={z_um:g} um within {timeout:g} s "
                     "(is the live engine running on this source, and the stage attached to it?)")