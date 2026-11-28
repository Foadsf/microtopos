"""Frame sources: UVC camera (OpenCV + native V4L2 controls on Linux) and the simulator.

Both run their own thread and keep only the most recent frame (pull model): a slow consumer never
causes a backlog, it just skips frames. `latest()` returns (frame_bgr, timestamp, sequence, truth).
"""

from __future__ import annotations

import glob
import math
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass

import cv2
import numpy as np

from .optics import Objective
from .simulator import SpecimenSimulator
from .stage import StageError, StageReadout

# Capture sizes offered when the device cannot be asked for its own list (no v4l2-ctl: Windows, macOS). The driver
# rounds a size it does not have to one it has; the GUI shows the size actually delivered in "Device reports".
DEFAULT_RESOLUTION = "1920x1080"
SUPPORTED_RESOLUTIONS = (
    DEFAULT_RESOLUTION,
    "1280x720",
    "1600x1200",
    "1440x1080",
    "1080x1080",
    "1024x768",
    "640x480",
    "3840x2160",
)


@dataclass
class CameraInfo:
    index: int
    name: str
    path: str = ""


def list_cameras() -> list[CameraInfo]:
    """Video capture devices. On Linux, uses sysfs names and skips metadata-only nodes."""
    cams: list[CameraInfo] = []
    if sys.platform.startswith("linux"):
        for dev in sorted(glob.glob("/dev/video*"), key=lambda p: int(re.sub(r"\D", "", p) or 0)):
            idx = int(re.sub(r"\D", "", dev))
            try:
                with open(f"/sys/class/video4linux/video{idx}/name") as fh:
                    name = fh.read().strip()
            except OSError:
                name = dev
            if _v4l2_is_capture(dev):
                cams.append(CameraInfo(idx, name, dev))
        return cams
    for idx in range(4):
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            cams.append(CameraInfo(idx, f"Camera {idx} ({cap.getBackendName()})"))
        cap.release()
    return cams


def _v4l2_is_capture(dev: str) -> bool:
    """Whether `dev` lists at least one capture pixel format (metadata-only nodes list none).

    Both layouts of `v4l2-ctl --list-formats` are accepted: current v4l-utils print `[0]: 'MJPG' ...`, older ones
    print `Index : 0 / Type : Video Capture / Pixel Format: 'MJPG'`. The second form was added in 59a96f2 so such
    systems do not show an empty camera list; checked against sample outputs only, NOT hardware-verified here.
    """
    if shutil.which("v4l2-ctl"):
        try:
            out = subprocess.run(["v4l2-ctl", "-d", dev, "--list-formats"], capture_output=True, text=True,
                                 timeout=3, check=False).stdout
            return "[0]:" in out or "Pixel Format" in out
        except (subprocess.SubprocessError, OSError):
            return False
    return True


def camera_modes(info: CameraInfo) -> list[str]:
    """Capture sizes ("WxH") the device offers for MJPG, largest first; MJPG is the format CameraSource requests.

    Read from `v4l2-ctl --list-formats-ext` (Linux). A device that lists no MJPG contributes all of its sizes. When
    the device cannot be asked (no v4l2-ctl, no device path, an error or an empty answer), SUPPORTED_RESOLUTIONS is
    returned: those sizes are requests that the driver may round to a mode it has."""
    if info.path and shutil.which("v4l2-ctl"):
        try:
            out = subprocess.run(["v4l2-ctl", "-d", info.path, "--list-formats-ext"], capture_output=True, text=True,
                                 timeout=3, check=False).stdout
        except (subprocess.SubprocessError, OSError):
            out = ""
        by_fmt: dict[str, list[str]] = {}
        fmt = ""
        for line in out.splitlines():
            # a format header in either v4l-utils layout ("[0]: 'MJPG' ..." or "Pixel Format: 'MJPG' ..."); the
            # FourCC is the four characters between the quotes, spaces included ('Y16 ', 'Y8  ')
            m = re.search(r"(?:\[\d+\]:|Pixel Format\s*:)\s*'([^']*)'", line)
            if m:
                fmt = m.group(1).strip()
                continue
            m = re.search(r"Size:\s*Discrete\s+(\d+)x(\d+)", line)
            if m and fmt:
                size = f"{int(m.group(1))}x{int(m.group(2))}"
                if size not in by_fmt.setdefault(fmt, []):
                    by_fmt[fmt].append(size)
        sizes = by_fmt.get("MJPG") or sorted({s for v in by_fmt.values() for s in v})
        if sizes:
            def area(s: str) -> int:
                w, h = s.split("x")
                return int(w) * int(h)
            return sorted(sizes, key=lambda s: (-area(s), s))
    return list(SUPPORTED_RESOLUTIONS)


def pick_resolution(modes: list[str], saved: str | None) -> str:
    """The size to open: the user's saved choice if this camera offers it, else full HD, else the largest offered."""
    if saved in modes:
        return str(saved)
    if DEFAULT_RESOLUTION in modes:
        return DEFAULT_RESOLUTION
    return modes[0]


class V4L2Controls:
    """Thin wrapper over `v4l2-ctl` for UVC controls and `gphoto2` for loopback DSLRs."""

    def __init__(self, path: str):
        self.path = path
        self.available = bool(path) and shutil.which("v4l2-ctl") is not None
        self.gphoto_available = shutil.which("gphoto2") is not None
        self._lock = threading.Lock()
        self.last_error: str | None = None

    def list(self) -> dict[str, dict]:
        if not self.available:
            return {}
        with self._lock:
            try:
                out = subprocess.run(["v4l2-ctl", "-d", self.path, "--list-ctrls"], capture_output=True, text=True,
                                     timeout=3, check=False).stdout
                self.last_error = None
            except (subprocess.SubprocessError, OSError) as e:
                self.last_error = str(e)
                return {}
        ctrls = {}
        for line in out.splitlines():
            m = re.match(r"\s*(\w+)\s+0x[0-9a-f]+\s+\((\w+)\)\s*:\s*(.*)", line)
            if m:
                d = dict(re.findall(r"(\w+)=(-?\d+)", m.group(3)))
                ctrls[m.group(1)] = {"type": m.group(2), **{k: int(v) for k, v in d.items()},
                                     "inactive": "inactive" in m.group(3)}
        return ctrls

    def set(self, **values) -> dict[str, int]:
        """Set controls in order and return what the device reports afterwards."""
        if not self.available:
            return {}
        with self._lock:
            for k, v in values.items():
                try:
                    subprocess.run(["v4l2-ctl", "-d", self.path, f"--set-ctrl={k}={int(v)}"],
                                   capture_output=True, timeout=3, check=False)
                except (subprocess.SubprocessError, OSError) as e:
                    self.last_error = str(e)
                    return {}
        cur = self.list()
        return {k: cur.get(k, {}).get("value") for k in values}


CAPTURE_THREAD_PREFIX = "capture:"  # a source's thread is named after it, so a thread left running is traceable


class FrameSource:
    _ids = 0

    def __init__(self):
        self._lock = threading.Lock()
        self._frame = None
        self._jpeg = None  # (seq, the camera's JPEG bytes or the recorded .jpg path) of the newest frame, when known
        self.camera_name = ""  # the physical camera behind the frames (live or recorded), "" when there is none
        self._seq = 0
        self._running = False
        self._thread: threading.Thread | None = None
        self.fps = 0.0
        self.error: str | None = None
        self.warnings: list[str] = []
        self._consumed = 0  # last sequence number the engine finished (used by the ordered replay mode)
        self.stage_driver = None  # optional StageDriver for live encoder / motorised Z readout
        # True only while stage.capture_z_sweep drives the stage: then, and only then, is the stage's Z the height
        # of each frame. Outside a motorised sweep the focus is moved by hand (or the frames are a replay), and a
        # stage label would replace the frame-index Z of a hand sweep with one constant stage Z.
        self.label_with_stage = False
        FrameSource._ids += 1
        self.source_id = f"{type(self).__name__}-{FrameSource._ids}"

    @property
    def consumed_seq(self) -> int:
        """Sequence number of the last frame the consumer reported as processed."""
        return self._consumed

    def consumed(self, seq: int) -> None:
        """Called by the consumer after processing frame `seq` (back-pressure for lossless replay)."""
        self._consumed = max(self._consumed, seq)

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True, name=f"{CAPTURE_THREAD_PREFIX}{self.source_id}")
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)

    def _publish(self, frame, truth=None, ts: float | None = None, step: int = 1, jpeg=None):
        """Publish a frame. ts: capture time in seconds (default: now); step: source frames since the previous
        published frame (> 1 when frames are known to be missing), so sequence gaps mean real gaps. jpeg: the
        compressed frame exactly as the camera sent it (bytes), or the recorded file holding those bytes (Path), so
        that still images can be restored from the JPEG data itself.

        During a motorised Z sweep (`label_with_stage`, set by `stage.capture_z_sweep`) with a StageDriver attached
        (`stage_driver`), a frame without truth of its own is labelled with the stage's pose read now
        (`StageReadout`, z in um). A stage that fails or is disconnected leaves the frame unlabelled: a missing
        label, never a made-up one.
        """
        if truth is None and self.label_with_stage and self.stage_driver is not None:
            try:
                p = self.stage_driver.pose()
                if p.connected:
                    truth = StageReadout(p.x_um, p.y_um, p.z_um)
            except (StageError, OSError):
                pass  # no label for this frame; the driver reports its failure to whoever commands it
        with self._lock:
            self._seq += max(1, int(step))
            self._frame = (frame, time.perf_counter() if ts is None else ts, self._seq, truth)
            self._jpeg = (self._seq, jpeg) if jpeg is not None else None

    def latest(self):
        with self._lock:
            return self._frame

    def jpeg_for(self, seq: int):
        """The JPEG bytes (or recorded .jpg path) of frame `seq` if it is still the newest one, else None."""
        with self._lock:
            return self._jpeg[1] if self._jpeg is not None and self._jpeg[0] == seq else None

    def _run(self):
        raise NotImplementedError


class CameraSource(FrameSource):
    """UVC / DirectShow / AVFoundation / loopback camera."""

    def __init__(self, info: CameraInfo, width: int = 1920, height: int = 1080, fps: int = 25):
        super().__init__()
        self.info = info
        self.camera_name = info.name
        self.req = (width, height, fps)
        self.controls = V4L2Controls(info.path)
        self.readback: dict = {}
        # _ctrl_lock guards only the request (_want, _apply_pending) and the result (applied, applied_seq).
        # It is never held while talking to the device: v4l2-ctl can take seconds, and the GUI thread asks here.
        self._ctrl_lock = threading.Lock()
        self._want: tuple[int | None, bool] | None = None
        self._apply_pending = False
        self.applied: dict = {}  # what the device reported after the last time the controls were applied
        self.applied_seq = 0  # incremented every time `applied` is replaced (the GUI refreshes its readback on it)

    def open(self) -> cv2.VideoCapture:
        if sys.platform.startswith("linux"):
            api = cv2.CAP_V4L2
        elif sys.platform == "win32":
            # DirectShow rather than OpenCV's default (MSMF) on Windows: the manual-exposure fallback in
            # _apply_controls writes CAP_PROP_EXPOSURE as log2(seconds), the DirectShow convention, and falls back
            # to CAP_ANY just below if the device cannot be opened that way. Introduced in 59a96f2 without a
            # recorded measurement; NOT hardware-verified here (no camera was attached when this was reviewed).
            api = cv2.CAP_DSHOW
        elif sys.platform == "darwin":
            api = cv2.CAP_AVFOUNDATION
        else:
            api = cv2.CAP_ANY
        cap = cv2.VideoCapture(self.info.index, api)
        if not cap.isOpened() and api != cv2.CAP_ANY:
            cap = cv2.VideoCapture(self.info.index, cv2.CAP_ANY)
        if not cap.isOpened():
            raise RuntimeError(f"cannot open camera {self.info.name}")
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.req[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.req[1])
        cap.set(cv2.CAP_PROP_FPS, self.req[2])
        fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
        self.readback = {
            "fourcc": "".join(chr((fourcc >> 8 * i) & 0xFF) for i in range(4)).strip() or "RAW",
            "size": f"{int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}",
            "fps_nominal": cap.get(cv2.CAP_PROP_FPS), "backend": cap.getBackendName(),
        }
        return cap

    def exposure_range(self) -> tuple[int, int] | None:
        """(min, max) of exposure_time_absolute in device units, when the driver reports it."""
        if self.controls.available:
            c = self.controls.list().get("exposure_time_absolute")
            if c and "min" in c and "max" in c:
                return int(c["min"]), int(c["max"])
        return None

    def set_manual_exposure(self, exposure: int | None, sharpness_off: bool = True) -> dict:
        """exposure in the device's units (UVC: 100 us). None = automatic exposure.

        Remembered and re-applied by the capture thread after the first frame and after every reconnect,
        because many UVC drivers reset or ignore controls set before streaming starts.

        While streaming this only records the request and returns at once: the capture thread applies it with
        the next frame, then replaces `applied` and increments `applied_seq`. The value returned in that case is
        the PREVIOUS device report; read `applied` after `applied_seq` has changed for the new one."""
        with self._ctrl_lock:
            self._want = (exposure, sharpness_off)
            self._apply_pending = True
            if self._running:
                return dict(self.applied)
        return self._apply_now()

    def _apply_now(self, cap=None) -> dict:
        """Apply the latest request to the device and publish what it reports. The lock is not held during I/O;
        a request that arrives meanwhile leaves `_apply_pending` set and is applied on the next call."""
        with self._ctrl_lock:
            self._apply_pending = False
            want = self._want
        try:
            applied = self._apply_controls(cap, want)
        except Exception as e:  # noqa: BLE001 - a missing control must not stop the stream
            applied = {"error": str(e)}
        with self._ctrl_lock:
            self.applied = applied
            self.applied_seq += 1
        return dict(applied)

    def _apply_controls(self, cap=None, want: tuple[int | None, bool] | None = None) -> dict:
        if want is None:
            return {}
        exposure, sharpness_off = want
        if self.controls.available:
            vals = {}
            ctrls = self.controls.list()
            if ctrls:
                if "auto_exposure" in ctrls:
                    vals["auto_exposure"] = 3 if exposure is None else 1
                if exposure is not None and "exposure_time_absolute" in ctrls:
                    c = ctrls["exposure_time_absolute"]
                    vals["exposure_time_absolute"] = int(np.clip(exposure, c.get("min", 1), c.get("max", 10000)))
                if sharpness_off and "sharpness" in ctrls:
                    vals["sharpness"] = ctrls["sharpness"].get("min", 0)
                if "backlight_compensation" in ctrls:
                    vals["backlight_compensation"] = 0
                if exposure is not None:
                    for k in ("exposure_dynamic_framerate", "exposure_auto_priority", "white_balance_automatic",
                              "white_balance_temperature_auto", "gain_automatic", "autogain"):
                        if k in ctrls:
                            vals[k] = 0
                return self.controls.set(**vals)
        # Cross-platform OpenCV property fallback (Windows DirectShow/MSMF, macOS AVFoundation, Linux loopback)
        if cap is not None:
            got = {}
            if exposure is None:
                cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75 if sys.platform.startswith("linux") else 1)
            else:
                cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25 if sys.platform.startswith("linux") else 0)
                v = math.log2(max(exposure, 1) * 1e-4) if sys.platform == "win32" else float(exposure)
                cap.set(cv2.CAP_PROP_EXPOSURE, v)
                cap.set(cv2.CAP_PROP_AUTO_WB, 0)
                # Gain is deliberately left to the device: 59a96f2 forced CAP_PROP_GAIN=0 here, which silently darkens
                # the image on cameras whose gain 0 is not "unity" and is not reported in the read-back. The V4L2 path
                # only switches automatic gain off (it keeps the current value); this path cannot do even that.
            if sharpness_off:
                cap.set(cv2.CAP_PROP_SHARPNESS, 0)
            got["exposure"] = round(float(cap.get(cv2.CAP_PROP_EXPOSURE)), 3)
            got["sharpness"] = round(float(cap.get(cv2.CAP_PROP_SHARPNESS)), 3)
            return got
        return {}

    def start_recording(self, folder) -> None:
        """Save every frame plus a manifest (see recording.py); replay with --replay. Raises if the folder is
        not empty. MJPEG bytes are saved as they came; decoded frames (no raw mode) as lossless PNG."""
        from .recording import RecordingWriter

        self.rec_writer = RecordingWriter(folder, self.info.name, dict(self.readback), dict(self.applied))
        self.rec_dir = self.rec_writer.folder
        self.rec_t0 = time.perf_counter()
        self.rec_error = None
        self.recording = True

    @property
    def rec_frames(self) -> int:
        w = getattr(self, "rec_writer", None)
        return w.n if w is not None else 0

    def stop_recording(self) -> int:
        self.recording = False
        w = getattr(self, "rec_writer", None)
        if w is None:
            return 0
        n = w.close()
        self.rec_error = w.error
        self.rec_writer = None
        return n

    def _run(self):
        try:
            cap = self.open()
        except RuntimeError as e:
            self.error = str(e)
            return
        cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)  # ask for MJPEG bytes; decode here
        self.recording = False
        self.rec_writer = None
        t0, n, n_total = time.perf_counter(), 0, 0
        last_ok = time.perf_counter()
        while self._running:
            ok, raw = cap.read()
            if not ok or raw is None:
                if time.perf_counter() - last_ok > 3.0:  # stalled or unplugged: say so and try to reopen
                    self.error = "no frames from the camera for 3 s; reconnecting (check the USB cable)"
                    cap.release()
                    time.sleep(1.0)
                    try:
                        cap = self.open()
                        cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)
                        n_total = 0  # re-apply exposure and sharpness on the reopened device
                    except RuntimeError:
                        pass
                    last_ok = time.perf_counter()
                time.sleep(0.01)
                continue
            last_ok = time.perf_counter()
            self.error = None
            if self._apply_pending or n_total == 0:
                self._apply_now(cap)
            n_total += 1
            if raw.ndim == 3:
                frame = raw
                self.readback["delivery"] = "decoded by the backend"
            else:
                frame = cv2.imdecode(raw, cv2.IMREAD_COLOR)
                if frame is None:
                    self.warnings.append("a camera frame did not decode")
                    del self.warnings[:-20]
                    continue
                self.readback["delivery"] = "MJPEG bytes"
            w = self.rec_writer  # read once: Stop on the GUI thread may clear it meanwhile
            if self.recording and w is not None:
                try:
                    w.add(raw, n_total, time.perf_counter() - self.rec_t0)
                except Exception as e:  # noqa: BLE001
                    if not (w.closed and w.error is None):
                        self.rec_error = str(e)
                        self.stop_recording()
                        self.rec_error = str(e)
            self._publish(frame, jpeg=raw.tobytes() if raw.ndim != 3 else None)
            n += 1
            dt = time.perf_counter() - t0
            if dt > 1.0:
                self.fps, t0, n = n / dt, time.perf_counter(), 0
        cap.release()
        if self.rec_writer is not None:  # the stream ended while recording: finalise the manifest
            self.stop_recording()


class SimulatorSource(FrameSource):
    """Renders the synthetic specimen at ~25 fps and follows scripted pan / sweep demos."""

    def __init__(self, seed: int, objective: Objective, width: int = 640, height: int = 360):
        super().__init__()
        self.lock = threading.Lock()
        self.size = (width, height)
        self.sim = SpecimenSimulator(seed, objective, width, height)
        self.script: list = []  # queue of (stage_x_um, stage_y_um, focus_z_um)
        self.seed = seed

    def set_objective(self, objective: Objective):
        with self.lock:
            self.sim = SpecimenSimulator(self.seed, objective, *self.size)
            self.script = []

    def _best_focus(self) -> float:
        _, hz = self.sim.truth_patch()
        return float(np.median(hz))

    def demo_pan(self, n: int = 150):
        with self.lock:
            s = self.sim
            fov = s.out_w * s.um_per_px
            t = np.linspace(0, 2 * np.pi, n)
            x0, y0 = s.stage_x_um, s.stage_y_um
            xs = x0 + 0.45 * fov * np.sin(t)
            ys = y0 + 0.3 * fov * (1 - np.cos(t))
            zf = self._best_focus()
            refocus = [(x0, y0, float(z)) for z in np.linspace(s.focus_z_um, zf, 20)]
            self.script = refocus + [(x, y, zf) for x, y in zip(xs, ys)]

    def demo_sweep(self, reversal: bool = False):
        with self.lock:
            s = self.sim
            zs = s.sweep_trajectory(reversal=reversal)
            # move to the start smoothly (as a hand would), pause briefly, then sweep
            approach = np.linspace(s.focus_z_um, zs[0], 25)
            p = lambda z: (s.stage_x_um, s.stage_y_um, float(z))
            back = np.linspace(zs[-1], self._best_focus(), 30)  # refocus afterwards, as a user would
            self.script = [p(z) for z in approach] + [p(zs[0])] * 6 + [p(z) for z in zs] + [p(zs[-1])] * 25 \
                + [p(z) for z in back]

    def relief_range_um(self) -> tuple[float, float]:
        """Focus heights (um) that bracket the visible relief by 3 DOF each side: a stage Z sweep over this
        range passes every pixel through focus (the same bracket as `SpecimenSimulator.sweep_trajectory`)."""
        with self.lock:
            _, hz = self.sim.truth_patch()
            dof = self.sim.objective.dof_um
        return float(hz.min()) - 3 * dof, float(hz.max()) + 3 * dof

    def busy(self) -> bool:
        return bool(self.script)

    def _run(self):
        period = 1 / 25.0
        t0, n = time.perf_counter(), 0
        while self._running:
            tick = time.perf_counter()
            with self.lock:
                s = self.sim
                if self.script:
                    s.stage_x_um, s.stage_y_um, s.focus_z_um = self.script.pop(0)
                bgr, truth = s.render()
            self._publish(bgr, truth)
            n += 1
            if time.perf_counter() - t0 > 1:
                self.fps, t0, n = n / (time.perf_counter() - t0), time.perf_counter(), 0
            time.sleep(max(0.0, period - (time.perf_counter() - tick)))


def opencv_threads(n: int | None = None) -> None:
    cv2.setNumThreads(n if n is not None else max(1, (os.cpu_count() or 2) - 1))


class ReplaySource(FrameSource):
    """Plays back a recording with full transport controls (pause, resume, seek, frame-step, speed, loop)."""

    def __init__(self, folder: str, speed: float = 1.0, loop: bool = False, exhaustive: bool = False, *,
                 clock=time.perf_counter, sleep=time.sleep):
        super().__init__()
        # The playback schedule reads time and waits only through these two, so a test can run it on a virtual clock
        # whose result does not depend on how fast the machine is (the app always uses the real ones).
        self._now = clock
        self._sleep = sleep
        from .recording import load_manifest

        self.manifest = load_manifest(folder)
        self.camera_name = str(self.manifest.meta.get("camera", "") or "")
        self.folder = self.manifest.folder
        self.files = self.manifest.files
        self.stamps = self.manifest.stamps
        self.warnings = list(self.manifest.warnings)
        self.speed = max(0.05, float(speed))
        self.loop = bool(loop)
        self.exhaustive = exhaustive
        self.readback = {"replay": str(self.folder), "frames": len(self.files), "mode": self.manifest.mode,
                         **self.manifest.meta.get("readback", {})}
        self.done = False
        self.corrupt = 0
        self.paused = False
        self.position = 0
        self._seek_target: int | None = None
        self._seek_dir = 1  # where a seek or step moves on when its target frame is damaged (+1 / -1)
        self._ctrl_lock = threading.Lock()
        self._last_pub: tuple[float, float] | None = None  # (wall time, media time) of the last published frame

    @property
    def total_frames(self) -> int:
        return len(self.files)

    @property
    def transport_supported(self) -> bool:
        """False in exhaustive (lossless) mode: the consumer paces that playback, so pause, seek, step and speed
        have no meaning and are refused (RuntimeError) instead of being silently ignored."""
        return not self.exhaustive

    def _need_transport(self, what: str) -> None:
        if self.exhaustive:
            raise RuntimeError(f"{what} is not available in exhaustive replay mode (the consumer paces the playback)")

    def pause(self) -> None:
        self._need_transport("pause")
        with self._ctrl_lock:
            self.paused = True

    def resume(self) -> None:
        self._need_transport("resume")
        with self._ctrl_lock:
            self.paused = False
            if self.done:
                self._seek_target = 0
                self.done = False

    def toggle_pause(self) -> bool:
        if self.paused:
            self.resume()
        else:
            self.pause()
        return self.paused

    def seek(self, index: int) -> None:
        """Jump to frame `index` (0 .. total_frames - 1) and publish it immediately."""
        self._need_transport("seek")
        with self._ctrl_lock:
            self._seek_target = int(np.clip(index, 0, max(0, len(self.files) - 1)))
            self._seek_dir = -1 if self._seek_target < self.position else 1
            self.done = False

    def step_frames(self, delta: int = 1) -> None:
        """Pause and step by `delta` frames from the last published position.

        The step bases on ``self.position`` (the frame the playback thread has
        actually published), never on a pending ``_seek_target``: ``resume()`` at
        EOF may leave ``_seek_target == 0`` until the thread wakes up to consume
        it, and ``pause()`` can arrive before that consumption. A step must never
        restart from that stale target. Consequently two steps issued before the
        thread has published the first one are one step (the second is computed from
        the same published position), and a frame the thread was already decoding when
        the step arrived is dropped rather than shown after it.
        """
        self._need_transport("step")
        with self._ctrl_lock:
            self.paused = True
            base = self.position
            self._seek_target = int(np.clip(base + delta, 0, max(0, len(self.files) - 1)))
            self._seek_dir = -1 if delta < 0 else 1
            self.done = False

    def set_speed(self, speed: float) -> None:
        """Playback speed multiplier. The schedule is re-anchored on the next loop pass, so the new speed applies
        from the frame published last: no stall when slowing down, no burst of catch-up frames when speeding up."""
        self._need_transport("speed change")
        with self._ctrl_lock:
            self.speed = max(0.05, float(speed))

    def set_loop(self, loop: bool) -> None:
        with self._ctrl_lock:
            self.loop = bool(loop)

    def frames(self):
        """Every decodable frame in order (damaged files are counted in self.corrupt and skipped)."""
        from .recording import decode

        for f in self.files:
            img = decode(f)
            if img is None:
                self.corrupt += 1
                continue
            yield img

    def _jpeg_file(self, idx: int):
        """The recorded file of frame idx when it holds the camera's JPEG bytes unchanged (mjpeg-raw .jpg), else None."""
        f = self.files[idx]
        return f if f.suffix.lower() == ".jpg" else None

    def _commit(self, idx: int, img, ts: float, step: int, *, seek: bool) -> bool:
        """Publish frame `idx` unless a newer transport request arrived while it was being decoded.

        The check and the publication are one step under the control lock, so a pause or a seek/step that returned
        before this call can never be followed by a frame decoded before it (a stale frame). A dropped frame is not
        lost: the thread comes back to it on resume, or the seek replaces it.
        """
        with self._ctrl_lock:
            if self._seek_target is not None or (self.paused and not seek):
                return False
            self.position = idx
            self._publish(img, ts=ts, step=step, jpeg=self._jpeg_file(idx))
            self._last_pub = (self._now(), ts)
            return True

    def _run(self):
        from .recording import decode

        if self.exhaustive:
            prev_seq = None
            while self._running:
                for idx, (f, ts, rseq) in enumerate(zip(self.files, self.stamps, self.manifest.seqs)):
                    if not self._running:
                        return
                    step = 1 if prev_seq is None or rseq <= prev_seq else rseq - prev_seq
                    while self._running and self._consumed < self._seq:
                        time.sleep(0.002)
                    img = decode(f)
                    if img is None:
                        self.corrupt += 1
                        self.error = f"{self.corrupt} damaged frame(s) skipped (last: {f.name})"
                        continue
                    self.position = idx
                    self._publish(img, ts=ts, step=step, jpeg=self._jpeg_file(idx))
                    prev_seq = rseq
                    self.fps = len(self.stamps) / max(self.stamps[-1], 1e-3)
                if not self.loop:
                    self.done = True
                    return
                prev_seq = None
            return

        idx = 0  # the next frame to publish
        prev_seq = None
        t_anchor = self._now()  # wall time that corresponds to media time ts_anchor
        ts_anchor = self.stamps[0] if self.stamps else 0.0
        anchor_speed = self.speed
        while self._running:
            with self._ctrl_lock:
                seek_to = self._seek_target
                seek_dir = self._seek_dir
                self._seek_target = None
                paused = self.paused
                spd = self.speed
            if spd != anchor_speed:
                # The schedule (t_anchor, ts_anchor) was laid out at the old speed. Re-anchor it on the frame published
                # last, so the new speed counts from there: elapsed wall time is not re-read as media time at the new rate.
                anchor_speed = spd
                last = self._last_pub
                if last is not None and idx < len(self.stamps) and self.stamps[idx] >= last[1]:
                    t_anchor, ts_anchor = last
                else:
                    t_anchor = self._now()
                    ts_anchor = self.stamps[min(idx, len(self.stamps) - 1)] if self.stamps else 0.0
            if seek_to is not None:
                idx = seek_to
                prev_seq = None
                img = decode(self.files[idx])
                while img is None:  # a damaged target is counted and passed in the direction of travel, as in playback
                    self.corrupt += 1
                    self.error = f"{self.corrupt} damaged frame(s) skipped (last: {self.files[idx].name})"
                    if not 0 <= idx + seek_dir < len(self.files):
                        break
                    idx += seek_dir
                    img = decode(self.files[idx])
                ts, rseq = self.stamps[idx], self.manifest.seqs[idx]
                if img is not None:
                    if not self._commit(idx, img, ts, 1, seek=True):
                        continue  # a newer seek or step arrived while decoding: serve that one only
                    prev_seq = rseq
                t_anchor = self._now()
                ts_anchor = ts
                idx += 1  # the shown frame is not published again on resume
                continue
            if paused:
                self._sleep(0.02)
                t_anchor = self._now()
                if 0 < idx <= len(self.stamps):
                    ts_anchor = self.stamps[idx - 1]  # resume continues from the frame shown, one interval later
                continue
            if idx >= len(self.files):
                if self.loop:
                    with self._ctrl_lock:
                        self.done = False  # loop may have been switched on after the end: playing again, not done
                    idx = 0
                    prev_seq = None
                    # frame 0 follows the last frame after one more recorded interval (the last one), like every
                    # other frame: re-anchoring at "now" showed the last frame for 0 s at every wrap
                    last = self._last_pub
                    gap = self.stamps[-1] - self.stamps[-2] if len(self.stamps) > 1 else 0.0
                    if last is not None and self.stamps and last[1] == self.stamps[-1]:
                        t_anchor = last[0] + max(gap, 0.0) / spd
                    else:
                        t_anchor = self._now()
                    ts_anchor = self.stamps[0] if self.stamps else 0.0
                    continue
                self.done = True
                self._sleep(0.05)
                continue
            f, ts, rseq = self.files[idx], self.stamps[idx], self.manifest.seqs[idx]
            wait = (ts - ts_anchor) / spd - (self._now() - t_anchor)
            if wait > 0:
                self._sleep(min(wait, 0.05))
                if wait > 0.05:
                    continue
            img = decode(f)
            if img is None:
                self.corrupt += 1
                self.error = f"{self.corrupt} damaged frame(s) skipped (last: {f.name})"
                idx += 1
                continue
            step = 1 if prev_seq is None or rseq <= prev_seq else rseq - prev_seq
            if not self._commit(idx, img, ts, step, seek=False):
                continue  # paused or seeked while decoding: this frame is stale now
            # More than one frame interval behind (a slow decode, an oversleep, a loaded machine): the schedule is
            # re-anchored on this frame instead of catching up, so playback never runs faster than the set speed
            # (catching up sent the overdue frames back to back: 2.9 ms/frame at a 5 ms interval on macOS CI).
            nxt = self.stamps[idx + 1] if idx + 1 < len(self.stamps) else ts
            if self._now() - (t_anchor + (ts - ts_anchor) / spd) > max(nxt - ts, 0.0) / spd:
                t_anchor, ts_anchor = self._now(), ts
            prev_seq = rseq
            self.fps = len(self.stamps) / max(self.stamps[-1], 1e-3) * spd
            idx += 1
