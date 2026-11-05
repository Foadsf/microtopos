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
                name = open(f"/sys/class/video4linux/video{idx}/name").read().strip()
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
    if shutil.which("v4l2-ctl"):
        try:
            out = subprocess.run(["v4l2-ctl", "-d", dev, "--list-formats"], capture_output=True, text=True,
                                 timeout=3).stdout
            return "[0]:" in out
        except (subprocess.SubprocessError, OSError):
            return False
    return True


class V4L2Controls:
    """Thin wrapper over `v4l2-ctl` for UVC controls OpenCV cannot reliably set (Linux only)."""

    def __init__(self, path: str):
        self.path = path
        self.available = bool(path) and shutil.which("v4l2-ctl") is not None

    def list(self) -> dict[str, dict]:
        if not self.available:
            return {}
        out = subprocess.run(["v4l2-ctl", "-d", self.path, "--list-ctrls"], capture_output=True, text=True,
                             timeout=3).stdout
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
        for k, v in values.items():
            subprocess.run(["v4l2-ctl", "-d", self.path, f"--set-ctrl={k}={int(v)}"], capture_output=True, timeout=3)
        cur = self.list()
        return {k: cur.get(k, {}).get("value") for k in values}


class FrameSource:
    _ids = 0

    def __init__(self):
        self._lock = threading.Lock()
        self._frame = None
        self._seq = 0
        self._running = False
        self._thread: threading.Thread | None = None
        self.fps = 0.0
        self.error: str | None = None
        self.warnings: list[str] = []
        self._consumed = 0  # last sequence number the engine finished (used by the ordered replay mode)
        FrameSource._ids += 1
        self.source_id = f"{type(self).__name__}-{FrameSource._ids}"

    def consumed(self, seq: int) -> None:
        """Called by the consumer after processing frame `seq` (back-pressure for lossless replay)."""
        self._consumed = max(self._consumed, seq)

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)

    def _publish(self, frame, truth=None, ts: float | None = None, step: int = 1):
        """Publish a frame. ts: capture time in seconds (default: now); step: source frames since the previous
        published frame (> 1 when frames are known to be missing), so sequence gaps mean real gaps."""
        with self._lock:
            self._seq += max(1, int(step))
            self._frame = (frame, time.perf_counter() if ts is None else ts, self._seq, truth)

    def latest(self):
        with self._lock:
            return self._frame

    def _run(self):
        raise NotImplementedError


class CameraSource(FrameSource):
    """UVC camera: MJPG at the requested size; on Linux, manual exposure and no in-camera sharpening."""

    def __init__(self, info: CameraInfo, width: int = 1920, height: int = 1080, fps: int = 25):
        super().__init__()
        self.info = info
        self.req = (width, height, fps)
        self.controls = V4L2Controls(info.path)
        self.readback: dict = {}

    def open(self) -> cv2.VideoCapture:
        api = cv2.CAP_V4L2 if sys.platform.startswith("linux") else cv2.CAP_ANY
        cap = cv2.VideoCapture(self.info.index, api)
        if not cap.isOpened():
            raise RuntimeError(f"cannot open camera {self.info.name}")
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.req[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.req[1])
        cap.set(cv2.CAP_PROP_FPS, self.req[2])
        fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
        self.readback = {
            "fourcc": "".join(chr((fourcc >> 8 * i) & 0xFF) for i in range(4)),
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
        because many UVC drivers reset or ignore controls set before streaming starts."""
        self._want = (exposure, sharpness_off)
        self._apply_pending = True
        return self._apply_controls()

    def _apply_controls(self, cap=None) -> dict:
        want = getattr(self, "_want", None)
        if want is None:
            return {}
        exposure, sharpness_off = want
        if self.controls.available:
            vals = {}
            ctrls = self.controls.list()
            if "auto_exposure" in ctrls:
                vals["auto_exposure"] = 3 if exposure is None else 1
            if exposure is not None and "exposure_time_absolute" in ctrls:
                c = ctrls["exposure_time_absolute"]
                vals["exposure_time_absolute"] = int(np.clip(exposure, c.get("min", 1), c.get("max", 10000)))
            if sharpness_off and "sharpness" in ctrls:
                vals["sharpness"] = ctrls["sharpness"].get("min", 0)
            if "backlight_compensation" in ctrls:
                vals["backlight_compensation"] = 0
            # anything else that changes brightness or colour between frames is locked during measurement
            if exposure is not None:
                for k in ("exposure_dynamic_framerate", "exposure_auto_priority", "white_balance_automatic",
                          "white_balance_temperature_auto", "gain_automatic", "autogain"):
                    if k in ctrls:
                        vals[k] = 0
            return self.controls.set(**vals)
        if cap is not None:  # no v4l2-ctl (Windows, macOS): best-effort OpenCV properties, from the capture thread
            got = {}
            if exposure is None:
                cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75 if sys.platform.startswith("linux") else 1)
            else:
                cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25 if sys.platform.startswith("linux") else 0)
                # DirectShow / MSMF take log2 seconds; 100 us units -> seconds
                v = math.log2(max(exposure, 1) * 1e-4) if sys.platform == "win32" else float(exposure)
                cap.set(cv2.CAP_PROP_EXPOSURE, v)
                cap.set(cv2.CAP_PROP_AUTO_WB, 0)
            if sharpness_off:
                cap.set(cv2.CAP_PROP_SHARPNESS, 0)
            got["exposure"] = cap.get(cv2.CAP_PROP_EXPOSURE)
            return got
        return {}

    def start_recording(self, folder) -> None:
        """Save every frame plus a manifest (see recording.py); replay with --replay. Raises if the folder is
        not empty. MJPEG bytes are saved as they came; decoded frames (no raw mode) as lossless PNG."""
        from .recording import RecordingWriter

        self.rec_writer = RecordingWriter(folder, self.info.name, dict(self.readback), dict(getattr(self, "applied", {})))
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
        cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)  # ask for MJPEG bytes; decode here (and record them if asked)
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
            if getattr(self, "_apply_pending", False) or n_total == 0:
                self._apply_pending = False
                try:
                    self.applied = self._apply_controls(cap)
                except Exception as e:  # a missing control must not stop the stream
                    self.applied = {"error": str(e)}
            n_total += 1
            if raw.ndim == 3:  # the backend decoded already: raw MJPEG is not available from this camera
                frame = raw
                self.readback["delivery"] = "decoded by the backend"
            else:
                frame = cv2.imdecode(raw, cv2.IMREAD_COLOR)
                if frame is None:
                    self.warnings.append("a camera frame did not decode")
                    del self.warnings[:-20]
                    continue
                self.readback["delivery"] = "MJPEG bytes"
            if self.recording and self.rec_writer is not None:
                try:
                    self.rec_writer.add(raw, n_total, time.perf_counter() - self.rec_t0)
                except Exception as e:  # noqa: BLE001 - disk full etc.: stop and say so
                    self.rec_error = str(e)
                    self.stop_recording()
                    self.rec_error = str(e)
            self._publish(frame)
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
            p = lambda z: (s.stage_x_um, s.stage_y_um, float(z))  # noqa: E731
            back = np.linspace(zs[-1], self._best_focus(), 30)  # refocus afterwards, as a user would
            self.script = [p(z) for z in approach] + [p(zs[0])] * 6 + [p(z) for z in zs] + [p(zs[-1])] * 25 \
                + [p(z) for z in back]

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
    """Plays back a recording (see recording.py).

    Real-time mode (default) follows the recorded timing; a slow consumer skips frames, as with a camera.
    Ordered mode (exhaustive=True) publishes each frame only after the consumer has finished the previous one,
    so every frame is analysed in order regardless of load (deterministic offline processing).
    """

    def __init__(self, folder: str, speed: float = 1.0, loop: bool = False, exhaustive: bool = False):
        super().__init__()
        from .recording import load_manifest

        self.manifest = load_manifest(folder)
        self.folder = self.manifest.folder
        self.files = self.manifest.files
        self.stamps = self.manifest.stamps
        self.warnings = list(self.manifest.warnings)
        self.speed, self.loop, self.exhaustive = speed, loop, exhaustive
        self.readback = {"replay": str(self.folder), "frames": len(self.files), "mode": self.manifest.mode,
                         **self.manifest.meta.get("readback", {})}
        self.done = False
        self.corrupt = 0

    def frames(self):
        """Every decodable frame in order (damaged files are counted in self.corrupt and skipped)."""
        from .recording import decode

        for f in self.files:
            img = decode(f)
            if img is None:
                self.corrupt += 1
                continue
            yield img

    def _run(self):
        from .recording import decode

        prev_seq = None
        while self._running:
            t0 = time.perf_counter()
            for f, ts, rseq in zip(self.files, self.stamps, self.manifest.seqs):
                if not self._running:
                    return
                step = 1 if prev_seq is None or rseq <= prev_seq else rseq - prev_seq
                if self.exhaustive:
                    while self._running and self._consumed < self._seq:
                        time.sleep(0.002)
                else:
                    wait = ts / self.speed - (time.perf_counter() - t0)
                    if wait > 0:
                        time.sleep(wait)
                img = decode(f)
                if img is None:
                    self.corrupt += 1
                    self.error = f"{self.corrupt} damaged frame(s) skipped (last: {f.name})"
                    continue
                self._publish(img, ts=ts, step=step)  # recorded time and gaps, not playback time
                prev_seq = rseq  # a damaged frame is not published, so it shows up as a gap in the sequence
                self.fps = len(self.stamps) / max(self.stamps[-1], 1e-3)
            if not self.loop:
                self.done = True
                return
            prev_seq = None
