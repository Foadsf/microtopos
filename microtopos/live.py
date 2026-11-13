"""Live engine: pulls the newest frame, corrects it, classifies motion, feeds the mosaic and the Z buffer.

Runs in its own thread (Qt-free). The GUI polls `snapshot()`; finished sweeps and objective-change hints
are delivered through callbacks, which the GUI marshals onto its own thread.
"""

from __future__ import annotations

import collections
import threading
import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from .calibration import CalibrationStore, RadiometricCorrector, bgr_to_luma, radiometric_key
from .camera import FrameSource
from .focus import focus_map
from .kinematics import Override, State, StateMachine
from .mosaic import Mosaic
from .optics import Objective

PREVIEW_W = 640
MAX_SWEEP_FRAMES = 240  # beyond this, the sweep is thinned 2:1 (keeps the full z range)
PREROLL = 12
HOLD_FRAMES = 50  # ~4 s at 12.5 fps


@dataclass
class SweepCapture:
    frames: list  # full-resolution corrected luma, uint8
    z: np.ndarray | None  # physical z per frame when known (simulator / motor), else None
    pose: tuple[float, float]  # mosaic camera position at sweep start (preview pixels)
    objective: str
    shape: tuple[int, int]
    simulated: bool
    completeness: float  # fraction of 6x6 field tiles whose focus peak lies inside the sweep (not at an end)
    frame_indices: np.ndarray | None = None  # original capture frame indices (preserves true spacing across thinning)


@dataclass
class Snapshot:
    preview: np.ndarray | None = None
    state: State = State.DWELL
    fps_in: float = 0.0
    fps_proc: float = 0.0
    shift_px: float = 0.0
    response: float = 0.0
    z_score: float = 0.0
    brenner_hist: list = field(default_factory=list)
    saturation: float = 0.0
    sweep_frames: int = 0
    coverage: np.ndarray | None = None  # per-pixel running max of focus energy during the sweep (preview res)
    in_focus: np.ndarray | None = None  # pixels currently within 70% of their running max
    frame_shape: tuple[int, int] = (0, 0)
    mosaic_version: int = 0
    mosaic_lost: bool = False
    dwell_frames: int = 0
    seq: int = 0
    mosaic_xy: tuple = (0.0, 0.0)  # live camera position in mosaic pixels (preview scale)
    preview_shape: tuple = (0, 0)
    sweep_completeness: float = 0.0  # live: fraction of the field that has gone blurry -> sharp -> blurry


class LiveEngine:
    def __init__(self, store: CalibrationStore, objective: Objective):
        self.store = store
        self.objective = objective
        self.sm = StateMachine()
        self.mosaic = Mosaic()
        self.source: FrameSource | None = None
        self.exposure: float | None = None
        self.linearize = False
        self.on_sweep = None  # callback(SweepCapture)
        self.on_objective_hint = None  # callback(ratio)
        self._lock = threading.Lock()
        self._snap = Snapshot()
        self.verbose = False  # also print the event log to stdout (--verbose)
        self._running = False
        self._thread = None
        self._last_seq = -1
        self._preroll = collections.deque(maxlen=PREROLL)
        self._sweep: list = []
        self._sweep_z: list = []
        self._sweep_pose = (0.0, 0.0)
        self._n_preroll = 0
        self._sweep_tiles: list = []
        self._sweep_indices: list[float] = []
        self._stride, self._zcount = 1, 0
        self._source_gen = 0
        self._held = None  # an incomplete sweep kept for HOLD_FRAMES in case focusing resumes
        self._pending_patches: list = []
        self._mosaic_view = ((np.full((8, 8), np.nan, np.float32), (0, 0)), None, 0, 0)
        self._view_version = -1
        self._view_t = 0.0
        self._areas_seen = 1
        self._cov = None
        self._dwell = collections.deque(maxlen=16)
        self._avg = collections.deque(maxlen=8)
        self._bhist = collections.deque(maxlen=200)
        self._calib_req = None  # (kind, n, frames)
        self.events: collections.deque = collections.deque(maxlen=500)  # (seq, text) for diagnostics
        self._last_state = None
        self._t0 = time.perf_counter()
        self._corrector = None
        self._corrector_key = None
        self.mosaic_version = 0

    # ------------------------------------------------------------------ control
    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)

    def set_source(self, src: FrameSource | None):
        with self._lock:
            self.source = src
            self._source_gen += 1
            self._last_seq = -1
            self._calib_req = None
            self._reset_motion()
            self.mosaic.reset()  # a different source shows a different specimen or scale
            self.mosaic_version += 1

    def set_objective(self, obj: Objective):
        with self._lock:
            self.objective = obj
            self._corrector_key = None
            self._calib_req = None
            self._reset_motion()

    def set_override(self, mode: Override):
        with self._lock:
            self.sm.override = mode

    def reset_mosaic(self):
        with self._lock:
            self.mosaic.reset()
            self._areas_seen = 1
            self.mosaic_version += 1

    def _reset_motion(self):
        self.sm.reset()
        self._held = None
        self._preroll.clear()
        self._sweep_tiles = []
        self._sweep_indices = []
        self._n_preroll = 0
        self._sweep.clear()
        self._sweep_z.clear()
        self._cov = None
        self._dwell.clear()
        self._avg.clear()

    def request_calibration(self, kind: str, n: int = 24):
        """Average the next n raw frames into a dark or flat frame for the current objective and exposure."""
        with self._lock:
            self._calib_req = [kind, n, [], self.objective.name, self._source_gen]

    def calibration_progress(self):
        c = self._calib_req
        return None if c is None else (c[0], len(c[2]), c[1])

    def dwell_frames(self) -> list[np.ndarray]:
        with self._lock:
            return list(self._dwell)

    def snapshot(self) -> Snapshot:
        return self._snap  # replaced atomically by the engine thread; never blocks the GUI

    def _emit(self, frames, zs, tiles, pose, n_pre, shape, simulated, indices=None):
        if len(frames) - n_pre < 8 or not self.on_sweep:
            return
        z = None if any(v is None for v in zs) else np.array(zs)
        idx_arr = np.array(indices, dtype=np.float64) if indices is not None and len(indices) == len(frames) else None
        cap = SweepCapture(
            list(frames), z, pose, self.objective.name, shape, simulated, _completeness(tiles), frame_indices=idx_arr
        )
        threading.Thread(target=self.on_sweep, args=(cap,), daemon=True).start()

    def log(self, text: str, seq: int = 0):
        line = f"{time.perf_counter() - self._t0:7.1f}s #{seq:<5d} {text}"
        self.events.append(line)  # shown in Diagnostics
        if self.verbose:
            print(line, flush=True)

    def mosaic_image(self):
        """Latest mosaic rendering, prepared by the engine thread: ((image, origin), height_mosaic, n_kf, n_patch)."""
        return self._mosaic_view

    def add_height_patch(self, pose, edf_preview, height_preview, z_unit):
        self._pending_patches.append((pose, edf_preview, height_preview, z_unit))  # applied by the engine thread

    # ------------------------------------------------------------------ loop
    def _corrector_for(self, shape):
        cal = self.store.get(self.objective)
        key = radiometric_key(shape, self.exposure)
        full_key = (self.objective.name, key, id(cal.darks.get(key)), id(cal.flats.get(key)))
        if full_key != self._corrector_key:
            self._corrector = RadiometricCorrector(cal.darks.get(key), cal.flats.get(key))
            self._corrector_key = full_key
        return self._corrector

    def _loop(self):
        t0, n = time.perf_counter(), 0
        fps_proc = 0.0
        while self._running:
            src = self.source
            item = src.latest() if src else None
            if item is None or item[2] == self._last_seq:
                time.sleep(0.004)
                continue
            bgr, ts, seq, truth = item
            self._last_seq = seq
            with self._lock:
                self._process(bgr, truth, src.fps, fps_proc, seq)
            n += 1
            if time.perf_counter() - t0 > 1:
                fps_proc, t0, n = n / (time.perf_counter() - t0), time.perf_counter(), 0

    def _process(self, bgr, truth, fps_in, fps_proc, seq):
        raw = bgr_to_luma(bgr, self.linearize)
        if self._calib_req is not None:
            kind, need, acc, req_obj, req_gen = self._calib_req
            if req_obj != self.objective.name or req_gen != self._source_gen:
                self._calib_req = None
            elif acc and raw.shape != acc[0].shape:
                self.log(f"{kind} calibration cancelled: frame size changed", seq)
                self._calib_req = None
            else:
                acc.append(raw)
                if len(acc) >= need:
                    cal = self.store.get(req_obj)
                    (cal.darks if kind == "dark" else cal.flats)[radiometric_key(raw.shape, self.exposure)] = np.mean(
                        acc, 0
                    )
                    self.store.save()
                    self._calib_req = None
        corr = self._corrector_for(raw.shape)(raw)
        cal_obj = self.store.get(self.objective)
        if truth is None and cal_obj.lens_model is not None:
            corr = cal_obj.lens_model.undistort(corr)
        sat = float(np.mean(raw >= 250))
        full_u8 = np.clip(corr + 0.5, 0, 255).astype(np.uint8)
        h, w = raw.shape
        pw = min(PREVIEW_W, w)
        prev = cv2.resize(corr, (pw, int(round(h * pw / w))), interpolation=cv2.INTER_AREA) if pw < w else corr

        while self._pending_patches:
            pose, e, hp, zu = self._pending_patches.pop(0)
            self.mosaic.add_height_patch(pose[0], pose[1], e, hp, zu)
            self.mosaic_version += 1
        um = self.store.get(self.objective).um_per_px(prev.shape[1])
        fm = focus_map(prev, "DoG band-pass", 9, self.objective.psf_sigma_px(um))
        th, tw = fm.shape[0] // 6, fm.shape[1] // 6
        tiles = fm[:th * 6, :tw * 6].reshape(6, th, 6, tw).mean(axis=(1, 3)).ravel()
        if self.sm.state == State.Z_SWEEP:
            self.sm.sweep_complete = _completeness(self._sweep_tiles + [tiles]) >= 0.5
        state, feat, events = self.sm.update(prev)
        if state != self._last_state:
            self.log(f"state {state.value}  (shift {feat.shift_px:.1f}px, corr {feat.response:.2f}, z {feat.z_score:+.1f})", seq)
            self._last_state = state
        self._bhist.append(feat.brenner)
        z_true = getattr(truth, "focus_z_um", None)
        for ev in events:
            if ev.kind == "sweep_started":
                if self._held is not None:  # focusing resumed soon after an incomplete sweep: continue it
                    (
                        self._sweep,
                        self._sweep_z,
                        self._sweep_tiles,
                        self._sweep_pose,
                        self._n_preroll,
                        _hseq,
                        _hshape,
                        _hsim,
                        self._sweep_indices,
                    ) = self._held
                    self._held = None
                    self._zcount = (int(self._sweep_indices[-1]) + 1) if self._sweep_indices else 0
                    self.log("sweep resumed (merged with the previous incomplete part)", seq)
                else:
                    self._sweep = [f for f, _, _ in self._preroll]
                    self._sweep_z = [z for _, z, _ in self._preroll]
                    self._sweep_tiles = [t for _, _, t in self._preroll]  # the blurry start is part of the sweep
                    self._n_preroll = len(self._sweep)
                    self._sweep_indices = [float(k) for k in range(-self._n_preroll, 0)]
                    self._sweep_pose = (self.mosaic.x, self.mosaic.y)
                    self._cov = None
                    self._zcount = 0
                self._stride = 1
            elif ev.kind == "sweep_ended":
                comp = _completeness(self._sweep_tiles)
                self.log(f"sweep ended: {len(self._sweep)} frames ({self._n_preroll} pre-roll), completeness "
                         f"{comp:.2f}{', ABORTED by motion' if ev.detail.get('aborted') else ''}", seq)
                if ev.detail.get("aborted"):
                    self._held = None
                elif comp < 0.5:  # incomplete: hold it for a few seconds in case focusing resumes
                    self._held = (
                        self._sweep,
                        self._sweep_z,
                        self._sweep_tiles,
                        self._sweep_pose,
                        self._n_preroll,
                        seq + HOLD_FRAMES,
                        (h, w),
                        truth is not None,
                        list(self._sweep_indices),
                    )
                else:
                    self._emit(
                        self._sweep,
                        self._sweep_z,
                        self._sweep_tiles,
                        self._sweep_pose,
                        self._n_preroll,
                        (h, w),
                        truth is not None,
                        self._sweep_indices,
                    )
                self._sweep, self._sweep_z, self._sweep_indices = [], [], []
            elif ev.kind == "objective_change":
                self.log(f"view changed (magnification ratio {ev.detail.get('magnification_ratio')})", seq)
            if ev.kind == "objective_change" and self.on_objective_hint:
                self.on_objective_hint(ev.detail.get("magnification_ratio"))
        self._preroll.append((full_u8, z_true, tiles))
        if self._held is not None and (seq > self._held[5] or state == State.PAN):
            held, self._held = self._held, None
            if state != State.PAN:  # the stage moved: the held part no longer belongs to this view
                self._emit(*held[:5], held[6], held[7], held[8])

        in_focus = None
        if state == State.Z_SWEEP:
            if self._zcount % self._stride == 0:
                if len(self._sweep) >= MAX_SWEEP_FRAMES:  # a long, slow sweep: thin it 2:1 instead of cutting it
                    self._sweep, self._sweep_z, self._sweep_tiles, self._sweep_indices = (
                        self._sweep[::2],
                        self._sweep_z[::2],
                        self._sweep_tiles[::2],
                        self._sweep_indices[::2],
                    )
                    self._n_preroll = (self._n_preroll + 1) // 2
                    self._stride *= 2
                self._sweep.append(full_u8)
                self._sweep_z.append(z_true)
                self._sweep_tiles.append(tiles)
                self._sweep_indices.append(float(self._zcount))
            self._zcount += 1
            self._cov = fm if self._cov is None else np.maximum(self._cov, fm)
            in_focus = fm >= 0.7 * self._cov
        if state in (State.PAN, State.DWELL, State.OBJECTIVE_CHANGE) and self.sm.override != Override.PAUSE:
            was_lost = self.mosaic.lost
            added, _ = self.mosaic.track(prev)
            if added and self.mosaic.areas != self._areas_seen:
                self._areas_seen = self.mosaic.areas
                self.log(f"mosaic: started separate area {self.mosaic.areas} (position of this view unknown)", seq)
            if self.mosaic.lost != was_lost:
                self.log("mosaic tracking LOST" if self.mosaic.lost else "mosaic tracking recovered", seq)
            if added:
                self.mosaic_version += 1
        now = time.perf_counter()
        if self.mosaic_version != self._view_version and now - self._view_t > 0.5:
            # render the mosaic here (not in the GUI thread), at most twice a second
            self._view_version, self._view_t = self.mosaic_version, now
            self._mosaic_view = (self.mosaic.image(), self.mosaic.height_mosaic(), len(self.mosaic.keyframes),
                                 len(self.mosaic.patches))
        if state == State.DWELL:
            self._dwell.append(full_u8)
            self._avg.append(prev)
            display = np.mean(self._avg, axis=0) if len(self._avg) > 1 else prev
        else:
            self._dwell.clear()
            self._avg.clear()
            display = prev
        self._snap = Snapshot(
            preview=display.astype(np.float32), state=state, fps_in=fps_in, fps_proc=fps_proc, shift_px=feat.shift_px,
            response=feat.response, z_score=feat.z_score, brenner_hist=list(self._bhist), saturation=sat,
            sweep_frames=len(self._sweep), coverage=self._cov, in_focus=in_focus, frame_shape=(h, w),
            mosaic_version=self._view_version, mosaic_lost=self.mosaic.lost, dwell_frames=len(self._dwell), seq=seq,
            mosaic_xy=(self.mosaic.x, self.mosaic.y), preview_shape=prev.shape,
            sweep_completeness=_completeness(self._sweep_tiles) if state == State.Z_SWEEP and seq % 4 == 0
            else (self._snap.sweep_completeness if state == State.Z_SWEEP else 0.0))


def _completeness(tiles: list) -> float:
    """Fraction of field tiles that went blurry -> sharp -> blurry at some point during the sweep.

    A tile counts if some frame k is clearly sharper (> 0.3 in log energy) than the blurriest frame before it
    AND than the blurriest frame after it. Later refocusing (sharp again at the end) does not undo that.
    """
    if len(tiles) < 5:
        return 0.0
    t = np.log(np.asarray(tiles) + 1e-6)
    before = np.minimum.accumulate(t, axis=0)
    after = np.minimum.accumulate(t[::-1], axis=0)[::-1]
    ok = ((t - before) > 0.3) & ((t - after) > 0.3)
    return float(np.mean(ok.any(axis=0)))
