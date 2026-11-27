"""Live engine: pulls the newest frame, corrects it, classifies motion, feeds the mosaic and the Z buffer.

Runs in its own thread (Qt-free). The GUI polls `snapshot()`; finished sweeps and objective-change hints
are delivered through callbacks, which the GUI marshals onto its own thread.
"""

from __future__ import annotations

import collections
import threading
import time
import traceback
from dataclasses import dataclass, field

import cv2
import numpy as np

from .calibration import CalibrationStore, RadiometricCorrector, bgr_to_luma, bgr_to_rgb_u8, radiometric_key
from .camera import FrameSource
from .focus import focus_map
from .kinematics import Override, State, StateMachine
from .mosaic import Mosaic, height_partitions, preview_factor, render_tiles, render_tiles_rgb
from .optics import Objective
from .stage import StageReadout

PREVIEW_W = 640
MAX_SWEEP_FRAMES = 240  # beyond this, the sweep is thinned 2:1 (keeps the full z range)
PREROLL = 12
HOLD_FRAMES = 50  # ~4 s at 12.5 fps
# A focus movement is a usable sweep once at least this fraction of the field went blurry -> sharp -> blurry.
# The live engine, the GUI and the batch tool all decide through this constant (SweepCapture.usable).
MIN_COMPLETENESS = 0.5


def _is_simulated(truth) -> bool:
    """Simulator ground truth is simulated; a stage readout (`StageReadout`) is a measurement, not a simulation."""
    return truth is not None and not isinstance(truth, StageReadout)


@dataclass
class SweepCapture:
    frames: list  # full-resolution corrected luma, uint8
    z: np.ndarray | None  # physical z per frame when known (simulator / motor), else None
    pose: tuple[float, float]  # mosaic camera position at sweep start (preview pixels)
    objective: str
    shape: tuple[int, int]
    simulated: bool
    completeness: float  # fraction of 6x6 field tiles whose focus peak lies inside the sweep (not at an end)
    seqs: list = field(default_factory=list)  # source sequence number of every frame (gaps = frames not kept)
    stamps: list = field(default_factory=list)  # capture time of every frame (s)
    source_id: str = ""
    source_generation: int = 0
    mosaic_generation: int = 0
    exposure: float | None = None
    linearize: bool = False
    preview_shape: tuple[int, int] = (0, 0)  # live preview size at capture (for pinning into the mosaic)
    optics: Objective | None = None  # the objective's optics at capture (the list may be edited before reconstruction)
    manual: bool = False  # True when captured under manual Z override (Space)
    color_frames: list | None = None  # raw colour (BGR) frames matching `frames`, when the engine runs in colour mode

    @property
    def usable(self) -> bool:
        """Whether this capture is a sweep to reconstruct: complete enough, or deliberately marked by the user.

        The engine also emits incomplete parts (a held half-sweep at the end of input, or one that timed out)
        so that the GUI can tell the user why nothing was reconstructed; consumers must ask here."""
        return self.manual or self.completeness >= MIN_COMPLETENESS


class SweepBuffer:
    """Frames of a sweep in progress with their z, tiles, sequence numbers and times, thinned together."""

    def __init__(self):
        self.frames, self.color_frames, self.z, self.tiles, self.seqs, self.stamps = [], [], [], [], [], []

    def append(self, frame, z, tiles, seq, ts, bgr=None):
        self.frames.append(frame)
        self.color_frames.append(bgr)
        self.z.append(z)
        self.tiles.append(tiles)
        self.seqs.append(seq)
        self.stamps.append(ts)

    def thin(self):
        for name in ("frames", "color_frames", "z", "tiles", "seqs", "stamps"):
            setattr(self, name, getattr(self, name)[::2])

    def drop_from(self, seq: int) -> None:
        """Remove every frame whose source sequence number is `seq` or later."""
        n = sum(1 for q in self.seqs if q < seq)
        for name in ("frames", "color_frames", "z", "tiles", "seqs", "stamps"):
            setattr(self, name, getattr(self, name)[:n])

    def __len__(self):
        return len(self.frames)


@dataclass
class Snapshot:
    preview: np.ndarray | None = None
    preview_rgb: np.ndarray | None = None  # colour preview matching `preview`, when colour mode is on
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
    mosaic_hold: str | None = None  # why live frames are not tracked into the mosaic (another objective or mode)
    dwell_frames: int = 0
    seq: int = 0
    mosaic_xy: tuple = (0.0, 0.0)  # live camera position in mosaic pixels (preview scale)
    preview_shape: tuple = (0, 0)
    sweep_completeness: float = 0.0  # live: fraction of the field that has gone blurry -> sharp -> blurry
    engine_error: str | None = None  # last processing failure (the engine keeps running)
    source_generation: int = 0
    mosaic_generation: int = 0
    raw_p995: float | None = None  # 99.5th percentile of uncorrected camera luma (for auto-exposure)


ENGINE_THREAD_NAME = "live-engine"  # the names let the test suite find a thread nobody stopped (tests/conftest.py)
SWEEP_THREAD_NAME = "sweep-consumer"


class LiveEngine:
    def __init__(self, store: CalibrationStore, objective: Objective):
        self.store = store
        self.objective = objective
        self.sm = StateMachine()
        self.mosaic = Mosaic()
        self.source: FrameSource | None = None
        self.exposure: float | None = None
        self.linearize = False
        self.color_mode = False  # when True, colour frames flow through SweepBuffer.color_frames and the RGB mosaic
        self.keep_overview_insets = True  # keep the previous objective's overview as an inset when the magnification changes
        self.on_sweep = None  # callback(SweepCapture)
        self.on_objective_hint = None  # callback(ratio)
        self.sync_callbacks = False  # True: call on_sweep in the engine thread (deterministic batch processing)
        self._lock = threading.Lock()
        self._snap = Snapshot()
        self.verbose = False  # also print the event log to stdout (--verbose)
        self._running = False
        self._thread = None
        self._last_seq = -1
        self._last_src = None
        self.source_generation = 0  # +1 whenever the source changes: older frames and requests are stale
        self.mosaic_generation = 0  # +1 whenever the mosaic restarts: late height maps are not pinned into it
        self.error: str | None = None
        self.errors = 0
        self._preroll = collections.deque(maxlen=PREROLL)
        self._seq_hist = collections.deque(maxlen=8)  # sequence numbers of the newest frames given to the state machine
        self._buf = SweepBuffer()
        self._sweep_pose = (0.0, 0.0)
        self._sweep_mgen = 0  # mosaic generation the open sweep's pose belongs to (sampled when it started)
        self._sweep_simulated = False
        self._sweep_manual = False  # the open sweep (if any) was captured under manual Z override
        self._n_preroll = 0
        self._stride, self._zcount = 1, 0
        self._held = None  # an incomplete sweep kept for HOLD_FRAMES in case focusing resumes
        self._pending_patches: list = []
        self._mosaic_view = ((np.full((8, 8), np.nan, np.float32), (0, 0), 1), [], 0, 0, None)
        self._mosaic_rgb_view: np.ndarray | None = None
        self._view_version = -1
        self._view_t = 0.0
        self._areas_seen = 1
        self._mosaic_hold: str | None = None  # see _mosaic_mismatch
        self._cov = None
        self._dwell = collections.deque(maxlen=16)
        self._avg = collections.deque(maxlen=8)
        self._bhist = collections.deque(maxlen=200)
        self._calib_req = None
        self.calib_error: str | None = None
        self.events: collections.deque = collections.deque(maxlen=500)  # (seq, text) for diagnostics
        self._last_state = None
        self._t0 = time.perf_counter()
        self._corrector = None
        self._corrector_key = None
        self.mosaic_version = 0

    # ------------------------------------------------------------------ control
    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True, name=ENGINE_THREAD_NAME)
        self._thread.start()

    def stop(self) -> bool:
        """Stop the engine thread; True when it has really stopped."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
            return not self._thread.is_alive()
        return True

    def set_source(self, src: FrameSource | None):
        with self._lock:
            self.source = src
            self.source_generation += 1
            self._last_seq, self._last_src = -1, None
            self._cancel_calibration("the source changed")
            self._reset_motion()
            self.mosaic.reset()  # a different source shows a different specimen or scale
            self.mosaic_generation += 1
            self.mosaic_version += 1

    def set_objective(self, obj: Objective):
        with self._lock:
            old_obj = self.objective
            if self.keep_overview_insets and len(self.mosaic.keyframes) >= 1 and obj.magnification != old_obj.magnification:
                img, origin = self.mosaic.image()
                if np.isfinite(img).any():
                    cal = self.store.get(old_obj)
                    pw = self._snap.preview_shape[1] or PREVIEW_W
                    um_old = cal.um_per_px(pw, old_obj, self._snap.preview_shape[0] or None)
                    self.mosaic.add_inset(old_obj.name, um_old, origin[0] * um_old, origin[1] * um_old, img)
            self.objective = obj
            self._corrector_key = None
            self._cancel_calibration("the objective changed")
            self._reset_motion()

    def set_override(self, mode: Override):
        with self._lock:
            self.sm.override = mode

    def reset_mosaic(self, keep_insets: bool = False):
        with self._lock:
            self.mosaic.reset(keep_insets=keep_insets)
            self._areas_seen = 1
            self._pending_patches.clear()
            self.mosaic_generation += 1
            self.mosaic_version += 1

    def mosaic_state(self) -> tuple[dict, dict[str, np.ndarray]]:
        """Serialized pose graph of the live mosaic, taken under the engine lock (the engine thread mutates the
        mosaic); safe to write to disk afterwards, the arrays are never modified in place."""
        with self._lock:
            return self.mosaic.to_state_arrays()

    def load_mosaic_state(self, meta: dict, arrays: dict[str, np.ndarray]):
        """Replace the mosaic with a serialized pose graph (keyframes, edges, patches, insets) and re-render.

        The replaced mosaic's coordinates mean nothing in the loaded one, so this is a new mosaic generation: a
        height map from a sweep captured before the load is not pinned into it (as after reset_mosaic)."""
        with self._lock:
            self.mosaic.load_state_arrays(meta, arrays)
            self._areas_seen = self.mosaic.areas
            self._pending_patches.clear()
            self.mosaic_generation += 1
            self.mosaic_version += 1
        self._render_now()

    def _render_now(self):
        """Render the current mosaic state into the snapshot view (called from the GUI thread). The canvas is
        snapshotted under the lock and rendered after it (copy-on-write: the engine thread cannot change it)."""
        with self._lock:
            req = self._render_request()
        self._render(*req)

    def _render_request(self):
        """Arguments of _render for the current state; call under the lock."""
        return self.mosaic.canvas_snapshot(), self.mosaic_version, len(self.mosaic.keyframes)

    def _reset_motion(self):
        self.sm.reset()
        self._held = None
        self._preroll.clear()
        self._seq_hist.clear()
        self._buf = SweepBuffer()
        self._sweep_simulated = False
        self._sweep_manual = False
        self._n_preroll = 0
        self._cov = None
        self._dwell.clear()
        self._avg.clear()

    def request_calibration(self, kind: str, n: int = 24):
        """Average the next n raw frames into a dark or flat frame. The source, objective, exposure and gamma
        setting are frozen now; if any changes before the average is complete, the request is cancelled."""
        from .camera import SimulatorSource

        with self._lock:
            if isinstance(self.source, SimulatorSource) and self.store.directory is not None:
                self.calib_error = f"{kind.capitalize()} frame cancelled: the simulator never writes to disk."
                self._calib_req = None
                return
            self.calib_error = None
            self._calib_req = {"kind": kind, "n": n, "frames": [], "gen": self.source_generation,
                               "objective": self.objective.name, "exposure": self.exposure,
                               "linearize": self.linearize, "shape": None}

    def _cancel_calibration(self, why: str):
        if self._calib_req is not None:
            self.calib_error = f"{self._calib_req['kind'].capitalize()} frame cancelled: {why}."
            self._calib_req = None

    def calibration_progress(self):
        c = self._calib_req
        return None if c is None else (c["kind"], len(c["frames"]), c["n"])

    def dwell_frames(self) -> list[np.ndarray]:
        with self._lock:
            return list(self._dwell)

    def snapshot(self) -> Snapshot:
        return self._snap  # replaced atomically by the engine thread; never blocks the GUI

    def _emit(self, buf: SweepBuffer, pose, n_pre, shape, simulated, prev_shape, manual: bool = False,
              mosaic_generation: int | None = None):
        """Deliver a sweep. `pose` is the mosaic position when the sweep started and `mosaic_generation` the
        generation of the mosaic that position is in (sampled at the same moment: a mosaic restarted or replaced
        during the sweep makes the pose stale, and the height map is then not pinned)."""
        if len(buf) - n_pre < 8 or not self.on_sweep:
            return
        z = None if any(v is None for v in buf.z) else np.array(buf.z)
        c_frames = list(buf.color_frames) if self.color_mode and all(c is not None for c in buf.color_frames) else None
        cap = SweepCapture(list(buf.frames), z, pose, self.objective.name, shape, simulated, _completeness(buf.tiles),
                           list(buf.seqs), list(buf.stamps), getattr(self.source, "source_id", ""),
                           self.source_generation,
                           self.mosaic_generation if mosaic_generation is None else mosaic_generation,
                           self.exposure, self.linearize, prev_shape,
                           self.objective, manual=manual, color_frames=c_frames)
        if self.sync_callbacks:
            self.on_sweep(cap)
        else:
            threading.Thread(target=self.on_sweep, args=(cap,), daemon=True, name=SWEEP_THREAD_NAME).start()

    def log(self, text: str, seq: int = 0):
        line = f"{time.perf_counter() - self._t0:7.1f}s #{seq:<5d} {text}"
        self.events.append(line)  # shown in Diagnostics
        if self.verbose:
            print(line, flush=True)

    def mosaic_image(self):
        """Latest mosaic rendering, prepared by the engine thread:
        ((preview image, origin, reduction d), height partitions, n_keyframes, n_patches)."""
        return self._mosaic_view[:4]

    def mosaic_view(self):
        """mosaic_image() plus the scale of that rendering's pixel grid (µm per full-resolution mosaic pixel, None
        when the mosaic has not recorded one), read together: (view, partitions, n_keyframes, n_patches, um)."""
        return self._mosaic_view

    def _mosaic_mismatch(self, shape) -> str | None:
        """Why a live preview of `shape` cannot be tracked into the mosaic, or None when it can.

        Mosaic pixels are preview pixels of one objective and camera mode (Mosaic.objective, frame_shape). A
        frame from another objective has another scale, one from another mode another size (and possibly field):
        tracking it into the mosaic would merge two scales into one pose graph or fail on the size. Such a mosaic
        (typically an opened session) is kept as it is and frames are not tracked until the objective and mode
        match again; Reset mosaic starts a new one for the current objective."""
        m = self.mosaic
        if not m.keyframes:
            return None
        if m.objective != self.objective.name:
            return f"the mosaic was made with {m.objective or 'an unrecorded objective'}, not {self.objective.name}"
        if m.frame_shape != tuple(shape):
            fh, fw = m.frame_shape
            return f"the mosaic was made at preview {fw}×{fh}, the camera now gives {shape[1]}×{shape[0]}"
        return None

    def mosaic_full(self):
        """Full-resolution mosaic and its height partitions, e.g. for saving. The canvas is snapshotted under
        the engine lock (briefly) and rendered after it: the image shows one state of the mosaic."""
        with self._lock:
            snap = self.mosaic.canvas_snapshot()
        img, origin, _ = render_tiles(snap.tiles, snap.tile, 1, snap.tiles_low)
        return (img, origin), height_partitions(snap.patches)

    def add_height_patch(self, pose, edf_preview, height_preview, z_unit, z_key=None, rid="",
                         mosaic_generation: int | None = None):
        """Queue a height map for the mosaic. It is applied by the engine thread only if the mosaic is still the
        one the sweep was captured in (same mosaic generation); a reset or source change makes it stale."""
        self._pending_patches.append(("add", pose, edf_preview, height_preview, z_unit, z_key, rid,
                                      self.mosaic_generation if mosaic_generation is None else mosaic_generation))

    def remove_height_patch(self, rid: str):
        self._pending_patches.append(("remove", rid))

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
            src, gen = self.source, self.source_generation
            item = src.latest() if src else None
            if item is None or (src is self._last_src and item[2] == self._last_seq):
                time.sleep(0.004)
                continue
            bgr, ts, seq, truth = item
            render = None
            with self._lock:
                if gen != self.source_generation or src is not self.source:  # switched while we fetched
                    continue
                self._last_src, self._last_seq = src, seq
                try:
                    render = self._process(bgr, truth, src.fps, fps_proc, seq, ts)
                except Exception as e:  # noqa: BLE001 - one bad frame must not kill the engine
                    self.errors += 1
                    self.error = f"{type(e).__name__}: {e}"
                    self.log(f"processing error ({self.errors}): {self.error}", seq)
                    if self.verbose:
                        traceback.print_exc()
                    self._snap = Snapshot(**{**self._snap.__dict__, "engine_error": self.error})
            src.consumed(seq)
            if render is not None:  # render outside the lock: controls never wait for it
                self._render(*render)
            n += 1
            if time.perf_counter() - t0 > 1:
                fps_proc, t0, n = n / (time.perf_counter() - t0), time.perf_counter(), 0

    def process_frame(self, bgr, ts: float, seq: int, truth=None, render: bool = False) -> None:
        """Synchronous processing of one frame (batch mode, tests). Exceptions propagate."""
        with self._lock:
            req = self._process(bgr, truth, 0.0, 0.0, seq, ts)
            self._last_seq = seq
        if render and req is not None:
            self._render(*req)

    def flush(self) -> None:
        """End of input (batch mode): deliver a held or still-running sweep instead of dropping it."""
        with self._lock:
            if self._held is not None:
                held, self._held = self._held, None
                self._emit(held[0], held[1], held[2], held[4], held[5], held[6], manual=held[7],
                           mosaic_generation=held[8])
            elif self.sm.state == State.Z_SWEEP and len(self._buf) \
                    and _completeness(self._buf.tiles) >= MIN_COMPLETENESS:
                h, w = self._buf.frames[0].shape
                sim = self._sweep_simulated
                self._emit(self._buf, self._sweep_pose, self._n_preroll, (h, w), sim, self._snap.preview_shape,
                           manual=self.sm.override == Override.Z, mosaic_generation=self._sweep_mgen)
                self._buf = SweepBuffer()

    def _render(self, snap, version, nkf):
        """Render a canvas snapshot (Mosaic.canvas_snapshot) into the views the GUI polls; runs without the lock."""
        tiles, tile, patches = snap.tiles, snap.tile, snap.patches
        tiles_rgb = snap.tiles_rgb if self.color_mode else None
        d = preview_factor(tiles, tile, 1600)
        view = render_tiles(tiles, tile, d, snap.tiles_low)  # (grey image, origin, d): the grey view is always there
        rgb = None
        if tiles_rgb:
            # the colour canvas, cut to the grey view's box so that both share its origin and its µm axes
            rgb = _crop_rgb_to_view(render_tiles_rgb(tiles_rgb, tile, d), tiles_rgb, tile, view)
        self._mosaic_view = (view, height_partitions(patches), nkf, len(patches), snap.um_per_px)
        self._mosaic_rgb_view = rgb
        self._view_version = version

    def mosaic_image_rgb(self) -> np.ndarray | None:
        """Colour rendering (RGBA uint8, alpha = valid) aligned pixel for pixel with `mosaic_image()`'s image, or
        None when no colour keyframes were added. Prepared by the engine thread like `mosaic_image`."""
        return self._mosaic_rgb_view

    # The GUI changes the mosaic through these, under the engine lock: the engine thread splats into the same tiles.
    def set_mosaic_blending(self, mode: str):
        with self._lock:
            self.mosaic.set_blending(mode)
            self.mosaic_version += 1

    def set_mosaic_track_rotation(self, on: bool):
        with self._lock:
            self.mosaic.track_rotation = bool(on)

    def add_mosaic_inset(self, objective: str, um_per_px: float, x_um: float, y_um: float, image: np.ndarray):
        with self._lock:
            self.mosaic.add_inset(objective, um_per_px, x_um, y_um, image)
            self.mosaic_version += 1

    def _process(self, bgr, truth, fps_in, fps_proc, seq, ts):
        raw = bgr_to_luma(bgr, self.linearize)
        if raw.shape != getattr(self, "_frame_shape", raw.shape):
            # a camera mode change (or reconnect at another size): motion, sweep and mosaic belong to the old mode
            self.log(f"frame size changed {self._frame_shape[1]}x{self._frame_shape[0]} -> "
                     f"{raw.shape[1]}x{raw.shape[0]}: motion tracking and the mosaic restart", seq)
            self._reset_motion()
            self.mosaic.reset()
            self.mosaic_generation += 1
            self.mosaic_version += 1
        self._frame_shape = raw.shape
        req = self._calib_req
        if req is not None:
            if req["shape"] is None:
                req["shape"] = raw.shape
            if (req["gen"] != self.source_generation or req["objective"] != self.objective.name
                    or req["exposure"] != self.exposure or req["linearize"] != self.linearize
                    or req["shape"] != raw.shape):
                self._cancel_calibration("the camera mode, exposure or objective changed while recording")
            elif _is_simulated(truth) and self.store.directory is not None:
                self._cancel_calibration("the simulator never writes to disk")
            else:
                req["frames"].append(raw)
                if len(req["frames"]) >= req["n"]:
                    cal = self.store.get(req["objective"])
                    avg = np.mean(np.stack(req["frames"]).astype(np.float64), 0).astype(np.float32)
                    with self.store._lock:
                        (cal.darks if req["kind"] == "dark" else cal.flats)[radiometric_key(raw.shape,
                                                                                            req["exposure"])] = avg
                        self.store.touch(req["objective"])
                        self.store.save()
                    self._corrector_key = None
                    self._calib_req = None
        raw_luma = bgr_to_luma(bgr, False) if self.linearize else raw
        raw_p995 = float(np.percentile(raw_luma[::4, ::4], 99.5))
        corr = self._corrector_for(raw.shape)(raw)
        sat = float(np.mean(raw_luma >= 250))
        full_u8 = np.clip(corr + 0.5, 0, 255).astype(np.uint8)
        h, w = raw.shape
        pw = min(PREVIEW_W, w)
        ph = round(h * pw / w)
        prev = cv2.resize(corr, (pw, ph), interpolation=cv2.INTER_AREA) if pw < w else corr
        prev_rgb = None
        if self.color_mode:
            rgb_full = bgr_to_rgb_u8(bgr)
            prev_rgb = cv2.resize(rgb_full, (pw, ph), interpolation=cv2.INTER_AREA) if pw < w else rgb_full

        while self._pending_patches:
            op = self._pending_patches.pop(0)
            if op[0] == "remove":
                if self.mosaic.remove_patch(op[1]):
                    self.mosaic_version += 1
                continue
            _, pose, e, hp, zu, zkey, rid, mgen = op
            if mgen != self.mosaic_generation:
                self.log("height map not pinned: its sweep was not captured in this mosaic (restarted or replaced "
                         "since, or not tracking at the time)", seq)
                continue
            self.mosaic.add_height_patch(pose[0], pose[1], e, hp, zu, zkey, rid)
            self.mosaic_version += 1
        um = self.store.get(self.objective).um_per_px(prev.shape[1], self.objective, prev.shape[0])
        hold = self._mosaic_mismatch(prev.shape)
        if hold != self._mosaic_hold:
            self.log(f"mosaic: live frames are not tracked into it: {hold}" if hold
                     else "mosaic: live frames are tracked into it again", seq)
            self._mosaic_hold = hold
        fm = focus_map(prev, "DoG band-pass", 9, self.objective.psf_sigma_px(um))
        th, tw = fm.shape[0] // 6, fm.shape[1] // 6
        tiles = fm[:th * 6, :tw * 6].reshape(6, th, 6, tw).mean(axis=(1, 3)).ravel()
        if self.sm.state == State.Z_SWEEP:
            self.sm.sweep_complete = _completeness(self._buf.tiles + [tiles]) >= MIN_COMPLETENESS
        self._seq_hist.append(seq)
        state, feat, events = self.sm.update(prev)
        if state != self._last_state:
            self.log(f"state {state.value}  (shift {feat.shift_px:.1f}px, corr {feat.response:.2f}, z {feat.z_score:+.1f})", seq)
            self._last_state = state
        self._bhist.append(feat.brenner)
        z_true = getattr(truth, "focus_z_um", None)
        simulated = _is_simulated(truth)
        for ev in events:
            if ev.kind == "sweep_started":
                if self._held is not None:  # focusing resumed soon after an incomplete sweep: continue it
                    self._buf, self._sweep_pose, self._n_preroll = self._held[:3]
                    self._sweep_simulated = self._held[5]
                    self._sweep_manual = self._held[7]
                    self._sweep_mgen = self._held[8]
                    self._held = None
                    self.log("sweep resumed (merged with the previous incomplete part)", seq)
                else:
                    self._buf = SweepBuffer()
                    for item in self._preroll:  # the blurry start is part of the sweep
                        self._buf.append(*item)
                    self._n_preroll = len(self._buf)
                    self._sweep_pose = (self.mosaic.x, self.mosaic.y)
                    # not tracking into the mosaic: the pose is not this view's, so the height map is never pinned
                    self._sweep_mgen = self.mosaic_generation if hold is None else -1
                    self._sweep_simulated = simulated
                    self._sweep_manual = self.sm.override == Override.Z
                    self._cov = None
                self._stride, self._zcount = 1, 0
            elif ev.kind == "sweep_ended":
                aborted = bool(ev.detail.get("aborted"))
                if aborted:
                    # the frames that showed the motion are not part of the sweep; everything before them is one view
                    k = min(int(ev.detail.get("moving_frames", 1)), len(self._seq_hist))
                    self._buf.drop_from(self._seq_hist[-k])
                comp = _completeness(self._buf.tiles)
                manual = bool(ev.detail.get("manual"))
                self.log(f"sweep ended: {len(self._buf)} frames ({self._n_preroll} pre-roll), completeness "
                         f"{comp:.2f}{', ABORTED by motion' if aborted else ''}"
                         f"{', manual' if manual else ''}", seq)
                sim_sweep = self._sweep_simulated or simulated
                if aborted:
                    # a sweep that had already passed the field through focus is a sweep whatever happened next
                    # (the user moved on at once, or the fully defocused image stopped matching itself); an
                    # incomplete one is dropped, since after motion it cannot be continued
                    self._held = None
                    if comp >= MIN_COMPLETENESS:
                        self.log("the frames before the motion are a complete sweep: kept", seq)
                        self._emit(self._buf, self._sweep_pose, self._n_preroll, (h, w), sim_sweep, prev.shape,
                                   mosaic_generation=self._sweep_mgen)
                elif manual or comp >= MIN_COMPLETENESS:
                    self._emit(self._buf, self._sweep_pose, self._n_preroll, (h, w), sim_sweep, prev.shape,
                               manual=manual, mosaic_generation=self._sweep_mgen)
                else:  # incomplete: hold it for a few seconds in case focusing resumes
                    self._held = (self._buf, self._sweep_pose, self._n_preroll, seq + HOLD_FRAMES, (h, w), sim_sweep,
                                  prev.shape, manual, self._sweep_mgen)
                self._buf = SweepBuffer()
            elif ev.kind == "objective_change":
                self.log(f"view changed (magnification ratio {ev.detail.get('magnification_ratio')})", seq)
            if ev.kind == "objective_change" and self.on_objective_hint:
                self.on_objective_hint(ev.detail.get("magnification_ratio"))
        self._preroll.append((full_u8, z_true, tiles, seq, ts, bgr if self.color_mode else None))
        if self._held is not None and (seq > self._held[3] or state == State.PAN):
            held, self._held = self._held, None
            if state != State.PAN:  # the stage moved: the held part no longer belongs to this view
                self._emit(held[0], held[1], held[2], held[4], held[5], held[6], mosaic_generation=held[8])

        in_focus = None
        if state == State.Z_SWEEP:
            if self._zcount % self._stride == 0:
                if len(self._buf) >= MAX_SWEEP_FRAMES:  # a long, slow sweep: thin it 2:1 instead of cutting it
                    self._buf.thin()  # sequence numbers are thinned too, so z spacing stays in source frames
                    self._n_preroll = (self._n_preroll + 1) // 2
                    self._stride *= 2
                self._buf.append(full_u8, z_true, tiles, seq, ts, bgr if self.color_mode else None)
            self._zcount += 1
            self._cov = fm if self._cov is None else np.maximum(self._cov, fm)
            in_focus = fm >= 0.7 * self._cov
        if hold is None and state in (State.PAN, State.DWELL, State.OBJECTIVE_CHANGE) \
                and self.sm.override != Override.PAUSE:
            # the mosaic's grid is this objective's preview at this mode: record (or refresh) what it is
            self.mosaic.objective, self.mosaic.um_per_px = self.objective.name, um
            was_lost = self.mosaic.lost
            added, _ = self.mosaic.track(prev, rgb=prev_rgb)
            ev = self.mosaic.last_event  # set by this track() call: a join also changes `areas`, the other way
            if ev == "new_area":
                self.log(f"mosaic: started separate area ({self.mosaic.areas} now; position of this view unknown)", seq)
            elif ev == "areas_joined":
                self.log(f"mosaic: this view links two areas; they are joined ({self.mosaic.areas} now)", seq)
            self._areas_seen = self.mosaic.areas
            if self.mosaic.lost != was_lost:
                self.log("mosaic tracking LOST" if self.mosaic.lost else "mosaic tracking recovered", seq)
            if added:
                self.mosaic_version += 1
        render = None
        now = time.perf_counter()
        if self.mosaic_version != self._view_version and now - self._view_t > 0.5:
            # rendering happens after the lock is released (at most twice a second, at display size)
            self._view_t = now
            render = self._render_request()
        if state == State.DWELL:
            self._dwell.append(full_u8)
            self._avg.append(prev)
            display = np.mean(self._avg, axis=0) if len(self._avg) > 1 else prev
        else:
            self._dwell.clear()
            self._avg.clear()
            display = prev
        self._snap = Snapshot(
            preview=display.astype(np.float32), preview_rgb=prev_rgb, state=state, fps_in=fps_in, fps_proc=fps_proc,
            shift_px=feat.shift_px,
            response=feat.response, z_score=feat.z_score, brenner_hist=list(self._bhist), saturation=sat,
            sweep_frames=len(self._buf), coverage=self._cov, in_focus=in_focus, frame_shape=(h, w),
            mosaic_version=self._view_version, mosaic_lost=self.mosaic.lost or hold is not None, mosaic_hold=hold,
            dwell_frames=len(self._dwell), seq=seq,
            mosaic_xy=(self.mosaic.x, self.mosaic.y), preview_shape=prev.shape,
            sweep_completeness=_completeness(self._buf.tiles) if state == State.Z_SWEEP and seq % 4 == 0
            else (self._snap.sweep_completeness if state == State.Z_SWEEP else 0.0),
            engine_error=self.error, source_generation=self.source_generation,
            mosaic_generation=self.mosaic_generation, raw_p995=raw_p995)
        return render


def _crop_rgb_to_view(rgb: np.ndarray | None, tiles_rgb: dict, tile: int, view) -> np.ndarray | None:
    """Cut the uncropped colour canvas (`render_tiles_rgb`, which starts at its own top-left tile) to the box of the
    grey view (`render_tiles`: image cropped to the valid area, origin in full-resolution mosaic pixels, reduction
    d). Pixels of that box with no colour data are transparent (alpha 0)."""
    if rgb is None:
        return None
    img, (ox, oy), d = view
    x0 = min(k[0] for k in tiles_rgb) * tile  # full-resolution position of the colour canvas's top-left pixel
    y0 = min(k[1] for k in tiles_rgb) * tile
    out = np.zeros((img.shape[0], img.shape[1], 4), np.uint8)
    cx, cy = (ox - x0) // d, (oy - y0) // d  # the grey box's top-left in colour-canvas pixels
    sx0, sy0 = max(0, cx), max(0, cy)
    sx1, sy1 = min(rgb.shape[1], cx + img.shape[1]), min(rgb.shape[0], cy + img.shape[0])
    if sx1 > sx0 and sy1 > sy0:
        out[sy0 - cy:sy1 - cy, sx0 - cx:sx1 - cx] = rgb[sy0:sy1, sx0:sx1]
    return out


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
