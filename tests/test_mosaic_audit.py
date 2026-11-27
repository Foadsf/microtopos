"""Mosaic and live-engine defects found by the November 2026 audit: stale height maps pinned into an opened
session, sessions of another camera mode or objective tracked into, renders mixing two canvas states, joins that
ignore rotation, leave height offsets unlevelled or rebuild the whole canvas under the engine lock, height maps
attached to another area, and insets that were stored but never shown.

Frames are windows of a known synthetic world (or the simulator), so every keyframe's true stage position is known;
races are forced with gates and monkeypatched hooks, never with timing."""

from __future__ import annotations

import copy
import threading

import cv2
import numpy as np

from microtopos import objectives
from microtopos.calibration import CalibrationStore
from microtopos.camera import SimulatorSource
from microtopos.kinematics import Override
from microtopos.live import LiveEngine, _crop_rgb_to_view
from microtopos.mosaic import Mosaic, preview_factor, render_tiles, render_tiles_rgb

VH, VW = 240, 320
OBJ = objectives.resolve("10x / 0.25", objset=objectives.default_set())


def _world(seed: int, h: int, w: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return cv2.GaussianBlur(rng.integers(0, 256, (h, w)).astype(np.float32), (0, 0), 1.2)


class _Stage:
    """A mosaic fed with windows of a world; records the stage position of every frame that added a keyframe."""

    def __init__(self, world: np.ndarray, y: int = 200, m: Mosaic | None = None):
        self.world, self.y = world, y
        self.m = m if m is not None else Mosaic()
        self.truth: list[tuple[float, float]] = []
        self.events: list[str] = []

    def frame(self, cx: float, cy: float | None = None) -> np.ndarray:
        cy = self.y if cy is None else cy
        return self.world[int(cy):int(cy) + VH, int(cx):int(cx) + VW].copy()

    def at(self, cx: float, n: int = 1, cy: float | None = None, rgb: bool = False) -> None:
        for _ in range(n):
            f = self.frame(cx, cy)
            added, _ = self.m.track(f, rgb=np.stack([f, 0.5 * f, 255.0 - f], -1) if rgb else None)
            if added:
                self.truth.append((cx, self.y if cy is None else cy))
            if self.m.last_event:
                self.events.append(self.m.last_event)

    def pan(self, a: float, b: float, step: float = 20.0, cy: float | None = None, rgb: bool = False) -> None:
        for cx in np.arange(a, b, step if b > a else -step):
            self.at(float(cx), cy=cy, rgb=rgb)


def _bgr(world: np.ndarray, cx: int, cy: int, h: int = VH, w: int = VW) -> np.ndarray:
    return np.dstack([world[cy:cy + h, cx:cx + w]] * 3).astype(np.uint8)


# ------------------------------------------------------------------------------------------ R1 open session


def test_a_height_map_from_the_mosaic_an_opened_session_replaced_is_not_pinned_into_it():
    """A sweep captured in the live mosaic finishes reconstructing after Open session replaced that mosaic. Its pose
    is in the replaced mosaic's coordinates (here x = 240 of world 1), meaningless in the opened one (world 2), so
    the engine must drop it, as it does after a reset; a map of the opened mosaic's own generation is pinned."""
    eng = LiveEngine(CalibrationStore(None), OBJ)
    w1 = _world(1, 800, 2000)
    _Stage(w1, m=eng.mosaic).pan(100, 400)
    gen_live = eng.mosaic_generation  # what a sweep captured now carries
    other = _Stage(_world(2, 800, 2000))
    other.pan(100, 600)
    eng.load_mosaic_state(*other.m.to_state_arrays())

    edf, hgt = np.full((VH, VW), 128, np.float32), np.zeros((VH, VW), np.float32)
    eng.add_height_patch((240.0, 0.0), edf, hgt, "um", None, "replaced", mosaic_generation=gen_live)
    eng.process_frame(_bgr(w1, 300, 200), 0.0, 1)
    assert [p.rid for p in eng.mosaic.patches] == []
    eng.add_height_patch((100.0, 0.0), edf, hgt, "um", None, "opened", mosaic_generation=eng.mosaic_generation)
    eng.process_frame(_bgr(w1, 300, 200), 0.04, 2)
    assert [p.rid for p in eng.mosaic.patches] == ["opened"]  # the generation gate itself was reached


def test_a_sweep_spanning_an_open_session_carries_the_generation_its_pose_belongs_to():
    """The sweep's pose is the mosaic position when it STARTED; a session opened halfway through replaces that
    mosaic. The capture must carry the generation of the mosaic its pose is in (so it is not pinned into the opened
    one), not the generation current when the sweep ended."""
    sim = SimulatorSource(4, OBJ, 160, 120)
    eng = LiveEngine(CalibrationStore(None), OBJ)
    eng.sync_callbacks = True
    caps: list = []
    eng.on_sweep = caps.append
    eng.set_source(sim)
    eng.sm.override = Override.Z
    other = _Stage(_world(2, 800, 2000))
    other.pan(100, 400)
    gen_at_start = None
    for i, z_um in enumerate(sim.sim.sweep_trajectory(n_frames=14)):
        if i == 7:
            gen_at_start = eng.mosaic_generation
            eng.load_mosaic_state(*other.m.to_state_arrays())
        sim.sim.focus_z_um = float(z_um)
        bgr, truth = sim.sim.render()
        eng.process_frame(bgr, i * 0.04, i + 1, truth=truth)
    eng.flush()
    assert len(caps) == 1
    assert caps[0].mosaic_generation == gen_at_start != eng.mosaic_generation


# ------------------------------------------------------------------------------------------ R3 torn renders


def _gate_render(monkeypatch):
    """Make live.render_tiles stop until released; returns (entered, release). The canvas is taken under the
    engine lock before render_tiles is called, so while it waits the engine thread is free to splat."""
    from microtopos import live

    real = live.render_tiles
    entered, release = threading.Event(), threading.Event()

    def gated(*a, **k):
        entered.set()
        assert release.wait(10)
        return real(*a, **k)

    monkeypatch.setattr(live, "render_tiles", gated)
    return entered, release


def _same_image(a: np.ndarray, b: np.ndarray) -> bool:
    return a.shape == b.shape and np.array_equal(a, b, equal_nan=True)


def _grow_until_new_tile(eng: LiveEngine, s: _Stage, cx: float, rgb: bool = False) -> None:
    """Engine thread: frames under the lock until one adds a keyframe that reaches into a tile not yet on the canvas
    (and so also changes tiles that existed when the GUI took its canvas)."""
    m = eng.mosaic
    n = len(m.tiles)
    with eng._lock:
        while len(m.tiles) == n:
            s.at(cx, rgb=rgb)
            cx += 20


def test_saving_renders_one_state_of_the_mosaic_while_the_engine_keeps_splatting(monkeypatch):
    """mosaic_full (Save session) takes the canvas under the lock and renders it after releasing it. The engine
    thread splats a keyframe in between that straddles a tile boundary; the saved image must be the mosaic of one
    state (the one at the time of the call), not old tiles with new content in some of them."""
    eng = LiveEngine(CalibrationStore(None), OBJ)
    s = _Stage(_world(4, 800, 3000), m=eng.mosaic, y=100)
    s.pan(100, 400)
    m = eng.mosaic
    before = render_tiles(copy.deepcopy(m.tiles), m.tile, 1)[0]
    entered, release = _gate_render(monkeypatch)
    out = {}
    t = threading.Thread(target=lambda: out.setdefault("img", eng.mosaic_full()[0][0]))
    t.start()
    assert entered.wait(10)
    _grow_until_new_tile(eng, s, 400.0)
    after = render_tiles(copy.deepcopy(m.tiles), m.tile, 1)[0]
    release.set()
    t.join(10)
    assert not _same_image(before, after)  # the engine really changed the canvas in between
    assert _same_image(out["img"], before)


def test_the_gui_render_of_the_view_shows_one_state_in_grey_and_colour(monkeypatch):
    """_render_now (GUI thread: Open session, tests) renders the grey and the colour view after releasing the lock;
    both must show the canvas of the moment the call took it, while the engine thread splats a new keyframe."""
    eng = LiveEngine(CalibrationStore(None), OBJ)
    eng.color_mode = True
    s = _Stage(_world(4, 800, 3000), m=eng.mosaic, y=100)
    s.pan(100, 400, rgb=True)
    m = eng.mosaic
    tiles0, rgb0 = copy.deepcopy(m.tiles), copy.deepcopy(m.tiles_rgb)
    view0 = render_tiles(tiles0, m.tile, preview_factor(tiles0, m.tile, 1600))
    rgb_expect = _crop_rgb_to_view(render_tiles_rgb(rgb0, m.tile, view0[2]), rgb0, m.tile, view0)
    entered, release = _gate_render(monkeypatch)
    t = threading.Thread(target=eng._render_now)
    t.start()
    assert entered.wait(10)
    _grow_until_new_tile(eng, s, 400.0, rgb=True)
    release.set()
    t.join(10)
    (img, origin, d), *_ = eng.mosaic_image()
    assert (origin, d) == (view0[1], view0[2]) and _same_image(img, view0[0])
    assert np.array_equal(eng.mosaic_image_rgb(), rgb_expect)
