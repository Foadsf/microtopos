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
import pytest

from microtopos import objectives
from microtopos.calibration import CalibrationStore
from microtopos.camera import SimulatorSource
from microtopos.kinematics import Override
from microtopos.live import LiveEngine, _crop_rgb_to_view
from microtopos.mosaic import Mosaic, MosaicInset, preview_factor, render_tiles, render_tiles_rgb, underlay_insets

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


# ------------------------------------------------------------------------------------------ R6 area of a patch


def _pan_b_over_a(record, blur=None):
    """Area A mapped at stage x 300..700; area B started at stage x 1900 and panned left until it joins A. B is
    placed to the right of A in mosaic coordinates, so panning it left moves its coordinates over A's keyframes.
    record(stage x, mosaic) is called after every frame of the B pan; frames at stage x in the range `blur` are
    blurred (motion blur: too soft for a keyframe, still tracked). Returns the stage."""
    s = _Stage(_world(8, 900, 3200), y=300)
    s.pan(300, 700)
    s.at(1900, 46)
    assert s.events == ["new_area"]
    for cx in np.arange(1900, 600, -20):
        if blur is not None and blur[0] <= cx <= blur[1]:
            added, _ = s.m.track(cv2.GaussianBlur(s.frame(float(cx)), (0, 0), 1.5))
            if added:
                s.truth.append((float(cx), s.y))
            s.events += [s.m.last_event] if s.m.last_event else []
        else:
            s.at(float(cx))
        record(float(cx), s.m)
    assert "areas_joined" in s.events and s.m.areas == 1
    return s


def test_a_height_map_pinned_in_a_separate_area_moves_with_it_when_the_areas_join():
    """The height map is pinned while the camera is in B at mosaic coordinates nearer to one of A's keyframes than
    to any of B's. Attached to that A keyframe, it stayed behind when B was moved into A's coordinates (824 px off).
    After the join it must sit where its view is in A's coordinates: stage x minus A's stage-to-mosaic offset."""
    pinned = {}

    def record(cx, m):
        if pinned or m.areas != 2:
            return
        d_other = min(np.hypot(k.x - m.x, k.y - m.y) for k in m.keyframes if k.area != m.current_area)
        d_own = min(np.hypot(k.x - m.x, k.y - m.y) for k in m.keyframes if k.area == m.current_area)
        if d_other < d_own:
            pinned["cx"] = cx
            m.add_height_patch(m.x, m.y, np.full((VH, VW), 80, np.float32), np.zeros((VH, VW), np.float32), "um")

    s = _pan_b_over_a(record)
    assert pinned, "the camera never came nearer to an A keyframe while in B (the case under test)"
    off_a = s.truth[0][0] - s.m.keyframes[0].x  # stage x - mosaic x in A
    assert abs(s.m.patches[0].x - (pinned["cx"] - off_a)) < 2.0


def test_the_tracking_reference_stays_in_the_cameras_area():
    """When the reference keyframe is about to leave the view and the frame is too soft for a new keyframe, the
    tracker switches to the nearest keyframe. Taken from any area, that was one of A's while the camera was in B
    (B's coordinates say nothing about A), and the next keyframe got an edge to it: a relative position between two
    unconnected areas. Every pose-graph edge must agree with the true relative stage position of its keyframes."""
    refs = []
    s = _pan_b_over_a(lambda cx, m: refs.append((m.keyframes[m.ref].area, m.current_area)), blur=(900, 1500))
    assert all(r == c for r, c in refs)
    sx = [cx for cx, _ in s.truth]
    bad = [(i, j, round(ex - (sx[j] - sx[i]), 1)) for i, j, ex, _ey, _w in s.m.edges if abs(ex - (sx[j] - sx[i])) > 3]
    assert bad == []


# ------------------------------------------------------------------------------------------ R4 R5 R9 joins

_PAD = 120


def _turned_view(world: np.ndarray, cx: float, cy: float, theta: float) -> np.ndarray:
    """The view whose CENTRE is at stage (cx, cy) with the specimen turned by theta degrees."""
    x0, y0 = int(cx - VW / 2 - _PAD), int(cy - VH / 2 - _PAD)
    crop = world[y0:y0 + VH + 2 * _PAD, x0:x0 + VW + 2 * _PAD]
    M = cv2.getRotationMatrix2D(((crop.shape[1] - 1) / 2, (crop.shape[0] - 1) / 2), theta, 1.0)
    return cv2.warpAffine(crop, M, (crop.shape[1], crop.shape[0]), flags=cv2.INTER_LINEAR)[
        _PAD:_PAD + VH, _PAD:_PAD + VW]


@pytest.mark.parametrize("theta", [1.5, -1.5])
def test_a_join_applies_the_measured_rotation_to_the_joined_area(theta):
    """Area B is the specimen put back turned by theta (an unconnected area defines its own orientation). B is panned
    1200 px and then into A, where it bridges. With rotation tracking the bridge measures the rotation; applying
    only the translation left B's far keyframes ~theta x distance off (31 px at 1.5 deg). Every keyframe centre
    must be at its stage centre plus A's one offset (< 3 px), so must the camera right after the join, and the
    canvas over B's far part must show the world there (the keyframe images turned into A's orientation, not just
    moved)."""
    world = _world(8, 1400, 3200)
    m = Mosaic()
    m.track_rotation = True
    truth, events, cam = [], [], {}

    def at(cx, th):
        if m.track(_turned_view(world, cx, 600, th))[0]:
            truth.append(cx)
        if m.last_event == "areas_joined":  # the camera (tracker) position right after the join
            cam.update(cx=cx, x=m.x, y=m.y)
        events.extend([m.last_event] if m.last_event else [])

    for cx in range(300, 700, 20):
        at(cx, 0.0)
    for _ in range(46):
        at(1900, theta)
    for cx in range(1900, 600, -20):
        at(cx, theta)
    assert events[:2] == ["new_area", "areas_joined"] and m.areas == 1
    k0 = m.keyframes[0]
    off = (k0.x + VW / 2 - truth[0], k0.y + VH / 2 - 600)  # mosaic centre = stage centre + off
    err = [np.hypot(k.x + VW / 2 - cx - off[0], k.y + VH / 2 - 600 - off[1]) for k, cx in zip(m.keyframes, truth)]
    assert max(err) < 3.0
    assert np.hypot(cam["x"] + VW / 2 - cam["cx"] - off[0], cam["y"] + VH / 2 - 600 - off[1]) < 3.0
    img, (ox, oy) = m.image()
    x0, y0 = round(1500 + off[0]) - ox, round(520 + off[1]) - oy
    canvas = img[y0:y0 + 160, x0:x0 + 400]
    assert np.isfinite(canvas).all()
    assert float(np.mean(np.abs(canvas - world[520:680, 1500:1900]))) < 4.0  # turned images: 2.0; moved only: 10.6


def test_a_join_levels_the_height_patches_it_brings_together():
    """Two height maps of the same surface (same z_key), one pinned in area A and one in a separate area B, read
    against z references 5 µm apart. When B joins A they overlap, and overlapping patches of one z_key are levelled
    (as when a patch is added or moved by relaxation); the join did not re-solve the offsets, which left a 5 µm
    step. Ground truth: H(stage x) = 0.01 x, and B reads H + 5."""
    s = _Stage(_world(5, 800, 3000))
    s.pan(100, 400)  # area A, camera ends at stage x 380
    m = s.m
    cols = np.arange(VW, dtype=np.float32)
    m.add_height_patch(m.x, m.y, np.full((VH, VW), 100, np.float32), np.tile(0.01 * (380 + cols), (VH, 1)), "um")
    s.at(700, 46)  # lost long enough: area B
    assert s.events == ["new_area"]
    s.pan(700, 620)  # camera at stage 640 (moved 60 px: no keyframe yet)
    m.add_height_patch(m.x, m.y, np.full((VH, VW), 100, np.float32),
                       np.tile(0.01 * (640 + cols), (VH, 1)) + 5.0, "um")
    s.pan(620, 300)  # B -> A: bridges
    assert "areas_joined" in s.events
    pa, pb = m.patches
    assert abs((pa.x - 380) - (pb.x - 640)) < 1.0  # both in A's coordinates
    assert abs((pa.offset - pb.offset) - 5.0) < 0.05
    hm, (hx0, _hy0) = m.height_mosaic()
    edge = round(pa.x) + VW - hx0  # right edge of A's patch, inside B's: the surface is smooth across it
    assert abs(float(np.nanmedian(hm[:, edge + 5] - hm[:, edge - 15])) - 0.2) < 0.05


def test_a_join_updates_only_the_joined_area_on_the_canvas():
    """A join used to rebuild the whole canvas inside track(), i.e. under the engine lock (0.5-1.8 s for a large
    mosaic, while every GUI handler that takes the lock waits). It must re-splat only the joined area (and what
    refers to it), and leave the canvas equal to a full rebuild."""
    s = _Stage(_world(12, 1000, 3400), y=0)
    for r, cy in enumerate((20, 140, 260)):  # area A: three rows
        if r:
            for yy in range(cy - 100, cy, 20):
                s.at(1260.0 if r % 2 else 20.0, cy=yy)
        s.pan(*((20, 1280) if r % 2 == 0 else (1260, 0)), cy=cy)
    n_a = len(s.m.keyframes)
    s.at(2600, 46, cy=280)  # B
    assert s.events[-1] == "new_area" and n_a >= 25
    m = s.m
    rebuilt, splats = [], []
    m._rebuild_canvas = lambda: rebuilt.append(1)
    real_splat = m._splat
    m._splat = lambda img, *a, **k: (splats.append(img.shape), real_splat(img, *a, **k))[1]
    for cx in np.arange(2600, 1000, -20):
        splats.clear()
        s.at(float(cx), cy=280)
        if s.events[-1] == "areas_joined":
            break
    assert s.events[-1] == "areas_joined" and not rebuilt
    n_b = len(m.keyframes) - n_a
    # un-splat + splat of B's keyframes (+ the new one and a neighbour); a full rebuild splats every keyframe once
    assert len(splats) <= 2 * (n_b + 2) < n_a + n_b
    del m._rebuild_canvas, m._splat
    incremental = m.image()
    m._rebuild_canvas()
    full = m.image()
    assert incremental[1] == full[1]
    a, b = incremental[0], full[0]
    assert np.array_equal(np.isfinite(a), np.isfinite(b)) and np.nanmax(np.abs(a - b)) < 0.05


# ------------------------------------------------------------------------------------------ R2 G5 whose grid


def _session_mosaic(world: np.ndarray, objective: str, um: float) -> Mosaic:
    """A mosaic as an opened session holds it: 320 x 240 views at stage x 100..480, made with `objective`."""
    s = _Stage(world)
    s.pan(100, 500)
    s.m.objective, s.m.um_per_px = objective, um
    return s.m


def _keyframe_poses(m: Mosaic) -> list:
    return [(k.x, k.y, k.area) for k in m.keyframes]


def test_a_session_of_another_camera_mode_is_kept_and_frames_keep_being_processed():
    """A session recorded at preview 320 x 240 is opened while the camera runs 320 x 180. Every frame raised a
    broadcast error in the tracker (the engine only restarts the mosaic when the CAMERA changes size). Frames
    must be processed (live view, state machine) and the opened mosaic kept unchanged and not tracked into: views
    of another size cannot be matched against it, and another mode may have another field. With the session's
    mode the same frames are tracked (the gate is the size, not the session)."""
    world = _world(3, 1200, 3000)
    saved = _session_mosaic(world, OBJ.name, OBJ.nominal_um_per_px(320))
    for h, tracked in ((180, False), (240, True)):
        eng = LiveEngine(CalibrationStore(None), OBJ)
        eng.sm.override = Override.PAN
        eng.process_frame(_bgr(world, 0, 0, h=h), 0.0, 1)  # the camera ran before the session was opened
        eng.load_mosaic_state(*saved.to_state_arrays())
        poses = _keyframe_poses(eng.mosaic)
        for i in range(2, 12):
            eng.process_frame(_bgr(world, 480 + 20 * i, 200, h=h), 0.04 * i, i)  # raises on e7c6d86 at h = 180
        snap = eng.snapshot()
        assert snap.seq == 11 and snap.preview.shape == (h, VW)
        if tracked:
            assert snap.mosaic_hold is None and len(eng.mosaic.keyframes) > len(poses)
        else:
            assert _keyframe_poses(eng.mosaic) == poses and eng.mosaic.frame_shape == (VH, VW)
            assert "320×240" in snap.mosaic_hold and snap.mosaic_lost


def test_frames_of_another_objective_are_not_tracked_into_an_opened_mosaic():
    """A session mapped at 10x is opened while 40x is selected. The live frames (at 4x the magnification) were
    tracked into the 10x pose graph. They must not be: the mosaic is kept unchanged, a sweep captured meanwhile is
    not pinned into it (its pose is not in the mosaic's coordinates), and once its objective is selected again the
    mosaic is tracked into as before (Open session continues where it stopped)."""
    world = _world(3, 1200, 3000)
    obj40 = objectives.resolve("40x / 0.85", objset=objectives.default_set())
    saved = _session_mosaic(world, OBJ.name, OBJ.nominal_um_per_px(320))
    eng = LiveEngine(CalibrationStore(None), obj40)
    eng.sync_callbacks = True
    caps: list = []
    eng.on_sweep = caps.append
    eng.load_mosaic_state(*saved.to_state_arrays())
    poses = _keyframe_poses(eng.mosaic)
    eng.sm.override = Override.PAN
    for i in range(1, 10):
        eng.process_frame(_bgr(world, 480 + 20 * i, 200), 0.04 * i, i)
    assert _keyframe_poses(eng.mosaic) == poses and eng.mosaic.objective == OBJ.name
    assert eng.snapshot().mosaic_hold and eng.mosaic.um_per_px == OBJ.nominal_um_per_px(320)
    sim = SimulatorSource(4, OBJ, VW, VH)
    eng.sm.override = Override.Z
    for i, z_um in enumerate(sim.sim.sweep_trajectory(n_frames=14)):
        sim.sim.focus_z_um = float(z_um)
        eng.process_frame(sim.sim.render()[0], 1 + i * 0.04, 20 + i)
    eng.flush()
    assert len(caps) == 1 and caps[0].mosaic_generation != eng.mosaic_generation  # never pinned
    eng.set_objective(OBJ)
    eng.sm.override = Override.PAN
    for i in range(1, 10):
        eng.process_frame(_bgr(world, 480 + 20 * i, 200), 2 + 0.04 * i, 40 + i)
    assert eng.snapshot().mosaic_hold is None and len(eng.mosaic.keyframes) > len(poses)


def test_a_mosaic_records_the_objective_and_scale_it_is_made_with():
    """The live engine's mosaic records the objective and preview scale its pixels have (the calibration's for that
    objective and camera mode; nominal here), and the saved state carries them, so an opened mosaic knows its scale
    whatever objective is selected later."""
    world = _world(3, 1200, 3000)
    eng = LiveEngine(CalibrationStore(None), OBJ)
    eng.sm.override = Override.PAN
    for i in range(12):
        eng.process_frame(_bgr(world, 100 + 30 * i, 200), 0.04 * i, i + 1)
    assert len(eng.mosaic.keyframes) >= 3 and eng.snapshot().mosaic_hold is None
    reopened = Mosaic()
    reopened.load_state_arrays(*eng.mosaic_state())
    assert (reopened.objective, reopened.um_per_px) == (OBJ.name, OBJ.nominal_um_per_px(VW))
    eng._render_now()
    assert eng.mosaic_view()[4] == OBJ.nominal_um_per_px(VW)


# ------------------------------------------------------------------------------------------ R7 R8 insets

OBJ40 = objectives.resolve("40x / 0.85", objset=objectives.default_set())


def _mapped_engine(world: np.ndarray, xs, y: int = 200, color: bool = False) -> LiveEngine:
    """An engine that has mapped the views at stage (x, y) for x in xs (320 x 240 previews, objective OBJ)."""
    eng = LiveEngine(CalibrationStore(None), OBJ)
    eng.color_mode = color
    eng.sm.override = Override.PAN
    for i, x in enumerate(xs):
        eng.process_frame(_bgr(world, int(x), y), 0.04 * i, i + 1)
    assert eng.snapshot().mosaic_hold is None and not eng.mosaic.lost
    return eng


def _checker(h: int, w: int, sq: int) -> np.ndarray:
    yy, xx = np.mgrid[0:h, 0:w]
    return np.where(((yy // sq) + (xx // sq)) % 2 == 0, 40.0, 220.0).astype(np.float32)


def test_insets_are_drawn_under_the_canvas_at_their_scale_and_place_in_grey_and_colour():
    """'Add inset' stored the view and saved it, but nothing ever drew it. An inset at half the magnification
    (2 mosaic px per inset px) placed below the mapped strip must appear there in the grey and the colour view,
    at its scale; another one over the mapped strip must stay under the canvas (the canvas is drawn over it)."""
    world = _world(1, 800, 2000)
    eng = _mapped_engine(world, range(100, 420, 20), color=True)
    um = eng.mosaic.um_per_px
    assert um == OBJ.nominal_um_per_px(VW)
    eng._render_now()
    (img0, o0, _), *_ = eng.mosaic_image()
    rgb0 = eng.mosaic_image_rgb()
    pattern = _checker(120, 160, 20)  # 6 x 8 squares of 20 inset px = 40 mosaic px
    x0, y0 = o0[0] - 8, o0[1] + img0.shape[0] + 12  # below the strip, on no tile
    eng.add_mosaic_inset("4x", 2 * um, x0 * um, y0 * um, pattern)
    eng.add_mosaic_inset("4x", 2 * um, o0[0] * um, o0[1] * um, np.full((60, 80), 255.0, np.float32))
    eng._render_now()
    (img, (ox, oy), d), *_ = eng.mosaic_image()
    rgb = eng.mosaic_image_rgb()
    assert d == 1 and rgb.shape[:2] == img.shape
    for j in range(6):  # the centre of every square, in mosaic px
        for i in range(8):
            X, Y = x0 + 40 * i + 20, y0 + 40 * j + 20
            v = pattern[20 * j + 10, 20 * i + 10]
            assert img[Y - oy, X - ox] == pytest.approx(v, abs=0.5)
            assert tuple(rgb[Y - oy, X - ox]) == (round(v),) * 3 + (255,)
    # the mapped strip is drawn over the bright inset: unchanged where it has data
    sy, sx = o0[1] - oy, o0[0] - ox
    canvas = img[sy:sy + img0.shape[0], sx:sx + img0.shape[1]]
    ok = np.isfinite(img0)
    assert np.array_equal(canvas[ok], img0[ok])
    assert np.array_equal(rgb[sy:sy + img0.shape[0], sx:sx + img0.shape[1]][rgb0[..., 3] > 0], rgb0[rgb0[..., 3] > 0])
    # an overview at a tenth of the magnification spans 3200 x 2400 mosaic px: the view is reduced to stay bounded
    eng.add_mosaic_inset("1x", 10 * um, (o0[0] - 1600) * um, (o0[1] - 1200) * um, _checker(240, 320, 40))
    eng._render_now()
    (img, (ox, oy), d), *_ = eng.mosaic_image()
    assert d == 2 and max(img.shape) <= 1600
    X, Y = o0[0] - 1600 + 200, o0[1] - 1200 + 200  # centre of the overview's first square (40 px x 10)
    assert img[(Y - oy) // d, (X - ox) // d] == pytest.approx(40.0, abs=0.5)
    # an overview inset is NaN where the old mosaic was empty: nothing is drawn there, in grey or colour
    holed = _checker(40, 40, 10)
    holed[:, :20] = np.nan
    (view, _origin, _d), rgb2 = underlay_insets((np.full((8, 8), np.nan, np.float32), (0, 0), 1),
                                                np.zeros((8, 8, 4), np.uint8), [MosaicInset("x", um, 0, 0, holed)], um)
    assert np.isnan(view[:, :20]).all() and (rgb2[:, :20, 3] == 0).all() and (rgb2[:, 20:, 3] == 255).all()


def test_an_inset_placed_in_a_separate_area_moves_with_it_when_the_areas_join():
    """An inset is placed in the coordinates of the area the camera is in; when that area joins another, its
    keyframes move into the other's coordinates and the inset must move with them, or it is drawn where the
    unconnected area used to be."""
    pinned = {}

    def record(cx, m):
        if not pinned and m.areas == 2 and cx <= 1600:
            m.um_per_px = 0.5
            pinned.update(cx=cx)
            m.add_inset("40x", 0.5, m.x * 0.5, m.y * 0.5, np.zeros((VH, VW), np.float32))

    s = _pan_b_over_a(record)
    off_a = s.truth[0][0] - s.m.keyframes[0].x  # stage x - mosaic x in A
    assert abs(s.m.insets[0].x_um / 0.5 - (pinned["cx"] - off_a)) < 2.0


@pytest.mark.parametrize("keep", [True, False])
def test_an_objective_change_keeps_the_overview_and_the_insets_in_the_new_mosaics_coordinates(keep):
    """Mapped at 10x, an inset added at the camera, then 40x is selected. The objective change added the old
    mosaic as an overview inset and the GUI's reset deleted it with every other inset. The overview must be kept,
    the user's inset as 'Keep insets on reset' says, both placed in the NEW mosaic's coordinates: its first
    keyframe is the next view (same stage position, centred objectives; the turret change lost the track, so the
    last known position), and 40x preview px are a quarter of 10x ones. Ground truth: frames are windows of a world whose pixels are the 10x preview's, so every inset's world
    position is known, and in the new mosaic world point P is at (first view centre) + (P - camera centre) x 4."""
    world = _world(1, 800, 2000)
    eng = _mapped_engine(world, range(100, 420, 20))
    m = eng.mosaic
    um10 = m.um_per_px
    eng.add_mosaic_inset("40x / 0.85", um10 / 4, m.x * um10, m.y * um10, np.zeros((VH, VW), np.float32))
    _img, origin = m.image()
    other = _world(2, 800, 2000)
    for i in range(4):  # the turret turns: views that match nothing, tracking is lost before 40x is selected
        eng.process_frame(_bgr(other, 300 + 50 * i, 300), 2 + 0.04 * i, 50 + i)
    assert m.lost and eng.snapshot().mosaic_hold is None
    to_world = (100 - m.keyframes[0].x, 200 - m.keyframes[0].y)  # the first view was at stage (100, 200)
    cam_world = (m.x + to_world[0] + (VW - 1) / 2, m.y + to_world[1] + (VH - 1) / 2)
    user_world = (m.x + to_world[0], m.y + to_world[1])
    over_world = (origin[0] + to_world[0], origin[1] + to_world[1])
    assert eng.set_objective(OBJ40, restart_mosaic=True, keep_insets=keep)
    assert not m.keyframes and m.objective == OBJ40.name
    um40 = OBJ40.nominal_um_per_px(VW)

    def expect(pw):  # µm in the new mosaic of world point pw
        return tuple(((VW - 1, VH - 1)[a] / 2) * um40 + (pw[a] - cam_world[a]) * um10 for a in (0, 1))

    got = {it.objective: (it.x_um, it.y_um) for it in m.insets}
    assert set(got) == ({"40x / 0.85", OBJ.name} if keep else {OBJ.name})
    assert got[OBJ.name] == pytest.approx(expect(over_world), abs=0.5 * um10)
    if keep:
        assert got["40x / 0.85"] == pytest.approx(expect(user_world), abs=0.5 * um10)


def test_reset_keeps_insets_where_they_were_on_the_specimen():
    """Reset mosaic with 'Keep insets on reset': the new mosaic starts at the camera's current view, so a kept inset
    must be moved by where the camera is; left at its old coordinates it is drawn off by the camera's travel."""
    world = _world(1, 800, 2000)
    eng = _mapped_engine(world, range(100, 300, 20))
    m = eng.mosaic
    um = m.um_per_px
    eng.add_mosaic_inset("40x / 0.85", um / 4, m.x * um, m.y * um, np.zeros((VH, VW), np.float32))
    inset_world = m.x + 100 - m.keyframes[0].x
    for i, x in enumerate(range(300, 700, 20)):  # travel on before resetting
        eng.process_frame(_bgr(world, x, 200), 1 + 0.04 * i, 100 + i)
    cam_world = m.x + 100 - m.keyframes[0].x
    eng.reset_mosaic(keep_insets=True)
    assert len(m.insets) == 1 and m.insets[0].x_um == pytest.approx((inset_world - cam_world) * um, abs=0.5 * um)
    eng.process_frame(_bgr(world, 680, 200), 3.0, 200)  # the next view is the new mosaic's first keyframe, at 0
    assert (m.keyframes[0].x, m.keyframes[0].y) == (0.0, 0.0)


def test_a_restart_drops_insets_it_cannot_place_and_keeps_an_opened_mosaic_of_the_selected_objective():
    """Insets are placed relative to the camera; when the camera is not tracked into the mosaic (an opened 10x
    session while 40x is selected) nothing tells where they belong in a new mosaic: they are not kept. Selecting
    the opened mosaic's own objective does not restart it (the GUI reset the mosaic on every objective change)."""
    world = _world(3, 1200, 3000)
    saved = _session_mosaic(world, OBJ.name, OBJ.nominal_um_per_px(320))
    saved.add_inset("4x", 1.0, 0.0, 0.0, np.zeros((8, 8), np.float32))
    eng = LiveEngine(CalibrationStore(None), OBJ40)
    eng.load_mosaic_state(*saved.to_state_arrays())
    eng.process_frame(_bgr(world, 500, 200), 0.0, 1)
    assert eng.snapshot().mosaic_hold
    assert eng.set_objective(OBJ, restart_mosaic=True, keep_insets=True) is False
    assert len(eng.mosaic.keyframes) == len(saved.keyframes) and len(eng.mosaic.insets) == 1
    eng.set_objective(OBJ40)
    eng.process_frame(_bgr(world, 500, 200), 0.04, 2)
    assert eng.snapshot().mosaic_hold
    eng.reset_mosaic(keep_insets=True)
    assert eng.mosaic.insets == [] and not eng.mosaic.keyframes
    assert any("without its insets" in e for e in eng.events)
