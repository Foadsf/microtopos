"""Mosaic regressions found by comparing v0.9.0 with the release after PR #14: keyframes yielding to each other
(holes inside keyframes, re-splat work growing with every pass), ungated cross-area matching, wrong area ids
after a re-localisation or a join, and canvas calls racing a blending switch. Frames are windows of a known
synthetic world, so every keyframe's true stage position is known."""

from __future__ import annotations

import cv2
import numpy as np
from _gen import closed_loop

from microtopos.mosaic import Mosaic

VH, VW = 240, 320


def _world(seed: int, h: int, w: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return cv2.GaussianBlur(rng.integers(0, 256, (h, w)).astype(np.float32), (0, 0), 1.2)


class _Stage:
    """A mosaic fed with windows of a world; records the stage x of every frame that added a keyframe."""

    def __init__(self, world: np.ndarray, y: int = 200):
        self.world, self.y = world, y
        self.m = Mosaic()
        self.truth: list[float] = []
        self.events: list[str] = []

    def at(self, cx: float, n: int = 1) -> None:
        for _ in range(n):
            added, _ = self.m.track(self.world[self.y:self.y + VH, int(cx):int(cx) + VW].copy())
            if added:
                self.truth.append(cx)
            event = getattr(self.m, "last_event", None)  # (absent before the fix: the area checks still run)
            if event is not None:
                self.events.append(event)

    def pan(self, a: float, b: float, step: float = 20.0) -> None:
        for cx in np.arange(a, b, step if b > a else -step):
            self.at(float(cx))


# ------------------------------------------------------------------------------------------ area ids


def test_area_ids_follow_relocalisation_and_stay_unique_after_a_join():
    """Map A, jump to B (new area), jump to C (new area), jump back over B (re-localises onto B's keyframe) and
    pan from B to A (B joins A); then jump to D (new area) and pan from D to C (D joins C).

    On main the re-localisation onto B left the area id at C's, so the B -> A chain was labelled as C's area and
    the join moved C's keyframe (never connected to A) into A's coordinates; and the next new area took the id
    `areas - 1`, which after the join was C's id, so D was silently treated as part of C. Every keyframe of an
    area must keep one constant offset between its mosaic x and its stage x (no keyframe moved by a join it
    did not take part in), the two joined areas keep distinct ids, and C's keyframe never moves."""
    s = _Stage(_world(5, 800, 3000))
    s.pan(100, 260)  # area A
    s.at(700, 46)  # B: lost long enough to start a separate area
    s.at(1500, 46)  # C: another one
    c_idx = len(s.m.keyframes) - 1
    c_pos = (s.m.keyframes[c_idx].x, s.m.keyframes[c_idx].y)
    s.at(720, 12)  # back over B: re-localised onto B's keyframe
    assert not s.m.lost
    s.pan(720, 160)  # B -> A: the chain bridges into A
    s.at(2300, 46)  # D: a new area again
    d_idx = len(s.m.keyframes) - 1
    assert s.m.keyframes[d_idx].area not in {k.area for k in s.m.keyframes[:d_idx]}, "D reused an existing area id"
    s.pan(2300, 1540)  # D -> C: the chain bridges into C

    m = s.m
    assert len(s.truth) == len(m.keyframes)
    assert (m.keyframes[c_idx].x, m.keyframes[c_idx].y) == c_pos, "C's keyframe was moved by a join it was not in"
    a_id, c_id = m.keyframes[0].area, m.keyframes[c_idx].area
    assert a_id != c_id and m.areas == 2
    offsets = {a_id: [], c_id: []}
    for i, (k, cx) in enumerate(zip(m.keyframes, s.truth)):
        expect = c_id if cx >= 1500 else a_id  # A, B and the B -> A chain vs. C, D and the D -> C chain
        assert k.area == expect, f"keyframe {i} (stage x {cx}) is in area {k.area}, expected {expect}"
        offsets[k.area].append(k.x - cx)
        assert abs(k.y - m.keyframes[0 if expect == a_id else c_idx].y) < 1.0
    for area, offs in offsets.items():
        assert max(offs) - min(offs) < 1.0, f"area {area}: mosaic x - stage x spans {min(offs):.1f}..{max(offs):.1f}"
    # what each track() call did to the areas, for the live engine's log: `areas` alone went 3 -> 2 -> 3 -> 2
    assert s.events == ["new_area", "new_area", "areas_joined", "new_area", "areas_joined"]


def test_leaving_a_new_area_at_once_does_not_reacquire_in_the_old_one():
    """Start a separate area B and leave it on the very next frame, back to where A was left. The tracker's last
    known-position frame was still A's, so the first frame back re-acquired in A's coordinates under B's area id;
    the keyframes made there were labelled B, and their bridge into A then relabelled B's seed (never connected
    to A) as part of A. B must stay a separate area, and the frames back in A must belong to A."""
    s = _Stage(_world(6, 800, 2000))
    s.pan(100, 260)  # area A, last frame at stage 240
    for _ in range(80):
        s.at(900)
        if s.events:
            break
    assert s.events == ["new_area"]
    b_idx = len(s.m.keyframes) - 1
    b_pos = (s.m.keyframes[b_idx].x, s.m.keyframes[b_idx].y)
    s.at(240, 4)  # straight back to A's last view
    assert not s.m.lost
    n0 = len(s.m.keyframes)
    s.pan(240, 60)
    m = s.m
    assert len(m.keyframes) > n0 and m.areas == 2
    a_id = m.keyframes[0].area
    assert m.keyframes[b_idx].area != a_id and (m.keyframes[b_idx].x, m.keyframes[b_idx].y) == b_pos
    for i, (k, cx) in enumerate(zip(m.keyframes, s.truth)):
        if i != b_idx:
            assert k.area == a_id and abs(k.x - (cx - s.truth[0])) < 1.0, f"keyframe {i} (stage x {cx}): {k}"


# ------------------------------------------------------------------------------------------ cross-area work


def _matches_while_panning_a_new_area(n_first: int) -> tuple[int, int]:
    """(Mosaic._match calls, keyframes added) while panning inside a new area, with n_first keyframes mapped in
    the first area; the new area never overlaps the first, so nothing can bridge."""
    s = _Stage(_world(9, 800, 4000))
    s.pan(100, 100 + 120 * n_first, 20.0)
    s.at(3000, 46)
    assert s.m.areas == 2 and sum(k.area == 0 for k in s.m.keyframes) >= n_first
    calls = [0]
    orig = s.m._match

    def counted(*a, **k):
        calls[0] += 1
        return orig(*a, **k)

    s.m._match = counted
    n0 = len(s.m.keyframes)
    s.pan(3000, 3500, 20.0)
    assert s.m.areas == 2
    return calls[0], len(s.m.keyframes) - n0


def test_cross_area_matching_does_not_grow_with_the_other_areas():
    """Panning inside a new area matched every new keyframe against EVERY keyframe of every other area (no
    distance gate is possible there): the work per keyframe add grew with the size of the mapped areas. The same
    pan must cost the same number of matches whether the first area holds 3 or 12 keyframes."""
    small, added_s = _matches_while_panning_a_new_area(3)
    large, added_l = _matches_while_panning_a_new_area(12)
    assert added_s == added_l >= 3
    assert large == small, f"{small} matches with 3 keyframes elsewhere, {large} with 12 ({added_l} keyframe adds)"


# ------------------------------------------------------------------------------------------ yielding


def _holes_inside_keyframes(m: Mosaic, margin: int = 8) -> int:
    """Empty (NaN) mosaic pixels inside some keyframe's view, `margin` px in from its border (the feather window
    is ~0 at the very border)."""
    img, (x0, y0) = m.image()
    inside = np.zeros(img.shape, bool)
    for k in m.keyframes:
        h, w = k.image.shape
        kx, ky = round(k.x) - x0, round(k.y) - y0
        inside[max(0, ky + margin):max(0, ky + h - margin), max(0, kx + margin):max(0, kx + w - margin)] = True
    return int((~np.isfinite(img) & inside).sum())


def test_yielding_keyframes_leave_no_hole_inside_a_keyframe():
    """A closed lap (simulated specimen, sensor noise, motion blur), so neighbouring keyframes overlap. After PR
    #14 an older keyframe also yielded to the new one wherever the new one looked sharper; the per-tile
    comparison of about equally sharp views is decided by noise, so most overlapping pairs yielded to EACH OTHER
    and where two yield masks met nothing was left: holes inside keyframes (10x seed 3: 83 px, 40x seed 2:
    5 px; v0.9.0: none). Every pixel a keyframe shows must stay in the mosaic."""
    for seed, oname in ((3, "10x / 0.25"), (2, "40x / 0.85")):
        _, m = closed_loop(seed, oname, 160)
        tag = f"{oname} seed {seed}"
        assert len(m.keyframes) >= 10 and m.areas == 1, tag
        holes = _holes_inside_keyframes(m)
        assert holes == 0, f"{tag}: {holes} empty pixels inside keyframes"


def test_keyframe_add_does_not_resplat_its_neighbours(monkeypatch):
    """Two raster passes over the same ground with static, equally sharp views (sensor noise only). After PR #14
    each add un-splatted and re-splatted most overlapping keyframes (pairs yielding to each other), so the work
    per add grew with every pass: up to 41 canvas splats for one add here (on a 640 x 360 raster the median CPU per
    add reached 2.0 s on the third pass, v0.9.0 0.4 s), all under the engine lock. Counted in splats, not wall time, so the bound holds on a
    loaded machine: one splat for the new keyframe, plus two for at most one older keyframe that the new one
    beats in every tile (a re-splat that can happen legitimately, e.g. under motion blur)."""
    world = _world(7, 700, 1000)
    rng = np.random.default_rng(7)
    per_add: list[int] = []
    splats = [0]
    orig_add, orig_splat = Mosaic._add_keyframe, Mosaic._splat

    def add(self, *a, **k):
        n0 = splats[0]
        out = orig_add(self, *a, **k)
        per_add.append(splats[0] - n0)
        return out

    def splat(self, *a, **k):
        splats[0] += 1
        return orig_splat(self, *a, **k)

    monkeypatch.setattr(Mosaic, "_add_keyframe", add)
    monkeypatch.setattr(Mosaic, "_splat", splat)
    m = Mosaic()
    for rep in range(2):
        for row, y in enumerate(range(50, 230, 70)):
            xs = range(50, 600, 20) if (row + rep) % 2 == 0 else range(600, 50, -20)
            for x in xs:
                m.track(world[y + rep:y + rep + VH, x:x + VW] + rng.normal(0, 2, (VH, VW)).astype(np.float32))
    monkeypatch.undo()
    assert len(per_add) >= 20 and m.areas == 1
    assert max(per_add) <= 3, f"splats per keyframe add: {per_add}"
    assert _holes_inside_keyframes(m) == 0


# ------------------------------------------------------------------------------------------ canvas vs. GUI thread


class _BlendingSwitchedMidSplat(Mosaic):
    """While armed, the first read of `blending` sees "feather" and every later read "multiband": the GUI thread's
    set_blending landing between two reads of the engine thread's splat."""

    def __init__(self):
        self._reads = None
        super().__init__()

    @property
    def blending(self):
        if self._reads is None:
            return self._mode
        self._reads += 1
        return "feather" if self._reads == 1 else "multiband"

    @blending.setter
    def blending(self, mode):
        self._mode = mode


def test_a_blending_switch_during_a_splat_does_not_break_it():
    """The splat read the mode once to prepare its bands and again for every canvas tile, so a switch in between
    used a high band that was never computed (UnboundLocalError in the engine thread). It must splat in one mode:
    the first frame then renders as itself."""
    frame = _world(3, 400, 400)[:VH, :VW].copy()
    m = _BlendingSwitchedMidSplat()
    m._reads = 0
    added, _ = m.track(frame)
    m._reads = None
    assert added and m.blending == "feather"
    img, origin = m.image()
    assert origin == (0, 0) and img.shape == frame.shape
    np.testing.assert_allclose(img, np.clip(frame + 0.5, 0, 255).astype(np.uint8), atol=1e-3)


def test_colour_render_survives_a_tile_added_while_it_renders():
    """image_rgb iterated the live tile dict: a tile the engine thread added meanwhile raised "dictionary changed
    size during iteration" in the GUI thread. It renders a snapshot: the colour mosaic as it was."""
    frame = _world(4, 400, 400)[:VH, :VW].copy()
    rgb = np.stack([frame, 0.5 * frame, 255.0 - frame], axis=-1)
    m = Mosaic()
    m.track(frame, rgb=rgb)
    expected = m.image_rgb()
    assert expected is not None and len(m.tiles_rgb) == 1
    # the render truncates num / den to uint8: one level below the stored colour where the quotient rounds down
    stored = np.clip(rgb + 0.5, 0, 255).astype(np.uint8).astype(int)
    assert np.abs(expected[:VH, :VW, :3].astype(int) - stored).max() <= 1 and (expected[:VH, :VW, 3] == 255).all()

    class _Den(np.ndarray):
        def __gt__(self, other):  # the engine thread's splat lands while this tile is being rendered
            m.tiles_rgb.setdefault((3, 3), [np.zeros((m.tile, m.tile, 3), np.float32),
                                            np.zeros((m.tile, m.tile), np.float32)])
            return np.asarray(self) > other

    key = next(iter(m.tiles_rgb))
    num, den = m.tiles_rgb[key]
    m.tiles_rgb[key] = [num, den.view(_Den)]
    out = m.image_rgb()
    np.testing.assert_array_equal(out, expected)
