"""Live X-Y mosaic: keyframe-relative tracking, unbounded tiled canvas, loop-closure relaxation, 2.5D heights.

Positions are camera positions over the sample in mosaic pixels. If phase correlation reports image content
moving by +d, the camera moved by -d, so the new frame is placed at keyframe_position - d.

In-plane rotation (track_rotation): angle_deg is the rotation of the VIEW relative to the mosaic orientation, in
the sense of cv2.getRotationMatrix2D: positive = the content appears turned counter-clockwise on the screen
(y down). A frame at angle_deg = a is the mosaic-oriented view warped with getRotationMatrix2D(centre, +a), so it
is brought back into the mosaic orientation with getRotationMatrix2D(centre, -a). registration.deformation
reports the rotation of the warp that maps the template onto the view, which is the opposite sign; the conversion
happens in ONE place (Mosaic._verified: `_last_rot = -deformation rotation`). Keyframes are stored derotated into
the mosaic orientation (their angle_deg records the rotation they had when captured), so a view matched against
a keyframe measures its absolute angle; a view matched against the previous (raw, rotated) frame measures an
increment, and its shift is turned into mosaic axes with that frame's angle.
"""

from __future__ import annotations

import collections
import copy
import math
from dataclasses import dataclass, field, replace

import cv2
import numpy as np

from .focus import brenner
from .registration import (
    PEAK_SEPARATION_MIN,
    correlation_peaks,
    correlation_surface,
    deformation,
    phase_shift,
    phase_surface,
    surface_peak,
    within_envelope,
)


@dataclass
class Keyframe:
    x: float
    y: float
    image: np.ndarray
    sharpness: float
    bp: np.ndarray | None = None
    # keyframes that show parts of this view more sharply: (other_kf_index, grid) where grid = YIELD_GRID x
    # YIELD_GRID tiles of this view where that keyframe is sharper (1) or not (0). Offsets are computed live
    # from current keyframe positions in _yield_mask - never frozen/stored.
    yields: list = field(default_factory=list)
    area: int = 0  # mosaic area this keyframe belongs to
    angle_deg: float = 0.0  # in-plane stage angle at capture (degrees)
    rgb: np.ndarray | None = None  # uint8 HxWx3 colour image matching image, if any
    # thumbnail spectrum for ranking bridge candidates (Mosaic._thumb_spectrum); derived, never persisted
    spectrum: np.ndarray | None = field(default=None, repr=False, compare=False)


@dataclass
class MosaicInset:
    """A reference view at its own scale (another objective, or the previous objective's whole mosaic), shown under
    the mosaic canvas (underlay_insets). (x_um, y_um) is its top-left in the mosaic's µm frame: mosaic pixels times
    Mosaic.um_per_px; um_per_px is the scale of its own pixels. It belongs to an area and moves with it on a join."""

    objective: str
    um_per_px: float
    x_um: float
    y_um: float
    image: np.ndarray
    area: int = 0


@dataclass
class HeightPatch:
    x: float
    y: float
    edf: np.ndarray
    height: np.ndarray  # in the mosaic's pixel grid (resampled), NaN = invalid
    z_unit: str  # canonical unit id ("um" or "frames")
    offset: float = 0.0
    kf: int | None = None  # keyframe the patch is attached to (moves with it when the pose graph relaxes)
    rel: tuple = (0.0, 0.0)  # patch position minus keyframe position
    z_key: str = ""  # comparability key: only patches with equal keys are levelled and blended together
    rid: str = ""  # id of the result the patch came from


@dataclass
class CanvasSnapshot:
    """What a render of the mosaic needs, frozen at one moment (Mosaic.canvas_snapshot).

    The tile dicts hold the canvas arrays as they were when the snapshot was taken: the mosaic copies a tile before
    its first write after a snapshot (copy-on-write), so these arrays are never modified afterwards and can be
    rendered without any lock. Patches are shallow copies (their arrays are never modified in place)."""

    tiles: dict
    tiles_low: dict | None  # only in "multiband" mode
    tiles_rgb: dict
    tile: int
    patches: list
    insets: list
    um_per_px: float | None  # scale of the mosaic's pixel grid (Mosaic.um_per_px)


@dataclass
class Mosaic:
    tile: int = 512
    tiles: dict = field(default_factory=dict)  # (tx, ty) -> [num, den]
    tiles_low: dict = field(default_factory=dict)  # multiband low-frequency band (only when blending == "multiband")
    tiles_rgb: dict = field(default_factory=dict)  # colour canvas, feather weights reused from tiles
    keyframes: list = field(default_factory=list)
    patches: list = field(default_factory=list)
    insets: list = field(default_factory=list)  # multi-scale inset reference views
    x: float = 0.0
    y: float = 0.0
    lost: bool = False
    prev_frame: np.ndarray | None = None
    prev_bp: np.ndarray | None = None
    prev_xy: tuple = (0.0, 0.0)
    ref: int = 0
    areas: int = 1
    angle_deg: float = 0.0  # current in-plane angle estimate (degrees)
    track_rotation: bool = False  # estimate & compensate in-plane rotation while tracking
    current_area: int = 0  # area new keyframes are added to
    # What the pixel grid is: mosaic pixels are preview pixels of ONE objective and camera mode (the keyframes'
    # shape). `objective` is its name ("" = not recorded) and `um_per_px` its lateral scale (None = unknown). They
    # travel with the saved state, so an opened mosaic is shown at its own scale whatever objective is selected,
    # and the live engine tracks a frame into it only when it comes from that objective and mode.
    objective: str = ""
    um_per_px: float | None = None
    # what the last track() call did to the areas: None, "new_area" (an unconnected view started a separate area)
    # or "areas_joined" (a bridge merged the current area into another one). `areas` alone cannot tell a join
    # from a new area: a join lowers it and the next new area raises it back to the same number.
    last_event: str | None = None
    # "feather": weighted average with a sine feather. "multiband": a TWO-band blend (not a Laplacian pyramid): the
    # image is split at Gaussian sigma 8 px; the low band is feather-weighted, the high band is weighted by
    # feather**4 so that overlaps take fine texture mostly from the keyframe whose centre is nearest. The id
    # "multiband" is kept for saved sessions and the GUI.
    blending: str = "feather"
    recent_b: collections.deque = field(default_factory=lambda: collections.deque(maxlen=12))
    edges: list = field(default_factory=list)  # (i, j, dx, dy, weight): position of j minus position of i
    lost_frames: int = 0
    velocity: tuple | None = None  # last per-frame camera motion (constant-velocity prediction)
    last_good: tuple | None = None  # (frame, band-pass, position) of the latest frame with a known position
    lost_view: tuple | None = None  # (frame, band-pass) of the latest lost frame, for the steady-view test
    rep_cache: dict = field(default_factory=dict, repr=False)  # id(band-pass) -> (band-pass, repeat distance)
    _last_rot: float = field(default=0.0, repr=False)  # rotation (angle_deg convention) measured by the last verified match
    # copy-on-write of the canvas (canvas_snapshot): a tile entry is written in place only if it was made in the
    # current epoch; an older one may be shared with a snapshot and is copied first
    _canvas_epoch: int = field(default=0, repr=False)
    _tile_epoch: dict = field(default_factory=dict, repr=False)  # (canvas name, tile key) -> epoch it was made in

    def reset(self, keep_insets: bool = False) -> None:
        self.tiles.clear()
        self.tiles_low.clear()
        self.tiles_rgb.clear()
        self.keyframes.clear()
        self.patches.clear()
        if not keep_insets:
            self.insets.clear()
        self.x = self.y = 0.0
        self.objective, self.um_per_px = "", None  # set again by whoever builds the next mosaic
        self.lost = False
        self.prev_frame = None
        self.edges.clear()
        self.ref = 0
        self.areas = 1
        self.angle_deg = 0.0
        self.current_area = 0
        self.last_event = None
        self._last_rot = 0.0
        self.lost_frames = 0
        self.lost_view = None
        self.velocity = None
        self.last_good = None
        self.prev_bp = None
        self.recent_b.clear()
        self.rep_cache.clear()
        self._tile_epoch.clear()

    @property
    def frame_shape(self) -> tuple[int, int] | None:
        """Shape of the views the mosaic is made of (every keyframe has it), None while it is empty."""
        return self.keyframes[0].image.shape if self.keyframes else None

    def canvas_snapshot(self) -> CanvasSnapshot:
        """The canvas of this moment, for rendering outside the lock that serialises access to the mosaic.

        Must be called where nothing else touches the mosaic (the live engine calls it under its lock). It costs
        dict copies only: the tile arrays are shared with the snapshot, and the next splat into a shared tile
        copies that tile first (_tile), so the snapshot keeps showing exactly one state of the canvas."""
        self._canvas_epoch += 1
        return CanvasSnapshot(dict(self.tiles), dict(self.tiles_low) if self.blending == "multiband" else None,
                              dict(self.tiles_rgb), self.tile, [copy.copy(p) for p in self.patches],
                              list(self.insets), self.um_per_px)

    def _tile(self, store: dict, name: str, key: tuple, shapes: tuple) -> list:
        """The [num, den] entry of tile `key` in canvas `store`, safe to write in place (see canvas_snapshot)."""
        e = store.get(key)
        if e is None:
            e = store[key] = [np.zeros(sh, np.float32) for sh in shapes]
        elif self._tile_epoch.get((name, key)) != self._canvas_epoch:
            e = store[key] = [a.copy() for a in e]  # may be shared with a snapshot: write into a private copy
        self._tile_epoch[(name, key)] = self._canvas_epoch
        return e

    def set_blending(self, mode: str) -> None:
        if mode not in ("feather", "multiband"):
            raise ValueError("blending must be 'feather' or 'multiband'")
        if self.blending != mode:
            self.blending = mode
            self._rebuild_canvas()

    def add_inset(self, objective: str, um_per_px: float, x_um: float, y_um: float, image: np.ndarray) -> None:
        """Add an inset placed in the coordinates of the area the camera is in (MosaicInset)."""
        self.insets.append(MosaicInset(objective, um_per_px, x_um, y_um, np.asarray(image).copy(), self.current_area))

    # ---------------------------------------------------------------- matching
    @staticmethod
    def _bp(img: np.ndarray) -> np.ndarray:
        """Half-resolution band-pass copy (structures ~3-24 px): removes the mean, illumination gradients and the
        static vignette (which would bias phase correlation toward zero shift) and survives moderate motion blur."""
        s = cv2.resize(img.astype(np.float32), (img.shape[1] // 2, img.shape[0] // 2), interpolation=cv2.INTER_AREA)
        return cv2.GaussianBlur(s, (0, 0), 1.5) - cv2.GaussianBlur(s, (0, 0), 6.0)

    @staticmethod
    def _ncc(a_bp: np.ndarray, b_bp: np.ndarray, dx: float, dy: float) -> float:
        """Normalised cross-correlation of the overlap of a and b after shifting b back by (dx, dy) (full-res px).

        This is the geometric check that rejects false phase-correlation peaks (e.g. from specular specks).
        """
        a_bp = a_bp.astype(np.float32, copy=False)
        b_bp = b_bp.astype(np.float32, copy=False)
        h, w = a_bp.shape
        hx, hy = dx / 2, dy / 2
        ix, iy = math.floor(hx), math.floor(hy)
        if abs(ix) > 0.75 * w or abs(iy) > 0.75 * h:
            return -1.0
        fx, fy = hx - ix, hy - iy
        if fx > 1e-3 or fy > 1e-3:  # sub-pixel part: shift b so that the remaining offset is integer
            b_bp = cv2.warpAffine(b_bp, np.float32([[1, 0, -fx], [0, 1, -fy]]), (w, h), borderMode=cv2.BORDER_REFLECT)
        # b(x) = a(x - d): compare a(x) with b(x + d)
        a = a_bp[max(0, -iy):h + min(0, -iy), max(0, -ix):w + min(0, -ix)]
        b = b_bp[max(0, iy):h + min(0, iy), max(0, ix):w + min(0, ix)]
        if a.size < 0.2 * h * w:
            return -1.0
        a = a - a.mean()
        b = b - b.mean()
        return float((a * b).sum() / (np.sqrt((a * a).sum() * (b * b).sum()) + 1e-9))

    # Peak separation (robust sigmas) of a pair's phase-only surface above which a periodic-looking view is matched
    # as aperiodic. Measured on band-pass pairs flagged periodic, 40-60 pairs each: the tests' 9 x 13 px grating max
    # 9.3, random-angle pure stripes max 8.2, 2-D gratings max 9.9; stripes over grain at 1 % of the stripe amplitude
    # min 28.0, at 2 % min 70.6, a 2-D grating over 2 % grain min 58.2.
    PAIR_UNIQUE_MIN = 18.0

    @staticmethod
    def _repeat_distance(a32: np.ndarray, hwin: np.ndarray) -> float:
        """Shortest translation (band-pass px) under which the view repeats, inf for an aperiodic view.

        A view that repeats under a translation p (grating, stage micrometer, regular array) matches itself, and so
        any frame of it, equally well at d and d + p: its autocorrelation has secondary peaks as high as the central
        one. Periodicity is the same peak-separation test register_pair applies (PEAK_SEPARATION_MIN robust sigmas):
        measured on band-pass frames, textured specimens separate by >= 9 (75 % overlap) and >= 14 for every other
        valid condition of the accuracy matrix, a 9 x 13 px grating by <= 0.2. The repeat distance is then the
        nearest secondary peak that is comparable (within PEAK_SEPARATION_MIN sigmas) to the central one."""
        ac = correlation_peaks(correlation_surface(a32, a32, hwin))
        if ac.separation >= PEAK_SEPARATION_MIN:
            return math.inf
        return min((math.hypot(sx, sy) for sx, sy, z in ac.candidates[1:] if z > -PEAK_SEPARATION_MIN),
                   default=math.inf)

    def _match(self, a_img, a_bp, f, f_bp, win, predict=None, gate=None, derot=True):
        """Shift of f relative to a (full preview pixels) from band-pass phase correlation, and its NCC check.

        A periodic view fixes the shift only modulo its repeat distance R: the NCC passes at a wrong period just as
        well (17/96 grating pairs on main were accepted up to 38 px off). Then the shift is refined from `predict`
        (expected content shift in full preview px, from a motion model) and accepted only if the correlation
        maximum found lies within `gate` px of the prediction and within R/4 of it, so that every other repeat is at
        least 3x farther from the prediction than the chosen one. Without a prediction the quality is 0.

        A view whose pattern repeats can still carry texture that does not (stripes on a machined surface with
        grain between them). The repeat test looks at the view alone, where the strong pattern dominates; the shift
        is chosen on the pair's phase-only surface, where the texture counts as much. So when that surface has one
        clear peak (PAIR_UNIQUE_MIN), the pair is not ambiguous and is matched like an aperiodic one; without
        this, a striped specimen with grain lost track at every pan start and could never recover."""
        a32 = a_bp.astype(np.float32, copy=False)
        f32 = f_bp.astype(np.float32, copy=False)
        hwin = cv2.createHanningWindow((a32.shape[1], a32.shape[0]), cv2.CV_32F)
        hit = self.rep_cache.get(id(a_bp))
        if hit is not None and hit[0] is a_bp:  # keyframes are matched against every frame: test them once
            rep = hit[1]
        elif not derot:  # the derotated re-match of a pair (temporary arrays): do not evict the keyframes' entries
            rep = self._repeat_distance(a32, hwin)
        else:
            rep = self._repeat_distance(a32, hwin)
            if len(self.rep_cache) >= 8:  # bounded: holds at most 8 band-pass frames
                self.rep_cache.clear()
            self.rep_cache[id(a_bp)] = (a_bp, rep)
        periodic = math.isfinite(rep)
        pc = phase_surface(a32, f32, hwin) if periodic else None
        if periodic and correlation_peaks(pc).separation >= self.PAIR_UNIQUE_MIN:
            # pinned by the grain: take the phase-only peak itself. Plain correlation (ECC) is a ridge along the
            # pattern and slides along it (measured: up to 16 px errors with ECC refinement allowed to move 8 px)
            dx, dy = surface_peak(pc)
            dx, dy = 2 * dx, 2 * dy
            return self._verified(a32, f32, dx, dy, periodic, derot)
        if periodic:
            if predict is None or gate is None:
                dx, dy, _ = phase_shift(a32, f32, hwin)
                return 2 * dx, 2 * dy, 0.0
            px, py = predict[0] / 2, predict[1] / 2
            tol = min(gate / 2, rep / 4)
            try:
                crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 1e-4)
                _, W = cv2.findTransformECC(a32, f32, np.float32([[1, 0, px], [0, 1, py]]), cv2.MOTION_TRANSLATION,
                                            crit, None, 1)
            except cv2.error:
                return predict[0], predict[1], 0.0
            dx, dy = float(W[0, 2]), float(W[1, 2])
            if not np.isfinite(W).all() or math.hypot(dx - px, dy - py) > tol:
                return predict[0], predict[1], 0.0
            dx, dy = 2 * dx, 2 * dy
            return self._verified(a32, f32, dx, dy, periodic, derot)
        dx, dy, _ = phase_shift(a32, f32, hwin)
        # Refine with ECC (correlation coefficient): motion blur is a box kernel whose sinc spectrum flips sign,
        # which splits the phase-only correlation peak; plain correlation still peaks at the true shift.
        try:
            crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 1e-4)
            _, W = cv2.findTransformECC(a32, f32, np.float32([[1, 0, dx], [0, 1, dy]]), cv2.MOTION_TRANSLATION,
                                        crit, None, 1)
            if np.isfinite(W).all() and math.hypot(W[0, 2] - dx, W[1, 2] - dy) < 8:
                dx, dy = float(W[0, 2]), float(W[1, 2])
        except cv2.error:
            pass
        dx, dy = 2 * dx, 2 * dy
        return self._verified(a32, f32, dx, dy, periodic, derot)

    # Largest in-plane rotation between two matched views (change since the previous frame, or since the angle
    # estimate the frame is derotated by) that the rotation estimate (affine ECC started from the translation,
    # then a derotated re-match) recovers reliably. Measured on a textured world, 3 worlds x 3 shifts x both signs
    # (tests/test_feat_mosaic.py::test_rotation_capture_range): <= 3 deg matched 18/18 with position error <= 0.10
    # px and angle error <= 0.02 deg; 4-7 deg matched only 6-17 of 18 (the first translation estimate of a strongly
    # rotated pair often locks on a false peak) and at 8 deg and beyond never. Larger rotations are reported as NOT
    # matched (q = 0) instead of a pose that could be wrong.
    ROT_CAPTURE_DEG = 3.5
    ROT_MIN_DEG = 0.2  # smaller rotations are within the translation matcher's tolerance and are not corrected

    @staticmethod
    def _to_mosaic(dx: float, dy: float, angle_deg: float) -> tuple[float, float]:
        """A shift measured in the axes of a view at angle_deg, expressed in mosaic axes (see module docstring)."""
        c, s = math.cos(math.radians(angle_deg)), math.sin(math.radians(angle_deg))
        return c * dx - s * dy, s * dx + c * dy

    def _verified(self, a32: np.ndarray, f32: np.ndarray, dx: float, dy: float, periodic: bool = False,
                  derot: bool = True) -> tuple[float, float, float]:
        """(dx, dy, NCC) of the overlap at the shift (full preview px); NCC is 0 when a translation cannot
        describe the pair.

        With track_rotation the measured rotation is removed and the pair matched again: the shift of the
        derotated pair is the exact shift of the view centre (the translation of the rotated pair is biased by
        the rotation) and replaces the first one. The rotation, in the angle_deg convention, is left in
        self._last_rot; a rotation beyond ROT_CAPTURE_DEG gives q = 0 (not matched).

        Under a rotation the NCC stays high enough to pass the gates (3 deg: NCC 0.59-0.69; on main 73/96 such
        pairs were tracked with up to 2.2 px error, 26/96 loop-closed), so a match that could be accepted also gets
        an affine ECC fit on the band-pass pair (at most 160 px wide: ~2-3 ms, rotation accurate to ~0.03 deg on the
        accuracy matrix) and must lie inside the envelope register_pair uses for translation-only results
        (registration.MAX_ROT_DEG, MAX_SCALE_DEV). Valid pairs measure <= 0.14 deg (rotation 1 deg: 0.97-1.02 deg,
        scale 1.02: 0.020), 3 deg rotations 2.96-3.02 deg. A periodic view (stripes) gets the rotation-only fit:
        the affine one is unconstrained along the stripes (see registration.deformation), and its scale is not
        checked."""
        q = self._ncc(a32, f32, dx, dy)
        self._last_rot = 0.0
        derotated = False
        if self.track_rotation and derot:
            d = deformation(a32, f32, dx / 2, dy / 2, rotation_only=True)
            if d is not None and abs(d[1]) > self.ROT_MIN_DEG:
                if abs(d[1]) > self.ROT_CAPTURE_DEG:
                    return dx, dy, 0.0
                h, w = f32.shape
                M = cv2.getRotationMatrix2D(((w - 1) / 2.0, (h - 1) / 2.0), float(d[1]), 1.0)
                f_derot = cv2.warpAffine(f32, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
                dx2, dy2, q2 = self._match(a32, a32, f_derot, f_derot, None, derot=False)
                if q2 > q:
                    dx, dy, q = dx2, dy2, q2
                    self._last_rot = -float(d[1])  # the one sign conversion (module docstring)
                    derotated = True
        if (
            q >= self.WEAK_NCC
            and not derotated
            and not within_envelope(deformation(a32, f32, dx / 2, dy / 2, rotation_only=periodic))
        ):
            return dx, dy, 0.0
        return dx, dy, q

    def _accept(self, dx, dy, q, base_xy, h, w) -> bool:
        """Strong matches are accepted; weaker ones (motion blur) only if they agree with the constant-velocity
        prediction of where the camera is now (a standard tracking gate)."""
        if q >= self.TRACK_NCC:
            return True
        if q < self.WEAK_NCC or self.velocity is None:
            return False
        px, py = self.prev_xy[0] + self.velocity[0], self.prev_xy[1] + self.velocity[1]
        return math.hypot(base_xy[0] - dx - px, base_xy[1] - dy - py) < 0.025 * min(h, w)

    # ---------------------------------------------------------------- tracking
    TRACK_NCC = 0.5
    WEAK_NCC = 0.25  # accepted only when consistent with the predicted motion
    LOOP_NCC = 0.65

    def track(self, frame: np.ndarray, rgb: np.ndarray | None = None) -> tuple[bool, float]:
        """Update the camera position; add a keyframe when it has moved far enough over steady, sharp views.

        Returns (added_keyframe, match quality). Positions come from the current reference keyframe when it
        still overlaps, else from the previous frame (dead reckoning); every match must pass the NCC check.
        `rgb`, when given, is the colour image matching `frame` and is stored with the new keyframe and
        blended into the colour canvas when blending is active.
        """
        self.last_event = None
        f = frame.astype(np.float32)
        fb = self._bp(f)
        b = brenner(f)
        h, w = f.shape
        rgb_u8 = np.clip(rgb + 0.5, 0, 255).astype(np.uint8) if rgb is not None else None
        if not self.keyframes:
            self._add_keyframe(f, fb, b, self.x, self.y, None, (0.0, 0.0), rgb=rgb_u8)
            self.ref = 0
            self.prev_frame, self.prev_bp, self.prev_xy = f, fb, (self.x, self.y)
            return True, 1.0
        win = cv2.createHanningWindow((w, h), cv2.CV_32F)
        kf = self.keyframes[self.ref]
        # predicted position (constant velocity; at rest before a velocity is known; none while lost), used only to
        # choose among the equally good shifts of a periodic view, with the same tolerance as the tracking gate
        pred, gate = None, 0.025 * min(h, w)
        if self.prev_frame is not None:
            vx, vy = self.velocity if self.velocity is not None else (0.0, 0.0)
            pred = (self.prev_xy[0] + vx, self.prev_xy[1] + vy)
        expect = (lambda x0, y0: {"predict": (x0 - pred[0], y0 - pred[1]), "gate": gate}) if pred else \
            (lambda x0, y0: {})
        # keyframes are in the mosaic orientation: bring the frame to the current angle estimate first, so that the
        # rotation to recover is the change since the last frame (ROT_CAPTURE_DEG), not the accumulated angle
        fb_k = self._derotate(fb)
        dx, dy, q = self._match(kf.image, kf.bp, f, fb_k, win, **expect(kf.x, kf.y))
        rot = self.angle_deg + self._last_rot if self.track_rotation else 0.0
        old_xy = (self.x, self.y)
        pm = None
        if self._accept(dx, dy, q, (kf.x, kf.y), h, w):
            self.x, self.y = kf.x - dx, kf.y - dy  # anchored to the reference keyframe
            rel = (-dx, -dy)
        elif self.prev_frame is not None and self._accept(
                *(pm := self._match_prev(f, fb, win, expect)), self.prev_xy, h, w):
            self.x, self.y = self.prev_xy[0] - pm[0], self.prev_xy[1] - pm[1]  # frame-to-frame dead reckoning
            rel, q, rot = None, pm[2], self._last_rot
        else:
            self.lost = True
            self.lost_frames += 1
            # the last frame with a KNOWN position (motion-blurred like the current one) is the best reference
            if self.last_good is not None:
                gf, gb, gxy, gang = self.last_good
                gdx, gdy, gq = self._match(gf, gb, f, fb, win)
                if self.track_rotation:
                    gdx, gdy = self._to_mosaic(gdx, gdy, gang)  # the last good frame is raw, at angle gang
                if gq >= self.WEAK_NCC and math.hypot(gdx, gdy) < 0.5 * min(h, w):
                    self.x, self.y = gxy[0] - gdx, gxy[1] - gdy
                    if self.track_rotation:
                        self.angle_deg = gang + self._last_rot
                    self.lost, self.lost_frames, self.lost_view = False, 0, None
                    self.prev_frame, self.prev_bp, self.prev_xy = f, fb, (self.x, self.y)
                    self.velocity = None
                    self.last_good = (f, fb, (self.x, self.y), self.angle_deg)
                    return False, gq
            if self.lost_frames % 3 == 0 and self._relocalize(f, fb, win):
                self.prev_frame, self.prev_bp, self.prev_xy = f, fb, (self.x, self.y)
                self.last_good = (f, fb, (self.x, self.y), self.angle_deg)
            elif self.lost_frames >= self.NEW_AREA_AFTER and self._steady_view(f, fb, win, b):
                self._new_area(f, fb, b, rgb_u8)
                return True, q
            # the position of a lost frame is unknown: it must not seed dead reckoning, only the steady-view test
            self.prev_frame, self.prev_bp = None, None
            self.velocity = None
            self.lost_view = (f, fb)
            return False, q
        if self.track_rotation:
            # a keyframe is stored in the mosaic orientation, so matching against it measures the absolute angle;
            # matching against the previous raw frame measures an increment
            self.angle_deg = rot if rel is not None else self.angle_deg + rot
        self.velocity = (self.x - old_xy[0], self.y - old_xy[1])
        self.last_good = (f, fb, (self.x, self.y), self.angle_deg)
        speed = math.hypot(self.x - self.prev_xy[0], self.y - self.prev_xy[1]) if self.prev_frame is not None else 0.0
        self.lost, self.lost_frames = False, 0
        self.prev_frame, self.prev_bp, self.prev_xy = f, fb, (self.x, self.y)
        moved = math.hypot(self.x - kf.x, self.y - kf.y)
        # motion blur check against the last frames (same region), not against other regions' keyframes
        self.recent_b.append(b)
        sharp_enough = b >= 0.8 * float(np.median(self.recent_b))
        steady = speed < 0.02 * min(h, w)  # avoid motion-blurred tiles: add keyframes when the stage slows down
        if moved > 0.2 * min(h, w) and sharp_enough and (steady or moved > 0.45 * min(h, w)):
            # edge to the reference keyframe: measured if anchored, else implied by dead reckoning (weaker)
            rel_xy = rel if rel is not None else (self.x - kf.x, self.y - kf.y)
            self._add_keyframe(f, fb, b, self.x, self.y, self.ref, rel_xy,
                               weight=1.0 if rel is not None else 0.3, rgb=rgb_u8)
            self.ref = len(self.keyframes) - 1
            self._loop_closure(win)
            return True, q
        if moved > 0.45 * min(h, w):  # reference about to leave the view: switch to the nearest keyframe of this area
            self.ref = self._nearest(self.x, self.y, area=self.current_area)
        return False, q

    def _derotate(self, bp: np.ndarray) -> np.ndarray:
        """A band-pass view turned back by the current angle estimate (into the mosaic orientation)."""
        if not self.track_rotation or abs(self.angle_deg) <= self.ROT_MIN_DEG:
            return bp
        h, w = bp.shape
        M = cv2.getRotationMatrix2D(((w - 1) / 2.0, (h - 1) / 2.0), -self.angle_deg, 1.0)
        return cv2.warpAffine(bp, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    def _match_prev(self, f, fb, win, expect):
        """Match against the previous raw frame; the shift is returned in mosaic axes (the previous frame is at
        self.angle_deg, the angle estimate not yet updated for the current frame)."""
        dx, dy, q = self._match(self.prev_frame, self.prev_bp, f, fb, win, **expect(*self.prev_xy))
        if self.track_rotation:
            dx, dy = self._to_mosaic(dx, dy, self.angle_deg)
        return dx, dy, q

    NEW_AREA_AFTER = 40  # lost frames (about 3 s at 12.5 fps) before starting a separate mosaic area

    def _steady_view(self, f, fb, win, b) -> bool:
        """The view is still and sharp (matches the previous frame closely), so it can seed a new area."""
        if self.lost_view is None:
            return False
        # the steadiness hypothesis is zero motion, within the 2 px tested below
        dx, dy, q = self._match(self.lost_view[0], self.lost_view[1], f, fb, win, predict=(0.0, 0.0), gate=2.0)
        self.recent_b.append(b)
        return q > 0.8 and math.hypot(dx, dy) < 2.0 and b >= 0.8 * float(np.median(self.recent_b))

    def _new_area(self, f, fb, b, rgb=None) -> None:
        """Place an unconnected view to the right of everything mapped so far (its true position is unknown)."""
        w = f.shape[1]
        x = max(k.x for k in self.keyframes) + 1.3 * w
        y = min(k.y for k in self.keyframes)
        # a fresh id: ids of joined areas are not reused, so after a join the area count is not a free id
        self.current_area = max(k.area for k in self.keyframes) + 1
        self.angle_deg = 0.0  # the orientation of an unconnected view is unknown: it defines its own
        self._add_keyframe(f, fb, b, x, y, None, (0.0, 0.0), rgb=rgb)
        self.ref = len(self.keyframes) - 1
        self.x, self.y = x, y
        self.areas = len({k.area for k in self.keyframes})
        self.last_event = "new_area"
        self.lost_view = None
        self.lost, self.lost_frames = False, 0
        self.prev_frame, self.prev_bp, self.prev_xy = f, fb, (x, y)
        # the seed view is now the latest frame with a known position; left on the previous area's last frame, a
        # re-acquisition from it would place frames in that area's coordinates under this area's id
        self.last_good = (f, fb, (x, y), self.angle_deg)

    def _nearest(self, x, y, k: int = 1, area: int | None = None):
        """Index (k = 1) or indices of the keyframes nearest (x, y); with `area`, only that area's keyframes.

        Areas are not connected: a position in one area's coordinates says nothing about where another area's
        keyframes are (a separate area is placed to the right of everything, and panning it moves its coordinates
        over the others' keyframes). Whatever is anchored to the camera position (the tracking reference, a height
        map) must use the camera's own area."""
        idx = [i for i, kf in enumerate(self.keyframes) if area is None or kf.area == area]
        d = [math.hypot(x - self.keyframes[i].x, y - self.keyframes[i].y) for i in idx]
        order = np.argsort(d, kind="stable")
        return idx[int(order[0])] if k == 1 else [idx[int(i)] for i in order[:k]]

    def _relocalize(self, f, fb, win) -> bool:
        """After losing track, try the keyframes nearest the last known position (bounded work per attempt)."""
        best = None
        fb_k = self._derotate(fb)
        for i in self._nearest(self.x, self.y, k=8):
            k = self.keyframes[i]
            dx, dy, q = self._match(k.image, k.bp, f, fb_k, win)
            if q >= self.LOOP_NCC and (best is None or q > best[0]):
                best = (q, i, dx, dy, self.angle_deg + self._last_rot)
        if best is None:
            return False
        _, i, dx, dy, rot = best
        k = self.keyframes[i]
        self.ref = i
        self.x, self.y = k.x - dx, k.y - dy
        # the view is now in that keyframe's coordinates, which may be another area's: new keyframes belong there
        self.current_area = k.area
        if self.track_rotation:  # keyframes are in the mosaic orientation: the measured rotation is absolute
            self.angle_deg = rot
        self.lost, self.lost_frames, self.lost_view = False, 0, None
        return True

    def _add_keyframe(self, f, fb, b, x, y, parent, rel, weight=1.0, rgb=None):
        # 8-bit image and half-precision band-pass: ~0.35 MB per keyframe instead of ~1.2 MB
        if self.track_rotation and abs(self.angle_deg) > 0.1:
            # store the frame derotated into the mosaic orientation (angle_deg is the view's rotation, so the
            # inverse rotation brings it back; module docstring) and take the matching band-pass from it
            hh, ww = f.shape
            M = cv2.getRotationMatrix2D(((ww - 1) / 2.0, (hh - 1) / 2.0), -self.angle_deg, 1.0)
            f = cv2.warpAffine(f, M, (ww, hh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
            fb = self._bp(f)
            if rgb is not None:
                rgb = cv2.warpAffine(rgb, M, (ww, hh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        u8 = np.clip(f + 0.5, 0, 255).astype(np.uint8)
        rgb_u8 = np.clip(rgb + 0.5, 0, 255).astype(np.uint8) if rgb is not None else None
        kf = Keyframe(x, y, u8, b, fb.astype(np.float16), area=self.current_area,
                      angle_deg=self.angle_deg, rgb=rgb_u8)
        kf_idx = len(self.keyframes)
        kf_yields, older_yields = self._sharper_overlaps(kf)
        kf.yields = kf_yields
        self.keyframes.append(kf)
        for k_idx, grid in older_yields:
            k = self.keyframes[k_idx]
            self._splat(k.image, k.x, k.y, weight=-1.0, mask=self._yield_mask(k), rgb=k.rgb)
            k.yields.append((kf_idx, grid))
            self._splat(k.image, k.x, k.y, mask=self._yield_mask(k), rgb=k.rgb)
        if parent is not None:
            self.edges.append((parent, kf_idx, rel[0], rel[1], weight))
        # splat the STORED 8-bit image: a later correction subtracts exactly the same contribution
        self._splat(kf.image, x, y, mask=self._yield_mask(kf), rgb=kf.rgb)

    @staticmethod
    def _relative_sharpness(img: np.ndarray) -> float:
        """Brenner energy per unit variance: sharpness of the same content compared between two frames, independent
        of exposure and lamp flicker (which scale both by the same gain squared)."""
        img = img.astype(np.float32)
        v = float(img.var())
        return brenner(img) / v if v > 1e-6 else 0.0

    # Sharpness is compared tile by tile (1/8 of the view per axis, >= 1000 pixels at 320 x 240): a revisit defocused
    # by a fraction of the depth of field is sharper where the surface is higher and blurrier where it is lower, so
    # one global score can call it equal while it blurs half of the view (measured: a 0.5 DOF revisit scoring 1.005x
    # the original overall still cut the Laplacian variance of part of the view by 25 %).
    YIELD_GRID = 8

    def _sharper_overlaps(self, kf: Keyframe) -> tuple[list, list]:
        """Compare kf with existing keyframes that share at least 1/4 width and height of overlap.

        Returns (kf_yields, older_yields) where:
        - kf_yields: [(k_idx, grid)] of tiles in kf where an existing keyframe k is sharper;
        - older_yields: [(k_idx, grid)] of tiles in existing keyframe k where the new kf is sharper, only for a k
          that kf beats in EVERY tile both views show entirely (a blurred first pass under a sharp revisit).

        A pair yields in one direction only. The per-tile comparison of two equally sharp views is decided by
        noise, so letting both views yield wherever the other looked sharper made most overlapping pairs yield to
        each other (40x closed loop, seed 2: 53 of 55 yielding pairs), and where their yield masks met nothing was
        left (NaN holes inside keyframes) while every keyframe add re-splatted most of its neighbours. As in v0.9.0
        the new view yields tile by tile to sharper existing views; an existing view yields to the new one only
        when the new one is sharper in every compared tile, so the new one yields nothing to it."""
        h, w = kf.image.shape
        g = self.YIELD_GRID
        xs = np.linspace(0, w, g + 1).astype(int)
        ys = np.linspace(0, h, g + 1).astype(int)
        kf_out, older_out = [], []
        for k_idx, k in enumerate(self.keyframes):
            kh, kw = k.image.shape
            ox, oy = round(k.x - kf.x), round(k.y - kf.y)
            x0, y0, x1, y1 = max(0, ox), max(0, oy), min(w, ox + kw), min(h, oy + kh)
            if x1 - x0 < w / 4 or y1 - y0 < h / 4:
                continue
            grid_kf = np.zeros((g, g), np.float32)
            compared = kf_sharper = 0
            for i in range(g):
                for j in range(g):
                    ta, tb, la, lb = ys[i], ys[i + 1], xs[j], xs[j + 1]
                    if ta < y0 or tb > y1 or la < x0 or lb > x1:  # only tiles that both views show entirely
                        continue
                    mine = self._relative_sharpness(kf.image[ta:tb, la:lb])
                    theirs = self._relative_sharpness(k.image[ta - oy:tb - oy, la - ox:lb - ox])
                    grid_kf[i, j] = float(mine < theirs)
                    compared += 1
                    kf_sharper += mine > theirs
            if grid_kf.any():
                kf_out.append((k_idx, grid_kf))
                continue
            if compared == 0 or kf_sharper < compared:
                continue
            xs_k = xs if kw == w else np.linspace(0, kw, g + 1).astype(int)
            ys_k = ys if kh == h else np.linspace(0, kh, g + 1).astype(int)
            kx0, ky0 = max(0, -ox), max(0, -oy)
            kx1, ky1 = min(kw, w - ox), min(kh, h - oy)
            grid_k = np.zeros((g, g), np.float32)
            for i in range(g):
                for j in range(g):
                    ta, tb, la, lb = ys_k[i], ys_k[i + 1], xs_k[j], xs_k[j + 1]
                    if ta < ky0 or tb > ky1 or la < kx0 or lb > kx1:
                        continue
                    theirs = self._relative_sharpness(k.image[ta:tb, la:lb])
                    mine = self._relative_sharpness(kf.image[ta + oy:tb + oy, la + ox:lb + ox])
                    grid_k[i, j] = float(theirs < mine)
            if grid_k.any():
                older_out.append((k_idx, grid_k))
        return kf_out, older_out

    def _yield_mask(self, kf: Keyframe) -> np.ndarray | None:
        """Canvas weight factor of kf: 1 - min(1, 2 f_k) * t_k under every keyframe k that is sharper in some tiles,
        with f_k that keyframe's own feather and t_k its tile grid interpolated smoothly over kf's view. Wherever k
        carries at least half of its peak blend weight (its central ~2/3 per axis) and is sharper, kf contributes
        nothing; only in k's outer band, where k's weight fades out anyway, and between tiles does kf take over,
        smoothly, so no seam is cut into the canvas."""
        if not kf.yields:
            return None
        h, w = kf.image.shape
        m = np.ones((h, w), np.float32)
        for k_idx, grid in kf.yields:
            k = self.keyframes[k_idx]
            kh, kw = k.image.shape
            ox, oy = round(k.x - kf.x), round(k.y - kf.y)
            x0, y0, x1, y1 = max(0, ox), max(0, oy), min(w, ox + kw), min(h, oy + kh)
            if x1 > x0 and y1 > y0:
                t = cv2.resize(grid, (w, h), interpolation=cv2.INTER_LINEAR)
                f = np.minimum(1.0, 2.0 * _feather(kh, kw)[y0 - oy:y1 - oy, x0 - ox:x1 - ox])
                m[y0:y1, x0:x1] *= 1.0 - f * t[y0:y1, x0:x1]
        return m

    def _loop_closure(self, win) -> None:
        """Verified matches between the new keyframe and older overlapping ones become extra pose-graph edges.

        A loop edge is accepted only if it passes the NCC check and disagrees with the current estimate by
        less than 10% of the frame (larger disagreement means a false match, not accumulated drift).
        """
        j = len(self.keyframes) - 1
        cur = self.keyframes[j]
        h, w = cur.image.shape
        linked = {e[0] for e in self.edges if e[1] == j}
        new = []
        # cross-area bridge first (the stage re-connected two previously disconnected areas): afterwards the joined
        # area's keyframes are in cur's coordinates and get the ordinary predicted loop-closure test below
        bridge = self._bridge(j, linked, win)
        if bridge is not None:
            new.append(bridge)
            linked.add(bridge[0])
        for i, k in enumerate(self.keyframes[:-1]):
            if i in linked or k.area != cur.area or math.hypot(cur.x - k.x, cur.y - k.y) > 0.6 * min(h, w):
                continue
            # predicted from the current estimate, within the drift this test tolerates
            dx, dy, q = self._match(k.image, k.bp, cur.image, cur.bp, win,
                                    predict=(k.x - cur.x, k.y - cur.y),
                                    gate=0.1 * min(h, w))
            if q < self.LOOP_NCC:
                continue
            ex, ey = -dx, -dy  # measured position of j relative to i
            if math.hypot(ex - (cur.x - k.x), ey - (cur.y - k.y)) > 0.1 * min(h, w):
                continue
            new.append((i, j, ex, ey, 1.0))
        if not new:
            return
        self.edges += new
        self._relax()

    # Keyframes of OTHER areas matched (Mosaic._match) per keyframe add when looking for a bridge. Their positions
    # relative to the current area are unknown, so no distance gate applies; matching all of them made every add
    # in a new area cost one full match per mapped keyframe (5 areas, 34 keyframes in the first: 35 matches,
    # 388-532 ms per add instead of 26-64 ms). They are ranked by the phase-only correlation peak of quarter-
    # resolution band-pass thumbnails (Mosaic._bridge_score, ~0.2 ms per pair; on a random-texture world a 50 %
    # overlap scored 0.23-0.25, non-overlapping views at most 0.09) and only the best are matched.
    BRIDGE_CANDIDATES = 3

    @staticmethod
    def _thumb_spectrum(k: Keyframe) -> np.ndarray:
        """Spectrum of the keyframe's windowed, zero-mean band-pass at half its resolution (computed once)."""
        if k.spectrum is None:
            bp = k.bp.astype(np.float32)
            t = cv2.resize(bp, (bp.shape[1] // 2, bp.shape[0] // 2), interpolation=cv2.INTER_AREA)
            win = cv2.createHanningWindow((t.shape[1], t.shape[0]), cv2.CV_32F)
            k.spectrum = np.fft.rfft2((t - t.mean()) * win)
        return k.spectrum

    @classmethod
    def _bridge_score(cls, a: Keyframe, b: Keyframe) -> float:
        """Height of the phase-only correlation peak of two keyframes' thumbnails: the fraction of their spectrum
        that agrees on one shift. Used only to rank bridge candidates; the decision is _match's."""
        A, B = cls._thumb_spectrum(a), cls._thumb_spectrum(b)
        X = np.conj(A) * B
        shape = (b.bp.shape[0] // 2, b.bp.shape[1] // 2)
        return float(np.fft.irfft2(X / (np.abs(X) + 1e-12), s=shape).max())

    def _bridge(self, j: int, linked: set, win) -> tuple | None:
        """Join cur's area to another area that cur's view overlaps; returns the bridge edge or None.

        Only the BRIDGE_CANDIDATES best-ranked keyframes of other areas are matched. On success cur's whole area
        is moved into the other area's coordinates by the measured rigid transform and takes its id (_join_area).
        With track_rotation the match also measures how cur's area is turned against the other one (an area
        started without a connection defines its own orientation, _new_area), and that rotation is applied too;
        without it a turned area's keyframes would land about angle x distance from their true position."""
        cur = self.keyframes[j]
        if cur.bp is None:
            return None
        cands = [i for i, k in enumerate(self.keyframes[:-1])
                 if k.area != cur.area and i not in linked and k.bp is not None and k.bp.shape == cur.bp.shape]
        if not cands:
            return None
        if len(cands) > self.BRIDGE_CANDIDATES:
            cands = sorted(cands, key=lambda i: -self._bridge_score(self.keyframes[i], cur))[:self.BRIDGE_CANDIDATES]
        for i in cands:
            k = self.keyframes[i]
            dx, dy, q = self._match(k.image, k.bp, cur.image, cur.bp, win)
            if q < max(self.LOOP_NCC, 0.72):
                continue
            # (dx, dy) is the shift of the view centre (of the derotated pair, with track_rotation), and _last_rot
            # the rotation of cur's view, i.e. of its area's orientation, against k's area (angle_deg convention)
            rot = self._last_rot if self.track_rotation else 0.0
            ex, ey = -dx, -dy  # measured position of j (cur) relative to i (k), in k's area's axes
            self._join_area(cur.area, k.area, (cur.x, cur.y), (k.x + ex, k.y + ey), rot, cur.image.shape)
            self.last_event = "areas_joined"
            return (i, j, ex, ey, 1.0)
        return None

    def _join_area(self, old: int, target: int, src: tuple, dst: tuple, rot: float, shape: tuple) -> None:
        """Move area `old` into area `target`'s coordinates: the rigid transform that takes a view of size `shape`
        at top-left `src` (old coordinates) to `dst` (target coordinates), turned by `rot` (angle_deg convention:
        old's axes are those of a view at angle `rot` in target's). Applied to the area's keyframes (pose, stored
        image, angle), the edges between them, its height patches, and the tracker (the camera is in that area).

        The canvas is updated incrementally (_move_keyframes): only the moved keyframes, the keyframes whose
        yield masks refer to them, and the moved patches are re-splatted, not the whole canvas, so the time the
        caller holds the engine lock grows with the joined area, not with the mosaic."""
        h, w = shape
        c0 = (src[0] + (w - 1) / 2.0, src[1] + (h - 1) / 2.0)  # the joining view's centre, old coordinates
        c1 = (dst[0] + (w - 1) / 2.0, dst[1] + (h - 1) / 2.0)  # ... and target coordinates

        def place(x: float, y: float, hh: int, ww: int) -> tuple[float, float]:
            """Top-left of an hh x ww rectangle at top-left (x, y), old -> target coordinates (about its centre)."""
            vx, vy = self._to_mosaic(x + (ww - 1) / 2.0 - c0[0], y + (hh - 1) / 2.0 - c0[1], rot)
            return c1[0] + vx - (ww - 1) / 2.0, c1[1] + vy - (hh - 1) / 2.0

        turn = rot != 0.0  # _verified leaves 0 below ROT_MIN_DEG: nothing is resampled for an unturned area
        members = [idx for idx, k in enumerate(self.keyframes) if k.area == old]
        member_set = set(members)
        self._move_keyframes({idx: place(self.keyframes[idx].x, self.keyframes[idx].y,
                                         *self.keyframes[idx].image.shape) for idx in members},
                             {idx: self._turned_views(self.keyframes[idx], rot) for idx in members} if turn else None)
        for idx in members:
            self.keyframes[idx].area = target
            self.keyframes[idx].angle_deg += rot
        if turn:  # relative positions measured inside the area are in its old axes
            self.edges = [(i, j, *self._to_mosaic(ex, ey, rot), wt) if i in member_set and j in member_set
                          else (i, j, ex, ey, wt) for i, j, ex, ey, wt in self.edges]
        for p in self.patches:  # height patches follow their keyframe, turned with it
            if p.kf is None or p.kf not in member_set:
                continue
            self._splat(p.edf, p.x, p.y, weight=-4.0)
            if turn:
                p.edf, p.height = _turn(p.edf, -rot), _turn(p.height, -rot, invalid=np.nan)
                p.rel = self._to_mosaic(p.rel[0], p.rel[1], rot)
            k = self.keyframes[p.kf]
            p.x, p.y = k.x + p.rel[0], k.y + p.rel[1]
            self._splat(p.edf, p.x, p.y, weight=4.0)
        self._solve_offsets()  # the joined patches may now overlap others of their z_key: level them together
        if self.um_per_px:  # insets placed in the joined area move with it (their position is in µm of this grid)
            um = self.um_per_px
            for n, it in enumerate(self.insets):
                if it.area != old:
                    continue
                ih, iw = np.asarray(it.image).shape[:2]
                x, y = place(it.x_um / um, it.y_um / um, ih * it.um_per_px / um, iw * it.um_per_px / um)
                self.insets[n] = replace(it, x_um=x * um, y_um=y * um, area=target,
                                         image=_turn(np.asarray(it.image, np.float32), -rot) if turn else it.image)
        # the tracker: the camera is at the joining view, in the old area's coordinates and orientation
        self.x, self.y = place(self.x, self.y, h, w)
        self.prev_xy = place(*self.prev_xy, h, w)
        if self.velocity is not None:
            self.velocity = self._to_mosaic(*self.velocity, rot)
        if self.last_good is not None:
            gf, gb, gxy, gang = self.last_good
            self.last_good = (gf, gb, place(*gxy, h, w), gang + rot)
        self.angle_deg += rot
        self.current_area = target
        self.areas = len({kf.area for kf in self.keyframes})

    def _turned_views(self, k: Keyframe, rot: float) -> tuple:
        """(image, band-pass, rgb) of keyframe k turned from its area's orientation into one turned by -rot, the
        same resampling _add_keyframe applies to a rotated view."""
        f = _turn(k.image.astype(np.float32), -rot)
        rgb = None if k.rgb is None else np.clip(_turn(k.rgb.astype(np.float32), -rot) + 0.5, 0, 255).astype(np.uint8)
        return np.clip(f + 0.5, 0, 255).astype(np.uint8), self._bp(f).astype(np.float16), rgb

    def _move_keyframes(self, poses: dict, views: dict | None = None) -> None:
        """Give keyframes new positions (and, with `views`, new (image, band-pass, rgb)), keeping the canvas equal
        to a rebuild: every keyframe whose contribution changes (the moved ones and those whose yield masks refer
        to them) is un-splatted with what it splatted, updated, and splatted again."""
        moved = set(poses)
        resplat = moved | {idx for idx, k in enumerate(self.keyframes) if any(j in moved for j, _ in k.yields)}
        # un-splat all affected keyframes BEFORE updating any coordinates so every mask matches what was splatted
        for idx in resplat:
            k = self.keyframes[idx]
            self._splat(k.image, k.x, k.y, weight=-1.0, mask=self._yield_mask(k), rgb=k.rgb)
        for idx, (x, y) in poses.items():
            k = self.keyframes[idx]
            k.x, k.y = float(x), float(y)
            if views is not None:
                k.image, k.bp, k.rgb = views[idx]
                k.spectrum = None
        for idx in resplat:
            k = self.keyframes[idx]
            self._splat(k.image, k.x, k.y, mask=self._yield_mask(k), rgb=k.rgb)

    # Robust pose graph (Mosaic._relax). Residual scale floor: an edge is only down-weighted when it disagrees with
    # the graph by more than registration error can explain, even when the graph is so consistent that the measured
    # residual scale is smaller. Accepted _match results over the accuracy matrix (tests/test_gen_registration.py)
    # are within 0.26 px of the truth for pure translations (+3 DOF defocus, the worst) and within 0.48 px under
    # 1 deg rotation or 1.02 scale, so sigma >= 0.5 px: with the Cauchy constant an edge 1.2 px off keeps half its
    # weight. A false loop edge that passes _loop_closure's gate can be anything up to 24 px off.
    RELAX_SIGMA_FLOOR = 0.5
    RELAX_ITERS = 10

    def _relax(self) -> None:
        """Robust least-squares pose graph: iteratively reweighted least squares with Cauchy weights on the edges.

        Row weights multiply residuals, so an edge of weight w enters the objective as w^2. A weak prior on every
        node's current position keeps unconnected areas where they are and fixes the gauge freedom of each
        connected component (keyframe 0 gets a strong prior). Plain least squares spreads a false loop edge (one
        that passed the NCC and 10 %-of-frame gates) over the whole graph; after the first solve every edge is
        reweighted by w = 1 / (1 + (r / c)^2) of its residual r, with c = 2.385 sigma (the Cauchy constant with
        95 % efficiency for Gaussian residuals) and sigma the robust residual scale (median / 1.1774 for 2-D
        Gaussian residual lengths), never below RELAX_SIGMA_FLOOR. Consistent edges keep weight ~1, so the clean
        solution is the least-squares one; an edge many sigma off loses its influence instead of bending the graph.
        """
        from scipy.sparse import coo_matrix
        from scipy.sparse.linalg import splu

        n = len(self.keyframes)
        m = len(self.edges)
        E = np.array([(i, j, ex, ey, wt) for i, j, ex, ey, wt in self.edges], np.float64).reshape(m, 5)
        ii, jj = E[:, 0].astype(int), E[:, 1].astype(int)
        wp = np.full(n, 0.01)
        wp[0] = 10.0
        px = np.array([k.x for k in self.keyframes])
        py = np.array([k.y for k in self.keyframes])
        rows = np.concatenate([np.arange(n), np.repeat(np.arange(n, n + m), 2)])
        cols = np.concatenate([np.arange(n), np.stack([ii, jj], 1).ravel()])

        def solve(rw):
            w = E[:, 4] * rw
            vals = np.concatenate([wp, np.stack([-w, w], 1).ravel()])
            A = coo_matrix((vals, (rows, cols)), shape=(n + m, n)).tocsc()
            bx = np.concatenate([wp * px, w * E[:, 2]])
            by = np.concatenate([wp * py, w * E[:, 3]])
            lu = splu((A.T @ A).tocsc())
            return lu.solve(A.T @ bx), lu.solve(A.T @ by)

        rw = np.ones(m)
        sx, sy = solve(rw)
        for _ in range(self.RELAX_ITERS if m else 0):
            r = np.hypot(sx[jj] - sx[ii] - E[:, 2], sy[jj] - sy[ii] - E[:, 3])
            sigma = max(float(np.median(r)) / 1.1774, self.RELAX_SIGMA_FLOOR)
            new = 1.0 / (1.0 + (r / (2.385 * sigma)) ** 2)
            done = np.max(np.abs(np.sqrt(new) - rw)) < 1e-3
            rw = np.sqrt(new)  # row weight: the squared residual is weighted by w
            sx, sy = solve(rw)
            if done:
                break
        corr = max(math.hypot(k.x - x, k.y - y) for k, x, y in zip(self.keyframes, sx, sy))
        if corr < 0.5:
            return
        dxc, dyc = sx[self.ref] - self.keyframes[self.ref].x, sy[self.ref] - self.keyframes[self.ref].y
        self._move_keyframes({idx: (sx[idx], sy[idx]) for idx, k in enumerate(self.keyframes)
                              if math.hypot(k.x - sx[idx], k.y - sy[idx]) > 0.5})
        moved = False
        for p in self.patches:  # height patches follow the keyframe they are attached to
            if p.kf is None or p.kf >= len(self.keyframes):
                continue
            k = self.keyframes[p.kf]
            nx, ny = k.x + p.rel[0], k.y + p.rel[1]
            if math.hypot(nx - p.x, ny - p.y) > 0.5:
                self._splat(p.edf, p.x, p.y, weight=-4.0)
                p.x, p.y = nx, ny
                self._splat(p.edf, p.x, p.y, weight=4.0)
                moved = True
        if moved:
            self._solve_offsets()
        self._shift_tracker(dxc, dyc)

    def _shift_tracker(self, dx: float, dy: float) -> None:
        """The graph moved under the camera (relaxation): move every position the tracker holds by the same amount
        (a join moves them with its rigid transform, _join_area). `last_good` is the reference for re-acquiring a lost track; left behind,
        the first frame after a loss would be placed where the graph used to be."""
        self.x, self.y = self.x + dx, self.y + dy
        self.prev_xy = (self.prev_xy[0] + dx, self.prev_xy[1] + dy)
        if self.last_good is not None:
            gf, gb, gxy, gang = self.last_good
            self.last_good = (gf, gb, (gxy[0] + dx, gxy[1] + dy), gang)

    # ---------------------------------------------------------------- canvas
    def _splat(
        self,
        img: np.ndarray,
        x: float,
        y: float,
        weight: float = 1.0,
        mask: np.ndarray | None = None,
        rgb: np.ndarray | None = None,
    ) -> None:
        h, w = img.shape
        # the mode is read ONCE: another thread switching it in the middle of a splat must not mix the two paths
        # (the high band would be used without having been computed)
        multiband = self.blending == "multiband"
        base_f = _feather(h, w)
        if mask is not None:
            base_f = base_f * mask
        feather = base_f * weight
        if multiband:
            img_f = img.astype(np.float32)
            low = cv2.GaussianBlur(img_f, (0, 0), 8.0)
            high = img_f - low
            sharp_w = (base_f**4) * weight
        x0, y0 = round(x), round(y)
        t = self.tile
        for ty in range(y0 // t, (y0 + h - 1) // t + 1):
            for tx in range(x0 // t, (x0 + w - 1) // t + 1):
                ax0, ay0 = max(x0, tx * t), max(y0, ty * t)
                ax1, ay1 = min(x0 + w, (tx + 1) * t), min(y0 + h, (ty + 1) * t)
                if ax1 <= ax0 or ay1 <= ay0:
                    continue
                src = (slice(ay0 - y0, ay1 - y0), slice(ax0 - x0, ax1 - x0))
                dst = (slice(ay0 - ty * t, ay1 - ty * t), slice(ax0 - tx * t, ax1 - tx * t))
                fea = feather[src]
                if multiband:
                    num, den = self._tile(self.tiles, "grey", (tx, ty), ((t, t), (t, t)))
                    low_num, sharp_den = self._tile(self.tiles_low, "low", (tx, ty), ((t, t), (t, t)))
                    num[dst] += high[src] * sharp_w[src]
                    low_num[dst] += low[src] * fea
                    den[dst] += fea
                    sharp_den[dst] += sharp_w[src]
                else:
                    num, den = self._tile(self.tiles, "grey", (tx, ty), ((t, t), (t, t)))
                    num[dst] += img[src] * fea
                    den[dst] += fea
                if rgb is not None and rgb.ndim == 3:
                    rnum, rden = self._tile(self.tiles_rgb, "rgb", (tx, ty), ((t, t, 3), (t, t)))
                    rnum[dst] += rgb[src].astype(np.float32) * fea[..., None]
                    rden[dst] += fea

    def _rebuild_canvas(self) -> None:
        """Re-splat every keyframe into the tile canvas after the graph or blending mode changed.

        The canvas is a pure function of the state: every keyframe at its pose with its yield mask, plus every
        height patch's all-in-focus image at its pose with weight 4 (what add_height_patch splats and _relax
        moves), so a rebuilt canvas equals the incrementally built one up to float32 summation order.
        """
        self.tiles.clear()
        self.tiles_low.clear()
        self.tiles_rgb.clear()
        self._tile_epoch.clear()
        for k in self.keyframes:
            self._splat(k.image, k.x, k.y, mask=self._yield_mask(k), rgb=k.rgb)
        for p in self.patches:
            self._splat(p.edf, p.x, p.y, weight=4.0)

    def image(self) -> tuple[np.ndarray, tuple[int, int]]:
        """Assembled full-resolution mosaic (NaN where empty) and the mosaic coordinate of its top-left pixel."""
        img, origin, _ = render_tiles(
            self.tiles, self.tile, 1, self.tiles_low if self.blending == "multiband" else None
        )
        return img, origin

    def image_rgb(self, d: int = 1) -> np.ndarray | None:
        """Assembled full-resolution colour mosaic (RGBA uint8; alpha marks valid area), or None when no colour
        keyframes have been added."""
        return render_tiles_rgb(self.tiles_rgb, self.tile, d)

    def preview(self, max_px: int = 1600) -> tuple[np.ndarray, tuple[int, int], int]:
        """Mosaic rendered at a power-of-two reduction d so that its larger side is <= max_px (bounded memory).

        Returns (image, origin in full-resolution mosaic pixels, d)."""
        return render_tiles(
            self.tiles,
            self.tile,
            preview_factor(self.tiles, self.tile, max_px),
            self.tiles_low if self.blending == "multiband" else None,
        )

    # ---------------------------------------------------------------- session persistence
    def to_state_arrays(self):
        """Serialize the full pose graph: keyframes (area, angle, rgb), edges, patches and insets."""
        meta = {
            "tile": self.tile,
            "objective": self.objective,
            "um_per_px": self.um_per_px,
            "blending": self.blending,
            "track_rotation": self.track_rotation,
            "angle_deg": self.angle_deg,
            "current_area": self.current_area,
            "areas": self.areas,
            "x": self.x,
            "y": self.y,
            "ref": self.ref,
            # tracker state, so that tracking continues exactly where it stopped
            "tracking": {
                "prev_xy": [float(self.prev_xy[0]), float(self.prev_xy[1])],
                "velocity": None if self.velocity is None else [float(self.velocity[0]), float(self.velocity[1])],
                "lost": bool(self.lost),
                "lost_frames": int(self.lost_frames),
                "recent_b": [float(v) for v in self.recent_b],
                "last_good_xy": None if self.last_good is None else [float(v) for v in self.last_good[2]],
                "last_good_angle": None if self.last_good is None else float(self.last_good[3]),
            },
            "keyframes": [
                {
                    "x": k.x,
                    "y": k.y,
                    "sharpness": float(k.sharpness),
                    "area": int(k.area),
                    "angle_deg": float(k.angle_deg),
                    "yields": [[int(i), np.asarray(g, np.float32).tolist()] for i, g in k.yields],
                    "has_rgb": k.rgb is not None,
                }
                for k in self.keyframes
            ],
            "edges": [[int(i), int(j), float(ex), float(ey), float(wt)] for i, j, ex, ey, wt in self.edges],
            "patches": [
                {
                    "x": float(p.x),
                    "y": float(p.y),
                    "z_unit": p.z_unit,
                    "offset": float(p.offset),
                    "kf": None if p.kf is None else int(p.kf),
                    "rel": [float(p.rel[0]), float(p.rel[1])],
                    "z_key": p.z_key,
                    "rid": p.rid,
                }
                for p in self.patches
            ],
            "insets": [
                {
                    "objective": it.objective,
                    "um_per_px": float(it.um_per_px),
                    "x_um": float(it.x_um),
                    "y_um": float(it.y_um),
                    "area": int(it.area),
                }
                for it in self.insets
            ],
        }
        arrays: dict[str, np.ndarray] = {}
        for i, k in enumerate(self.keyframes):
            arrays[f"kf_img_{i}"] = k.image
            if k.bp is not None:
                arrays[f"kf_bp_{i}"] = k.bp.astype(np.float32)
            if k.rgb is not None:
                arrays[f"kf_rgb_{i}"] = k.rgb
        for i, p in enumerate(self.patches):
            arrays[f"patch_edf_{i}"] = p.edf
            arrays[f"patch_h_{i}"] = p.height
        for i, it in enumerate(self.insets):
            arrays[f"inset_img_{i}"] = it.image
        for name, v in (("prev_frame", self.prev_frame), ("prev_bp", self.prev_bp)):
            if v is not None:
                arrays[name] = np.asarray(v, np.float32)
        if self.last_good is not None:
            arrays["last_good_f"] = np.asarray(self.last_good[0], np.float32)
            arrays["last_good_bp"] = np.asarray(self.last_good[1], np.float32)
        if self.lost_view is not None:
            arrays["lost_view_f"] = np.asarray(self.lost_view[0], np.float32)
            arrays["lost_view_bp"] = np.asarray(self.lost_view[1], np.float32)
        return meta, arrays

    def load_state_arrays(self, meta: dict, arrays: dict) -> None:
        """Restore a serialized pose graph onto this mosaic and rebuild its canvas.

        All-or-nothing: the state is built on a scratch mosaic and only taken over when it is complete, so a
        damaged state raises and leaves this mosaic (the live one) untouched. After the call the mosaic keeps
        tracking exactly where the saved one stopped (previous frame, velocity, last good and lost views are
        part of the state); a state saved without them (older sessions) continues from the reference keyframe.
        """
        new = Mosaic()
        new._load_state(meta, arrays)
        self.__dict__.update(new.__dict__)

    def _load_state(self, meta: dict, arrays: dict) -> None:
        self.reset()
        self.tile = int(meta.get("tile", self.tile))
        self.objective = str(meta.get("objective") or "")  # absent before the mosaic recorded it
        um = meta.get("um_per_px")
        self.um_per_px = float(um) if um is not None and math.isfinite(float(um)) and float(um) > 0 else None
        self.blending = str(meta.get("blending", "feather"))
        self.track_rotation = bool(meta.get("track_rotation", False))
        self.angle_deg = float(meta.get("angle_deg", 0.0))
        self.current_area = int(meta.get("current_area", 0))
        self.areas = int(meta.get("areas", 1))
        self.x = float(meta.get("x", 0.0))
        self.y = float(meta.get("y", 0.0))
        n = len(meta.get("keyframes", []))
        self.ref = min(int(meta.get("ref", 0)), n - 1) if n else 0
        for i in range(n):
            km = meta["keyframes"][i]
            img = arrays[f"kf_img_{i}"]
            bp = arrays.get(f"kf_bp_{i}")
            if bp is not None:
                bp = bp.astype(np.float16)
            rgb = arrays.get(f"kf_rgb_{i}")
            kf = Keyframe(float(km["x"]), float(km["y"]), img, float(km["sharpness"]), bp,
                          [(int(j), np.asarray(g, np.float32)) for j, g in km.get("yields", [])],
                          int(km.get("area", 0)), float(km.get("angle_deg", 0.0)), rgb)
            self.keyframes.append(kf)
        for e in meta.get("edges", []):
            self.edges.append((int(e[0]), int(e[1]), float(e[2]), float(e[3]), float(e[4])))
        for i, pm in enumerate(meta.get("patches", [])):
            hgt = arrays[f"patch_h_{i}"]
            edf = arrays[f"patch_edf_{i}"]
            p = HeightPatch(float(pm["x"]), float(pm["y"]), edf.astype(np.float32),
                            hgt.astype(np.float32), str(pm.get("z_unit", "um")),
                            float(pm.get("offset", 0.0)),
                            None if pm.get("kf") is None else int(pm["kf"]),
                            (float(pm["rel"][0]), float(pm["rel"][1])),
                            str(pm.get("z_key", "um")), str(pm.get("rid", "")))
            self.patches.append(p)
        for i, im in enumerate(meta.get("insets", [])):
            self.insets.append(MosaicInset(str(im["objective"]), float(im["um_per_px"]),
                                           float(im["x_um"]), float(im["y_um"]),
                                           arrays[f"inset_img_{i}"], int(im.get("area", 0))))
        if self.keyframes:
            self.prev_frame = self.keyframes[self.ref].image.astype(np.float32)
            self.prev_bp = self.keyframes[self.ref].bp
            self.prev_xy = (self.x, self.y)
        tr = meta.get("tracking")
        if tr is not None:  # state written by this version: the exact tracker state
            self.prev_xy = (float(tr["prev_xy"][0]), float(tr["prev_xy"][1]))
            self.velocity = None if tr["velocity"] is None else (float(tr["velocity"][0]), float(tr["velocity"][1]))
            self.lost, self.lost_frames = bool(tr["lost"]), int(tr["lost_frames"])
            self.recent_b.extend(float(v) for v in tr["recent_b"])
            self.prev_frame = arrays.get("prev_frame")
            self.prev_bp = arrays.get("prev_bp")
            if tr["last_good_xy"] is not None:
                self.last_good = (arrays["last_good_f"], arrays["last_good_bp"],
                                  (float(tr["last_good_xy"][0]), float(tr["last_good_xy"][1])),
                                  float(tr["last_good_angle"]))
            if "lost_view_f" in arrays:
                self.lost_view = (arrays["lost_view_f"], arrays["lost_view_bp"])
        self._rebuild_canvas()

    # ---------------------------------------------------------------- 2.5D
    def add_height_patch(self, x: float, y: float, edf: np.ndarray, height: np.ndarray, z_unit: str,
                         z_key: str | None = None, rid: str = "") -> None:
        """Pin a height map at camera position (x, y). Patches are only levelled and blended with patches of the
        same z_key (same quantity, unit and scale); z_key defaults to the canonical unit."""
        from . import units

        unit = units.canon(z_unit)
        # the camera's own area: a keyframe of another area does not move with this view when the areas are joined
        mine = any(k.area == self.current_area for k in self.keyframes)
        kf = self._nearest(x, y, area=self.current_area) if mine else None
        rel = (x - self.keyframes[kf].x, y - self.keyframes[kf].y) if kf is not None else (0.0, 0.0)
        self.patches.append(HeightPatch(x, y, edf.astype(np.float32), height.astype(np.float32), unit, 0.0, kf, rel,
                                        z_key or unit, rid))
        self._splat(edf.astype(np.float32), x, y, weight=4.0)
        self._solve_offsets()

    def remove_patch(self, rid: str) -> bool:
        """Remove the height patch of a result (e.g. before re-adding it rescaled)."""
        for i, p in enumerate(self.patches):
            if p.rid and p.rid == rid:
                self._splat(p.edf, p.x, p.y, weight=-4.0)
                del self.patches[i]
                self._solve_offsets()
                return True
        return False

    def _solve_offsets(self) -> None:
        """Least-squares vertical offsets so overlapping height patches of the same z_key agree."""
        for key in {p.z_key for p in self.patches}:
            self._solve_offsets_for([p for p in self.patches if p.z_key == key])

    @staticmethod
    def _solve_offsets_for(P: list) -> None:
        n = len(P)
        if n < 2:
            for p in P:
                p.offset = 0.0
            return
        rows, rhs, pairs = [], [], []
        for i in range(n):
            for j in range(i + 1, n):
                a, b = P[i], P[j]
                ax, ay, bx, by = round(a.x), round(a.y), round(b.x), round(b.y)
                x0, y0 = max(ax, bx), max(ay, by)
                x1 = min(ax + a.height.shape[1], bx + b.height.shape[1])
                y1 = min(ay + a.height.shape[0], by + b.height.shape[0])
                if x1 - x0 < 16 or y1 - y0 < 16:  # bounding boxes first: cheap rejection of distant pairs
                    continue
                da = a.height[y0 - ay:y1 - ay, x0 - ax:x1 - ax]
                db = b.height[y0 - by:y1 - by, x0 - bx:x1 - bx]
                ok = np.isfinite(da) & np.isfinite(db)
                if ok.sum() < 50:
                    continue
                r = np.zeros(n)
                r[i], r[j] = 1, -1  # (h_i + o_i) - (h_j + o_j) = 0
                rows.append(r)
                rhs.append(-float(np.median(da[ok] - db[ok])))
                pairs.append((i, j))
        if not rows:
            for p in P:
                p.offset = 0.0
            return
        adj: list[list[int]] = [[] for _ in range(n)]
        for i, j in pairs:
            adj[i].append(j)
            adj[j].append(i)
        seen = [False] * n
        anchors = []
        for start in range(n):
            if seen[start]:
                continue
            stack = [start]
            seen[start] = True
            while stack:
                u = stack.pop()
                for v in adj[u]:
                    if not seen[v]:
                        seen[v] = True
                        stack.append(v)
            r0 = np.zeros(n)
            r0[start] = 1.0
            anchors.append(r0)
        sol = np.linalg.lstsq(np.vstack(anchors + rows), np.array([0.0] * len(anchors) + rhs), rcond=None)[0]
        for p, o in zip(P, sol):
            p.offset = float(o)

    def height_keys(self) -> list[str]:
        """Comparability keys present, most recent patch's key first."""
        return height_keys(self.patches)

    def height_mosaic(self, key: str | None = None) -> tuple[np.ndarray, tuple[int, int]] | None:
        """Blended heights of the patches with one z_key (default: the most recent patch's key)."""
        return height_mosaic(self.patches, key)

    def height_partitions(self) -> list[tuple[str, np.ndarray, tuple[int, int], int]]:
        """Every comparability group: (z_key, blended heights, origin, number of patches), most recent first."""
        return height_partitions(self.patches)


def height_keys(patches: list) -> list[str]:
    keys: list[str] = []
    for p in reversed(patches):
        if p.z_key not in keys:
            keys.append(p.z_key)
    return keys


def height_mosaic(patches: list, key: str | None = None) -> tuple[np.ndarray, tuple[int, int]] | None:
    """Feathered blend of the patches with z_key == key (default: the most recent patch's key). Patches of other
    keys are never mixed in: a µm map and a focus-frame map are different quantities."""
    if not patches:
        return None
    key = key if key is not None else patches[-1].z_key
    P = [p for p in patches if p.z_key == key]
    if not P:
        return None
    xs = [round(p.x) for p in P]
    ys = [round(p.y) for p in P]
    X0, Y0 = min(xs), min(ys)
    W = max(x + p.height.shape[1] for x, p in zip(xs, P)) - X0
    H = max(y + p.height.shape[0] for y, p in zip(ys, P)) - Y0
    num = np.zeros((H, W), np.float32)
    den = np.zeros((H, W), np.float32)
    for x, y, p in zip(xs, ys, P):
        h, w = p.height.shape
        fw = _feather(h, w)  # patch edges (least reliable, border effects) get less weight
        ok = np.isfinite(p.height)
        sl = (slice(y - Y0, y - Y0 + h), slice(x - X0, x - X0 + w))
        num[sl] += np.where(ok, (p.height + p.offset) * fw, 0)
        den[sl] += np.where(ok, fw, 0)
    return np.where(den > 1e-6, num / np.maximum(den, 1e-6), np.nan).astype(np.float32), (X0, Y0)


def height_partitions(patches: list) -> list[tuple[str, np.ndarray, tuple[int, int], int]]:
    out = []
    for key in height_keys(patches):
        hm = height_mosaic(patches, key)
        if hm is not None:
            out.append((key, hm[0], hm[1], sum(p.z_key == key for p in patches)))
    return out


_FEATHER: dict = {}


def _feather(h: int, w: int) -> np.ndarray:
    """Separable sine feather window, cached by shape."""
    f = _FEATHER.get((h, w))
    if f is None:
        f = np.outer(np.sin(np.linspace(0.05, np.pi - 0.05, h, dtype=np.float32)),
                     np.sin(np.linspace(0.05, np.pi - 0.05, w, dtype=np.float32)))
        if len(_FEATHER) > 16:
            _FEATHER.clear()
        _FEATHER[(h, w)] = f
    return f


def _turn(img: np.ndarray, deg: float, invalid: float | None = None) -> np.ndarray:
    """img turned about its centre by deg (cv2.getRotationMatrix2D convention), same size. Corners take the
    reflected border (as for derotated keyframes) or, with `invalid`, that value (NaN for heights)."""
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D(((w - 1) / 2.0, (h - 1) / 2.0), deg, 1.0)
    if invalid is None:
        return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                          borderValue=invalid)


def inset_boxes(insets: list, um_per_px: float) -> list[tuple[float, float, float, float]]:
    """(x0, y0, width, height) of every inset in full-resolution pixels of a mosaic whose pixel is um_per_px µm."""
    out = []
    for it in insets:
        h, w = np.asarray(it.image).shape[:2]
        s = it.um_per_px / um_per_px
        out.append((it.x_um / um_per_px, it.y_um / um_per_px, w * s, h * s))
    return out


def preview_factor(tiles: dict, tile: int, max_px: int, boxes: list = ()) -> int:
    """Smallest power-of-two reduction (<= tile) that keeps the rendered mosaic, together with `boxes` drawn with
    it ((x0, y0, w, h) in full-resolution pixels: inset_boxes), within max_px."""
    if not tiles and not boxes:
        return 1
    x0s, y0s, x1s, y1s = [], [], [], []
    if tiles:
        txs = [k[0] for k in tiles]
        tys = [k[1] for k in tiles]
        x0s, y0s, x1s, y1s = [min(txs) * tile], [min(tys) * tile], [(max(txs) + 1) * tile], [(max(tys) + 1) * tile]
    for bx, by, bw, bh in boxes:
        x0s.append(bx)
        y0s.append(by)
        x1s.append(bx + bw)
        y1s.append(by + bh)
    side = max(max(x1s) - min(x0s), max(y1s) - min(y0s))
    d = 1
    while side / d > max_px and d < tile:
        d *= 2
    return d


def render_tiles(tiles: dict, tile: int, d: int = 1,
                 tiles_low: dict | None = None) -> tuple[np.ndarray, tuple[int, int], int]:
    """Assemble tiles reduced by d (weighted: num and den are reduced separately). When tiles_low is given
    ("multiband" mode, a two-band blend), each tile's high-frequency band is divided by sharp_den and added to the
    feathered low-frequency band; empty canvas is NaN and the result is cropped to the valid bounding box, exactly
    as in feather mode. The tile arrays are read in place: nothing may splat into them during the call, so render
    the tiles of a Mosaic.canvas_snapshot (taken under the lock) or hold the lock (the dict is copied only so that
    bounds and tiles come from the same set of keys). Returns (image, full-resolution origin, d)."""
    tiles = dict(tiles)  # bounds and tiles from the same snapshot
    if not tiles:
        return np.full((8, 8), np.nan, np.float32), (0, 0), d
    t = tile
    td = t // d
    txs = [k[0] for k in tiles]
    tys = [k[1] for k in tiles]
    X0, Y0 = min(txs), min(tys)
    out = np.full(((max(tys) - Y0 + 1) * td, (max(txs) - X0 + 1) * td), np.nan, np.float32)
    for (tx, ty), (num, den) in tiles.items():
        if d > 1:
            num = cv2.resize(num, (td, td), interpolation=cv2.INTER_AREA)
            den = cv2.resize(den, (td, td), interpolation=cv2.INTER_AREA)
        sl = (slice((ty - Y0) * td, (ty - Y0 + 1) * td), slice((tx - X0) * td, (tx - X0 + 1) * td))
        if tiles_low is not None:
            low_num, sharp_den = tiles_low[(tx, ty)]
            if d > 1:
                low_num = cv2.resize(low_num, (td, td), interpolation=cv2.INTER_AREA)
                sharp_den = cv2.resize(sharp_den, (td, td), interpolation=cv2.INTER_AREA)
            # empty canvas (no feather weight) is NaN exactly as in the feather render, so that the bounding-box
            # crop below and every consumer of the NaN mask behave identically in both modes
            out[sl] = np.where(den > 1e-3,
                               np.where(sharp_den > 1e-6, num / np.maximum(sharp_den, 1e-6), 0.0)
                               + low_num / np.maximum(den, 1e-6),
                               np.nan)
        else:
            out[sl] = np.where(den > 1e-3, num / np.maximum(den, 1e-6), np.nan)
    rows = np.where(np.isfinite(out).any(1))[0]
    cols = np.where(np.isfinite(out).any(0))[0]
    if rows.size == 0:
        return out, (X0 * t, Y0 * t), d
    out = out[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]
    return out, (X0 * t + int(cols[0]) * d, Y0 * t + int(rows[0]) * d), d


def underlay_insets(view: tuple, rgb: np.ndarray | None, insets: list, um_per_px: float):
    """A rendered view with the insets drawn UNDER it, each at its own scale and position.

    `view` is render_tiles' (image, origin, d) and `rgb` its colour version on the same box (RGBA, or None). The
    box is extended to the insets' boxes (the origin stays on the view's pixel grid); every inset is resampled to
    the view's scale (inset µm/px over the mosaic's, divided by d) and painted in order, later ones on top, and
    the canvas is painted over them wherever it has data (grey: finite; colour: alpha > 0). Insets are grey: in
    the colour view they are opaque grey, except where they are NaN (empty). Returns (view, rgb) in the same
    forms."""
    img, (ox, oy), d = view
    boxes = inset_boxes(insets, um_per_px)
    has = bool(np.isfinite(img).any())
    x0 = min([b[0] for b in boxes] + ([ox] if has else []))
    y0 = min([b[1] for b in boxes] + ([oy] if has else []))
    x1 = max([b[0] + b[2] for b in boxes] + ([ox + img.shape[1] * d] if has else []))
    y1 = max([b[1] + b[3] for b in boxes] + ([oy + img.shape[0] * d] if has else []))
    X0, Y0 = math.floor(x0), math.floor(y0)
    if has:  # keep the canvas on whole pixels of the reduced view
        X0, Y0 = ox - d * math.ceil((ox - X0) / d), oy - d * math.ceil((oy - Y0) / d)
    H, W = math.ceil((y1 - Y0) / d), math.ceil((x1 - X0) / d)
    out = np.full((H, W), np.nan, np.float32)
    out_rgb = None if rgb is None else np.zeros((H, W, 4), np.uint8)
    for it, (bx, by, bw, bh) in zip(insets, boxes):
        tw, th = round(bw / d), round(bh / d)
        if tw < 1 or th < 1:
            continue
        im = np.asarray(it.image, np.float32)
        im = im.mean(axis=2) if im.ndim == 3 else im
        im = cv2.resize(im, (tw, th), interpolation=cv2.INTER_AREA if tw < im.shape[1] else cv2.INTER_LINEAR)
        px, py = round((bx - X0) / d), round((by - Y0) / d)
        sx0, sy0, sx1, sy1 = max(0, -px), max(0, -py), min(tw, W - px), min(th, H - py)
        if sx1 <= sx0 or sy1 <= sy0:
            continue
        part = im[sy0:sy1, sx0:sx1]
        ok = np.isfinite(part)  # an overview inset is NaN where the old mosaic was empty: nothing is drawn there
        out[py + sy0:py + sy1, px + sx0:px + sx1][ok] = part[ok]
        if out_rgb is not None:
            g = np.clip(part[ok] + 0.5, 0, 255).astype(np.uint8)
            out_rgb[py + sy0:py + sy1, px + sx0:px + sx1][ok] = np.stack([g, g, g, np.full_like(g, 255)], -1)
    if has:
        vx, vy = (ox - X0) // d, (oy - Y0) // d
        sub = out[vy:vy + img.shape[0], vx:vx + img.shape[1]]
        ok = np.isfinite(img)
        sub[ok] = img[ok]
        if out_rgb is not None:
            subr = out_rgb[vy:vy + rgb.shape[0], vx:vx + rgb.shape[1]]
            a = rgb[..., 3] > 0
            subr[a] = rgb[a]
    return (out, (X0, Y0), d), out_rgb


def render_tiles_rgb(tiles_rgb: dict, tile: int, d: int = 1) -> np.ndarray | None:
    """Assemble the colour canvas (RGBA uint8; alpha = 255 where valid) reduced by d. Reads the tile arrays in
    place, like render_tiles: pass the tiles of a Mosaic.canvas_snapshot or hold the lock."""
    tiles_rgb = dict(tiles_rgb)
    if not tiles_rgb:
        return None
    td = tile // d
    txs = [k[0] for k in tiles_rgb]
    tys = [k[1] for k in tiles_rgb]
    X0, Y0 = min(txs), min(tys)
    out = np.full(((max(tys) - Y0 + 1) * td, (max(txs) - X0 + 1) * td, 4), np.nan, np.float32)
    for (tx, ty), (num, den) in tiles_rgb.items():
        if d > 1:
            num = cv2.resize(num, (td, td), interpolation=cv2.INTER_AREA)
            den = cv2.resize(den, (td, td), interpolation=cv2.INTER_AREA)
        sl = (slice((ty - Y0) * td, (ty - Y0 + 1) * td), slice((tx - X0) * td, (tx - X0 + 1) * td))
        out[sl][..., :3] = np.clip(np.where(den[..., None] > 1e-3, num / np.maximum(den[..., None], 1e-6), np.nan), 0, 255)
        out[sl][..., 3] = np.where(den > 1e-3, 255.0, 0.0)
    return np.clip(np.nan_to_num(out, nan=0.0), 0, 255).astype(np.uint8)
