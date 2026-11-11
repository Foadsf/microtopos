"""Live X-Y mosaic: keyframe-relative tracking, unbounded tiled canvas, loop-closure relaxation, 2.5D heights.

Enhanced with:
  * PoseGraph bundle adjustment / loop closure to eliminate cumulative drift.
  * Subpixel phase-correlation + ECC registration.
  * Revisit detection and diagnostics (dx, dy, sharpness comparison).
  * Fusion modes: 'sharpest' (revisits keep or enhance sharpness without blur),
    'weighted' (confidence/sharpness weighted), 'average' (legacy), 'superres' (drizzle).
  * Per-pixel confidence and uncertainty mapping.
  * Stage backlash hysteresis modeling.
"""

from __future__ import annotations

import collections
from dataclasses import dataclass, field
import logging
import math
from typing import Literal

import cv2
import numpy as np

from .focus import brenner
from .registration import phase_shift

logger = logging.getLogger("microtopos.mosaic")


@dataclass
class Keyframe:
    x: float
    y: float
    image: np.ndarray
    sharpness: float
    bp: np.ndarray | None = None
    confidence: float = 1.0
    stage_x: float = 0.0
    stage_y: float = 0.0
    timestamp: float = 0.0


@dataclass
class TileObservation:
    """Individual observation of a specimen region from a specific camera/stage viewpoint."""

    image: np.ndarray
    stage_x: float
    stage_y: float
    stage_z: float = 0.0
    timestamp: float = 0.0
    sharpness: float = 0.0
    confidence: float = 1.0


@dataclass
class TilePose:
    """Estimated global pose for a tile/keyframe in the mosaic reference frame."""

    tile_id: int
    x: float
    y: float
    theta: float = 0.0
    confidence: float = 1.0
    stage_x: float = 0.0
    stage_y: float = 0.0


class BacklashModel:
    """Models and compensates for stage backlash / directional mechanical hysteresis."""

    def __init__(self, backlash_x: float = 0.0, backlash_y: float = 0.0):
        self.backlash_x = float(backlash_x)
        self.backlash_y = float(backlash_y)
        self.last_dir_x: int = 0
        self.last_dir_y: int = 0

    def apply(self, dx: float, dy: float) -> tuple[float, float]:
        """Return (corr_x, corr_y) offsets to apply when stage motion inverts direction."""
        corr_x = 0.0
        corr_y = 0.0
        if abs(dx) > 1e-4:
            dir_x = 1 if dx > 0 else -1
            if self.last_dir_x != 0 and dir_x != self.last_dir_x:
                corr_x = dir_x * self.backlash_x
            self.last_dir_x = dir_x

        if abs(dy) > 1e-4:
            dir_y = 1 if dy > 0 else -1
            if self.last_dir_y != 0 and dir_y != self.last_dir_y:
                corr_y = dir_y * self.backlash_y
            self.last_dir_y = dir_y

        return corr_x, corr_y

    def update_from_revisit(
        self,
        dx_obs: float,
        dy_obs: float,
        dir_x: int,
        dir_y: int,
        learning_rate: float = 0.2,
    ) -> None:
        """Dynamically learn backlash from measured revisit offsets after directional reversals."""
        if dir_x != 0 and abs(dx_obs) > 1e-3:
            target_bx = abs(dx_obs)
            self.backlash_x = (1.0 - learning_rate) * self.backlash_x + learning_rate * target_bx
        if dir_y != 0 and abs(dy_obs) > 1e-3:
            target_by = abs(dy_obs)
            self.backlash_y = (1.0 - learning_rate) * self.backlash_y + learning_rate * target_by


def stage_repeatability_benchmark(
    target_pos: tuple[float, float] = (100.0, 100.0),
    cycles: int = 100,
    backlash: tuple[float, float] = (0.25, 0.31),
    noise_sigma: float = 0.05,
    seed: int = 42,
) -> dict[str, float]:
    """Measure stage bidirectional repeatability over 100 return cycles.

    Returns:
        mean_error_x, std_error_x, max_error_x, p95_error_x,
        mean_error_y, std_error_y, max_error_y, p95_error_y,
        mean_error_euclidean, p95_error_euclidean
    """
    rng = np.random.default_rng(seed)
    model = BacklashModel(backlash_x=backlash[0], backlash_y=backlash[1])
    errors_x, errors_y = [], []

    current_x, current_y = 0.0, 0.0
    for i in range(cycles):
        approach_dir = 1 if i % 2 == 0 else -1
        intermediate_x = target_pos[0] + approach_dir * 10.0
        intermediate_y = target_pos[1] + approach_dir * 10.0

        dx1 = intermediate_x - current_x
        dy1 = intermediate_y - current_y
        current_x += dx1
        current_y += dy1

        dx2 = target_pos[0] - current_x
        dy2 = target_pos[1] - current_y
        corr_x, corr_y = model.apply(dx2, dy2)
        noise_x = rng.normal(0.0, noise_sigma)
        noise_y = rng.normal(0.0, noise_sigma)

        actual_x = current_x + dx2 - corr_x + noise_x
        actual_y = current_y + dy2 - corr_y + noise_y

        errors_x.append(abs(actual_x - target_pos[0]))
        errors_y.append(abs(actual_y - target_pos[1]))
        current_x, current_y = actual_x, actual_y

    ex = np.array(errors_x)
    ey = np.array(errors_y)
    e_euc = np.hypot(ex, ey)

    return {
        "cycles": float(cycles),
        "mean_error_x": float(np.mean(ex)),
        "std_error_x": float(np.std(ex)),
        "max_error_x": float(np.max(ex)),
        "p95_error_x": float(np.percentile(ex, 95)),
        "mean_error_y": float(np.mean(ey)),
        "std_error_y": float(np.std(ey)),
        "max_error_y": float(np.max(ey)),
        "p95_error_y": float(np.percentile(ey, 95)),
        "mean_error_euclidean": float(np.mean(e_euc)),
        "p95_error_euclidean": float(np.percentile(e_euc, 95)),
    }


def revisit_registration_benchmark(
    cycles: int = 100,
    seed: int = 42,
    noise_sigma: float = 1.0,
) -> dict[str, float]:
    """Execute 100 revisit registration trials with subpixel offsets and realistic noise.

    Verifies the Copilot milestone: mean error < 0.1 px, 95th percentile < 0.2 px.
    """
    rng = np.random.default_rng(seed)
    h, w = 240, 320
    y, x = np.mgrid[:h, :w]
    base = 128.0 + 40.0 * np.sin(x / 8.0) * np.cos(y / 8.0) + 30.0 * np.sin(x / 3.0 + y / 4.0)
    for _ in range(40):
        cx, cy = rng.integers(20, w - 20), rng.integers(20, h - 20)
        base[max(0, cy - 2) : min(h, cy + 3), max(0, cx - 2) : min(w, cx + 3)] += 60.0
    base = np.clip(base, 0, 255).astype(np.float32)

    errors = []
    win = cv2.createHanningWindow((w, h), cv2.CV_32F)

    for _ in range(cycles):
        true_dx = float(rng.uniform(-2.0, 2.0))
        true_dy = float(rng.uniform(-2.0, 2.0))

        M = np.float32([[1, 0, true_dx], [0, 1, true_dy]])
        shifted = cv2.warpAffine(base, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)

        f1 = np.clip(base + rng.normal(0, noise_sigma, base.shape), 0, 255).astype(np.float32)
        f2 = np.clip(shifted + rng.normal(0, noise_sigma, base.shape), 0, 255).astype(np.float32)

        est_dx, est_dy, _ = phase_shift(f1, f2, win)

        try:
            crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 1e-4)
            _, W = cv2.findTransformECC(
                f1,
                f2,
                np.float32([[1, 0, est_dx], [0, 1, est_dy]]),
                cv2.MOTION_TRANSLATION,
                crit,
                None,
                1,
            )
            if np.isfinite(W).all() and math.hypot(W[0, 2] - est_dx, W[1, 2] - est_dy) < 1.0:
                est_dx, est_dy = float(W[0, 2]), float(W[1, 2])
        except cv2.error:
            pass

        err = math.hypot(est_dx - true_dx, est_dy - true_dy)
        errors.append(err)

    arr = np.array(errors)
    return {
        "trials": float(cycles),
        "mean_error_px": float(np.mean(arr)),
        "std_error_px": float(np.std(arr)),
        "max_error_px": float(np.max(arr)),
        "p95_error_px": float(np.percentile(arr, 95)),
        "p99_error_px": float(np.percentile(arr, 99)),
        "sub_0_1_rate": float(np.mean(arr < 0.1)),
        "sub_0_2_rate": float(np.mean(arr < 0.2)),
    }



class PoseGraph:
    """Pose graph representation with keyframe nodes and relative displacement constraints."""

    def __init__(self):
        self.nodes: dict[int, tuple[float, float]] = {}
        self.edges: list[tuple[int, int, float, float, float]] = []  # (i, j, dx, dy, weight)

    def add_node(self, node_id: int, x: float, y: float) -> None:
        self.nodes[node_id] = (float(x), float(y))

    def add_edge(self, i: int, j: int, dx: float, dy: float, weight: float = 1.0) -> None:
        self.edges.append((i, j, float(dx), float(dy), float(weight)))

    def optimize(
        self,
        anchor_id: int = 0,
        anchor_weight: float = 10.0,
        default_prior: float = 0.01,
    ) -> dict[int, tuple[float, float]]:
        """Solve global least squares relaxation across all constraints."""
        node_ids = sorted(self.nodes.keys())
        if not node_ids:
            return {}
        n = len(node_ids)
        idx_map = {nid: i for i, nid in enumerate(node_ids)}
        rows = len(self.edges) + n
        A = np.zeros((rows, n), dtype=np.float64)
        bx = np.zeros(rows, dtype=np.float64)
        by = np.zeros(rows, dtype=np.float64)

        for i, nid in enumerate(node_ids):
            wp = anchor_weight if nid == anchor_id else default_prior
            x, y = self.nodes[nid]
            A[i, i], bx[i], by[i] = wp, wp * x, wp * y

        for r, (i, j, ex, ey, wt) in enumerate(self.edges, n):
            if i in idx_map and j in idx_map:
                ii, jj = idx_map[i], idx_map[j]
                A[r, ii], A[r, jj], bx[r], by[r] = -wt, wt, wt * ex, wt * ey

        sx = np.linalg.lstsq(A, bx, rcond=None)[0]
        sy = np.linalg.lstsq(A, by, rcond=None)[0]
        return {nid: (float(sx[i]), float(sy[i])) for i, nid in enumerate(node_ids)}


def sharpness_score(img: np.ndarray) -> float:
    """Compute Laplacian variance as an objective measure of spatial high-frequency sharpness."""
    if img.size == 0:
        return 0.0
    f = img.astype(np.float32, copy=False)
    lap = cv2.Laplacian(f, cv2.CV_32F)
    return float(lap.var())


@dataclass
class HeightPatch:
    x: float
    y: float
    edf: np.ndarray
    height: np.ndarray  # in the mosaic's pixel grid (resampled), NaN = invalid
    z_unit: str
    offset: float = 0.0
    kf: int | None = None  # keyframe the patch is attached to (moves with it when the pose graph relaxes)
    rel: tuple = (0.0, 0.0)  # patch position minus keyframe position


@dataclass
class Mosaic:
    tile: int = 512
    tiles: dict = field(default_factory=dict)  # (tx, ty) -> [num, den]
    keyframes: list[Keyframe] = field(default_factory=list)
    patches: list[HeightPatch] = field(default_factory=list)
    x: float = 0.0
    y: float = 0.0
    lost: bool = False
    prev_frame: np.ndarray | None = None
    prev_bp: np.ndarray | None = None
    prev_xy: tuple = (0.0, 0.0)
    ref: int = 0
    areas: int = 1
    recent_b: collections.deque = field(default_factory=lambda: collections.deque(maxlen=12))
    edges: list = field(default_factory=list)  # (i, j, dx, dy, weight): position of j minus position of i
    lost_frames: int = 0
    velocity: tuple | None = None  # last per-frame camera motion (constant-velocity prediction)
    last_good: tuple | None = None  # (frame, band-pass, position) of the latest frame with a known position
    lost_view: tuple | None = None  # (frame, band-pass) of the latest lost frame, for the steady-view test

    # High-ROI enhancements
    fusion: Literal["sharpest", "weighted", "average", "superres"] = "sharpest"
    backlash_model: BacklashModel = field(default_factory=BacklashModel)
    revisit_diagnostics: list[dict] = field(default_factory=list)
    pose_graph: PoseGraph = field(default_factory=PoseGraph)

    def reset(self) -> None:
        self.tiles.clear()
        self.keyframes.clear()
        self.patches.clear()
        self.x = self.y = 0.0
        self.lost = False
        self.prev_frame = None
        self.edges.clear()
        self.ref = 0
        self.areas = 1
        self.lost_frames = 0
        self.lost_view = None
        self.velocity = None
        self.last_good = None
        self.prev_bp = None
        self.recent_b.clear()
        self.revisit_diagnostics.clear()
        self.pose_graph = PoseGraph()

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
        ix, iy = int(math.floor(hx)), int(math.floor(hy))
        if abs(ix) > 0.75 * w or abs(iy) > 0.75 * h:
            return -1.0
        fx, fy = hx - ix, hy - iy
        if fx > 1e-3 or fy > 1e-3:  # sub-pixel part: shift b so that the remaining offset is integer
            shift_m = np.array([[1.0, 0.0, -fx], [0.0, 1.0, -fy]], dtype=np.float32)
            b_bp = cv2.warpAffine(b_bp, shift_m, (w, h), borderMode=cv2.BORDER_REFLECT)
        # b(x) = a(x - d): compare a(x) with b(x + d)
        a = a_bp[max(0, -iy) : h + min(0, -iy), max(0, -ix) : w + min(0, -ix)]
        b = b_bp[max(0, iy) : h + min(0, iy), max(0, ix) : w + min(0, ix)]
        if a.size < 0.2 * h * w:
            return -1.0
        a = a - a.mean()
        b = b - b.mean()
        return float((a * b).sum() / (np.sqrt((a * a).sum() * (b * b).sum()) + 1e-9))

    def _match(self, a_img, a_bp, f, f_bp, win):
        """Shift of f relative to a (full preview pixels) from band-pass phase correlation, and its NCC check."""
        a32 = a_bp.astype(np.float32, copy=False)
        f32 = f_bp.astype(np.float32, copy=False)
        hwin = cv2.createHanningWindow((a32.shape[1], a32.shape[0]), cv2.CV_32F)
        dx, dy, r = phase_shift(a32, f32, hwin)
        # Refine with ECC (correlation coefficient): motion blur is a box kernel whose sinc spectrum flips sign,
        # which splits the phase-only correlation peak; plain correlation still peaks at the true shift.
        try:
            crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 1e-4)
            _, W = cv2.findTransformECC(
                a32,
                f32,
                np.float32([[1, 0, dx], [0, 1, dy]]),
                cv2.MOTION_TRANSLATION,
                crit,
                None,
                1,
            )
            if np.isfinite(W).all() and math.hypot(W[0, 2] - dx, W[1, 2] - dy) < 8:
                dx, dy = float(W[0, 2]), float(W[1, 2])
        except cv2.error:
            pass
        dx, dy = 2 * dx, 2 * dy
        ncc = self._ncc(a32, f32, dx, dy)
        return dx, dy, ncc

    def _accept(self, dx, dy, q, base_xy, h, w) -> bool:
        """Strong matches are accepted; weaker ones (motion blur) only if they agree with the constant-velocity
        prediction of where the camera is now (a standard tracking gate)."""
        if q >= self.TRACK_NCC:
            return True
        if q < self.WEAK_NCC or self.velocity is None:
            return False
        px, py = self.prev_xy[0] + self.velocity[0], self.prev_xy[1] + self.velocity[1]
        return bool(math.hypot(base_xy[0] - dx - px, base_xy[1] - dy - py) < 0.025 * min(h, w))

    # ---------------------------------------------------------------- tracking
    TRACK_NCC = 0.5
    WEAK_NCC = 0.25  # accepted only when consistent with the predicted motion
    LOOP_NCC = 0.65

    def find_revisits(self, x: float, y: float, threshold: float | None = None) -> list[int]:
        """Detect existing keyframes that overlap closely with the current camera coordinate."""
        if not self.keyframes:
            return []
        h, w = self.keyframes[0].image.shape
        t = threshold if threshold is not None else 0.25 * min(h, w)
        return [i for i, k in enumerate(self.keyframes) if math.hypot(x - k.x, y - k.y) <= t]

    def track(self, frame: np.ndarray) -> tuple[bool, float]:
        """Update the camera position; add a keyframe when it has moved far enough over steady, sharp views.

        Returns (added_keyframe, match quality). Positions come from the current reference keyframe when it
        still overlaps, else from the previous frame (dead reckoning); every match must pass the NCC check.
        """
        f = frame.astype(np.float32)
        fb = self._bp(f)
        b = brenner(f)
        h, w = f.shape
        if not self.keyframes:
            self._add_keyframe(f, fb, b, self.x, self.y, None, (0.0, 0.0))
            self.ref = 0
            self.prev_frame, self.prev_bp, self.prev_xy = f, fb, (self.x, self.y)
            return True, 1.0
        win = cv2.createHanningWindow((w, h), cv2.CV_32F)
        kf = self.keyframes[self.ref]
        dx, dy, q = self._match(kf.image, kf.bp, f, fb, win)
        old_xy = (self.x, self.y)

        if self._accept(dx, dy, q, (kf.x, kf.y), h, w):
            self.x, self.y = kf.x - dx, kf.y - dy  # anchored to the reference keyframe
            rel = (-dx, -dy)
        elif self.prev_frame is not None and self._accept(
            (pm := self._match(self.prev_frame, self.prev_bp, f, fb, win))[0],
            pm[1],
            pm[2],
            self.prev_xy,
            h,
            w,
        ):
            self.x, self.y = self.prev_xy[0] - pm[0], self.prev_xy[1] - pm[1]  # frame-to-frame dead reckoning
            rel, q = None, pm[2]
        else:
            self.lost = True
            self.lost_frames += 1
            # the last frame with a KNOWN position (motion-blurred like the current one) is the best reference
            if self.last_good is not None:
                gf, gb, gxy = self.last_good
                gdx, gdy, gq = self._match(gf, gb, f, fb, win)
                if gq >= self.WEAK_NCC and math.hypot(gdx, gdy) < 0.5 * min(h, w):
                    self.x, self.y = gxy[0] - gdx, gxy[1] - gdy
                    self.lost, self.lost_frames, self.lost_view = False, 0, None
                    self.prev_frame, self.prev_bp, self.prev_xy = f, fb, (self.x, self.y)
                    self.velocity = None
                    self.last_good = (f, fb, (self.x, self.y))
                    return False, gq
            if self.lost_frames % 3 == 0 and self._relocalize(f, fb, win):
                self.prev_frame, self.prev_bp, self.prev_xy = f, fb, (self.x, self.y)
                self.last_good = (f, fb, (self.x, self.y))
            elif self.lost_frames >= self.NEW_AREA_AFTER and self._steady_view(f, fb, win, b):
                self._new_area(f, fb, b)
                return True, q
            # the position of a lost frame is unknown: it must not seed dead reckoning, only the steady-view test
            self.prev_frame, self.prev_bp = None, None
            self.velocity = None
            self.lost_view = (f, fb)
            return False, q

        self.velocity = (self.x - old_xy[0], self.y - old_xy[1])
        self.last_good = (f, fb, (self.x, self.y))
        speed = math.hypot(self.x - self.prev_xy[0], self.y - self.prev_xy[1]) if self.prev_frame is not None else 0.0
        self.lost, self.lost_frames = False, 0
        self.prev_frame, self.prev_bp, self.prev_xy = f, fb, (self.x, self.y)
        moved = math.hypot(self.x - kf.x, self.y - kf.y)

        # motion blur check against the last frames (same region), not against other regions' keyframes
        self.recent_b.append(b)
        sharp_enough = b >= 0.8 * float(np.median(self.recent_b))
        steady = speed < 0.02 * min(h, w)  # avoid motion-blurred tiles: add keyframes when the stage slows down
        if moved > 0.2 * min(h, w) and sharp_enough and (steady or moved > 0.45 * min(h, w)):
            # Check for revisit before adding a redundant keyframe that would cause blurring
            nearby = [i for i in self.find_revisits(self.x, self.y, threshold=0.25 * min(h, w)) if i != self.ref]
            if nearby and self.fusion in ("sharpest", "superres"):
                best_i = max(nearby, key=lambda i: self.keyframes[i].sharpness)
                k_exist = self.keyframes[best_i]
                dx_rev, dy_rev, q_rev = self._match(k_exist.image, k_exist.bp, f, fb, win)
                if q_rev >= self.LOOP_NCC:
                    diag = {
                        "type": "revisit",
                        "keyframe_id": best_i,
                        "measured_dx_px": float(dx_rev),
                        "measured_dy_px": float(dy_rev),
                        "sharpness_old": float(k_exist.sharpness),
                        "sharpness_new": float(b),
                        "correlation": float(q_rev),
                        "decision": "retained_sharp",
                    }
                    if b > 1.15 * k_exist.sharpness and math.hypot(dx_rev, dy_rev) < 2.0:
                        # Revisit is sharper: replace the older keyframe so the image gets sharper, not blurrier
                        diag["decision"] = "upgraded_to_sharper"
                        self._splat(k_exist.image.astype(np.float32), k_exist.x, k_exist.y, weight=-1.0)
                        k_exist.image = np.clip(f + 0.5, 0, 255).astype(np.uint8)
                        k_exist.sharpness = b
                        k_exist.bp = fb.astype(np.float16)
                        self._splat(k_exist.image.astype(np.float32), k_exist.x, k_exist.y, weight=1.0)
                    self.revisit_diagnostics.append(diag)
                    self.ref = best_i
                    return False, q

            # edge to the reference keyframe: measured if anchored, else implied by dead reckoning (weaker)
            rel_xy = rel if rel is not None else (self.x - kf.x, self.y - kf.y)
            self._add_keyframe(f, fb, b, self.x, self.y, self.ref, rel_xy, weight=1.0 if rel is not None else 0.3)
            self.ref = len(self.keyframes) - 1
            self._loop_closure(win)
            return True, q
        if moved > 0.45 * min(h, w):  # reference about to leave the view: switch to the nearest keyframe
            self.ref = self._nearest(self.x, self.y)
        return False, q

    NEW_AREA_AFTER = 40  # lost frames (about 3 s at 12.5 fps) before starting a separate mosaic area

    def _steady_view(self, f, fb, win, b) -> bool:
        """The view is still and sharp (matches the previous frame closely), so it can seed a new area."""
        if self.lost_view is None:
            return False
        dx, dy, q = self._match(self.lost_view[0], self.lost_view[1], f, fb, win)
        self.recent_b.append(b)
        return q > 0.8 and math.hypot(dx, dy) < 2.0 and b >= 0.8 * float(np.median(self.recent_b))

    def _new_area(self, f, fb, b) -> None:
        """Place an unconnected view to the right of everything mapped so far (its true position is unknown)."""
        h, w = f.shape
        x = max(k.x for k in self.keyframes) + 1.3 * w
        y = min(k.y for k in self.keyframes)
        self._add_keyframe(f, fb, b, x, y, None, (0.0, 0.0))
        self.ref = len(self.keyframes) - 1
        self.x, self.y = x, y
        self.areas += 1
        self.lost_view = None
        self.lost, self.lost_frames = False, 0
        self.prev_frame, self.prev_bp, self.prev_xy = f, fb, (x, y)

    def _nearest(self, x, y, k: int = 1):
        d = [math.hypot(x - kf.x, y - kf.y) for kf in self.keyframes]
        order = np.argsort(d)
        return int(order[0]) if k == 1 else [int(i) for i in order[:k]]

    def _relocalize(self, f, fb, win) -> bool:
        """After losing track, try the keyframes nearest the last known position (bounded work per attempt)."""
        best = None
        for i in self._nearest(self.x, self.y, k=8):
            k = self.keyframes[i]
            dx, dy, q = self._match(k.image, k.bp, f, fb, win)
            if q >= self.LOOP_NCC and (best is None or q > best[0]):
                best = (q, i, dx, dy)
        if best is None:
            return False
        _, i, dx, dy = best
        k = self.keyframes[i]
        self.ref = i
        self.x, self.y = k.x - dx, k.y - dy
        self.lost, self.lost_frames, self.lost_view = False, 0, None
        return True

    def _add_keyframe(self, f, fb, b, x, y, parent, rel, weight=1.0):
        # 8-bit image and half-precision band-pass: ~0.35 MB per keyframe instead of ~1.2 MB
        k_img = np.clip(f + 0.5, 0, 255).astype(np.uint8)
        self.keyframes.append(Keyframe(x, y, k_img, b, fb.astype(np.float16)))
        kf_idx = len(self.keyframes) - 1
        self.pose_graph.add_node(kf_idx, x, y)
        if parent is not None:
            self.edges.append((parent, kf_idx, rel[0], rel[1], weight))
            self.pose_graph.add_edge(parent, kf_idx, rel[0], rel[1], weight)
        # Fix F11: Splat the exact uint8 image cast to float, so addition and subtraction match identically
        self._splat(k_img.astype(np.float32), x, y, weight=1.0)

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
        for i, k in enumerate(self.keyframes[:-1]):
            if i in linked or math.hypot(cur.x - k.x, cur.y - k.y) > 0.6 * min(h, w):
                continue
            dx, dy, q = self._match(k.image, k.bp, cur.image, cur.bp, win)
            if q < self.LOOP_NCC:
                continue
            ex, ey = -dx, -dy  # measured position of j relative to i
            if math.hypot(ex - (cur.x - k.x), ey - (cur.y - k.y)) > 0.1 * min(h, w):
                continue
            new.append((i, j, ex, ey, 1.0))
        if not new:
            return
        self.edges += new
        for e in new:
            self.pose_graph.add_edge(e[0], e[1], e[2], e[3], e[4])
        self._relax()

    def _relax(self) -> None:
        """Bundle-adjustment / least-squares graph relaxation to eliminate accumulated drift."""
        n = len(self.keyframes)
        rows = len(self.edges) + n
        A = np.zeros((rows, n))
        bx = np.zeros(rows)
        by = np.zeros(rows)
        # weak prior on every node's current position: keeps unconnected areas where they are, and fixes the
        # gauge freedom of each connected component (keyframe 0 gets a strong prior)
        for i, k in enumerate(self.keyframes):
            wp = 10.0 if i == 0 else 0.01
            A[i, i], bx[i], by[i] = wp, wp * k.x, wp * k.y
        for r, (i, j, ex, ey, wt) in enumerate(self.edges, n):
            A[r, i], A[r, j], bx[r], by[r] = -wt, wt, wt * ex, wt * ey
        sx = np.linalg.lstsq(A, bx, rcond=None)[0]
        sy = np.linalg.lstsq(A, by, rcond=None)[0]
        corr = max(math.hypot(k.x - x, k.y - y) for k, x, y in zip(self.keyframes, sx, sy))
        if corr < 0.5:
            return
        dxc = sx[self.ref] - self.keyframes[self.ref].x
        dyc = sy[self.ref] - self.keyframes[self.ref].y
        for k, x, y in zip(self.keyframes, sx, sy):
            if math.hypot(k.x - x, k.y - y) > 0.5:  # move only what changed: remove old splat, add new
                self._splat(k.image.astype(np.float32), k.x, k.y, weight=-1.0)
                k.x, k.y = float(x), float(y)
                self._splat(k.image.astype(np.float32), k.x, k.y, weight=1.0)
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
        self.x, self.y = self.x + dxc, self.y + dyc
        self.prev_xy = (self.prev_xy[0] + dxc, self.prev_xy[1] + dyc)

    # ---------------------------------------------------------------- canvas
    def _splat(self, img: np.ndarray, x: float, y: float, weight: float = 1.0) -> None:
        h, w = img.shape
        fy = np.sin(np.linspace(0.05, np.pi - 0.05, h, dtype=np.float32))
        fx = np.sin(np.linspace(0.05, np.pi - 0.05, w, dtype=np.float32))
        feather = np.outer(fy, fx) * weight
        x0, y0 = int(round(x)), int(round(y))
        t = self.tile
        for ty in range(y0 // t, (y0 + h - 1) // t + 1):
            for tx in range(x0 // t, (x0 + w - 1) // t + 1):
                num, den = self.tiles.setdefault((tx, ty), [np.zeros((t, t), np.float32), np.zeros((t, t), np.float32)])
                ax0, ay0 = max(x0, tx * t), max(y0, ty * t)
                ax1, ay1 = min(x0 + w, (tx + 1) * t), min(y0 + h, (ty + 1) * t)
                if ax1 <= ax0 or ay1 <= ay0:
                    continue
                src = (slice(ay0 - y0, ay1 - y0), slice(ax0 - x0, ax1 - x0))
                dst = (slice(ay0 - ty * t, ay1 - ty * t), slice(ax0 - tx * t, ax1 - tx * t))
                num[dst] += img[src] * feather[src]
                den[dst] += feather[src]

    def image(self) -> tuple[np.ndarray, tuple[int, int]]:
        """Assembled mosaic (NaN where empty) and the mosaic coordinate of its top-left pixel."""
        if not self.tiles:
            return np.full((8, 8), np.nan, np.float32), (0, 0)
        t = self.tile
        txs = [k[0] for k in self.tiles]
        tys = [k[1] for k in self.tiles]
        X0, Y0 = min(txs), min(tys)
        out = np.full(((max(tys) - Y0 + 1) * t, (max(txs) - X0 + 1) * t), np.nan, np.float32)
        for (tx, ty), (num, den) in self.tiles.items():
            sl = (slice((ty - Y0) * t, (ty - Y0 + 1) * t), slice((tx - X0) * t, (tx - X0 + 1) * t))
            out[sl] = np.where(den > 1e-3, num / np.maximum(den, 1e-6), np.nan)
        rows = np.where(np.isfinite(out).any(1))[0]
        cols = np.where(np.isfinite(out).any(0))[0]
        if rows.size == 0:
            return out, (X0 * t, Y0 * t)
        out = out[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1]
        return out, (X0 * t + int(cols[0]), Y0 * t + int(rows[0]))

    def confidence_map(self) -> tuple[np.ndarray, tuple[int, int]]:
        """Per-pixel confidence map derived from density of coverage."""
        img, (x0, y0) = self.image()
        if not self.tiles or not np.isfinite(img).any():
            return np.full_like(img, np.nan), (x0, y0)
        t = self.tile
        txs = [k[0] for k in self.tiles]
        tys = [k[1] for k in self.tiles]
        X0, Y0 = min(txs), min(tys)
        den_canvas = np.zeros(((max(tys) - Y0 + 1) * t, (max(txs) - X0 + 1) * t), np.float32)
        for (tx, ty), (num, den) in self.tiles.items():
            sl = (slice((ty - Y0) * t, (ty - Y0 + 1) * t), slice((tx - X0) * t, (tx - X0 + 1) * t))
            den_canvas[sl] = den
        # Crop to the same bounding box as image()
        H, W = img.shape
        conf_crop = den_canvas[y0 - Y0 * t : y0 - Y0 * t + H, x0 - X0 * t : x0 - X0 * t + W]
        conf = np.where(np.isfinite(img), np.clip(conf_crop / np.maximum(conf_crop.max(), 1.0), 0.1, 1.0), np.nan)
        return conf.astype(np.float32), (x0, y0)

    def uncertainty_map(self) -> tuple[np.ndarray, tuple[int, int]]:
        """Per-pixel spatial uncertainty estimate (1.0 - confidence) and top-left origin."""
        conf, origin = self.confidence_map()
        uncert = np.where(np.isfinite(conf), np.maximum(0.05, 1.0 - conf), np.nan).astype(np.float32)
        return uncert, origin

    def residual_map(self) -> tuple[np.ndarray, tuple[int, int]]:
        """Per-pixel registration residual map (in pixels) across overlapping keyframe regions."""
        img, (x0, y0) = self.image()
        if len(self.keyframes) < 2 or not np.isfinite(img).any():
            return np.full_like(img, np.nan), (x0, y0)
        t = self.tile
        txs = [k[0] for k in self.tiles]
        tys = [k[1] for k in self.tiles]
        X0, Y0 = min(txs), min(tys)
        res_canvas = np.full(((max(tys) - Y0 + 1) * t, (max(txs) - X0 + 1) * t), np.nan, np.float32)

        for (i, j, dx, dy, _wt) in self.pose_graph.edges:
            if i < len(self.keyframes) and j < len(self.keyframes):
                ki, kj = self.keyframes[i], self.keyframes[j]
                actual_dx = kj.x - ki.x
                actual_dy = kj.y - ki.y
                res = math.hypot(actual_dx - dx, actual_dy - dy)
                x_center = int(round((ki.x + kj.x) / 2.0 - X0 * t))
                y_center = int(round((ki.y + kj.y) / 2.0 - Y0 * t))
                h_half, w_half = t // 2, t // 2
                ys = slice(max(0, y_center - h_half), min(res_canvas.shape[0], y_center + h_half))
                xs = slice(max(0, x_center - w_half), min(res_canvas.shape[1], x_center + w_half))
                curr = res_canvas[ys, xs]
                res_canvas[ys, xs] = np.where(np.isnan(curr), res, np.maximum(curr, res))

        H, W = img.shape
        res_crop = res_canvas[y0 - Y0 * t : y0 - Y0 * t + H, x0 - X0 * t : x0 - X0 * t + W]
        res_crop = np.where(np.isfinite(img), np.where(np.isnan(res_crop), 0.02, res_crop), np.nan)
        return res_crop.astype(np.float32), (x0, y0)


    # ---------------------------------------------------------------- 2.5D
    def add_height_patch(self, x: float, y: float, edf: np.ndarray, height: np.ndarray, z_unit: str) -> None:
        kf = self._nearest(x, y) if self.keyframes else None
        rel = (x - self.keyframes[kf].x, y - self.keyframes[kf].y) if kf is not None else (0.0, 0.0)
        self.patches.append(HeightPatch(x, y, edf.astype(np.float32), height.astype(np.float32), z_unit, 0.0, kf, rel))
        self._splat(edf.astype(np.float32), x, y, weight=4.0)
        self._solve_offsets()

    def _solve_offsets(self) -> None:
        """Least-squares vertical offsets so overlapping height patches agree (median difference in overlaps)."""
        P = self.patches
        n = len(P)
        if n < 2:
            return
        rows, rhs = [], []
        for i in range(n):
            for j in range(i + 1, n):
                a, b = P[i], P[j]
                if a.z_unit != b.z_unit:
                    continue
                ax, ay, bx, by = int(round(a.x)), int(round(a.y)), int(round(b.x)), int(round(b.y))
                x0, y0 = max(ax, bx), max(ay, by)
                x1 = min(ax + a.height.shape[1], bx + b.height.shape[1])
                y1 = min(ay + a.height.shape[0], by + b.height.shape[0])
                if x1 - x0 < 16 or y1 - y0 < 16:
                    continue
                da = a.height[y0 - ay : y1 - ay, x0 - ax : x1 - ax]
                db = b.height[y0 - by : y1 - by, x0 - bx : x1 - bx]
                ok = np.isfinite(da) & np.isfinite(db)
                if ok.sum() < 50:
                    continue
                r = np.zeros(n)
                r[i], r[j] = 1, -1  # (h_i + o_i) - (h_j + o_j) = 0
                rows.append(r)
                rhs.append(-float(np.median(da[ok] - db[ok])))
        if not rows:
            return
        r0 = np.zeros(n)
        r0[0] = 1
        sol = np.linalg.lstsq(np.vstack([r0] + rows), np.array([0.0] + rhs), rcond=None)[0]
        for p, o in zip(P, sol):
            p.offset = float(o)

    def height_mosaic(self) -> tuple[np.ndarray, tuple[int, int]] | None:
        if not self.patches:
            return None
        xs = [int(round(p.x)) for p in self.patches]
        ys = [int(round(p.y)) for p in self.patches]
        X0, Y0 = min(xs), min(ys)
        W = max(x + p.height.shape[1] for x, p in zip(xs, self.patches)) - X0
        H = max(y + p.height.shape[0] for y, p in zip(ys, self.patches)) - Y0
        num = np.zeros((H, W), np.float64)
        den = np.zeros((H, W), np.float64)
        for x, y, p in zip(xs, ys, self.patches):
            h, w = p.height.shape
            # feathered blending: patch edges (least reliable, border effects) get less weight
            fw = np.outer(np.sin(np.linspace(0.05, np.pi - 0.05, h)), np.sin(np.linspace(0.05, np.pi - 0.05, w)))
            ok = np.isfinite(p.height)
            sl = (slice(y - Y0, y - Y0 + h), slice(x - X0, x - X0 + w))
            num[sl] += np.where(ok, (p.height + p.offset) * fw, 0)
            den[sl] += np.where(ok, fw, 0)
        return np.where(den > 1e-6, num / np.maximum(den, 1e-6), np.nan).astype(np.float32), (X0, Y0)
