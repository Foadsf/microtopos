"""Live X-Y mosaic: keyframe-relative tracking, unbounded tiled canvas, loop-closure relaxation, 2.5D heights.

Positions are camera positions over the sample in mosaic pixels. If phase correlation reports image content
moving by +d, the camera moved by -d, so the new frame is placed at keyframe_position - d.
"""

from __future__ import annotations

import collections
import math
from dataclasses import dataclass, field

import cv2
import numpy as np

from .focus import brenner
from .registration import (PEAK_SEPARATION_MIN, correlation_peaks, correlation_surface, deformation, phase_shift,
                           within_envelope)


@dataclass
class Keyframe:
    x: float
    y: float
    image: np.ndarray
    sharpness: float
    bp: np.ndarray | None = None


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
class Mosaic:
    tile: int = 512
    tiles: dict = field(default_factory=dict)  # (tx, ty) -> [num, den]
    keyframes: list = field(default_factory=list)
    patches: list = field(default_factory=list)
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
            b_bp = cv2.warpAffine(b_bp, np.float32([[1, 0, -fx], [0, 1, -fy]]), (w, h), borderMode=cv2.BORDER_REFLECT)
        # b(x) = a(x - d): compare a(x) with b(x + d)
        a = a_bp[max(0, -iy):h + min(0, -iy), max(0, -ix):w + min(0, -ix)]
        b = b_bp[max(0, iy):h + min(0, iy), max(0, ix):w + min(0, ix)]
        if a.size < 0.2 * h * w:
            return -1.0
        a = a - a.mean()
        b = b - b.mean()
        return float((a * b).sum() / (np.sqrt((a * a).sum() * (b * b).sum()) + 1e-9))

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

    def _match(self, a_img, a_bp, f, f_bp, win, predict=None, gate=None):
        """Shift of f relative to a (full preview pixels) from band-pass phase correlation, and its NCC check.

        A periodic view fixes the shift only modulo its repeat distance R: the NCC passes at a wrong period just as
        well (17/96 grating pairs on main were accepted up to 38 px off). Then the shift is refined from `predict`
        (expected content shift in full preview px, from a motion model) and accepted only if the correlation
        maximum found lies within `gate` px of the prediction and within R/4 of it, so that every other repeat is at
        least 3x farther from the prediction than the chosen one. Without a prediction the quality is 0."""
        a32 = a_bp.astype(np.float32, copy=False)
        f32 = f_bp.astype(np.float32, copy=False)
        hwin = cv2.createHanningWindow((a32.shape[1], a32.shape[0]), cv2.CV_32F)
        rep = self._repeat_distance(a32, hwin)
        if math.isfinite(rep):
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
            return dx, dy, self._verified(a32, f32, dx, dy)
        dx, dy, r = phase_shift(a32, f32, hwin)
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
        return dx, dy, self._verified(a32, f32, dx, dy)

    def _verified(self, a32: np.ndarray, f32: np.ndarray, dx: float, dy: float) -> float:
        """NCC of the overlap at the shift (full preview px), or 0 when a translation cannot describe the pair.

        Under a rotation the NCC stays high enough to pass the gates (3 deg: NCC 0.59-0.69; on main 73/96 such
        pairs were tracked with up to 2.2 px error, 26/96 loop-closed), so a match that could be accepted also gets
        an affine ECC fit on the band-pass pair (at most 160 px wide: ~2-3 ms, rotation accurate to ~0.03 deg on the
        accuracy matrix) and must lie inside the envelope register_pair uses for translation-only results
        (registration.MAX_ROT_DEG, MAX_SCALE_DEV). Valid pairs measure <= 0.14 deg (rotation 1 deg: 0.97-1.02 deg,
        scale 1.02: 0.020), 3 deg rotations 2.96-3.02 deg."""
        q = self._ncc(a32, f32, dx, dy)
        if q >= self.WEAK_NCC and not within_envelope(deformation(a32, f32, dx / 2, dy / 2, max_side=160)):
            return 0.0
        return q

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
        # predicted position (constant velocity; at rest before a velocity is known; none while lost), used only to
        # choose among the equally good shifts of a periodic view, with the same tolerance as the tracking gate
        pred, gate = None, 0.025 * min(h, w)
        if self.prev_frame is not None:
            vx, vy = self.velocity if self.velocity is not None else (0.0, 0.0)
            pred = (self.prev_xy[0] + vx, self.prev_xy[1] + vy)
        expect = (lambda x0, y0: {"predict": (x0 - pred[0], y0 - pred[1]), "gate": gate}) if pred else \
            (lambda x0, y0: {})
        dx, dy, q = self._match(kf.image, kf.bp, f, fb, win, **expect(kf.x, kf.y))
        old_xy = (self.x, self.y)
        if self._accept(dx, dy, q, (kf.x, kf.y), h, w):
            self.x, self.y = kf.x - dx, kf.y - dy  # anchored to the reference keyframe
            rel = (-dx, -dy)
        elif self.prev_frame is not None and self._accept(*(pm := self._match(self.prev_frame, self.prev_bp, f, fb, win,
                                                                               **expect(*self.prev_xy))),
                                                          self.prev_xy, h, w):
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
        # the steadiness hypothesis is zero motion, within the 2 px tested below
        dx, dy, q = self._match(self.lost_view[0], self.lost_view[1], f, fb, win, predict=(0.0, 0.0), gate=2.0)
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
        kf = Keyframe(x, y, np.clip(f + 0.5, 0, 255).astype(np.uint8), b, fb.astype(np.float16))
        self.keyframes.append(kf)
        if parent is not None:
            self.edges.append((parent, len(self.keyframes) - 1, rel[0], rel[1], weight))
        # splat the STORED 8-bit image: a later correction subtracts exactly the same contribution
        self._splat(kf.image, x, y)

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
            # predicted from the current estimate, within the drift this test tolerates
            dx, dy, q = self._match(k.image, k.bp, cur.image, cur.bp, win, predict=(k.x - cur.x, k.y - cur.y),
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
        for k, x, y in zip(self.keyframes, sx, sy):
            if math.hypot(k.x - x, k.y - y) > 0.5:  # move only what changed: remove old splat, add new
                self._splat(k.image, k.x, k.y, weight=-1.0)
                k.x, k.y = float(x), float(y)
                self._splat(k.image, k.x, k.y)
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
        feather = _feather(h, w) * weight
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
        """Assembled full-resolution mosaic (NaN where empty) and the mosaic coordinate of its top-left pixel."""
        img, origin, _ = render_tiles(self.tiles, self.tile, 1)
        return img, origin

    def preview(self, max_px: int = 1600) -> tuple[np.ndarray, tuple[int, int], int]:
        """Mosaic rendered at a power-of-two reduction d so that its larger side is <= max_px (bounded memory).

        Returns (image, origin in full-resolution mosaic pixels, d)."""
        return render_tiles(self.tiles, self.tile, preview_factor(self.tiles, self.tile, max_px))

    # ---------------------------------------------------------------- 2.5D
    def add_height_patch(self, x: float, y: float, edf: np.ndarray, height: np.ndarray, z_unit: str,
                         z_key: str | None = None, rid: str = "") -> None:
        """Pin a height map at camera position (x, y). Patches are only levelled and blended with patches of the
        same z_key (same quantity, unit and scale); z_key defaults to the canonical unit."""
        from . import units

        unit = units.canon(z_unit)
        kf = self._nearest(x, y) if self.keyframes else None
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
        rows, rhs = [], []
        for i in range(n):
            for j in range(i + 1, n):
                a, b = P[i], P[j]
                ax, ay, bx, by = int(round(a.x)), int(round(a.y)), int(round(b.x)), int(round(b.y))
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
        if not rows:
            return
        r0 = np.zeros(n)
        r0[0] = 1
        sol = np.linalg.lstsq(np.vstack([r0] + rows), np.array([0.0] + rhs), rcond=None)[0]
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
    xs = [int(round(p.x)) for p in P]
    ys = [int(round(p.y)) for p in P]
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


def preview_factor(tiles: dict, tile: int, max_px: int) -> int:
    """Smallest power-of-two reduction (<= tile) that keeps the rendered mosaic within max_px."""
    if not tiles:
        return 1
    txs = [k[0] for k in tiles]
    tys = [k[1] for k in tiles]
    side = max(max(txs) - min(txs) + 1, max(tys) - min(tys) + 1) * tile
    d = 1
    while side / d > max_px and d < tile:
        d *= 2
    return d


def render_tiles(tiles: dict, tile: int, d: int = 1) -> tuple[np.ndarray, tuple[int, int], int]:
    """Assemble tiles reduced by d (weighted: num and den are reduced separately). Works on a shallow copy of the
    tile dict, so the engine can render outside its lock. Returns (image, full-resolution origin, d)."""
    if not tiles:
        return np.full((8, 8), np.nan, np.float32), (0, 0), d
    t = tile
    td = t // d
    txs = [k[0] for k in tiles]
    tys = [k[1] for k in tiles]
    X0, Y0 = min(txs), min(tys)
    out = np.full(((max(tys) - Y0 + 1) * td, (max(txs) - X0 + 1) * td), np.nan, np.float32)
    for (tx, ty), (num, den) in list(tiles.items()):
        if d > 1:
            num = cv2.resize(num, (td, td), interpolation=cv2.INTER_AREA)
            den = cv2.resize(den, (td, td), interpolation=cv2.INTER_AREA)
        sl = (slice((ty - Y0) * td, (ty - Y0 + 1) * td), slice((tx - X0) * td, (tx - X0 + 1) * td))
        out[sl] = np.where(den > 1e-3, num / np.maximum(den, 1e-6), np.nan)
    rows = np.where(np.isfinite(out).any(1))[0]
    cols = np.where(np.isfinite(out).any(0))[0]
    if rows.size == 0:
        return out, (X0 * t, Y0 * t), d
    out = out[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]
    return out, (X0 * t + int(cols[0]) * d, Y0 * t + int(rows[0]) * d), d
