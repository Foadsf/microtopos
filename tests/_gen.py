"""Scene, pair and trajectory generators for the generalisation tests (test_gen_*.py).

Everything is built from the package's own physics simulator (microtopos.simulator), drawn from explicit seeds,
so a test run is deterministic and independent of test order: every render reseeds the simulator's noise
generator from (seed, draw, condition) before drawing. Simulators are cached per (seed, objective, variant)
because building a world costs ~0.4 s; tests reset the stage, focus and noise state before use.

Truth conventions (pinned by the self-test): moving the stage by +s um moves image content by -s/um_per_px px;
register_pair's warp translation and Mosaic._match's (dx, dy) are the content shift d with b(x) = a(x - d).
"""

from __future__ import annotations

import functools
import math
import zlib
from dataclasses import dataclass, replace

import cv2
import numpy as np

from microtopos.calibration import bgr_to_luma
from microtopos.mosaic import Mosaic
from microtopos.objectives import default_set, resolve
from microtopos.registration import register_pair
from microtopos.simulator import SpecimenSimulator

W, H = 320, 240
OBJECTIVES = ("10x / 0.25", "40x / 0.85")


def obj(name: str):
    return resolve(name, objset=default_set())


@functools.lru_cache(maxsize=40)
def sim(seed: int, oname: str, textureless: bool = False, world_margin: float = 2.2) -> SpecimenSimulator:
    s = SpecimenSimulator(seed, obj(oname), W, H, textureless_patch=textureless, world_margin=world_margin)
    # render() and truth_patch() sample only the prefiltered worlds; drop the raw ones to halve the cache's memory
    s.world_albedo = s.world_height = None
    return s


def fresh(seed: int, oname: str, rng_key, textureless: bool = False, world_margin: float = 2.2) -> SpecimenSimulator:
    """Cached simulator with its mutable state reset: stage at the origin, focus at the median surface height,
    noise generator reseeded from rng_key (so the result does not depend on what ran before)."""
    s = sim(seed, oname, textureless, world_margin)
    s.stage_x_um = s.stage_y_um = 0.0
    s.focus_z_um = median_height(s)
    s.rng = np.random.default_rng(_key(rng_key))
    return s


def _key(k) -> list[int]:
    """Stable integer seed sequence for a tuple of ints/strings (str hash() is salted per process)."""
    out = []
    for v in (k if isinstance(k, tuple) else (k,)):
        out.append(zlib.crc32(v.encode()) if isinstance(v, str) else int(v))
    return out


@functools.lru_cache(maxsize=96)
def _median_height(seed: int, oname: str, textureless: bool, world_margin: float) -> float:
    s = sim(seed, oname, textureless, world_margin)
    _, hz = s.truth_patch(0.0, 0.0)
    return float(np.median(hz))


def median_height(s: SpecimenSimulator) -> float:
    return _median_height(s.seed, s.objective.name, s.textureless_patch, s.world_margin)


def luma(s: SpecimenSimulator, **kw) -> np.ndarray:
    kw.setdefault("tremor", False)
    kw.setdefault("breathing", False)
    return bgr_to_luma(s.render(**kw)[0]).astype(np.float32)


# ------------------------------------------------------------------------------------------- registration pairs
@dataclass(frozen=True)
class Cond:
    """One registration condition. Shifts are content shifts of the moving frame in px (truth)."""
    name: str
    shift: float = 3.0  # uniform in [-shift, shift] on each axis
    dz_dof: float = 0.0  # extra defocus of the moving frame, in depths of field
    noise: float = 0.0  # extra Gaussian noise sigma (grey levels) on both frames
    motion: float = 0.0  # linear motion blur length (px) on the moving frame
    deg: float = 0.0  # rotation of the moving frame about the image centre
    scale: float = 1.0  # scale of the moving frame about the image centre
    contrast: float = 1.0  # contrast factor on both frames (with 1.5 grey levels of extra noise)
    textureless: bool = False  # simulator variant with a flat 1/3 x 1/3 FOV patch at the centre
    fixed: tuple | None = None  # fixed content shift magnitude per axis, random sign (partial-overlap pairs)
    far: tuple | None = None  # fixed content shift for a no-overlap pair (truth unknown to the matcher)
    other_specimen: bool = False  # moving frame from a different specimen (seed + 500)
    grating: bool = False  # synthetic 9 x 13 px periodic grating instead of the specimen


def make_pair(seed: int, oname: str, draw: int, c: Cond):
    """(template, moving, truth content shift or None for a negative control)."""
    rng = np.random.default_rng(_key((seed, oname, draw, c.name)))
    tdx, tdy = rng.uniform(-c.shift, c.shift, 2)
    if c.grating:
        return _grating(rng), _grating(rng, tdx, tdy), (tdx, tdy)
    margin = 3.5 if c.far else 2.2  # a world large enough that a far frame is real specimen, not a reflection
    s = fresh(seed, oname, (seed, oname, draw, c.name, 1), c.textureless, margin)
    z0 = s.focus_z_um
    a = luma(s, noise=True)
    if c.fixed is not None:
        tdx, tdy = c.fixed[0] * rng.choice([-1, 1]), c.fixed[1] * rng.choice([-1, 1])
    if c.far is not None:
        tdx, tdy = c.far
    s.stage_x_um, s.stage_y_um = -tdx * s.um_per_px, -tdy * s.um_per_px
    s.focus_z_um = z0 + c.dz_dof * s.objective.dof_um
    if c.other_specimen:
        s2 = fresh(seed + 500, oname, (seed, oname, draw, c.name, 2))
        b = luma(s2, noise=True)
    else:
        b = luma(s, noise=True, motion_px=(c.motion, 0.0))
    s.stage_x_um = s.stage_y_um = 0.0
    s.focus_z_um = z0
    if c.deg or c.scale != 1.0:
        M = cv2.getRotationMatrix2D(((W - 1) / 2, (H - 1) / 2), c.deg, c.scale)
        b = cv2.warpAffine(b, M, (W, H), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
    if c.contrast != 1.0:
        a = a.mean() + c.contrast * (a - a.mean())
        b = b.mean() + c.contrast * (b - b.mean())
    sig = c.noise or (1.5 if c.contrast != 1.0 else 0.0)
    if sig:
        a = np.clip(a + rng.normal(0, sig, a.shape), 0, 255).astype(np.float32)
        b = np.clip(b + rng.normal(0, sig, b.shape), 0, 255).astype(np.float32)
    truth = None if (c.far is not None or c.other_specimen) else (float(tdx), float(tdy))
    return a, b, truth


def _grating(rng, dx=0.0, dy=0.0):
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    g = 128 + 50 * np.sin(2 * np.pi * (xx - dx) / 9.0) + 40 * np.sin(2 * np.pi * (yy - dy) / 13.0)
    return (g + rng.normal(0, 2, g.shape)).astype(np.float32)


@dataclass
class PairResult:
    rp_shift: tuple  # register_pair (factor 1, translation) warp translation
    rp_ok: bool
    m_shift: tuple  # Mosaic._match content shift
    m_q: float  # Mosaic._match NCC
    truth: tuple | None

    @property
    def rp_err(self) -> float:
        return math.hypot(self.rp_shift[0] - self.truth[0], self.rp_shift[1] - self.truth[1])

    @property
    def m_err(self) -> float:
        return math.hypot(self.m_shift[0] - self.truth[0], self.m_shift[1] - self.truth[1])


def register_both(a, b, truth) -> PairResult:
    r = register_pair(a, b, factor=1, model="translation")
    m = Mosaic()
    win = cv2.createHanningWindow((W, H), cv2.CV_32F)
    dx, dy, q = m._match(a, m._bp(a), b, m._bp(b), win)
    return PairResult((float(r.warp[0, 2]), float(r.warp[1, 2])), bool(r.ok), (float(dx), float(dy)), float(q), truth)


def run_matrix(c: Cond, seeds, objectives=OBJECTIVES, draws: int = 4) -> list[PairResult]:
    """Results in the order objective x seed x draw (cached: several tests judge the same matrix)."""
    return list(_run_matrix(c, tuple(seeds), tuple(objectives), draws))


@functools.lru_cache(maxsize=64)
def _run_matrix(c, seeds, objectives, draws):
    return tuple(register_both(*make_pair(seed, o, d, c)) for o in objectives for seed in seeds for d in range(draws))


# ------------------------------------------------------------------------------------------- mosaic trajectories
def closed_loop(seed: int, oname: str, n_frames: int, blur_frac: float = 0.3):
    """Elliptical closed loop (0.8 x 0.5 FOV), as the self-test's mosaic check, at n_frames per lap, with motion
    blur of blur_frac x the per-frame travel, tremor and breathing on. Returns (per-frame position errors px,
    mosaic)."""
    s = fresh(seed, oname, (seed, oname, n_frames, "loop"))
    fov = s.out_w * s.um_per_px
    m = Mosaic()
    errs = []
    prev = (0.0, 0.0)
    for a in np.linspace(0, 2 * np.pi, n_frames):
        s.stage_x_um, s.stage_y_um = 0.4 * fov * math.sin(a), 0.25 * fov * (1 - math.cos(a))
        px, py = s.stage_x_um / s.um_per_px, s.stage_y_um / s.um_per_px
        mv = (-(px - prev[0]) * blur_frac, -(py - prev[1]) * blur_frac)
        prev = (px, py)
        m.track(bgr_to_luma(s.render(motion_px=mv)[0]))
        errs.append(math.hypot(m.x - px, m.y - py))
    s.stage_x_um = s.stage_y_um = 0.0
    return np.array(errs), m


def out_and_back(seed: int, step: float, bias: float, oname: str = "10x / 0.25", reach: float = 360.0):
    """Pan +reach px along x and back, `step` px per frame, with `bias` px added to every Mosaic._match x estimate
    (a systematic registration error). Returns (keyframe position errors at the end, number of loop edges,
    final camera position error, mosaic)."""
    s = fresh(seed, oname, (seed, oname, step, bias, "oab"), world_margin=3.0)
    m = Mosaic()
    orig = m._match

    def biased(*a, **k):
        dx, dy, q = orig(*a, **k)
        return dx + bias, dy, q

    m._match = biased
    n = round(reach / step)
    xs = [0.0] * 10 + [step * i for i in range(1, n + 1)] + [step * n] * 10 + \
         [step * (n - i) for i in range(1, n + 1)] + [0.0] * 15
    truth = []
    for xp in xs:
        s.stage_x_um = xp * s.um_per_px
        added, _ = m.track(luma(s, noise=True, motion_px=(0.3 * step, 0.0)))
        if added:
            truth.append(xp)
    s.stage_x_um = 0.0
    kx = np.array([k.x for k in m.keyframes])
    ky = np.array([k.y for k in m.keyframes])
    # a camera at stage +x sees content at -x: the camera position in mosaic px is +x
    err = np.hypot(kx - np.array(truth), ky)
    loops = sum(1 for e in m.edges if abs(e[1] - e[0]) > 1)
    return err, loops, abs(m.x - xs[-1]), m


@functools.lru_cache(maxsize=256)
def revisit(seed: int, offset: float, dz_dof: float, oname: str = "40x / 0.85", noise: bool = False):
    """Map the origin sharply, pan 64 px away (a keyframe there), come back to `offset` px from the origin and
    dwell with the focus off by dz_dof depths of field. Returns (sharpness ratio of the mosaic over the original
    view vs. the original sharp frame, number of keyframes placed within 20 px of the origin).

    Frames are noise-free by default: with sensor noise the Laplacian-variance ratio mostly measures the noise
    averaging of blended keyframes (measured on main: 0.70-0.72 at 0 DOF with noise, 0.97 without), not blur."""
    s = fresh(seed, oname, (seed, oname, offset, dz_dof, noise, "revisit"))
    z0 = s.focus_z_um

    def grab(x, z):
        s.stage_x_um, s.focus_z_um = x * s.um_per_px, z
        return luma(s, noise=noise)

    f0 = grab(0.0, z0)
    m = Mosaic()
    for _ in range(8):
        m.track(f0)
    for x in np.arange(8.0, 64.0 + 1e-9, 8.0):
        m.track(grab(x, z0))
    for _ in range(5):
        m.track(grab(64.0, z0))
    for x in np.arange(56.0, offset, -8.0):
        m.track(grab(x, z0))
    for _ in range(6):
        m.track(grab(offset, z0 + dz_dof * s.objective.dof_um))
    s.stage_x_um, s.focus_z_um = 0.0, z0
    img, (X0, Y0) = m.image()
    ys = slice(int(0.2 * H) - Y0, int(0.8 * H) - Y0)
    xs = slice(int(0.1 * W) - X0, int(0.5 * W) - X0)
    crop = img[ys, xs]
    ref = f0[int(0.2 * H):int(0.8 * H), int(0.1 * W):int(0.5 * W)]
    lv = lambda im: float(cv2.Laplacian(im.astype(np.float32), cv2.CV_32F).var())
    near = sum(1 for k in m.keyframes if math.hypot(k.x, k.y) < 20)
    return lv(np.nan_to_num(crop, nan=float(np.nanmean(crop)))) / lv(ref), near


def stats(v) -> dict:
    v = np.asarray(v, float)
    return {"n": int(v.size), "med": round(float(np.median(v)), 4), "p95": round(float(np.percentile(v, 95)), 4),
            "max": round(float(v.max()), 4)}


__all__ = [
    "OBJECTIVES",
    "Cond",
    "H",
    "PairResult",
    "W",
    "closed_loop",
    "fresh",
    "luma",
    "make_pair",
    "obj",
    "out_and_back",
    "register_both",
    "replace",
    "revisit",
    "run_matrix",
    "sim",
    "stats",
]
