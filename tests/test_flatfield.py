"""Estimated flat field from a panned specimen (microtopos.flatfield), checked against a known shading field.

Ground truth: the simulator's vignette is replaced by V(x, y), a cos^4 falloff about an off-centre axis times a
linear illumination tilt (32 % corner drop). Frames of the simulated specimen are rendered while the stage pans,
and the motion fed to the estimator is measured by the app's own registration (register_pair) unless a test says
otherwise. Everything runs on the simulator; nothing here exercises a real camera.
"""

from __future__ import annotations

import functools

import cv2
import numpy as np
import pytest

from microtopos.calibration import RadiometricCorrector, bgr_to_luma
from microtopos.flatfield import MIN_PAIRS, MIN_TRAVEL_FIELDS, FlatEstimator
from microtopos.objectives import default_set, resolve
from microtopos.registration import register_pair
from microtopos.simulator import SpecimenSimulator

W, H = 320, 240
SEEDS = (1, 2, 3)


def vignette(w: int = W, h: int = H) -> np.ndarray:
    """cos^4 law about an axis at (0.56 w, 0.46 h) with focal length 1.3 w, times a 12 % / 8 % illumination tilt."""
    y, x = np.mgrid[0:h, 0:w].astype(np.float64)
    r2 = ((x - 0.56 * w) ** 2 + (y - 0.46 * h) ** 2) / (1.3 * w) ** 2
    v = (1.0 + r2) ** -2 * (1.0 + 0.12 * (x / w - 0.5) - 0.08 * (y / h - 0.5))
    return (v / v.max()).astype(np.float32)


V = vignette()
G = V / V.mean()


def corner_centre(a: np.ndarray) -> float:
    c = np.mean([a[:8, :8].mean(), a[:8, -8:].mean(), a[-8:, :8].mean(), a[-8:, -8:].mean()])
    return float(c / a[H // 2 - 8:H // 2 + 8, W // 2 - 8:W // 2 + 8].mean())


def gain_rms_error(gain: np.ndarray, truth: np.ndarray = G) -> float:
    """RMS relative deviation of the estimated gain from the true one (default V / mean(V)), up to a constant."""
    r = gain / truth
    return float(np.sqrt(np.mean((r / r.mean() - 1.0) ** 2)))


@functools.lru_cache(maxsize=8)
def simulator(seed: int) -> SpecimenSimulator:
    s = SpecimenSimulator(seed, resolve("10x / 0.25", objset=default_set()), W, H)
    s.vignette = V
    s.gain = 0.85  # keeps the brightest grains below saturation, so only the test's own glints saturate
    s.tremor0 = s.tremor_px
    return s


def fresh(seed: int, tremor_px: float | None = None) -> SpecimenSimulator:
    s = simulator(seed)
    s.vignette = V
    s.tremor_px = s.tremor0 if tremor_px is None else tremor_px
    s.stage_x_um = s.stage_y_um = 0.0
    s.rng = np.random.default_rng(1000 + seed)
    return s


def lissajous(n: int, amp: float) -> np.ndarray:
    """View positions in field units (x in widths, y in heights): a Lissajous figure, i.e. motion in 2-D."""
    t = np.linspace(0.0, 1.0, n)
    return np.stack([amp * np.sin(2 * np.pi * t + 0.3), amp * np.sin(3 * np.pi * t + 1.1)], 1)


def pan(seed: int, path: np.ndarray, glints=None, registered: bool = True, tremor_px: float | None = None):
    """Render the path; yield (luma, shift) with shift in register_pair's convention (None for the first frame).

    registered=False passes the true content shift (stage motion plus hand tremor) instead of a measured one."""
    s = fresh(seed, tremor_px)
    prev = None
    for fx, fy in path:
        s.stage_x_um, s.stage_y_um = fx * W * s.um_per_px, fy * H * s.um_per_px
        bgr, truth = s.render(blur=False, breathing=False)
        luma = bgr_to_luma(bgr)
        if glints is not None:
            luma = glints(luma, fx, fy)
        pos = np.array([-fx * W + truth.content_shift_px[0], -fy * H + truth.content_shift_px[1]])
        if prev is None:
            shift = None
        elif registered:
            reg = register_pair(prev[1], luma)
            shift = (float(reg.warp[0, 2]), float(reg.warp[1, 2]))
        else:
            shift = tuple(pos - prev[0])
        prev = (pos, luma)
        yield luma, shift


def estimates(seed: int, path: np.ndarray, *options: dict, glints=None, registered: bool = True,
              tremor_px: float | None = None) -> list[FlatEstimator]:
    """One estimator per options dict (FlatEstimator keyword arguments), all fed the same rendered frames."""
    dark = np.full((H, W), fresh(seed).dark_level, np.float32)
    ests = [FlatEstimator((H, W), 4, dark=dark, **o) for o in (options or ({},))]
    for luma, shift in pan(seed, path, glints, registered, tremor_px):
        for est in ests:
            est.add(luma, shift)
    return ests


def estimate(seed: int, path: np.ndarray, **kw) -> FlatEstimator:
    return estimates(seed, path, **kw)[0]


@functools.lru_cache(maxsize=8)
def panned_result(seed: int) -> dict:
    """4.4 fields of 2-D travel, about twice the readiness threshold."""
    return estimate(seed, lissajous(100, 0.6)).result()


# ---------------------------------------------------------------------------------------------- recovery
@pytest.mark.parametrize("seed", SEEDS)
def test_recovers_known_vignetting(seed):
    r = panned_result(seed)
    assert r["travel"] >= 2 * MIN_TRAVEL_FIELDS
    # measured 0.12 - 0.39 % RMS over seeds 1..6; a 5 % error in the shading depth fails the corner check below
    assert gain_rms_error(r["gain"]) < 0.006
    assert corner_centre(G) == pytest.approx(0.679, abs=0.002)  # the ground truth is the stated 32 % drop
    assert abs(corner_centre(r["gain"]) - corner_centre(G)) < 0.006  # measured <= 0.003
    assert r["gain"].mean() == pytest.approx(1.0, abs=1e-4)


@pytest.mark.parametrize("seed", SEEDS)
def test_error_at_the_readiness_threshold_is_below_one_percent(seed):
    est = estimate(seed, lissajous(60, 0.3))
    p = est.progress()
    assert p["ready"]
    assert p["pairs"] < MIN_PAIRS + 3 and p["travel"] < MIN_TRAVEL_FIELDS + 0.3  # really at the threshold
    r = est.result()
    assert gain_rms_error(r["gain"]) < 0.01  # measured 0.10 - 0.83 % over seeds 1..6
    assert abs(corner_centre(r["gain"]) - corner_centre(G)) < 0.006


def test_flat_has_the_form_the_corrector_expects():
    s = fresh(SEEDS[0])
    r = panned_result(SEEDS[0])
    assert r["flat"].shape == (H, W) and r["flat"].dtype == np.float32
    sig = r["flat"] - s.dark_level
    np.testing.assert_allclose(sig / sig.mean(), r["gain"], rtol=1e-5)
    # default level: the mean of raw - dark over the accepted frames, which is about gain * mean albedo * mean V
    assert 40.0 < sig.mean() < 200.0
    lvl = estimate(SEEDS[0], lissajous(60, 0.3)).result(level=100.0)["flat"] - s.dark_level
    assert lvl.mean() == pytest.approx(100.0, rel=1e-5)


@pytest.mark.parametrize("seed", SEEDS)
def test_corrector_with_the_estimate_removes_the_shading(seed):
    s = fresh(seed)
    r = panned_result(seed)
    s.stage_x_um, s.stage_y_um = -0.9 * W * s.um_per_px, 0.8 * H * s.um_per_px  # a view not on the path
    shaded = bgr_to_luma(s.render(tremor=False, noise=False, breathing=False, blur=False)[0])
    s.vignette = np.ones_like(V)
    unshaded = bgr_to_luma(s.render(tremor=False, noise=False, breathing=False, blur=False)[0])
    s.vignette = V
    dark = np.full((H, W), s.dark_level, np.float32)
    corrected = RadiometricCorrector(dark, r["flat"])(shaded)

    def shading_rms(img):
        """RMS of the large-scale (16 x 16 px block) log ratio to the unshaded view: the specimen cancels."""
        ratio = img / (unshaded - s.dark_level)
        blocks = ratio.reshape(H // 16, 16, W // 16, 16).mean(axis=(1, 3))
        return float(np.std(np.log(blocks)))

    before, after = shading_rms(shaded - s.dark_level), shading_rms(corrected)
    assert before > 0.08  # the shading that was there
    assert after < 0.05 * before  # measured 1.5 - 4 % of the original
    assert after < 0.004


# ---------------------------------------------------------------------------------------------- refusal
def test_still_view_is_refused_and_the_motion_gate_is_what_refuses_it():
    """A view that never moves, held by a trembling hand (0.8 px RMS per frame) for 500 frames. With the gate it
    is one frame and no pair, refused. Without it every frame is a pair, the jitter adds up to more than the
    travel threshold in perfectly mixed directions, and the estimator declares the still view ready; its answer
    is then several times worse than a real pan's (measured 1.1 % against 0.1 - 0.4 %)."""
    seed = SEEDS[0]
    gated, ungated = estimates(seed, np.zeros((500, 2)), {}, {"min_step_fraction": 0.0}, registered=False,
                               tremor_px=0.8)
    p = gated.progress()
    assert p["frames"] == 1 and p["pairs"] == 0 and p["travel"] == 0.0
    assert not p["ready"]
    with pytest.raises(ValueError, match="not ready"):
        gated.result()
    assert ungated.progress()["ready"]
    assert gain_rms_error(ungated.result()["gain"]) > 2.5 * gain_rms_error(panned_result(seed)["gain"])


def test_pan_along_one_line_is_refused():
    """A pan in X leaves the vertical profile undetermined, however long the travel: refused, not guessed."""
    t = np.linspace(0.0, 1.0, 120)
    path = np.stack([0.8 * np.sin(4 * np.pi * t), np.zeros_like(t)], 1)
    est = estimate(SEEDS[0], path)
    p = est.progress()
    assert p["travel"] > 2 * MIN_TRAVEL_FIELDS and p["pairs"] > 2 * MIN_PAIRS  # plenty of motion ...
    assert p["isotropy"] < 0.1  # ... all of it along one line
    assert not p["ready"] and any("one line" in m for m in p["missing"])
    with pytest.raises(ValueError, match="isotropy"):
        est.result()


# ---------------------------------------------------------------------------------------------- outliers
class Glints:
    """Saturated specular glints fixed to the specimen (they move with it); each shows in a random half of the
    frames that contain it, as glints do when the illumination angle seen by a facet changes. Profile: a Gaussian
    of sigma 3 px peaking at three times full scale, so a saturated core of 4.5 px radius with a bright rim."""

    R = 10

    def __init__(self, seed: int, n: int):
        r = np.random.default_rng(seed)
        self.xy = r.uniform([-1.6 * W, -1.6 * H], [1.6 * W, 1.6 * H], size=(n, 2))  # specimen coordinates, px
        self.r = r

    def __call__(self, luma, fx, fy):
        out = np.pad(luma, self.R)
        for gx, gy in self.xy:
            u, v = gx - fx * W + W / 2, gy - fy * H + H / 2  # content moves by -stage
            if 0 <= u < W and 0 <= v < H and self.r.random() < 0.5:  # centre in view
                iu, iv = int(u), int(v)
                yy, xx = np.mgrid[iv - self.R:iv + self.R + 1, iu - self.R:iu + self.R + 1]
                spot = 765.0 * np.exp(-((xx - u) ** 2 + (yy - v) ** 2) / (2 * 3.0**2))
                win = out[iv:iv + 2 * self.R + 1, iu:iu + 2 * self.R + 1]
                np.maximum(win, spot, out=win)
        return np.minimum(out[self.R:-self.R, self.R:-self.R], 255.0).astype(np.float32)


def test_saturated_glints_do_not_bias_the_estimate():
    """About 20 saturated glints per frame (1.5 % of the pixels). Clipped at full scale, a glint seen in two frames
    gives a ratio of 1 wherever it sits, which pulls the estimate towards no shading; a glint present in only one
    frame of a pair gives a large positive ratio. The per-pixel exclusion removes both."""
    seed = SEEDS[0]
    with_glints, no_exclusion = estimates(seed, lissajous(100, 0.6), {}, {"saturation": 1e9}, glints=Glints(7, 400))
    g, g_off = with_glints.result()["gain"], no_exclusion.result()["gain"]
    assert gain_rms_error(g) < 0.006  # measured 0.21 - 0.23 % over seeds 1..3, as without glints
    assert abs(corner_centre(g) - corner_centre(G)) < 0.006
    # the same frames with the exclusion switched off (nothing counts as saturated): shading underestimated
    assert corner_centre(g_off) - corner_centre(G) > 0.008  # measured +0.010 .. +0.012
    assert gain_rms_error(g_off) > 2 * gain_rms_error(g)  # measured 0.63 - 0.68 %


# ---------------------------------------------------------------------------------------------- closed forms
class Synthetic:
    """Crops of a high-contrast log-normal texture (sigma 0.3 in log) at offsets that are multiples of the 4 px
    analysis cell, so a pair needs no interpolation, times V = exp(P) with P a known degree-4 polynomial in the
    estimator's normalised coordinates. Options: per-pixel log-normal noise, per-frame brightness factors, black
    holes in the specimen (raw = dark, 11 % of the area) and a caller-given path."""

    COEF = (((1, 0), 0.06), ((0, 1), -0.04), ((2, 0), -0.12), ((0, 2), -0.10), ((1, 1), 0.02), ((2, 2), -0.03),
            ((4, 0), -0.02))  # ((i, j), c): c * x^i * y^j

    def __init__(self, seed: int = 5):
        r = np.random.default_rng(seed)
        z = cv2.GaussianBlur(r.normal(0.0, 1.0, (1400, 1600)).astype(np.float32), (0, 0), 1.5)
        self.world = (90.0 * np.exp(0.3 * z / z.std())).astype(np.float32)
        holes = cv2.GaussianBlur(r.normal(0.0, 1.0, z.shape).astype(np.float32), (0, 0), 6.0)
        self.holed = np.where(holes > 1.2 * holes.std(), 0.0, self.world).astype(np.float32)
        x = np.linspace(-1.0, 1.0, W)[None, :]
        y = np.linspace(-1.0, 1.0, H)[:, None]
        self.logv = sum(c * x**i * y**j for (i, j), c in self.COEF)
        self.gain = np.exp(self.logv) / np.exp(self.logv).mean()
        self.r = r

    def run(self, n: int = 120, noise: float = 0.0, brightness: float = 0.0, holes: bool = False,
            offsets: np.ndarray | None = None):
        """Returns the estimator and the log brightness of each accepted frame. offsets: (n, 2) crop corners."""
        if offsets is None:
            t = np.linspace(0.0, 1.0, n)
            offsets = np.stack([600 + 500 * np.sin(2 * np.pi * t), 500 + 400 * np.sin(3 * np.pi * t + 0.7)], 1)
        offsets = 4 * np.round(np.asarray(offsets) / 4).astype(int)
        world = self.holed if holes else self.world
        est = FlatEstimator((H, W), 4)
        logb, prev = [], None
        for x0, y0 in offsets:
            b = self.r.normal(0.0, brightness) if brightness else 0.0
            img = world[y0:y0 + H, x0:x0 + W] * np.exp(self.logv + b)
            if noise:
                img = img * np.exp(self.r.normal(0.0, noise, img.shape))
            pos = np.array([-x0, -y0], float)  # content moves opposite to the crop window
            if est.add(img.astype(np.float32), None if prev is None else tuple(pos - prev)):
                logb.append(b)
            prev = pos
        return est, np.array(logb)


def test_specimen_cancels_exactly_without_interpolation():
    """No noise, no brightness change, cell-aligned shifts: the result is the polynomial itself, although the
    specimen has 30 % contrast (a per-pixel average of so few frames would carry several % of texture)."""
    est, _ = Synthetic().run()
    r = est.result()
    g = Synthetic().gain
    assert float(np.max(np.abs(r["gain"] / g - 1.0))) < 1e-4  # measured 7e-6
    assert r["residual_rms"] < 1e-5 and r["brightness_rms"] < 1e-6


def test_residual_rms_is_the_pair_noise_per_cell():
    """Per-pixel log noise sigma, averaged over a 4 x 4 cell (sigma / 4) and differenced between two frames
    (times sqrt 2): residual_rms must be sigma * sqrt(2) / 4."""
    sigma = 0.04
    est, _ = Synthetic().run(noise=sigma)
    r = est.result()
    assert r["residual_rms"] == pytest.approx(sigma * np.sqrt(2) / 4, rel=0.005)  # measured within 0.01 %
    assert gain_rms_error(r["gain"], Synthetic().gain) < 0.001  # measured 0.011 %


def test_brightness_rms_measures_the_frame_to_frame_brightness_change():
    est, logb = Synthetic().run(brightness=0.02)
    expected = float(np.sqrt(np.mean(np.diff(logb) ** 2)))  # the injected change between accepted frames
    r = est.result()
    assert expected > 0.02
    assert r["brightness_rms"] == pytest.approx(expected, rel=0.005)  # measured within 0.03 %


def test_black_specimen_regions_are_excluded():
    """Holes where raw = dark carry no shading; counted, they would pull the estimate towards a flat field."""
    est, _ = Synthetic().run(holes=True)
    r = est.result()
    assert float(np.max(np.abs(r["gain"] / Synthetic().gain - 1.0))) < 1e-4


def test_pairs_with_too_little_overlap_do_not_count():
    """A view that jumps by 85 % of the field between frames shares 15 % of it with the previous one: such pairs
    neither enter the fit nor count towards readiness, however many there are."""
    square = np.array([[0.0, 0.0], [0.85 * W, 0.0], [0.85 * W, 0.85 * H], [0.0, 0.85 * H]]) + [300.0, 300.0]
    est, _ = Synthetic().run(offsets=np.tile(square, (10, 1)))
    p = est.progress()
    assert p["frames"] == 40 and p["pairs"] == 0 and p["travel"] == 0.0 and not p["ready"]
