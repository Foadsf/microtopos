"""Aligned, outlier-robust stacking of still frames (microtopos.stacking), checked against ground truth.

The truth is a band-limited periodic random texture (Gaussian spectrum, correlation length 3 px). A frame with
content shift d is that texture sampled exactly at x - d (a Fourier phase ramp, exact because the spectrum is
negligible at Nyquist), so every frame and the reference's truth are known without any interpolation.
"""
import cv2
import numpy as np
import pytest

from microtopos.calibration import bgr_to_luma
from microtopos.objectives import resolve
from microtopos.simulator import SpecimenSimulator
from microtopos.stacking import stack_frames
from microtopos.superres import MIN_SR_FRAMES

H, W = 240, 320


def _texture(seed: int, corr_px: float = 3.0, std: float = 20.0, mean: float = 120.0):
    """Return at(dx, dy, blur_px=0): the texture with its content moved by (+dx, +dy) px, float64, H x W."""
    r = np.random.default_rng(seed)
    fy = np.fft.fftfreq(H)[:, None]
    fx = np.fft.rfftfreq(W)[None, :]
    spec = np.fft.rfft2(r.normal(size=(H, W))) * np.exp(-2 * (np.pi * corr_px) ** 2 * (fx ** 2 + fy ** 2))
    spec *= std / np.fft.irfft2(spec, s=(H, W)).std()

    def at(dx: float, dy: float, blur_px: float = 0.0) -> np.ndarray:
        """blur_px: convolved with a Gaussian of this sigma (exact, in the Fourier domain)."""
        g = np.exp(-2 * (np.pi * blur_px) ** 2 * (fx ** 2 + fy ** 2))
        return mean + np.fft.irfft2(spec * g * np.exp(-2j * np.pi * (fx * dx + fy * dy)), s=(H, W))
    return at


def _noisy(at, shifts, sigma, rng):
    return [(at(*d) + rng.normal(0, sigma, (H, W))).astype(np.float32) for d in shifts]


def _rms(a, b, mask):
    return float(np.sqrt(np.mean((np.asarray(a, np.float64) - b)[mask] ** 2)))


def _lanczos_weights(f: float) -> np.ndarray:
    """The 8 Lanczos-4 weights for samples floor(x) - 3 .. floor(x) + 4 at fractional position f, from the kernel
    L(t) = sinc(t) sinc(t / 4), normalised to sum 1, with f snapped to OpenCV's 1/32 grid (written here from the
    formula, not taken from OpenCV)."""
    f = np.round((f - np.floor(f)) * 32.0) / 32.0
    if f in (0.0, 1.0):
        w = np.zeros(8)
        w[3 if f == 0.0 else 4] = 1.0
        return w
    t = np.arange(8) - 3 - f
    w = np.sinc(t) * np.sinc(t / 4.0)
    return w / w.sum()


def _lanczos_noise_gain(f: float) -> float:
    """Variance factor of white noise resampled by Lanczos-4 at fractional position f (1-D)."""
    return float(np.sum(_lanczos_weights(f) ** 2))


def _lanczos_shift(img: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """out(x) = img(x + d) by separable Lanczos-4 interpolation of the periodic image."""
    out = np.zeros_like(img, dtype=np.float64)
    tmp = np.zeros_like(out)
    ix, iy = int(np.floor(dx)), int(np.floor(dy))
    for k, wk in enumerate(_lanczos_weights(dx)):
        tmp += wk * np.roll(img, -(ix + k - 3), axis=1)
    for k, wk in enumerate(_lanczos_weights(dy)):
        out += wk * np.roll(tmp, -(iy + k - 3), axis=0)
    return out


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_noise_falls_as_expected_for_sixteen_shifted_frames(seed):
    """16 frames, sub-pixel shifts within +-2.5 px, Gaussian noise s = 6 DN.

    Expected RMS error of the stack against the truth on the common area: sqrt(noise^2 + bias^2), where
      noise = (s / N) sqrt(sum_k g(fx_k) g(fy_k)), g(f) = sum of the squared Lanczos-4 weights at fraction f, the
        variance that resampling leaves of white noise (the reference, f = 0, keeps all of it), and
      bias = RMS of (mean of the noiseless frames resampled by Lanczos-4 at their true shifts) - truth: the
        smoothing interpolation does to the signal, computed independently of the code.
    Measured over 8 seeds: ratio 1.002..1.012 (noise 1.35..1.39 DN, bias 0.014..0.038 DN). Tolerance +-2.5 %: the RMS over ~65 000 pixels varies by ~0.3 %,
    registration error (~0.015 px) adds ~0.2 %, and a 5 % error in the noise reduction fails. Returning frame 0
    (RMS ~ s) or averaging without registration (2.5..5.7 DN) is far outside.
    """
    rng = np.random.default_rng(100 + seed)
    at = _texture(seed)
    s, n = 6.0, 16
    t = rng.uniform(-2.5, 2.5, (n, 2))
    res = stack_frames(_noisy(at, t, s, rng))
    assert res["frames_used"] == n and res["frames_rejected"] == 0
    r = res["reference"]
    truth = at(*t[r])
    d = t - t[r]
    noise = s / n * np.sqrt(sum(_lanczos_noise_gain(dx) * _lanczos_noise_gain(dy) for dx, dy in d))
    resampled = np.mean([_lanczos_shift(at(*t[k]), *d[k]) for k in range(n)], axis=0)
    va = res["valid"]
    bias = _rms(resampled, truth, va)
    expected = float(np.hypot(noise, bias))
    measured = _rms(res["image"], truth, va)
    assert measured == pytest.approx(expected, rel=0.025)
    # the reported stack noise is the frame-to-frame (random) part only: it must match `noise`, not include bias
    assert res["noise_sigma_stack"] == pytest.approx(noise, rel=0.025)
    # one resampled sample's frame-to-frame noise: s times the RMS bilinear noise gain of the shifts
    assert res["noise_sigma_single"] == pytest.approx(noise * np.sqrt(n), rel=0.03)


def test_known_subpixel_shifts_are_measured_and_aligned():
    """Shifts up to +-3 px: the reported shifts equal the true ones (sign included) and the stack is aligned."""
    rng = np.random.default_rng(7)
    at = _texture(7)
    t = rng.uniform(-3.0, 3.0, (12, 2))
    frames = _noisy(at, t, 2.0, rng)
    res = stack_frames(frames)
    r = res["reference"]
    true_rel = t[res["frame_indices"]] - t[r]
    np.testing.assert_allclose(np.array(res["shifts"]), true_rel, atol=0.03)
    truth = at(*t[r])
    va = res["valid"]
    unregistered = _rms(np.mean(frames, axis=0), truth, va)
    registered = _rms(res["image"], truth, va)
    assert unregistered > 3.0  # the shifts are large enough to matter
    assert registered < 0.15 * unregistered and registered < 0.6


def test_border_pixels_use_only_frames_that_cover_them():
    """A strip that only some frames cover is the merge of those frames: count is right and no border fill leaks."""
    rng = np.random.default_rng(5)
    at = _texture(5)
    # half of the frames shifted +2.4 px in x: near the borders some frames' Lanczos taps fall outside them
    t = np.array([(0.1 * k, 0.05 * k) if k % 2 else (2.4 + 0.05 * k, -0.04 * k) for k in range(8)])
    res = stack_frames(_noisy(at, t, 1.0, rng))
    r = res["reference"]
    truth = at(*t[r])
    d = t - t[r]
    def covered(n, dk):  # every Lanczos tap (floor - 3 .. floor + 4; just the sample on a whole pixel) inside
        q = np.round((np.arange(n) + dk) * 32.0) / 32.0
        base = np.floor(q)
        reach = np.where(q == base, 0, 1)
        return (base - 3 * reach >= 0) & (base + 4 * reach <= n - 1)
    expect = np.array([covered(H, dk[1])[:, None] & covered(W, dk[0])[None, :] for dk in d])
    np.testing.assert_array_equal(res["count"], expect.sum(axis=0))
    # the zero border of the warp would pull edge pixels towards 0 (truth ~120); every pixel stays near the truth
    assert np.abs(res["image"] - truth).max() < 6.0


def test_transient_and_foreign_frame_leave_no_ghost():
    """Frame 0 is another scene (so the first registration target is unusable), frame 6 carries a +100 DN square.

    The foreign frame must be rejected and the square must leave less than 1 DN in any 20 x 20 window; a plain
    mean of the registered frames would leave 100/15 = 6.7 DN there.
    """
    rng = np.random.default_rng(11)
    at = _texture(11)
    t = rng.uniform(-2.0, 2.0, (16, 2))
    frames = _noisy(at, t, 4.0, rng)
    frames[0] = (_texture(999)(0, 0) + rng.normal(0, 4.0, (H, W))).astype(np.float32)
    frames[6][100:120, 150:170] += 100.0
    res = stack_frames(frames)
    assert 0 not in res["frame_indices"] and res["frames_rejected"] == 1
    assert res["reference"] != 0
    truth = at(*t[res["reference"]])
    y0, y1, x0, x1 = res["common_box"]
    resid = (res["image"] - truth)[y0:y1, x0:x1].astype(np.float32)
    window = cv2.blur(resid, (20, 20))[10:-10, 10:-10]
    assert np.abs(window).max() < 1.0
    # the rejected samples are the square (400 pixels of one frame) and the 3-sigma tails, nothing like a frame
    assert 400 / (15 * H * W) < res["pixels_clipped_fraction"] < 0.01


def test_colour_stack_uses_the_luma_warps_and_rejections():
    """Colour frames get the same warps: each channel matches its truth, and luma of the colour stack = luma stack."""
    rng = np.random.default_rng(21)
    chans = [_texture(21 + c, mean=90.0 + 40 * c) for c in range(3)]
    t = rng.uniform(-2.0, 2.0, (8, 2))
    colour = [np.dstack([ch(*d) + rng.normal(0, 3.0, (H, W)) for ch in chans]).astype(np.float32) for d in t]
    wts = np.array([0.114, 0.587, 0.299], np.float32)  # Rec.601 luma of B, G, R
    luma = [(c * wts).sum(axis=2).astype(np.float32) for c in colour]
    res = stack_frames(luma, colour=colour)
    r = res["reference"]
    va = res["valid"]
    col = res["colour"]
    assert col.shape == (H, W, 3) and col.dtype == np.float32
    for c in range(3):
        truth = chans[c](*t[r])
        unaligned = _rms(np.mean([f[..., c] for f in colour], axis=0), truth, va)
        assert _rms(col[..., c], truth, va) < 1.0 and unaligned > 2.5
    np.testing.assert_allclose((col * wts).sum(axis=2), res["image"], atol=1e-3)


def test_too_few_usable_frames_are_refused():
    rng = np.random.default_rng(3)
    at = _texture(3)
    few = _noisy(at, rng.uniform(-1, 1, (MIN_SR_FRAMES - 1, 2)), 2.0, rng)
    with pytest.raises(ValueError, match=f"at least {MIN_SR_FRAMES}"):
        stack_frames(few)
    # 16 frames, only 3 of the same scene: whatever frame is tried as the registration target, 3 < MIN_SR_FRAMES
    frames = [(_texture(500 + k)(0, 0) + rng.normal(0, 2.0, (H, W))).astype(np.float32) for k in range(16)]
    for k in (0, 5, 9):
        frames[k] = (at(*rng.uniform(-1, 1, 2)) + rng.normal(0, 2.0, (H, W))).astype(np.float32)
    with pytest.raises(ValueError, match="registered reliably"):
        stack_frames(frames)
    # reference="first" stands or falls with frame 0: a foreign frame 0 leaves nothing to register to
    good = _noisy(at, rng.uniform(-1, 1, (8, 2)), 2.0, rng)
    good[0] = (_texture(777)(0, 0) + rng.normal(0, 2.0, (H, W))).astype(np.float32)
    with pytest.raises(ValueError, match="registered reliably"):
        stack_frames(good, reference="first")
    assert stack_frames(good)["frames_used"] == 7  # the default reference passes over the foreign frame


def test_deconvolution_sharpens_towards_the_unblurred_truth():
    """Frames blurred by a known Gaussian (sigma 1.2 px): deconvolving the stack with that sigma moves it closer
    to the sharp truth than the stack itself, and without the option nothing is deconvolved."""
    rng = np.random.default_rng(31)
    sharp = _texture(31, corr_px=1.2, std=25.0)
    t = rng.uniform(-2.0, 2.0, (16, 2))
    blur = 1.2
    frames = [(sharp(*d, blur_px=blur) + rng.normal(0, 1.0, (H, W))).astype(np.float32) for d in t]
    plain = stack_frames(frames)
    np.testing.assert_array_equal(plain["image"], plain["stacked"])
    assert plain["rl_iterations"] == 0
    res = stack_frames(frames, deconvolve_psf_sigma=blur)
    truth = sharp(*t[res["reference"]])
    inner = np.zeros((H, W), bool)
    inner[20:-20, 20:-20] = True
    inner &= res["valid"]
    before, after = _rms(res["stacked"], truth, inner), _rms(res["image"], truth, inner)
    assert res["rl_iterations"] >= 1
    assert after < 0.5 * before


def test_jpeg_simulator_frames_register_to_the_true_tremor():
    """Realistic input: 16 uint8 luma frames from the simulator (sensor noise + JPEG, 1.5 px hand tremor). The
    reported shifts match the simulator's true content shifts relative to the reference frame."""
    obj = resolve("4x / 0.10")
    sim = SpecimenSimulator(3, obj, 320, 180)
    sim.tremor_px = 1.5
    _, hz = sim.truth_patch()
    sim.focus_z_um = float(np.median(hz))
    out = [sim.render(breathing=False) for _ in range(16)]
    frames = [bgr_to_luma(bgr).astype(np.uint8) for bgr, _ in out]
    true = np.array([ft.content_shift_px for _, ft in out])
    res = stack_frames(frames)
    assert res["frames_used"] >= 14
    rel = true[res["frame_indices"]] - true[res["reference"]]
    assert np.abs(rel).max() > 2.0  # enough motion that a 5 % scale error would exceed the tolerance below
    np.testing.assert_allclose(np.array(res["shifts"]), rel, atol=0.05)


def test_spatially_correlated_noise_is_reported_at_its_true_level():
    """Camera noise is correlated between neighbouring pixels (demosaicing, JPEG). Still frames with noise blurred by
    a Gaussian of 1.2 px: the reported single-frame noise is the per-pixel standard deviation of that noise (measured
    on the noise fields themselves), and the stack's is that over sqrt(16). A white-noise estimator reads it ~3x low."""
    rng = np.random.default_rng(21)
    at = _texture(21)
    n = 16
    fy = np.fft.fftfreq(H)[:, None]
    fx = np.fft.rfftfreq(W)[None, :]
    g = np.exp(-2 * (np.pi * 1.2) ** 2 * (fx ** 2 + fy ** 2))
    noise = [np.fft.irfft2(np.fft.rfft2(rng.normal(0, 8.0, (H, W))) * g, s=(H, W)) for _ in range(n)]
    sigma = float(np.std(np.array(noise)))  # the true per-pixel noise of one frame
    frames = [(at(0.0, 0.0) + e).astype(np.float32) for e in noise]
    res = stack_frames(frames)
    assert res["noise_sigma_single"] == pytest.approx(sigma, rel=0.03)
    assert res["noise_sigma_stack"] == pytest.approx(sigma / np.sqrt(n), rel=0.03)
    assert _rms(res["image"], at(0.0, 0.0), res["valid"]) == pytest.approx(sigma / np.sqrt(n), rel=0.05)


def test_frame_brightness_flicker_is_divided_out():
    """Lamp flicker / automatic exposure: frames scaled by up to +-4 %. The reported gains equal the true ones relative
    to the reference within 0.2 %, and the stack matches the reference-brightness truth as closely as without flicker
    (a merge that ignores the gains is off by the mean gain error over the image, ~1-2 DN, and is clipped unevenly)."""
    rng = np.random.default_rng(5)
    at = _texture(5)
    n, s = 16, 2.0
    t = rng.uniform(-1.5, 1.5, (n, 2))
    k = rng.uniform(0.96, 1.04, n)
    frames = [((at(*d)) * kk + rng.normal(0, s, (H, W))).astype(np.float32) for d, kk in zip(t, k)]
    res = stack_frames(frames)
    r = res["reference"]
    true_gains = [k[i] / k[r] for i in res["frame_indices"]]
    assert res["gains"] == pytest.approx(true_gains, rel=0.002)
    steady = stack_frames([(at(*d) * k[r] + rng.normal(0, s, (H, W))).astype(np.float32) for d in t])
    truth = at(*t[r]) * k[r]
    e_flicker = _rms(res["image"], truth, res["valid"])
    e_steady = _rms(steady["image"], truth, steady["valid"])
    assert e_flicker < 1.1 * e_steady
