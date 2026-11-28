"""Rolling-shutter readout fraction and deskewing against a synthetic rolling-shutter camera with known rho.

The camera renders the specimen simulator's world on a grid 3x finer than the output, shifts every output row by
the content motion at that row's exposure time (row r exposed at rho * (r - c) / H frame intervals from the middle
row c = (H - 1) / 2), and averages down to the output grid (INTER_AREA), then adds vignetting, gain, dark level and
Gaussian read noise. The rolling shutter is built here from that definition only; nothing in it uses the module's
closed forms. Motion blur within a row's exposure is not simulated.
"""
import math

import cv2
import numpy as np
import pytest

from microtopos import rollingshutter as rs
from microtopos.objectives import resolve
from microtopos.simulator import SpecimenSimulator

W, H, K = 240, 180, 3
C = (H - 1) / 2.0


class RollingShutterCamera:
    def __init__(self, seed: int, rho: float):
        self.sim = SpecimenSimulator(seed, resolve("10x / 0.25"), W, H)
        self.rho = rho
        self.rng = np.random.default_rng(seed + 1000)
        self.s = self.sim.um_per_px / self.sim.world_um_per_px  # world px per output px
        jj, ii = np.meshgrid(np.arange(W * K), np.arange(H * K))
        self.u = (jj + 0.5) / K - 0.5  # output-pixel coordinate of each fine sample
        self.v = (ii + 0.5) / K - 0.5
        self.row = ii // K  # the sensor row the fine sample belongs to

    def frame(self, pos, vel) -> np.ndarray:
        """Frame whose middle row sees the content shifted by `pos` (px), panning at `vel` px/frame."""
        t = self.rho * (self.row - C) / H  # exposure time of the row, frame intervals from the middle row
        dx, dy = pos[0] + vel[0] * t, pos[1] + vel[1] * t  # content shift at that time
        sim = self.sim
        mx = (sim.world_w / 2 + self.s * (self.u - dx - (W - 1) / 2)).astype(np.float32)
        my = (sim.world_h / 2 + self.s * (self.v - dy - (H - 1) / 2)).astype(np.float32)
        fine = cv2.remap(sim.world_albedo_pf, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        img = cv2.resize(fine, (W, H), interpolation=cv2.INTER_AREA)
        img = img * sim.vignette * sim.gain + sim.dark_level + self.rng.normal(0, 1.5, img.shape)
        return np.clip(img + 0.5, 0, 255).astype(np.uint8)


X_PANS = [(4.0, 0.0), (8.0, 0.0), (-11.0, 0.0)]
Y_PANS = [(0.0, 5.0), (0.0, -8.0), (0.0, 10.0)]


def _pairs_against_still(cam, velocities):
    still = cam.frame((0.0, 0.0), (0.0, 0.0))
    return [(still, (0.0, 0.0), cam.frame((1.3 - 0.2 * i, -0.6 + 0.3 * i), v), v) for i, v in enumerate(velocities)]


@pytest.mark.parametrize("rho", [0.6, 0.9])
@pytest.mark.parametrize("pans", ["x", "y", "mixed"])
def test_readout_fraction_recovered_within_5_percent(rho, pans):
    cam = RollingShutterCamera(3, rho)
    if pans == "x":
        pairs = _pairs_against_still(cam, X_PANS)
    elif pans == "y":
        pairs = _pairs_against_still(cam, Y_PANS)
    else:  # neither frame at rest: the fit must use both velocities, not the difference of one from zero
        pairs = [(cam.frame((0.0, 0.0), (5.0, 0.0)), (5.0, 0.0), cam.frame((0.8, 0.4), (-4.0, 6.0)), (-4.0, 6.0)),
                 (cam.frame((0.0, 0.0), (0.0, -6.0)), (0.0, -6.0), cam.frame((-0.5, 0.9), (7.0, 3.0)), (7.0, 3.0)),
                 (cam.frame((0.0, 0.0), (3.0, 3.0)), (3.0, 3.0), cam.frame((0.3, -0.7), (-6.0, -5.0)), (-6.0, -5.0))]
    r = rs.estimate_readout(pairs)
    assert r["observable"], r["reason"]
    assert r["n_pairs"] == 3
    assert abs(r["rho"] - rho) < 0.05 * rho, r
    assert r["rho_se"] < 0.01, r


def test_zero_readout_camera_gives_zero_within_its_uncertainty():
    cam = RollingShutterCamera(5, 0.0)
    r = rs.estimate_readout(_pairs_against_still(cam, X_PANS + Y_PANS))
    assert r["observable"], r["reason"]
    assert abs(r["rho"]) <= 3 * r["rho_se"] and r["rho_se"] < 0.01, r


def test_pairs_at_equal_velocity_are_refused():
    """Same velocity, same skew: the pair's relative transform is a translation and rho does not show."""
    cam = RollingShutterCamera(3, 0.9)
    pairs = [(cam.frame((0.0, 0.0), v), v, cam.frame((1.2, -0.8), v), v) for v in [(8.0, 0.0), (0.0, -7.0), (5.0, 5.0)]]
    r = rs.estimate_readout(pairs)
    assert not r["observable"]
    assert r["rho"] is None and r["rho_se"] is None
    assert r["n_low_dv"] == 3 and r["n_pairs"] == 0
    # a difference below MIN_PAIR_DV_PX is not used either
    v1, v2 = (8.0, 0.0), (8.0 + 0.8 * rs.MIN_PAIR_DV_PX, 0.0)
    r = rs.estimate_readout([(cam.frame((0.0, 0.0), v1), v1, cam.frame((0.5, 0.5), v2), v2)])
    assert not r["observable"] and r["n_low_dv"] == 1


def test_skews_inconsistent_with_the_velocities_are_refused():
    """Two pairs of equal stated velocity difference whose shears have opposite signs determine no rho."""
    cam = RollingShutterCamera(3, 0.9)
    still = cam.frame((0.0, 0.0), (0.0, 0.0))
    pairs = [(still, (0.0, 0.0), cam.frame((0.5, 0.2), (8.0, 0.0)), (8.0, 0.0)),
             (still, (0.0, 0.0), cam.frame((0.5, 0.2), (-8.0, 0.0)), (8.0, 0.0))]  # mislabelled velocity
    r = rs.estimate_readout(pairs)
    assert not r["observable"] and r["n_pairs"] == 2
    assert "standard error" in r["reason"]


def _rms_difference(a, b, margin=12):
    """RMS pixel difference over the frame's interior (the deskew replicates borders in the outer rows)."""
    d = (a.astype(np.float32) - b.astype(np.float32))[margin:-margin, margin:-margin]
    return float(np.sqrt(np.mean(d * d)))


@pytest.mark.parametrize("vel", [(12.0, 0.0), (0.0, 10.0), (-8.0, -8.0)])
def test_deskewed_fast_pan_matches_the_global_shutter_frame(vel):
    """Deskewing with the true rho removes the skew and leaves the frame where the middle row saw it."""
    rho = 0.9
    cam = RollingShutterCamera(3, rho)
    gs = RollingShutterCamera(3, 0.0).frame((0.7, -0.4), (0.0, 0.0))  # global shutter, same position
    skewed = cam.frame((0.7, -0.4), vel)
    fixed = rs.deskew(skewed, vel, rho)

    before = rs.measure_pair(gs, skewed)[0]
    after = rs.measure_pair(gs, fixed)[0]
    skew_before = max(abs(before[0, 1]), abs(before[1, 1] - 1))
    skew_after = max(abs(after[0, 1]), abs(after[1, 1] - 1), abs(after[1, 0]), abs(after[0, 0] - 1))
    assert skew_before > 0.03
    assert skew_after < skew_before / 100, (skew_before, skew_after)
    # about the middle row: the deskewed frame is not displaced relative to the global-shutter frame
    centre = np.array([(W - 1) / 2, C])
    assert np.hypot(*(after[:, :2] @ centre + after[:, 2] - centre)) < 0.1, after
    # and pixel for pixel it is the global-shutter frame up to noise (read noise 1.5 DN per frame, 2.1 DN in a
    # difference of two frames)
    err_fixed, err_skewed = _rms_difference(gs, fixed), _rms_difference(gs, skewed)
    assert err_fixed < 3.0 and err_fixed < err_skewed / 3, (err_fixed, err_skewed)


def test_deskew_with_the_wrong_sign_makes_the_skew_worse():
    rho, vel = 0.9, (10.0, 0.0)
    cam = RollingShutterCamera(3, rho)
    gs = RollingShutterCamera(3, 0.0).frame((0.0, 0.0), (0.0, 0.0))
    skewed = cam.frame((0.0, 0.0), vel)
    shear = rs.measure_pair(gs, skewed)[0][0, 1]
    wrong = rs.measure_pair(gs, rs.deskew(skewed, vel, -rho))[0][0, 1]
    assert abs(shear - rho * vel[0] / H) < 0.05 * rho * vel[0] / H  # the camera's shear, positive for vx > 0
    assert abs(wrong - 2 * shear) < 0.1 * abs(shear), (shear, wrong)


@pytest.mark.parametrize("vel", [(9.0, 0.0), (0.0, 9.0), (-6.0, -7.0)])
def test_deskew_points_inverts_the_camera_geometry(vel):
    """A scene point at global-shutter position (X, Y) is recorded in the row y where y = Y + vy rho (y - c) / H,
    displaced in x by vx rho (y - c) / H. The recorded position is found here by fixed-point iteration."""
    rho = 0.8
    rng = np.random.default_rng(1)
    gs_pts = np.column_stack([rng.uniform(0, W, 50), rng.uniform(0, H, 50)])
    y = gs_pts[:, 1].copy()
    for _ in range(200):
        y = gs_pts[:, 1] + vel[1] * rho * (y - C) / H
    x = gs_pts[:, 0] + vel[0] * rho * (y - C) / H
    got = rs.deskew_points(np.column_stack([x, y]), vel, rho, H)
    assert np.abs(got - gs_pts).max() < 1e-6
    assert np.abs(np.column_stack([x, y]) - gs_pts).max() > 1.0  # the skew itself is not negligible


def test_vertical_speed_at_the_readout_speed_is_rejected():
    with pytest.raises(ValueError):
        rs.skew_matrix((0.0, H / 0.5), 0.5, H)
    assert math.isfinite(rs.skew_matrix((0.0, 0.99 * H / 0.5), 0.5, H)[1, 1])
