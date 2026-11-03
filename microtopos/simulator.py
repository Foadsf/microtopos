"""Physics-based synthetic specimen and camera, used for demos and for ground-truth self-tests.

Every instrument unknown (dark level, gain, noise, JPEG quality, tremor, focus breathing, knob speed,
sweep range) is drawn from the seed, so no algorithm constant can be tuned to one fixed simulator.
The specimen relief is scaled to the objective's depth of field so every objective can be demonstrated;
this is a property of the demo, not of real samples.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np
from scipy.spatial import cKDTree

from .optics import Objective


@dataclass
class FrameTruth:
    """What the simulator did to produce a frame (content shift is in output pixels)."""
    stage_x_um: float
    stage_y_um: float
    focus_z_um: float
    content_shift_px: tuple[float, float]
    breathing_scale: float


class SpecimenSimulator:
    def __init__(self, seed: int, objective: Objective, out_w: int = 640, out_h: int = 360,
                 textureless_patch: bool = False, world_margin: float = 2.2):
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)
        self.objective = objective
        self.out_w, self.out_h = out_w, out_h
        self.um_per_px = objective.nominal_um_per_px(out_w)
        r = self.rng
        self.dark_level = float(r.uniform(3.0, 16.0))
        self.gain = float(r.uniform(0.75, 1.2))
        self.read_noise = float(r.uniform(0.8, 2.5))
        self.shot_k = float(r.uniform(0.05, 0.2))
        self.jpeg_quality = int(r.integers(72, 92))
        self.tremor_px = float(r.uniform(0.1, 0.45))
        self.breathing_per_um = float(r.uniform(2e-4, 1.5e-3)) / max(objective.dof_um, 1e-3)
        self.flicker = float(r.uniform(0.002, 0.01))
        self.textureless_patch = textureless_patch
        self.world_margin = world_margin
        self.stage_x_um = 0.0
        self.stage_y_um = 0.0
        self.focus_z_um = 0.0
        self._build_world()

    # ------------------------------------------------------------------ world
    def _build_world(self) -> None:
        r = self.rng
        self.world_um_per_px = self.um_per_px / 2.0
        ww = int(self.out_w * 2 * self.world_margin)
        wh = int(self.out_h * 2 * self.world_margin)
        self.world_w, self.world_h = ww, wh
        px = self.world_um_per_px
        fov_um = self.out_w * self.um_per_px

        # Voronoi grains (computed on a 1/4 grid, nearest-neighbour upsampled)
        n_grains = int(r.integers(60, 140))
        seeds = r.uniform(0, 1, size=(n_grains, 2)) * [wh, ww]
        gy, gx = np.mgrid[0:wh:4, 0:ww:4]
        _, gid = cKDTree(seeds).query(np.column_stack([gy.ravel(), gx.ravel()]))
        gid = gid.reshape(gy.shape).astype(np.float32)
        gid = cv2.resize(gid, (ww, wh), interpolation=cv2.INTER_NEAREST).astype(np.int32)
        edges = (np.abs(np.diff(gid, axis=0, prepend=gid[:1])) + np.abs(np.diff(gid, axis=1, prepend=gid[:, :1]))) > 0
        boundary = cv2.GaussianBlur(edges.astype(np.float32), (0, 0), 1.2)

        yy, xx = np.mgrid[0:wh, 0:ww].astype(np.float32)
        ux, uy = xx * px, yy * px
        relief = float(np.clip(4.0 * self.objective.dof_um, 1.5, 120.0))
        g_h = r.normal(0.0, 0.25, n_grains).astype(np.float32)
        h = cv2.GaussianBlur(g_h[gid], (0, 0), 2.0)
        k = 2 * math.pi / (fov_um * r.uniform(0.6, 1.2))
        h += 0.35 * np.sin(k * ux + r.uniform(0, 6)) * np.cos(0.8 * k * uy + r.uniform(0, 6))
        ang = r.uniform(0, math.pi)
        rot = ux * math.cos(ang) + uy * math.sin(ang)
        c0 = float(np.mean(rot))
        scar = np.exp(-((rot - c0) ** 2) / (2 * (0.12 * fov_um) ** 2))
        h += -0.45 * scar + 0.08 * scar * np.sin(rot * 2 * math.pi / (0.02 * fov_um))
        for _ in range(int(r.integers(8, 20))):  # pits
            cy, cx = r.uniform(0, wh), r.uniform(0, ww)
            rad = r.uniform(0.01, 0.03) * fov_um / px
            h -= 0.3 * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * rad**2))
        h -= 0.05 * boundary
        h = (h - h.mean()) / (h.std() + 1e-6) * (relief / 4.0)
        self.world_height = h.astype(np.float32)

        # reflectance: grain shade + boundaries + band-limited micro texture (2-6 output px wavelength)
        g_a = r.uniform(0.45, 0.85, n_grains).astype(np.float32)
        alb = g_a[gid] * (1 - 0.45 * boundary) - 0.12 * scar
        tex = np.zeros_like(alb)
        for s, a in ((1.0, 0.10), (2.0, 0.08), (4.0, 0.06)):
            n = cv2.GaussianBlur(r.normal(0, 1, (wh, ww)).astype(np.float32), (0, 0), s)
            tex += a * n / (n.std() + 1e-6)
        alb = alb + tex
        if self.textureless_patch:
            cy, cx = wh // 2, ww // 2
            hh, hw = self.out_h // 3, self.out_w // 3  # in world px this is a 1/3 x 1/3 FOV region
            alb[cy - hh:cy + hh, cx - hw:cx + hw] = 0.7
        slope = cv2.Sobel(self.world_height, cv2.CV_32F, 1, 1, ksize=3) / (8 * px)
        alb = alb * np.clip(1.0 + 0.1 * slope, 0.6, 1.4)
        self.world_albedo = np.clip(alb * 200.0, 10.0, 240.0).astype(np.float32)
        # anti-alias prefilter for the 2:1 world->camera resampling
        self.world_albedo_pf = cv2.GaussianBlur(self.world_albedo, (0, 0), 0.5 * 2.0)
        self.world_height_pf = cv2.GaussianBlur(self.world_height, (0, 0), 1.0)
        vy, vx = np.mgrid[-1:1:complex(0, self.out_h), -1:1:complex(0, self.out_w)].astype(np.float32)
        self.vignette = np.clip(1.0 - r.uniform(0.15, 0.3) * (vx**2 * 0.8 + vy**2), 0.5, 1.0)

    # ------------------------------------------------------------------ geometry
    def _matrix(self, scale: float, shift_px: tuple[float, float], sx_um: float, sy_um: float) -> np.ndarray:
        """2x3 map from output pixel (u, v) to world pixel, for WARP_INVERSE_MAP.

        Content shift (dx, dy) means the output image content moves by (+dx, +dy) px.
        """
        s = (self.um_per_px / self.world_um_per_px) / scale
        cx = self.world_w / 2.0 + sx_um / self.world_um_per_px
        cy = self.world_h / 2.0 + sy_um / self.world_um_per_px
        u0 = (self.out_w - 1) / 2.0 + shift_px[0]
        v0 = (self.out_h - 1) / 2.0 + shift_px[1]
        return np.array([[s, 0, cx - s * u0], [0, s, cy - s * v0]], np.float32)

    def _sample(self, world: np.ndarray, M: np.ndarray) -> np.ndarray:
        return cv2.warpAffine(world, M, (self.out_w, self.out_h), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                              borderMode=cv2.BORDER_REFLECT)

    def truth_patch(self, stage_x_um: float | None = None, stage_y_um: float | None = None):
        """(albedo, height_um) seen at the stage position without tremor, breathing or blur."""
        sx = self.stage_x_um if stage_x_um is None else stage_x_um
        sy = self.stage_y_um if stage_y_um is None else stage_y_um
        M = self._matrix(1.0, (0.0, 0.0), sx, sy)
        return self._sample(self.world_albedo_pf, M), self._sample(self.world_height_pf, M)

    def truth_albedo_upsampled(self, scale: int) -> np.ndarray:
        """Ground-truth reflectance on a grid `scale` times finer than the camera (for super-resolution tests)."""
        s = (self.um_per_px / scale) / self.world_um_per_px
        W, H = self.out_w * scale, self.out_h * scale
        cx = self.world_w / 2.0 + self.stage_x_um / self.world_um_per_px
        cy = self.world_h / 2.0 + self.stage_y_um / self.world_um_per_px
        # output pixel j of the fine grid covers camera coordinate (j + 0.5) / scale - 0.5
        u0 = ((self.out_w - 1) / 2.0 + 0.5) * scale - 0.5
        v0 = ((self.out_h - 1) / 2.0 + 0.5) * scale - 0.5
        M = np.array([[s, 0, cx - s * u0], [0, s, cy - s * v0]], np.float32)
        return cv2.warpAffine(self.world_albedo, M, (W, H), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                              borderMode=cv2.BORDER_REFLECT)

    # ------------------------------------------------------------------ rendering
    def render(self, tremor: bool = True, noise: bool = True, breathing: bool = True,
               blur: bool = True) -> tuple[np.ndarray, FrameTruth]:
        r = self.rng
        shift = (float(r.normal(0, self.tremor_px)), float(r.normal(0, self.tremor_px))) if tremor else (0.0, 0.0)
        scale = 1.0 + self.breathing_per_um * self.focus_z_um if breathing else 1.0
        M = self._matrix(scale, shift, self.stage_x_um, self.stage_y_um)
        alb = self._sample(self.world_albedo_pf, M)
        if blur:
            z = self._sample(self.world_height_pf, M)
            sig = self.objective.defocus_sigma_um(z - self.focus_z_um) / self.um_per_px
            img = _variable_blur(alb, sig)
        else:
            img = alb
        flick = 1.0 + (float(r.normal(0, self.flicker)) if noise else 0.0)
        img = img * self.vignette * self.gain * flick + self.dark_level
        if noise:
            sd = np.sqrt(self.read_noise**2 + self.shot_k * np.maximum(img, 0))
            img = img + r.normal(0, 1, img.shape).astype(np.float32) * sd
        u8 = np.clip(img + 0.5, 0, 255).astype(np.uint8)
        bgr = cv2.merge([np.clip(u8 * 0.93, 0, 255).astype(np.uint8), u8, np.clip(u8 * 1.05, 0, 255).astype(np.uint8)])
        if noise:
            ok, enc = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
            bgr = cv2.imdecode(enc, cv2.IMREAD_COLOR)
        return bgr, FrameTruth(self.stage_x_um, self.stage_y_um, self.focus_z_um, shift, scale)

    # ------------------------------------------------------------------ trajectories
    def sweep_trajectory(self, reversal: bool = False, n_frames: int | None = None,
                         back_fraction: tuple[float, float] = (0.25, 0.5)) -> np.ndarray:
        """Hand-turned focus sweep covering the visible relief +/- 3 DOF, with variable speed and a pause."""
        r = self.rng
        _, hz = self.truth_patch()
        dof = self.objective.dof_um
        z0, z1 = float(hz.min()) - 3 * dof * r.uniform(0.8, 1.2), float(hz.max()) + 3 * dof * r.uniform(0.8, 1.2)
        if r.random() < 0.5:
            z0, z1 = z1, z0
        n = int(n_frames or r.integers(40, 70))
        t = np.linspace(0, 1, n)
        vel = 1 + r.uniform(0.2, 0.6) * np.sin(2 * np.pi * (r.uniform(0.5, 2) * t + r.uniform(0, 1)))
        tp = r.uniform(0.3, 0.7)
        vel *= 1 - 0.95 * np.exp(-((t - tp) / r.uniform(0.02, 0.05)) ** 2)  # pause
        vel = np.maximum(vel, 0.02)
        cum = np.concatenate([[0], np.cumsum(vel[1:])])
        z = z0 + (z1 - z0) * cum / cum[-1]
        if reversal:
            m = int(r.integers(max(8, int(n * back_fraction[0])), max(9, int(n * back_fraction[1]))))
            back = z[-1] - (z1 - z0) * np.linspace(1, m, m) / n * r.uniform(0.8, 1.2)
            z = np.concatenate([z, back])
        return z.astype(np.float64)


def _variable_blur(img: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """Spatially varying Gaussian blur by interpolating a bank of uniformly blurred images."""
    s_lo, s_hi = float(np.min(sigma)), float(np.max(sigma))
    s_hi = min(s_hi, 40.0)
    levels = [s_lo]
    while levels[-1] < s_hi:
        levels.append(levels[-1] * 1.35 + 0.15)
    levels = np.array(levels, np.float32)
    bank = [cv2.GaussianBlur(img, (0, 0), float(max(s, 0.3))) for s in levels]
    sig = np.clip(sigma, levels[0], levels[-1]).astype(np.float32)
    idx = np.clip(np.searchsorted(levels, sig) - 1, 0, len(levels) - 2)
    out = np.zeros_like(img)
    if len(levels) == 1:
        return bank[0]
    for i in range(len(levels) - 1):
        m = idx == i
        if np.any(m):
            a = (sig[m] - levels[i]) / (levels[i + 1] - levels[i])
            out[m] = (1 - a) * bank[i][m] + a * bank[i + 1][m]
    return out
