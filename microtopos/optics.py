"""Optical model of the objectives: sampling, diffraction, depth of field and defocus blur."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

SENSOR_FULL_WIDTH_PX = 1920
# Unverified fallback. The real object-space scale depends on the camera's relay optics and the
# microscope tube length, so it must come from a stage-micrometer calibration.
NOMINAL_SENSOR_PITCH_UM = 3.0
WAVELENGTH_UM = 0.55


@dataclass(frozen=True)
class Objective:
    name: str
    magnification: float
    na: float
    immersion_n: float = 1.0
    cover: str = ""  # cover-glass engraving: "0.17" (mm), "∞/0" or "0" (uncovered), "-" (either), "" unknown
    maker: str = ""
    series: str = ""

    def to_dict(self) -> dict:
        return {"name": self.name, "magnification": self.magnification, "na": self.na,
                "immersion_n": self.immersion_n, "cover": self.cover, "maker": self.maker, "series": self.series}

    @classmethod
    def from_dict(cls, d: dict) -> "Objective":
        """Build from a stored dict; raises ValueError for a missing or non-numeric field (never guesses)."""
        try:
            return cls(name=str(d["name"]), magnification=float(d["magnification"]), na=float(d["na"]),
                       immersion_n=float(d.get("immersion_n", 1.0)), cover=str(d.get("cover", "") or ""),
                       maker=str(d.get("maker", "") or ""), series=str(d.get("series", "") or ""))
        except (KeyError, TypeError, ValueError) as e:
            raise ValueError(f"objective entry {d!r} is incomplete or not numeric: {e}") from None

    @property
    def is_oil(self) -> bool:
        return self.immersion_n > 1.0

    def nominal_um_per_px(self, frame_width: int = SENSOR_FULL_WIDTH_PX) -> float:
        """Nominal object-space pixel size for a frame of the given width (full sensor = 1920 px)."""
        return NOMINAL_SENSOR_PITCH_UM / self.magnification * SENSOR_FULL_WIDTH_PX / float(frame_width)

    @property
    def rayleigh_um(self) -> float:
        return 0.61 * WAVELENGTH_UM / self.na

    @property
    def lateral_cutoff_um(self) -> float:
        """Abbe period cutoff lambda / (2 NA): the shortest lateral wavelength the optics transfer."""
        return WAVELENGTH_UM / (2.0 * self.na)

    @property
    def dof_um(self) -> float:
        """Classic wave + geometric depth of field: n*lambda/NA^2 + n*e/(M*NA), e = sensor pitch."""
        n = self.immersion_n
        return n * WAVELENGTH_UM / self.na**2 + n * NOMINAL_SENSOR_PITCH_UM / (self.magnification * self.na)

    @property
    def airy_sigma_um(self) -> float:
        """Gaussian approximation of the in-focus Airy PSF (sigma = 0.21 lambda / NA)."""
        return 0.21 * WAVELENGTH_UM / self.na

    @property
    def half_angle_rad(self) -> float:
        return math.asin(min(0.999, self.na / self.immersion_n))

    def defocus_sigma_um(self, dz_um):
        """Gaussian sigma of the defocused PSF in object space.

        Geometric blur circle radius = |dz| tan(theta); a uniform disc of radius R is approximated by a
        Gaussian with sigma = R / 2. Added in quadrature to the diffraction-limited sigma.
        Works on scalars and numpy arrays.
        """
        geo = np.abs(dz_um) * (math.tan(self.half_angle_rad) / 2.0)
        return np.sqrt(self.airy_sigma_um**2 + geo**2)

    def psf_sigma_px(self, um_per_px: float) -> float:
        return max(0.5, self.airy_sigma_um / um_per_px)

    def focus_window_px(self, um_per_px: float) -> int:
        """Odd focus-measure window covering at least 3 PSF sigmas each side, minimum 7 px."""
        side = int(math.ceil(6.0 * self.psf_sigma_px(um_per_px)))
        side = max(7, side)
        return side | 1


# The program's original objective set: the default for a new installation and the "Generic" preset. The user's
# own set lives in objectives.json (see objectives.py); look objectives up with objectives.resolve(), not here.
BUILTIN_OBJECTIVES: tuple[Objective, ...] = (
    Objective("4x / 0.10", 4.0, 0.10),
    Objective("10x / 0.25", 10.0, 0.25),
    Objective("40x / 0.85", 40.0, 0.85),
    Objective("100x / 1.25 oil", 100.0, 1.25, immersion_n=1.515),
)
