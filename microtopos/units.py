"""Canonical unit identifiers. Display strings ("µm") are for people; code compares canonical ids only.

Height units:
  "um"      physical micrometres (simulator truth, a measured z drive, or a per-sweep knob-travel calibration)
  "frames"  relative focus progression in source camera frames, assuming constant knob speed; its scale differs
            between sweeps and its orientation (up or down) is unknown unless the user states it
"""

from __future__ import annotations

UM = "um"
FRAMES = "frames"
_ALIASES = {"um": UM, "µm": UM, "μm": UM, "micrometre": UM, "micrometer": UM, "frames": FRAMES, "frame": FRAMES}


def canon(unit: str | None) -> str:
    """Canonical id for a unit string; raises ValueError for an unknown unit (never guesses)."""
    if unit is None:
        raise ValueError("missing unit")
    u = _ALIASES.get(str(unit).strip())
    if u is None:
        raise ValueError(f"unknown unit {unit!r}")
    return u


def is_um(unit: str | None) -> bool:
    try:
        return canon(unit) == UM
    except ValueError:
        return False


def label(unit: str | None) -> str:
    """Display label: 'µm' or 'frames'."""
    try:
        return "µm" if canon(unit) == UM else "frames"
    except ValueError:
        return str(unit)
