"""Still-image products of the live view: the one place that turns the engine's still frames into their inputs.

Stacking and super-resolution both start from `LiveEngine.dwell_still()`. This module decides which data they get:
the corrected luma the live pipeline used, plus white-balanced colour when there is colour. Every product goes
through `prepare`, so they cannot disagree about corrections.
"""

from __future__ import annotations

import numpy as np

from .calibration import apply_white_balance
from .live import DwellStill


def prepare(still: DwellStill) -> dict:
    """Luma frames (float32, dark/flat corrected exactly as live), colour frames (float32 BGR, white-balanced) or
    None, and notes saying what was applied."""
    if not still.frames:
        raise ValueError("no still frames: hold the view still for a moment")
    frames = [f.astype(np.float32) for f in still.frames]
    notes = ["luma: the live pipeline's corrected frames (8-bit)"]
    colour = None
    if still.colour is not None:
        colour = [apply_white_balance(c.astype(np.float32), still.wb_gains) for c in still.colour]
        notes.append("colour: white-balanced" if still.wb_gains else "colour: camera balance")
    return {"frames": frames, "colour": colour, "notes": notes}
