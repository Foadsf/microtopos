"""Still-image products of the live view: the one place that turns the engine's still frames into their inputs.

Stacking and super-resolution both start from `LiveEngine.dwell_still()`, and both go through `prepare`, so they
cannot disagree about corrections. When every still frame comes with the camera's JPEG data (a camera that sends
MJPEG, or a raw recording), the frames are rebuilt from it (`jpegq.restore_frame`): the decoder's exact luma,
deblocked inside the quantisation intervals the camera's own tables allow, and chroma upsampled along the luma
edges. Otherwise the live pipeline's 8-bit corrected luma is used. Dark/flat correction (and gamma linearisation,
when the engine has it on) is applied exactly as live; white balance only to colour.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from . import jpegq
from .calibration import apply_white_balance, bgr_to_luma
from .live import DwellStill

DEBLOCK_ITERATIONS = 1  # jpegq.restore_frame's default: one projection round (tests/test_jpegq.py measures its gain)


def _jpeg_bytes(j) -> bytes:
    return bytes(j) if isinstance(j, (bytes, bytearray, memoryview)) else Path(j).read_bytes()


def prepare(still: DwellStill, restore_jpeg: bool = True) -> dict:
    """Luma frames (float32, dark/flat corrected), colour frames (float32 BGR, white-balanced) or None, and notes
    saying what was applied."""
    if not still.frames:
        raise ValueError("no still frames: hold the view still for a moment")
    colour = None
    if still.jpegs is not None and restore_jpeg:
        need_bgr = still.color_mode or still.linearize
        with ThreadPoolExecutor(max(1, min(4, (os.cpu_count() or 2) - 1))) as ex:
            rs = list(ex.map(lambda j: jpegq.restore_frame(_jpeg_bytes(j), DEBLOCK_ITERATIONS, colour=need_bgr),
                             still.jpegs))
        if still.linearize:  # the same luminance the live pipeline measures, from the restored colour
            lum = [bgr_to_luma(np.clip(r["bgr"], 0, 255).astype(np.float32), True, still.response) for r in rs]
        else:
            lum = [r["y"] for r in rs]  # the JPEG's own luma Y' (Rec.601), as live, without the 8-bit BGR round trip
        frames = [still.corrector(y) if still.corrector is not None else y.astype(np.float32) for y in lum]
        info = rs[0]["info"]
        q = info.estimated_quality(0)
        notes = [f"rebuilt from the camera's JPEG data ({', '.join(rs[0]['applied'])}; quality ≈ {q})"
                 if q is not None else f"rebuilt from the camera's JPEG data ({', '.join(rs[0]['applied'])})"]
        if still.color_mode and all(r["bgr"] is not None for r in rs):
            colour = [apply_white_balance(r["bgr"], still.wb_gains) for r in rs]
    else:
        frames = [f.astype(np.float32) for f in still.frames]
        notes = ["the live pipeline's corrected 8-bit frames" + (" (no JPEG data)" if still.jpegs is None else "")]
        if still.colour is not None:
            colour = [apply_white_balance(c.astype(np.float32), still.wb_gains) for c in still.colour]
    if colour is not None and still.lateral_ca is not None:
        colour = [still.lateral_ca.apply(c) for c in colour]
        notes.append(f"colour fringes corrected (lateral chromatic aberration, {still.lateral_ca.fringe_px_at_corner():.2f} "
                     "px at the corner)")
    if colour is not None:
        notes.append("colour white-balanced" if still.wb_gains else "colour: the camera's balance")
    return {"frames": frames, "colour": colour, "notes": notes}
