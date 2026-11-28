"""Software white balance: the MikrOkular exposes no white-balance control, so colour images are balanced here.

The gains act on colour outputs only (preview, sweep colour frames, colour mosaic); luma, and so every measurement,
dark and flat, must not change with them.
"""
import numpy as np
import pytest

from microtopos.calibration import CalibrationStore, apply_white_balance, bgr_to_luma, white_balance_gains
from microtopos.live import LiveEngine, Override
from microtopos.objectives import resolve
from microtopos.simulator import SpecimenSimulator

OBJ = resolve("10x / 0.25")
TINT = np.array([0.80, 1.00, 1.25], np.float32)  # B, G, R: a warm lamp


def _tinted(seed=5, shape=(120, 160)):
    """A grey specimen (neutral by construction) seen through a warm tint, in 20..190 DN."""
    sim = SpecimenSimulator(seed, OBJ, shape[1], shape[0])
    _, hz = sim.truth_patch()
    sim.focus_z_um = float(np.median(hz))
    y = bgr_to_luma(sim.render(breathing=False)[0])  # the simulator's specimen is slightly warm: take its luma
    grey = np.repeat((20 + 150 * (y - y.min()) / max(1.0, float(np.ptp(y))))[..., None], 3, axis=2)
    return np.clip(grey * TINT + 0.5, 0, 255).astype(np.uint8), grey


def test_grey_world_gains_undo_a_known_tint():
    bgr, _ = _tinted()
    g = np.array(white_balance_gains(bgr))
    assert g[1] == 1.0
    assert g == pytest.approx(1.0 / TINT, rel=0.01)  # B 1.25, R 0.80 within 1 %
    out = apply_white_balance(bgr, tuple(g)).astype(np.float32)
    m = out.reshape(-1, 3).mean(axis=0)
    assert m[0] == pytest.approx(m[1], rel=0.01) and m[2] == pytest.approx(m[1], rel=0.01)


def test_clipped_and_black_pixels_do_not_set_the_balance():
    """A blown-out specular patch (255 in R and G, less in B) would pull the grey-world mean towards blue."""
    bgr, _ = _tinted()
    bgr[:40, :60] = (200, 255, 255)
    bgr[-30:, -50:] = 0
    assert np.array(white_balance_gains(bgr)) == pytest.approx(1.0 / TINT, rel=0.01)


def test_a_dark_view_is_refused():
    with pytest.raises(ValueError, match="adjust the exposure"):
        white_balance_gains(np.full((50, 50, 3), 4, np.uint8))


def _tinted_sweep(n=14):
    """A focus sweep through the simulator's specimen, made neutral (its luma) and seen through the warm tint."""
    sim = SpecimenSimulator(4, OBJ, 160, 120)
    out = []
    for z_um in sim.sweep_trajectory(n_frames=n):
        sim.focus_z_um = float(z_um)
        bgr, truth = sim.render()
        y = bgr_to_luma(bgr)[..., None].repeat(3, axis=2)
        out.append((np.clip(y * 0.8 * TINT + 0.5, 0, 255).astype(np.uint8), truth))
    return out


def test_engine_balances_colour_outputs_but_never_luma():
    sweep = _tinted_sweep()
    gains = white_balance_gains(sweep[0][0])
    runs = {}
    for wb in (None, gains):
        eng = LiveEngine(CalibrationStore(None), OBJ)
        eng.color_mode = True
        eng.sync_callbacks = True
        eng.wb_gains = wb
        caps = []
        eng.on_sweep = caps.append
        eng.sm.override = Override.Z
        for i, (bgr, truth) in enumerate(sweep):
            eng.process_frame(bgr, i * 0.04, i + 1, truth=truth)
        eng.flush()
        runs[wb is None] = (eng.snapshot(), caps)
    (raw_snap, raw_caps), (wb_snap, wb_caps) = runs[True], runs[False]
    m = wb_snap.preview_rgb.reshape(-1, 3).astype(np.float32).mean(axis=0)  # RGB order
    assert m[0] == pytest.approx(m[1], rel=0.015) and m[2] == pytest.approx(m[1], rel=0.015)
    r = raw_snap.preview_rgb.reshape(-1, 3).astype(np.float32).mean(axis=0)
    assert r[0] / r[1] == pytest.approx(TINT[2], rel=0.02)  # without gains the tint is still there
    np.testing.assert_array_equal(wb_snap.preview, raw_snap.preview)  # luma untouched
    assert len(wb_caps) == 1 and len(raw_caps) == 1
    assert wb_caps[0].white_balance == pytest.approx(gains) and raw_caps[0].white_balance is None
    c = wb_caps[0].color_frames[0].reshape(-1, 3).astype(np.float32).mean(axis=0)  # BGR
    assert c[0] == pytest.approx(c[1], rel=0.015) and c[2] == pytest.approx(c[1], rel=0.015)
    for a, b in zip(wb_caps[0].frames, raw_caps[0].frames):
        np.testing.assert_array_equal(a, b)
