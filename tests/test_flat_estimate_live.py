"""The estimated flat field, from the engine's own motion and registration, against the simulator's known shading.

The simulator multiplies every frame by its `vignette` (a 15-30 % fall-off): a pan through the live engine with
start_flat_estimate must store a flat whose gain is that vignette, labelled as estimated, persisted, and used.
"""
import numpy as np
import pytest

from microtopos.calibration import FLAT_ESTIMATED, CalibrationStore, flat_source, radiometric_key
from microtopos.live import LiveEngine, Override
from microtopos.objectives import resolve
from microtopos.simulator import SpecimenSimulator

OBJ = resolve("10x / 0.25")
W, H = 320, 240


def _pan(eng, sim, path, seq0=1):
    for i, (fx, fy) in enumerate(path):
        sim.stage_x_um, sim.stage_y_um = fx * W * sim.um_per_px, fy * H * sim.um_per_px
        bgr, _ = sim.render(breathing=False)
        eng.process_frame(bgr, (seq0 + i) * 0.04, seq0 + i, truth=None)
    return seq0 + len(path)


def _lissajous(n, amp):
    t = np.linspace(0.0, 1.0, n)
    return np.stack([amp * np.sin(2 * np.pi * t + 0.3), amp * np.sin(3 * np.pi * t + 1.1)], 1)


def _gain_error(gain, truth):
    r = gain / truth
    return float(np.sqrt(np.mean((r / r.mean() - 1.0) ** 2)))


@pytest.mark.parametrize("seed", [1, 2])
def test_a_pan_through_the_engine_stores_the_simulators_vignette(tmp_path, seed):
    sim = SpecimenSimulator(seed, OBJ, W, H)
    sim.gain = 0.85
    _, hz = sim.truth_patch()
    sim.focus_z_um = float(np.median(hz))
    store = CalibrationStore(tmp_path / "cal")
    eng = LiveEngine(store, OBJ)
    eng.sm.override = Override.PAN
    eng.start_flat_estimate()
    _pan(eng, sim, _lissajous(400, 0.6))
    assert eng.flat_estimate_progress() is None and eng.flat_status.startswith("Flat stored"), eng.flat_status
    key = radiometric_key((H, W), None)
    cal = store.get(OBJ)
    flat = cal.flats[key]
    truth = sim.vignette / sim.vignette.mean()
    err = _gain_error(flat / flat.mean(), truth)
    assert err < 0.015, err  # the vignette falls off by 15-30 %
    assert _gain_error(np.ones_like(truth), truth) > 0.04  # a flat "flat" is far outside the bound
    assert flat_source(cal, key).startswith(FLAT_ESTIMATED)
    again = CalibrationStore(tmp_path / "cal").get(OBJ)  # persisted, with its provenance
    assert flat_source(again, key) == flat_source(cal, key) and np.allclose(again.flats[key], flat)
    # the live correction now uses it: a fresh view's shading is gone from the corrected frame
    sim.stage_x_um, sim.stage_y_um = -2.0 * W * sim.um_per_px, 1.5 * H * sim.um_per_px
    shade = eng._corrector_for((H, W))(np.full((H, W), 100.0, np.float32) * sim.vignette)
    assert float(np.std(shade) / np.mean(shade)) < 0.25 * float(np.std(sim.vignette) / np.mean(sim.vignette))


def test_a_still_view_never_completes_an_estimate(tmp_path):
    sim = SpecimenSimulator(3, OBJ, W, H)
    eng = LiveEngine(CalibrationStore(tmp_path / "cal"), OBJ)
    eng.start_flat_estimate()
    _pan(eng, sim, np.zeros((200, 2)))
    p = eng.flat_estimate_progress()
    assert p is not None and not p["ready"] and p["pairs"] == 0
    assert radiometric_key((H, W), None) not in eng.store.get(OBJ).flats


def test_an_exposure_change_cancels_the_estimate(tmp_path):
    sim = SpecimenSimulator(4, OBJ, W, H)
    eng = LiveEngine(CalibrationStore(tmp_path / "cal"), OBJ)
    eng.sm.override = Override.PAN
    eng.start_flat_estimate()
    seq = _pan(eng, sim, _lissajous(60, 0.6)[:12])  # a few pairs, far from ready
    p = eng.flat_estimate_progress()
    assert p is not None and 0 < p["pairs"] and not p["ready"]
    eng.exposure = 300.0
    _pan(eng, sim, _lissajous(5, 0.1), seq)
    assert eng.flat_estimate_progress() is None and "cancelled" in eng.flat_status
