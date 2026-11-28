"""Rolling shutter through the live engine: measured from the engine's own motion, then used to straighten
fast-pan frames before the mosaic stitches them (against a synthetic rolling-shutter camera with a known readout)."""
import numpy as np
import pytest
import test_rollingshutter as trs

from microtopos import rollingshutter as rs
from microtopos.calibration import CalibrationStore
from microtopos.camera import FrameSource
from microtopos.live import READOUT_PAIRS, LiveEngine, Override
from microtopos.objectives import resolve

RHO = 0.7


class _Named(FrameSource):
    def __init__(self):
        super().__init__()
        self.camera_name = "RS camera"

    def _run(self):
        pass


def _script(cam, n_cycles=6):
    """Still, a fast pan, still, another direction...: (frame, velocity) per frame, positions continuous."""
    pos = np.zeros(2)
    out = []
    pans = [(9.0, 0.0), (0.0, 8.0), (-10.0, 0.0), (0.0, -9.0), (7.0, 6.0), (-8.0, -5.0)]
    for c in range(n_cycles):
        for _ in range(4):
            out.append((cam.frame(tuple(pos), (0.0, 0.0)), (0.0, 0.0)))
        v = np.array(pans[c % len(pans)])
        for _ in range(4):
            pos = pos + v
            out.append((cam.frame(tuple(pos), tuple(v)), tuple(v)))
    return out


def _engine(tmp_path):
    eng = LiveEngine(CalibrationStore(tmp_path / "cal"), resolve("10x / 0.25"))
    eng.set_source(_Named())
    return eng


def test_the_engine_collects_pairs_that_give_the_cameras_readout(tmp_path):
    cam = trs.RollingShutterCamera(3, RHO)
    eng = _engine(tmp_path)
    eng.start_readout_estimate()
    for i, (f, _v) in enumerate(_script(cam)):
        eng.process_frame(np.dstack([f] * 3), i * 0.04, i + 1)
        if eng.readout_progress() is not None and eng.readout_progress()[0] >= READOUT_PAIRS:
            break
    got = eng.take_readout_pairs()
    assert got is not None, eng.readout_progress()
    pairs, shape = got
    assert shape == (trs.H, trs.W)
    r = rs.estimate_readout(pairs)
    assert r["observable"] and r["rho"] == pytest.approx(RHO, rel=0.05), r


def test_with_a_stored_readout_the_mosaic_gets_straightened_fast_pan_frames(tmp_path):
    cam = trs.RollingShutterCamera(4, RHO)
    eng = _engine(tmp_path)
    eng.store.set_camera_value("readout", "RS camera", (trs.H, trs.W), {"rho": RHO})
    seen = []
    real = eng.mosaic.track
    eng.mosaic.track = lambda img, rgb=None: (seen.append(img.copy()), real(img, rgb=rgb))[1]
    eng.sm.override = Override.PAN
    still = cam.frame((0.0, 0.0), (0.0, 0.0))
    v = (10.0, 0.0)
    for i, p in enumerate([(0.0, 0.0), (10.0, 0.0), (20.0, 0.0)]):
        f = cam.frame(p, (0.0, 0.0) if i == 0 else v)
        eng.process_frame(np.dstack([f] * 3), i * 0.04, i + 1)
    raw_last = cam.frame((20.0, 0.0), v)
    def shear(img):
        return rs.measure_pair(still, img)[0][0, 1]

    assert abs(shear(raw_last)) > 0.02  # rho v / H = 0.039: the raw frame is visibly sheared
    assert abs(shear(seen[-1])) < 0.2 * abs(shear(raw_last))


def test_frames_the_engine_skipped_do_not_change_the_measured_readout(tmp_path):
    """The engine processes the newest frame and skips others when busy: here every other frame (sequence gap 2).
    The skew follows the speed per source frame, so the measurement divides by the gap (without it: rho / 2)."""
    cam = trs.RollingShutterCamera(6, RHO)
    eng = _engine(tmp_path)
    eng.start_readout_estimate()
    script = _script(cam, n_cycles=12)
    for i in range(0, len(script), 2):
        f = script[i][0]
        eng.process_frame(np.dstack([f] * 3), i * 0.04, i + 1)
        if eng.readout_progress()[0] >= READOUT_PAIRS:
            break
    got = eng.take_readout_pairs()
    assert got is not None, eng.readout_progress()
    r = rs.estimate_readout(got[0])
    assert r["observable"] and r["rho"] == pytest.approx(RHO, rel=0.05), r
