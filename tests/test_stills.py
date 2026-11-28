"""Still-image inputs rebuilt from the camera's JPEG data (stills.prepare), checked against ground truth.

The MikrOkular sends 4:2:2 JPEG at quality about 43 (measured, 2026-11-28). Frames encoded the same way from the
simulator's specimen, with its sensor noise, stand in for it; the truth is the mean of 64 uncompressed renders.
"""
import time

import cv2
import numpy as np
import pytest

from microtopos import stills
from microtopos.calibration import CalibrationStore, bgr_to_luma
from microtopos.camera import ReplaySource
from microtopos.live import DwellStill, LiveEngine
from microtopos.objectives import resolve
from microtopos.recording import RecordingWriter
from microtopos.simulator import SpecimenSimulator
from microtopos.stacking import stack_frames

OBJ = resolve("10x / 0.25")
C = (slice(16, -16), slice(16, -16))  # away from the image border


def _camera_like(seed, n=12, w=640, h=360):
    """n still frames as the camera would send them (JPEG bytes), the live pipeline's 8-bit luma of them, and the
    noiseless truth (mean luma of 64 more renders, residual noise ~3/8 DN)."""
    sim = SpecimenSimulator(seed, OBJ, w, h)
    sim.tremor_px = 0.0
    _, hz = sim.truth_patch()
    sim.focus_z_um = float(np.median(hz))
    renders = [sim.render(breathing=False)[0] for _ in range(n + 64)]
    truth = np.mean([bgr_to_luma(r) for r in renders[n:]], axis=0)
    params = [cv2.IMWRITE_JPEG_QUALITY, 43, cv2.IMWRITE_JPEG_SAMPLING_FACTOR, cv2.IMWRITE_JPEG_SAMPLING_FACTOR_422]
    jpegs = [cv2.imencode(".jpg", r, params)[1].tobytes() for r in renders[:n]]
    live = [np.clip(bgr_to_luma(cv2.imdecode(np.frombuffer(j, np.uint8), cv2.IMREAD_COLOR)) + 0.5, 0, 255)
            .astype(np.uint8) for j in jpegs]
    return jpegs, live, truth


def _rms(a, truth):
    return float(np.sqrt(np.mean((np.asarray(a, np.float64)[C] - truth[C]) ** 2)))


@pytest.mark.parametrize("seed", [3, 11])
def test_frames_rebuilt_from_the_jpeg_are_closer_to_the_truth(seed):
    """Measured: one frame 16 % closer (4.09 -> 3.44 DN, 3.61 -> 3.08 DN), a 12-frame stack 7-10 % closer (2.21 ->
    2.06, 2.29 -> 2.06; what remains is detail the quantisation removed in every frame alike). Bounds: 10 % and 4 %."""
    jpegs, live, truth = _camera_like(seed)
    p = stills.prepare(DwellStill(live, jpegs, None, None, False, None, None, OBJ, color_mode=True))
    assert p["notes"][0].startswith("rebuilt from the camera's JPEG data") and "quality ≈ 43" in p["notes"][0]
    assert _rms(p["frames"][0], truth) < 0.90 * _rms(live[0], truth)
    s_live = stack_frames([f.astype(np.float32) for f in live])["image"]
    s_rest = stack_frames(p["frames"])["image"]
    assert _rms(s_rest, truth) < 0.96 * _rms(s_live, truth)
    assert p["colour"] is not None and p["colour"][0].shape == live[0].shape + (3,)


def test_without_jpeg_data_the_live_frames_are_used_unchanged():
    _, live, _ = _camera_like(5, n=8, w=320, h=180)
    p = stills.prepare(DwellStill(live, None, None, None, False, None, None, OBJ))
    assert all(np.array_equal(a, b) for a, b in zip(p["frames"], live))
    assert p["notes"] == ["the live pipeline's corrected 8-bit frames (no JPEG data)"] and p["colour"] is None


def test_a_raw_recording_replayed_hands_its_jpeg_files_to_the_still_products(tmp_path):
    """Recorded MJPEG bytes -> ReplaySource -> engine dwell buffer -> dwell_still().jpegs are the recorded files,
    and prepare rebuilds the frames from them."""
    jpegs, _, _ = _camera_like(7, n=40, w=320, h=180)
    wr = RecordingWriter(tmp_path / "rec", "test")
    for i, j in enumerate(jpegs):
        wr.add(np.frombuffer(j, np.uint8), seq=i, t=0.04 * i)
    wr.close()
    eng = LiveEngine(CalibrationStore(None), OBJ)
    src = ReplaySource(str(tmp_path / "rec"), exhaustive=True)
    eng.set_source(src)
    src.start()
    eng.start()
    try:
        t0 = time.time()
        while time.time() - t0 < 30 and not (src.done and len(eng.dwell_frames()) >= 8):
            time.sleep(0.05)
        still = eng.dwell_still()
    finally:
        eng.stop()
        src.stop()
    assert len(still.frames) >= 8 and still.jpegs is not None
    assert all(p.suffix == ".jpg" and p.parent == tmp_path / "rec" for p in still.jpegs)
    p = stills.prepare(still)
    assert p["notes"][0].startswith("rebuilt from the camera's JPEG data")
    # the rebuilt frame is the same picture as the live one (the same JPEG), within the deblocking change
    assert float(np.mean(np.abs(p["frames"][-1] - still.frames[-1].astype(np.float32)))) < 2.0


class _MjpegCap:
    """cv2.VideoCapture with CONVERT_RGB off, as on the MikrOkular: read() hands over the JPEG bytes (1-D uint8)."""

    def __init__(self, jpeg: bytes):
        self.buf = np.frombuffer(jpeg, np.uint8)

    def read(self):
        time.sleep(0.005)
        return True, self.buf.copy()

    def set(self, *a):
        return True

    def get(self, *a):
        return 0.0

    def getBackendName(self):
        return "fake"

    def release(self):
        pass


def test_the_camera_source_publishes_the_bytes_it_received(monkeypatch):
    from microtopos.camera import CameraInfo, CameraSource

    jpegs, _, _ = _camera_like(9, n=1, w=64, h=48)
    src = CameraSource(CameraInfo(0, "cam", "/dev/video0"))
    monkeypatch.setattr(src, "open", lambda: _MjpegCap(jpegs[0]))
    monkeypatch.setattr(src, "_apply_controls", lambda *a: {})
    src.start()
    try:
        t0 = time.time()
        while time.time() - t0 < 5 and src.latest() is None:
            time.sleep(0.01)
        frame, _ts, seq, _truth = src.latest()
        assert frame.shape == (48, 64, 3)
        got = src.jpeg_for(seq)
        assert got == jpegs[0] or src.jpeg_for(src.latest()[2]) == jpegs[0]  # a newer frame may have replaced seq
        assert src.jpeg_for(-1) is None
    finally:
        src.stop()


def test_with_gamma_linearisation_on_the_rebuilt_frames_are_linear_luminance():
    """The engine's 'Linearise gamma' measures linear-light luminance; rebuilt frames must be the same quantity
    (within the restoration's change), not the JPEG's gamma-encoded Y', which differs by tens of DN mid-range."""
    jpegs, _, _ = _camera_like(13, n=8, w=320, h=180)
    dec = [cv2.imdecode(np.frombuffer(j, np.uint8), cv2.IMREAD_COLOR) for j in jpegs]
    live_lin = [bgr_to_luma(b, True) for b in dec]
    p = stills.prepare(DwellStill([np.zeros((180, 320), np.uint8)] * 8, jpegs, None, None, True, None, None, OBJ))
    err_lin = np.mean([np.mean(np.abs(a - b)[C]) for a, b in zip(p["frames"], live_lin)])
    err_y = np.mean([np.mean(np.abs(a - bgr_to_luma(b))[C]) for a, b in zip(p["frames"], dec)])
    # measured: 1.66 DN mean change from deblocking and guided chroma (amplified ~1.2-1.6x by the 2.2 curve in the
    # highlights), against 68 DN between linear luminance and Y'
    assert err_lin < 2.5 and err_y > 20 * err_lin
