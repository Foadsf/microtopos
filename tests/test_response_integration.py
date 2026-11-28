"""A measured camera response, once stored, is what 'Linearise gamma' applies (instead of assuming gamma 2.2)."""
import json

import numpy as np
import pytest

from microtopos.calibration import CalibrationStore, bgr_to_luma, linear_luma
from microtopos.camera import FrameSource
from microtopos.live import LiveEngine
from microtopos.objectives import resolve
from microtopos.response import ResponseCurve

DN = np.arange(256, dtype=np.float64)


def _curve(gamma=1.8, channels="bgr"):
    """A response with the closed form g(DN) = (DN / 128) ** gamma, every level measured."""
    row = (DN / 128.0) ** gamma
    lut = np.tile(row, (3 if channels == "bgr" else 1, 1))
    return ResponseCurve(lut, channels, (0.01, 0.02, 0.04), ((0, 255),) * lut.shape[0], 128, "test", 0.0)


@pytest.mark.parametrize("channels", ["bgr", "luma"])
def test_linear_luminance_follows_the_measured_curve(channels):
    """A grey frame of DN v gives 255 (v / 255) ** 1.8 through a gamma-1.8 response, not the 2.2 assumption."""
    v = np.array([16, 64, 128, 200, 250], np.uint8)
    grey = np.repeat(v[None, :, None], 3, axis=2)
    expect = 255.0 * (v / 255.0) ** 1.8
    assert linear_luma(grey, _curve(channels=channels))[0] == pytest.approx(expect, rel=1e-4)
    assert bgr_to_luma(grey, True, _curve(channels=channels))[0] == pytest.approx(expect, rel=1e-4)
    assert not np.allclose(bgr_to_luma(grey, True)[0], expect, rtol=0.05)  # the assumption differs by > 5 %


def test_responses_are_stored_per_camera_and_size_and_survive_a_restart(tmp_path):
    s = CalibrationStore(tmp_path)
    s.set_response("Cam A", (1080, 1920), _curve(1.8))
    s.set_response("Cam A", (720, 1280), _curve(2.0))
    again = CalibrationStore(tmp_path)
    assert again.get_response("Cam A", (1080, 1920)).gamma_equivalent() == pytest.approx(1.8, abs=0.01)
    assert again.get_response("Cam A", (720, 1280)).gamma_equivalent() == pytest.approx(2.0, abs=0.01)
    assert again.get_response("Cam B", (1080, 1920)) is None and again.get_response("", (1080, 1920)) is None
    data = json.loads((tmp_path / "camera_response.json").read_text())
    data["Cam A|1920x1080"]["lut"] = "damaged"
    (tmp_path / "camera_response.json").write_text(json.dumps(data))
    assert CalibrationStore(tmp_path).get_response("Cam A", (1080, 1920)) is None  # never a half-read curve


class _Named(FrameSource):
    def __init__(self):
        super().__init__()
        self.camera_name = "Cam A"

    def _run(self):
        pass


def test_the_engine_linearises_with_the_stored_response_of_its_camera(tmp_path):
    store = CalibrationStore(tmp_path)
    store.set_response("Cam A", (60, 80), _curve(1.8))
    eng = LiveEngine(store, resolve("10x / 0.25"))
    eng.set_source(_Named())
    eng.linearize = True
    frame = np.full((60, 80, 3), 128, np.uint8)
    eng.process_frame(frame, 0.0, 1)
    assert float(eng.snapshot().preview.mean()) == pytest.approx(255 * (128 / 255) ** 1.8, rel=1e-3)
    assert eng.linearization((60, 80)).startswith("measured camera response (gamma-equivalent 1.80")
    eng2 = LiveEngine(CalibrationStore(tmp_path / "none"), resolve("10x / 0.25"))
    eng2.set_source(_Named())
    eng2.linearize = True
    eng2.process_frame(frame, 0.0, 1)
    assert float(eng2.snapshot().preview.mean()) == pytest.approx(255 * (128 / 255) ** 2.2, rel=1e-3)
    assert eng2.linearization((60, 80)).startswith("gamma 2.2 assumed")
