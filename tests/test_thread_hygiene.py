"""No thread the application starts may outlive the test module that started it.

A leaked source thread renders with OpenCV; still running at interpreter exit it made the whole test process abort
(exit 134, "terminate called without an active exception") on Linux with Python 3.12 after every test had passed."""

import os
import pathlib
import subprocess
import sys

from conftest import started_threads, threads_left_running

from microtopos import calibration, live, objectives
from microtopos.camera import CAPTURE_THREAD_PREFIX, SimulatorSource


def _source() -> SimulatorSource:
    return SimulatorSource(1, objectives.default_set().objectives[1], 64, 48)


def test_a_started_source_has_a_thread_named_after_it_and_stop_ends_it():
    before = set(started_threads())
    src = _source()
    src.start()
    try:
        mine = [t for t in started_threads() if t not in before]
        assert [t.name for t in mine] == [f"{CAPTURE_THREAD_PREFIX}{src.source_id}"]
    finally:
        src.stop()
    assert threads_left_running(before, grace_s=0) == []


def test_the_engine_thread_is_named_and_stops():
    before = set(started_threads())
    eng = live.LiveEngine(calibration.CalibrationStore(None), objectives.default_set().objectives[1])
    eng.start()
    try:
        assert [t.name for t in started_threads() if t not in before] == [live.ENGINE_THREAD_NAME]
    finally:
        assert eng.stop()
    assert threads_left_running(before, grace_s=0) == []


def test_a_source_that_is_dropped_without_stop_is_reported_until_it_is_stopped():
    before = set(started_threads())
    src = _source()
    src.start()
    try:
        left = threads_left_running(before, grace_s=0.2)
        assert [t.name for t in left] == [f"{CAPTURE_THREAD_PREFIX}{src.source_id}"]
    finally:
        src.stop()
    assert threads_left_running(before, grace_s=0.2) == []


def _run_module_with_conftest(tmp_path, test_src: str) -> str:
    (tmp_path / "conftest.py").write_text((pathlib.Path(__file__).parent / "conftest.py").read_text(encoding="utf-8"),
                                          encoding="utf-8")
    (tmp_path / "test_x.py").write_text(test_src, encoding="utf-8")
    env = {k: v for k, v in os.environ.items() if not k.startswith(("MICROTOPOS_", "PYTEST_"))}
    p = subprocess.run([sys.executable, "-m", "pytest", "-p", "no:cacheprovider", "-q", "test_x.py"],
                       cwd=tmp_path, env=env, capture_output=True, text=True, timeout=120, check=False)
    return p.stdout + p.stderr


_LEAKY = (
    "import conftest\n"
    "from microtopos import objectives\n"
    "from microtopos.camera import SimulatorSource\n"
    "conftest.THREAD_GRACE_S = 0.3\n"
    "def test_starts_a_source_and_forgets_it():\n"
    "    src = SimulatorSource(1, objectives.default_set().objectives[1], 64, 48)\n"
    "    src.start()\n"
    "    {after}\n"
)


def test_a_module_that_leaves_a_source_running_fails(tmp_path):
    out = _run_module_with_conftest(tmp_path, _LEAKY.format(after="pass  # never stopped"))
    assert "1 passed" in out and "1 error" in out, out  # the test itself passes; the module is what fails
    assert "still running" in out and "capture:SimulatorSource-" in out, out


def test_a_module_that_stops_its_source_passes(tmp_path):
    out = _run_module_with_conftest(tmp_path, _LEAKY.format(after="src.stop()"))
    assert "1 passed" in out and "error" not in out, out
