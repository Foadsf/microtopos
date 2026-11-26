"""The release build must ship what the README says it ships, and one version number must mean one thing."""

import json
import re
import sys
import tomllib
import types
from pathlib import Path

import pytest

import microtopos
from microtopos import selftest, stage

ROOT = Path(__file__).resolve().parent.parent


@pytest.mark.characterization("passes whenever the three numbers agree; it exists to catch a half-done version bump")
def test_package_version_is_the_same_in_pyproject_init_and_lock():
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]["version"]
    lock = re.search(r'name = "microtopos"\r?\nversion = "([^"]+)"', (ROOT / "uv.lock").read_text(encoding="utf-8"))
    assert lock, "uv.lock has no entry for the project itself"
    assert pyproject == microtopos.__version__ == lock.group(1)


@pytest.mark.characterization("tests_bite swaps only the microtopos package; it fails against the old release.yml and spec "
                              "(checked by hand: restoring both made it fail)")
def test_the_release_build_installs_the_serial_extra_and_bundles_it():
    workflow = (ROOT / ".github" / "workflows" / "release.yml").read_text(encoding="utf-8")
    assert re.search(r"uv sync [^\n]*--extra serial", workflow), "release binaries would have no Grbl driver"
    spec = (ROOT / "packaging" / "microtopos.spec").read_text(encoding="utf-8")
    assert 'collect_submodules("serial")' in spec
    extras = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]["optional-dependencies"]
    assert any(req.startswith("pyserial") for req in extras["serial"])


@pytest.mark.characterization("tests_bite swaps only the microtopos package; it fails against the old release.yml "
                              "(checked by hand: restoring it made it fail)")
def test_the_release_job_fails_when_the_executable_has_no_serial_library():
    workflow = (ROOT / ".github" / "workflows" / "release.yml").read_text(encoding="utf-8")
    assert ".pyserial != null" in workflow
    assert workflow.index(".pyserial != null") < workflow.index("upload-artifact")


def _pretend_pyserial(monkeypatch, version: str | None):
    """CI does not install the `serial` extra, so the library is simulated: None = not installed."""
    monkeypatch.setattr(stage, "_HAS_PYSERIAL", version is not None)
    if version is not None:
        monkeypatch.setitem(sys.modules, "serial", types.SimpleNamespace(__version__=version))


def test_pyserial_version_is_none_without_the_library_and_the_library_version_with_it(monkeypatch):
    _pretend_pyserial(monkeypatch, None)
    assert stage.pyserial_version() is None
    _pretend_pyserial(monkeypatch, "3.5")
    assert stage.pyserial_version() == "3.5"


@pytest.mark.parametrize("version", ["3.5", None])
def test_the_selftest_report_records_the_serial_library_version_or_its_absence(monkeypatch, tmp_path, version):
    _pretend_pyserial(monkeypatch, version)
    report = tmp_path / "selftest.json"
    assert selftest.run((11,), out=lambda *_: None, only="check_conventions", json_path=str(report))
    data = json.loads(report.read_text(encoding="utf-8"))
    assert data["pyserial"] == version
