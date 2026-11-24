"""Test-suite skip policy (one place).

A skip is legitimate only when a capability is absent from the machine (no OpenCL device, no Xvfb, an optional
package that is not installed). A skip must never turn a WRONG result into "not run".

* `require_opencl()` skips with the reason "no OpenCL device" when OpenCV has no usable OpenCL device. When the
  environment variable MICROTOPOS_REQUIRE_OPENCL=1 is set (CI on a runner that has an OpenCL runtime), the same
  condition FAILS instead, so a broken runtime cannot hide behind a skip.
* MICROTOPOS_STRICT_SKIPS=1 turns every skip whose reason does not match ALLOWED_SKIP_REASONS into a failure that
  quotes the reason (collection-time skips such as a module-level importorskip included).
"""

from __future__ import annotations

import os
import re

import cv2
import numpy as np
import pytest

# Reasons that describe an absent capability. Never add a reason that describes a wrong or diverging result.
ALLOWED_SKIP_REASONS = (
    r"^no OpenCL device$",
    r"needs Xvfb and a window manager",
    r"could not import 'pyqtgraph'",  # optional GUI dependency not installed
    r"^Windows memory query ",  # a Windows-only API test on another platform
)


def skip_reason_allowed(reason: str) -> bool:
    return any(re.search(p, reason) for p in ALLOWED_SKIP_REASONS)


def _env_on(name: str) -> bool:
    return os.environ.get(name, "") == "1"


def opencl_device_available() -> bool:
    """True when OpenCV reports OpenCL and a default device can actually be created."""
    try:
        if not cv2.ocl.haveOpenCL():
            return False
        cv2.ocl.setUseOpenCL(True)
        cv2.UMat(np.zeros((8, 8), np.float32)).get()  # forces context/device creation
        return bool(cv2.ocl.Device.getDefault().available())
    except cv2.error:
        return False


def require_opencl() -> None:
    """Skip ("no OpenCL device") when there is no OpenCL device; fail instead when MICROTOPOS_REQUIRE_OPENCL=1."""
    if opencl_device_available():
        return
    if _env_on("MICROTOPOS_REQUIRE_OPENCL"):
        pytest.fail("MICROTOPOS_REQUIRE_OPENCL=1 but no usable OpenCL device was found", pytrace=False)
    pytest.skip("no OpenCL device")


@pytest.fixture
def opencl():
    """Fixture form of require_opencl(): the test runs only on a machine with an OpenCL device."""
    require_opencl()


def _strict_outcome(rep, what: str) -> None:
    if not (_env_on("MICROTOPOS_STRICT_SKIPS") and rep.skipped):
        return
    reason = rep.longrepr[2] if isinstance(rep.longrepr, tuple) else str(rep.longrepr)
    reason = re.sub(r"^Skipped: ", "", reason)
    if not skip_reason_allowed(reason):
        rep.outcome = "failed"
        rep.longrepr = f"MICROTOPOS_STRICT_SKIPS: {what} skipped for a reason that is not an absent capability: {reason!r}"


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    _strict_outcome(outcome.get_result(), "test")


@pytest.hookimpl(hookwrapper=True)
def pytest_make_collect_report(collector):
    outcome = yield
    _strict_outcome(outcome.get_result(), "collection")

