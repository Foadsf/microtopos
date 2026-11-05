"""The self-test's verdict gate: statuses, totals and the process exit code must agree."""

import os
import subprocess
import sys

import numpy as np
import pytest

from microtopos.verdict import Status, from_row, normalize, summarize


@pytest.mark.parametrize("value,expected", [(True, Status.PASS), (False, Status.FAIL), (np.bool_(True), Status.PASS),
                                            (np.bool_(False), Status.FAIL), (None, Status.INFO)])
def test_normalize(value, expected):
    assert normalize(value) is expected


@pytest.mark.parametrize("bad", ["yes", 1, 0.0, np.float64(1.0), [True]])
def test_invalid_verdicts_are_errors(bad):
    with pytest.raises(TypeError):
        normalize(bad)
    assert from_row(("x", 0, "m", "t", bad)).status is Status.ERROR


def test_numpy_false_counts_as_failure():
    rows = [from_row(("a", 0, "", "", np.bool_(False))), from_row(("b", 0, "", "", True))]
    s = summarize(rows)
    assert (s.passed, s.failed, s.ok) == (1, 1, False)


def test_empty_and_info_only_sets_do_not_pass():
    assert not summarize([]).ok
    assert not summarize([from_row(("a", 0, "", "", None))]).ok


def test_exception_in_check_is_a_failure():
    from microtopos.selftest import run_check

    def boom(_seed):
        raise RuntimeError("deliberate")

    res = run_check(boom, 1)
    assert len(res) == 1 and res[0].status is Status.ERROR and res[0].failed
    assert run_check(lambda s: [], 1)[0].status is Status.ERROR


@pytest.mark.parametrize("kind", ["numpy_false", "python_false", "string"])
def test_injected_failure_exits_nonzero(kind):
    """Subprocess control: a deliberately failing row must make the CLI exit 1, and the fast check alone exit 0."""
    env = {**os.environ, "MICROTOPOS_SELFTEST_INJECT": kind}
    cmd = [sys.executable, "-m", "microtopos.selftest", "--quick", "--only", "check_sfilter"]
    bad = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=300)
    assert bad.returncode == 1, bad.stdout[-2000:]
    assert "SELF-TEST FAILED" in bad.stdout
    env.pop("MICROTOPOS_SELFTEST_INJECT")
    good = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=300)
    assert good.returncode == 0, good.stdout[-2000:]
