"""pytest wrapper around the ground-truth self-tests (one seed; `microtopos-selftest` runs three).

Rows are normalised by `verdict.from_row`, exactly as the CLI does, so a NumPy False fails here too.
"""

import pytest

from microtopos import selftest
from microtopos.verdict import Status

CHECKS = selftest.CHECKS + selftest.ONCE


KNOWN: dict[str, str] = {}  # check name -> reason, for checks expected to fail (none at present)


def _id(c):
    return getattr(c, "__name__", "check")


@pytest.mark.parametrize("check", [pytest.param(c, marks=pytest.mark.xfail(reason=KNOWN[_id(c)], strict=False))
                                   if _id(c) in KNOWN else c for c in CHECKS], ids=_id)
def test_check(check):
    results = selftest.run_check(check, 11)
    assert results, "a check must return at least one row"
    failed = [r for r in results if r.failed]
    assert not failed, "\n".join(f"{r.check}: measured {r.measured}, threshold {r.threshold} ({r.status.value})"
                                 for r in failed)
    assert any(r.status in (Status.PASS, Status.INFO) for r in results)


def test_gui_smoke():
    from microtopos.gui.smoke import run_gui_smoke

    assert run_gui_smoke()


def test_sweep_completeness():
    """A sweep counts once the field went blurry -> sharp -> blurry; refocusing afterwards must not undo it."""
    import numpy as np

    from microtopos.live import _completeness

    z = np.linspace(-3, 3, 60)

    def tiles(path):
        return [np.full(36, 1 + 5 * np.exp(-x * x)) for x in path]

    assert _completeness(tiles(z)) == 1.0
    assert _completeness(tiles(np.r_[z, np.linspace(3, 0, 30), np.zeros(15)])) == 1.0  # sweep, then refocus
    assert _completeness(tiles(np.r_[np.linspace(-3, 0, 30), np.zeros(20)])) == 0.0  # refocus only
    assert _completeness(tiles(np.r_[np.zeros(10), np.linspace(0, 0.3, 10), np.zeros(10)])) == 0.0  # wiggle
