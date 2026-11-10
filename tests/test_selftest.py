"""pytest wrapper around the ground-truth self-tests (one seed; `microtopos-selftest` runs three)."""

import pytest

from microtopos import selftest

CHECKS = selftest.CHECKS + selftest.ONCE


KNOWN: dict[str, str] = {}  # check name -> reason, for checks expected to fail (none at present)


def _id(c):
    return getattr(c, "__name__", "check_sff_10x")


@pytest.mark.parametrize("check", [pytest.param(c, marks=pytest.mark.xfail(reason=KNOWN[_id(c)], strict=False))
                                   if _id(c) in KNOWN else c for c in CHECKS], ids=_id)
def test_check(check):
    rows = check(11)
    failed = [r for r in rows if r[4] is not None and not bool(r[4])]
    assert not failed, "\n".join(f"{r[0]}: measured {r[2]}, threshold {r[3]}" for r in failed)


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
