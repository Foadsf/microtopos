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
    failed = [r for r in rows if r[4] is False]
    assert not failed, "\n".join(f"{r[0]}: measured {r[2]}, threshold {r[3]}" for r in failed)


def test_gui_smoke():
    from microtopos.gui.smoke import run_gui_smoke

    assert run_gui_smoke()
