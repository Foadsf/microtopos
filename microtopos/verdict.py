"""Typed check results for the self-test. One normalisation, used by the CLI, pytest and the JSON log.

A check function returns rows (name, seed, measured, threshold, ok). `ok` must be a Python or NumPy boolean
(PASS/FAIL) or None (INFO); anything else is an error, never silently a pass. Counting uses the normalised
status, so a NumPy False cannot slip past an identity test.
"""

from __future__ import annotations

import enum
from dataclasses import asdict, dataclass

import numpy as np


class Status(str, enum.Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    INFO = "info"
    ERROR = "ERROR"  # the check itself raised: counts as a failure


def normalize(ok) -> Status:
    """None -> INFO; bool / numpy.bool_ -> PASS or FAIL; anything else raises TypeError."""
    if ok is None:
        return Status.INFO
    if isinstance(ok, (bool, np.bool_)):
        return Status.PASS if bool(ok) else Status.FAIL
    raise TypeError(f"check verdict must be a boolean or None, got {type(ok).__name__}: {ok!r}")


@dataclass(frozen=True)
class CheckResult:
    check: str
    seed: int
    measured: str
    threshold: str
    status: Status

    @property
    def failed(self) -> bool:
        return self.status in (Status.FAIL, Status.ERROR)

    def as_dict(self) -> dict:
        d = asdict(self)
        d["status"] = self.status.value
        return d


def from_row(row) -> CheckResult:
    name, seed, measured, threshold, ok = row
    try:
        st = normalize(ok)
    except TypeError as e:
        return CheckResult(str(name), int(seed), str(measured), f"{threshold} [{e}]", Status.ERROR)
    return CheckResult(str(name), int(seed), str(measured), str(threshold), st)


@dataclass(frozen=True)
class Summary:
    passed: int
    failed: int
    info: int
    errors: int

    @property
    def ok(self) -> bool:
        """True only for a non-empty set with at least one pass and no failure or error."""
        return self.failed == 0 and self.errors == 0 and self.passed > 0


def summarize(results: list[CheckResult]) -> Summary:
    return Summary(sum(r.status == Status.PASS for r in results), sum(r.status == Status.FAIL for r in results),
                   sum(r.status == Status.INFO for r in results), sum(r.status == Status.ERROR for r in results))
