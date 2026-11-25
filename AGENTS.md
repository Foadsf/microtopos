# Rules for contributors and coding agents

MicroTopos produces measurements. A number shown in µm, a parameter named after an ISO clause, or a line in the README is a claim someone will rely on. These rules exist because a review of agent-written branches (November 2026) found features that were unreachable from the app, numerically wrong, or covered by tests that could not fail. Each rule below closes one of the ways that happened. They apply to people and to AI agents alike; CI enforces the ones a machine can check.

## A change is done when

1. **It is reachable.** A GUI control, a CLI flag, or a public function the app calls reaches the new code. Code nothing calls is not a feature: wire it or delete it.
2. **A test fails without it.** Every new or changed test must fail on the base branch. CI runs `tools/tests_bite.py` on every pull request and rejects tests that pass there. A test that is meant to pass on the base (a guard for behaviour that was already right) carries `@pytest.mark.characterization("reason")`.
3. **The test checks the result, not the plumbing.** Numerical code is compared with a closed form or an independently built ground truth, with a tolerance tight enough that a 5 % scale error or a sign flip fails. `is not None`, a shape, `isfinite`, `> 0` or "does not raise" is never the only check of a numerical claim. Do not assert on private attributes when the behaviour is observable.
4. **You broke it on purpose and the test noticed.** Before you call a feature tested, flip a sign or bypass the new branch and watch the test go red. Say in the pull request which mutation you tried.
5. **The words match the code.** README, docs, docstrings, UI labels and the pull-request text describe what the code does, in the terms a metrologist would use. No "calibrated" for a textbook formula, no "out-of-core" for an in-memory loop, no standard's name on something that is not that standard's definition. If something was only verified against a simulator or a fake device, say exactly that.

## Never

* **Never turn a wrong result into a skip.** `pytest.skip`, `importorskip`, `skipif` and `pytest.xfail()` are not allowed in test modules (`tests/test_suite_integrity.py` enforces this). The only skips are the capability fixtures in `tests/conftest.py` (`opencl`, `windows_memory_query`, `xvfb_and_window_manager`), whose reasons are allowlisted there; a package the app depends on is imported plainly, so a broken install fails instead of skipping; CI runs with `MICROTOPOS_STRICT_SKIPS=1`, which turns any other skip into a failure. A known defect is recorded as `pytest.mark.xfail(strict=True, reason=...)` in the tables that exist for that (`KNOWN` in `tests/test_selftest.py`, `KNOWN_BAD` in the generalisation tests), so fixing it turns the build red until the entry is removed.
* **Never loosen a tolerance, widen an `except`, add `noqa` / `type: ignore`, or narrow an exception handler to get a tool to print zero.** Fix the cause. An `except Exception` that keeps a worker thread or the GUI alive is there on purpose.
* **Never add a second way to decide something.** Before adding a threshold, a unit conversion or a guard, search for the existing one and use it (example: `live.MIN_COMPLETENESS` / `SweepCapture.usable` is the single decision whether a focus movement is a sweep).
* **Never guess a scale or a unit.** If the lateral scale, the Z unit or the camera mode is not known, the result says so (NOMINAL, frames, "not calibrated for this mode"); it does not show a plausible number.
* **Never hold a lock while a device or subprocess operation is in progress** (take it for one request/reply at most, so cancel and status calls are never blocked behind a move), and never block the GUI thread on the engine thread.
* **Never report a hardware feature as working unless it ran on that hardware.** Cameras and stage controllers are exercised in tests through fakes that model the documented protocol; the docstring and the docs say "verified against a simulated device only".

## Commands

```
uv run --group dev pytest -q -rs               # default suite; -rs prints every skip and its reason
uv run --group dev pytest -q -m slow           # full generalisation matrix
uv run --group dev ruff check microtopos tests tools # lint (project configuration)
uv run --group dev python tools/tests_bite.py --base origin/main   # do the new tests fail on the base?
uv run microtopos-selftest                     # ground-truth self-tests, three seeds
```

Set `MICROTOPOS_STRICT_SKIPS=1` to run with the CI skip policy, and `MICROTOPOS_REQUIRE_OPENCL=1` on a machine that has an OpenCL device so that the OpenCL parity tests cannot skip.

## Pull requests

Fill in the template. For every claim give the entry point that reaches it, the test that fails without it, and what was not verified. A pull request that adds a skip reason to the allowlist, a characterization mark, or an entry to a known-failure table says so in its description.
